from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.utils.data

from prml_denoise.config import ResUNetConfig
from prml_denoise.data_io import download_datasets, load_audio
from prml_denoise.dsp import frame_signal_row_major, mix_at_snr, overlap_add_row_major
from prml_denoise.models.resunet import DenoiseLoss, ResUNet


def parse_args() -> ResUNetConfig:
    parser = argparse.ArgumentParser(description="Run ResUNet speech denoising pipeline")
    parser.add_argument("--target-sr", type=int, default=16_000)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--mix-snr-db", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/resunet"))
    a = parser.parse_args()
    return ResUNetConfig(
        target_sr=a.target_sr,
        frame_size=a.frame_size,
        hop_size=a.hop_size,
        mix_snr_db=a.mix_snr_db,
        num_samples=a.num_samples,
        random_seed=a.random_seed,
        batch_size=a.batch_size,
        epochs=a.epochs,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        train_ratio=a.train_ratio,
        output_dir=a.output_dir,
    )


def build_frame_tensors(pairs: list[dict], frame_size: int, hop_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    noisy_list = []
    clean_list = []
    for p in pairs:
        c = frame_signal_row_major(p["clean"], frame_size, hop_size)
        n = frame_signal_row_major(p["noisy"], frame_size, hop_size)
        if len(c) == 0 or len(n) == 0:
            continue
        clean_list.append(c)
        noisy_list.append(n)

    if not noisy_list:
        raise RuntimeError("No valid frames created")

    x = torch.tensor(np.concatenate(noisy_list), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.concatenate(clean_list), dtype=torch.float32).unsqueeze(1)
    return x, y


def run_epoch(model, loader, loss_fn, optimizer, device, train_mode: bool):
    model.train(train_mode)
    total = 0.0
    total_wav = 0.0
    total_spec = 0.0
    batches = 0

    grad_context = torch.enable_grad() if train_mode else torch.no_grad()
    with grad_context:
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            if train_mode:
                optimizer.zero_grad()

            pred = model(bx)
            loss, wav_l, spec_l = loss_fn(pred, by)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total += float(loss.item())
            total_wav += float(wav_l.item())
            total_spec += float(spec_l.item())
            batches += 1

    batches = max(1, batches)
    return {
        "total": total / batches,
        "wav": total_wav / batches,
        "spec": total_spec / batches,
    }


def run(config: ResUNetConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ls_root, us_root = download_datasets()

    speech_files = sorted(ls_root.rglob("*.flac"))[: config.num_samples]
    noise_files = sorted(us_root.rglob("*.wav"))[: config.num_samples]
    pair_count = min(len(speech_files), len(noise_files))
    if pair_count < 2:
        raise RuntimeError("Need at least 2 speech/noise pairs")

    pairs = []
    for i, (sp, npth) in enumerate(zip(speech_files, noise_files), start=1):
        clean = load_audio(sp, config.target_sr)
        noise = load_audio(npth, config.target_sr)
        noisy, _ = mix_at_snr(clean, noise, config.mix_snr_db)
        pairs.append({"name": f"case_{i:02d}", "clean": clean, "noisy": noisy})

    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * config.train_ratio))
    split_idx = min(split_idx, len(pairs) - 1)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    x_train, y_train = build_frame_tensors(train_pairs, config.frame_size, config.hop_size)
    x_val, y_val = build_frame_tensors(val_pairs, config.frame_size, config.hop_size)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_val, y_val), batch_size=config.batch_size, shuffle=False
    )

    model = ResUNet().to(device)
    loss_fn = DenoiseLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = float("inf")
    for _ in range(config.epochs):
        run_epoch(model, train_loader, loss_fn, optimizer, device, train_mode=True)
        val_m = run_epoch(model, val_loader, loss_fn, optimizer, device, train_mode=False)
        if val_m["total"] < best_val:
            best_val = val_m["total"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, config.output_dir / "best_resunet.pt")

    model.eval()
    with torch.no_grad():
        for p in pairs:
            n_frames = frame_signal_row_major(p["noisy"], config.frame_size, config.hop_size)
            x = torch.tensor(n_frames, dtype=torch.float32).unsqueeze(1).to(device)
            pred_frames = model(x).squeeze(1).cpu().numpy()
            den = overlap_add_row_major(pred_frames, config.hop_size, len(p["clean"]))

            sf.write(config.output_dir / f"{p['name']}_clean.wav", p["clean"], config.target_sr)
            sf.write(config.output_dir / f"{p['name']}_noisy.wav", p["noisy"], config.target_sr)
            sf.write(config.output_dir / f"{p['name']}_denoised.wav", den, config.target_sr)


if __name__ == "__main__":
    run(parse_args())
