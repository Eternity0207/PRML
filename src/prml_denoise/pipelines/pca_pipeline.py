from __future__ import annotations

import argparse
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from prml_denoise.config import PcaConfig
from prml_denoise.data_io import AUDIO_EXTENSIONS, download_datasets, list_audio_files, load_audio
from prml_denoise.dsp import frame_signal, mix_at_snr, overlap_add_column_major
from prml_denoise.models.pca_model import PcaSpeechDenoiser

matplotlib.use("Agg")


def parse_args() -> PcaConfig:
    parser = argparse.ArgumentParser(description="Run PCA speech denoising pipeline")
    parser.add_argument("--target-sr", type=int, default=16_000)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--variance-thresh", type=float, default=0.95)
    parser.add_argument("--mix-snr-db", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pca"))
    a = parser.parse_args()
    return PcaConfig(
        target_sr=a.target_sr,
        frame_size=a.frame_size,
        hop_size=a.hop_size,
        variance_thresh=a.variance_thresh,
        mix_snr_db=a.mix_snr_db,
        num_samples=a.num_samples,
        random_seed=a.random_seed,
        output_dir=a.output_dir,
    )


def _plot_representative_spectrogram(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, sr: int, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True, constrained_layout=True)
    titles = ["Clean", "Noisy", "Denoised"]
    sigs = [clean, noisy, denoised]
    img = None
    for ax, title, sig in zip(axes, titles, sigs):
        d = librosa.stft(sig, n_fft=512, hop_length=128)
        s = librosa.amplitude_to_db(np.abs(d), ref=np.max)
        img = librosa.display.specshow(s, sr=sr, hop_length=128, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(title)
    fig.colorbar(img, ax=axes.tolist(), format="%+2.0f dB", shrink=0.8)
    plt.savefig(save_path, dpi=160)
    plt.close()


def run(config: PcaConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.random_seed)

    ls_root, us_root = download_datasets()
    speech_files = list_audio_files(ls_root, AUDIO_EXTENSIONS)
    noise_files = list_audio_files(us_root, AUDIO_EXTENSIONS)

    pair_count = min(config.num_samples, len(speech_files), len(noise_files))
    selected_speech = rng.sample(speech_files, pair_count)
    selected_noise = rng.sample(noise_files, pair_count)

    cases = []
    train_frames = []
    for idx, (sp, npth) in enumerate(zip(selected_speech, selected_noise), start=1):
        clean = load_audio(sp, config.target_sr)
        noise = load_audio(npth, config.target_sr)
        noisy, _ = mix_at_snr(clean, noise, config.mix_snr_db)

        length = min(len(clean), len(noisy))
        clean = clean[:length]
        noisy = noisy[:length]
        if length < config.frame_size:
            continue

        x = frame_signal(noisy, config.frame_size, config.hop_size)
        train_frames.append(x)
        cases.append({"id": idx, "clean": clean, "noisy": noisy, "x": x, "len": length, "sp": sp.name, "np": npth.name})

    if not cases:
        raise RuntimeError("No valid sample produced. Try reducing frame-size")

    x_train = np.concatenate(train_frames, axis=1)
    denoiser = PcaSpeechDenoiser(config.variance_thresh)
    denoiser.fit(x_train)

    weights_path = Path(__file__).resolve().parents[3] / "pca_weights.npz"
    np.savez(
        weights_path,
        V_s=denoiser.v_s.detach().cpu().numpy().astype(np.float32),
        mean=denoiser.mean_.detach().cpu().numpy().astype(np.float32),
    )

    c = cases[0]
    x_hat = denoiser.denoise(c["x"])
    den = overlap_add_column_major(x_hat, config.hop_size, c["len"])

    n = min(len(c["clean"]), len(c["noisy"]), len(den))
    representative = {
        "clean": c["clean"][:n],
        "noisy": c["noisy"][:n],
        "den": den[:n],
    }

    if representative is None:
        raise RuntimeError("Representative sample unavailable")

    rep_dir = config.output_dir / "representative_case"
    rep_dir.mkdir(parents=True, exist_ok=True)
    sf.write(rep_dir / "01_clean.wav", representative["clean"], config.target_sr)
    sf.write(rep_dir / "02_noisy.wav", representative["noisy"], config.target_sr)
    sf.write(rep_dir / "03_denoised_pca.wav", representative["den"], config.target_sr)

    denoiser.save_scree_plot(config.output_dir / "scree_plot.png")
    _plot_representative_spectrogram(
        representative["clean"], representative["noisy"], representative["den"], config.target_sr, config.output_dir / "representative_spectrogram.png"
    )

    print(f"Saved PCA weights: {weights_path}")


if __name__ == "__main__":
    run(parse_args())
