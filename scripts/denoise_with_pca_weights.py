from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from prml_denoise.data_io import AUDIO_EXTENSIONS, list_audio_files, load_audio
from prml_denoise.dsp import (
    compute_pesq,
    compute_snr,
    frame_signal,
    frame_signal_row_major,
    mix_at_snr,
    overlap_add_column_major,
    overlap_add_row_major,
)
from prml_denoise.models.resunet import ResUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create noisy mixtures from signal+noise pairs, denoise with PCA weights, and compare metrics."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--signal-subdir", type=str, default="signal")
    parser.add_argument("--noise-subdir", type=str, default="noise")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--weights-path", type=Path, default=Path("pca_weights.npz"))
    parser.add_argument("--target-sr", type=int, default=16_000)
    parser.add_argument("--hop-size", type=int, default=256)
    parser.add_argument("--mix-snr-db", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means use all available pairs")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--compare-resunet", action="store_true", help="Compare PCA denoising against ResUNet denoising")
    parser.add_argument("--resunet-weights-path", type=Path, default=Path("outputs/resunet/best_resunet.pt"))
    parser.add_argument("--resunet-frame-size", type=int, default=512)
    parser.add_argument("--resunet-hop-size", type=int, default=256)
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 generation")
    return parser.parse_args()


def resolve_ffmpeg_binary() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is not None:
        return ffmpeg_bin

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg executable not found in PATH.\n"
            "Note: 'pip install ffmpeg' does not install the ffmpeg command-line binary.\n"
            "Fix one of the following and run again:\n"
            "  1) brew install ffmpeg\n"
            "  2) pip install imageio-ffmpeg"
        ) from exc

    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    if not ffmpeg_bin or not Path(ffmpeg_bin).exists():
        raise RuntimeError(
            "Unable to resolve ffmpeg binary. Install ffmpeg with:\n"
            "  brew install ffmpeg"
        )
    return ffmpeg_bin


def load_pca_weights(weights_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(weights_path, allow_pickle=True)
    if "V_s" not in data or "mean" not in data:
        raise ValueError("pca_weights.npz must contain keys: 'V_s' and 'mean'")

    v_s = data["V_s"].astype(np.float32)
    mean = data["mean"].astype(np.float32)

    if mean.ndim == 1:
        mean = mean[:, None]

    if v_s.ndim != 2 or mean.ndim != 2:
        raise ValueError("Invalid PCA weights shape")

    if mean.shape[1] != 1:
        raise ValueError("Expected mean shape (frame_size, 1)")

    if v_s.shape[0] != mean.shape[0]:
        raise ValueError("Mismatch between V_s rows and mean rows")

    return v_s, mean


def pca_denoise_signal(x: np.ndarray, v_s: np.ndarray, mean: np.ndarray, hop_size: int) -> np.ndarray:
    frame_size = v_s.shape[0]
    original_len = len(x)

    if original_len < frame_size:
        pad = frame_size - original_len
        x = np.pad(x, (0, pad), mode="constant")

    x_frames = frame_signal(x, frame_size=frame_size, hop_size=hop_size)
    x_centered = x_frames - mean
    x_hat = v_s @ (v_s.T @ x_centered) + mean

    denoised = overlap_add_column_major(x_hat, hop_size=hop_size, signal_length=len(x))
    denoised = denoised[:original_len]

    peak = float(np.max(np.abs(denoised)))
    if peak > 1e-9:
        denoised = denoised / peak

    return denoised.astype(np.float32)


def load_resunet_model(weights_path: Path) -> tuple[ResUNet, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResUNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def resunet_denoise_signal(
    x: np.ndarray,
    model: ResUNet,
    device: torch.device,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    original_len = len(x)
    if original_len < frame_size:
        x = np.pad(x, (0, frame_size - original_len), mode="constant")

    frames = frame_signal_row_major(x, frame_size=frame_size, hop_size=hop_size)
    if len(frames) == 0:
        return np.zeros(original_len, dtype=np.float32)

    with torch.no_grad():
        x_tensor = torch.tensor(frames, dtype=torch.float32).unsqueeze(1).to(device)
        pred_frames = model(x_tensor).squeeze(1).cpu().numpy()

    denoised = overlap_add_row_major(pred_frames, hop_size=hop_size, signal_length=len(x))
    denoised = denoised[:original_len]

    peak = float(np.max(np.abs(denoised)))
    if peak > 1e-9:
        denoised = denoised / peak

    return denoised.astype(np.float32)


def create_waveform_image(cleaned: np.ndarray, sr: int, image_path: Path, title: str) -> None:
    t = np.arange(len(cleaned)) / float(sr)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.plot(t, cleaned, color="#00E5FF", linewidth=0.8)
    ax.set_title(title, color="white", fontsize=18, pad=12)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Amplitude", color="white")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.2, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close(fig)


def make_video_from_audio(ffmpeg_bin: str, image_path: Path, audio_path: Path, video_path: Path) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-i",
        str(audio_path),
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)


def collect_signal_noise_pairs(input_dir: Path, signal_subdir: str, noise_subdir: str) -> list[tuple[Path, Path]]:
    signal_dir = input_dir / signal_subdir
    noise_dir = input_dir / noise_subdir

    if not signal_dir.is_dir():
        raise FileNotFoundError(f"Signal directory not found: {signal_dir}")
    if not noise_dir.is_dir():
        raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

    signal_files = list_audio_files(signal_dir, AUDIO_EXTENSIONS)
    noise_files = list_audio_files(noise_dir, AUDIO_EXTENSIONS)

    if len(signal_files) != len(noise_files):
        raise ValueError(
            f"Signal/noise file count mismatch: {len(signal_files)} signal files vs {len(noise_files)} noise files"
        )

    return list(zip(signal_files, noise_files))


def select_pairs(pairs: list[tuple[Path, Path]], num_samples: int, random_seed: int) -> list[tuple[Path, Path]]:
    if num_samples <= 0 or num_samples >= len(pairs):
        return pairs
    rng = random.Random(random_seed)
    return rng.sample(pairs, num_samples)


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"\s+", "_", path.stem.strip())
    stem = re.sub(r"[^A-Za-z0-9._-]", "", stem)
    return stem or "sample"


def print_case_metrics(
    case: int,
    signal_name: str,
    noise_name: str,
    snr_noisy: float,
    snr_denoised: float,
    pesq_noisy: float,
    pesq_denoised: float,
    snr_resunet: float | None = None,
    pesq_resunet: float | None = None,
) -> None:
    print("\n" + "=" * 72)
    print(f"Case {case:02d}: signal={signal_name} | noise={noise_name}")
    print("-" * 72)
    print(f"SNR  noisy    : {snr_noisy:8.3f} dB")
    print(f"SNR  pca      : {snr_denoised:8.3f} dB")
    print(f"dSNR (pca)    : {snr_denoised - snr_noisy:+8.3f} dB")
    if snr_resunet is not None:
        print(f"SNR  resunet  : {snr_resunet:8.3f} dB")
        print(f"dSNR (resunet): {snr_resunet - snr_noisy:+8.3f} dB")
    print(f"PESQ noisy    : {pesq_noisy:8.3f}")
    print(f"PESQ pca      : {pesq_denoised:8.3f}")
    print(f"dPESQ (pca)   : {pesq_denoised - pesq_noisy:+8.3f}")
    if pesq_resunet is not None:
        print(f"PESQ resunet  : {pesq_resunet:8.3f}")
        print(f"dPESQ (resunet): {pesq_resunet - pesq_noisy:+8.3f}")


def summarize_metrics(rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return

    def mean_std(values: np.ndarray) -> tuple[float, float]:
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return float("nan"), float("nan")
        return float(np.mean(valid)), float(np.std(valid))

    include_resunet = "snr_resunet" in rows[0]

    snr_noisy = np.array([float(r["snr_noisy"]) for r in rows], dtype=np.float64)
    snr_denoised = np.array([float(r["snr_denoised"]) for r in rows], dtype=np.float64)
    pesq_noisy = np.array([float(r["pesq_noisy"]) for r in rows], dtype=np.float64)
    pesq_denoised = np.array([float(r["pesq_denoised"]) for r in rows], dtype=np.float64)

    snr_n_mu, snr_n_std = mean_std(snr_noisy)
    snr_d_mu, snr_d_std = mean_std(snr_denoised)
    pesq_n_mu, pesq_n_std = mean_std(pesq_noisy)
    pesq_d_mu, pesq_d_std = mean_std(pesq_denoised)

    print("\n" + "=" * 72)
    print("AGGREGATE METRICS")
    print("=" * 72)
    print(f"SNR  noisy    : {snr_n_mu:8.3f} +/- {snr_n_std:8.3f} dB")
    print(f"SNR  pca      : {snr_d_mu:8.3f} +/- {snr_d_std:8.3f} dB")
    print(f"dSNR (pca)    : {(snr_d_mu - snr_n_mu):8.3f} dB")
    print(f"PESQ noisy    : {pesq_n_mu:8.3f} +/- {pesq_n_std:8.3f}")
    print(f"PESQ pca      : {pesq_d_mu:8.3f} +/- {pesq_d_std:8.3f}")
    print(f"dPESQ (pca)   : {(pesq_d_mu - pesq_n_mu):8.3f}")

    if include_resunet:
        snr_resunet = np.array([float(r["snr_resunet"]) for r in rows], dtype=np.float64)
        pesq_resunet = np.array([float(r["pesq_resunet"]) for r in rows], dtype=np.float64)
        snr_r_mu, snr_r_std = mean_std(snr_resunet)
        pesq_r_mu, pesq_r_std = mean_std(pesq_resunet)
        print(f"SNR  resunet  : {snr_r_mu:8.3f} +/- {snr_r_std:8.3f} dB")
        print(f"dSNR (resunet): {(snr_r_mu - snr_n_mu):8.3f} dB")
        print(f"PESQ resunet  : {pesq_r_mu:8.3f} +/- {pesq_r_std:8.3f}")
        print(f"dPESQ (resunet): {(pesq_r_mu - pesq_n_mu):8.3f}")

    print("=" * 72)


def _to_float_array(rows: list[dict[str, float | str]], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=np.float64)


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def save_aggregate_metrics(rows: list[dict[str, float | str]], save_path: Path) -> None:
    include_resunet = "snr_resunet" in rows[0]

    snr_noisy = _to_float_array(rows, "snr_noisy")
    snr_denoised = _to_float_array(rows, "snr_denoised")
    delta_snr = _to_float_array(rows, "delta_snr")
    pesq_noisy = _to_float_array(rows, "pesq_noisy")
    pesq_denoised = _to_float_array(rows, "pesq_denoised")
    delta_pesq = _to_float_array(rows, "delta_pesq")

    snr_n_stats = _finite_stats(snr_noisy)
    snr_d_stats = _finite_stats(snr_denoised)
    dsnr_stats = _finite_stats(delta_snr)
    pesq_n_stats = _finite_stats(pesq_noisy)
    pesq_d_stats = _finite_stats(pesq_denoised)
    dpesq_stats = _finite_stats(delta_pesq)

    row = {
        "num_cases": len(rows),
        "snr_noisy_mean": snr_n_stats["mean"],
        "snr_noisy_std": snr_n_stats["std"],
        "snr_noisy_median": snr_n_stats["median"],
        "snr_noisy_min": snr_n_stats["min"],
        "snr_noisy_max": snr_n_stats["max"],
        "snr_denoised_mean": snr_d_stats["mean"],
        "snr_denoised_std": snr_d_stats["std"],
        "snr_denoised_median": snr_d_stats["median"],
        "snr_denoised_min": snr_d_stats["min"],
        "snr_denoised_max": snr_d_stats["max"],
        "delta_snr_mean": dsnr_stats["mean"],
        "delta_snr_std": dsnr_stats["std"],
        "delta_snr_median": dsnr_stats["median"],
        "delta_snr_min": dsnr_stats["min"],
        "delta_snr_max": dsnr_stats["max"],
        "snr_improved_cases": int(np.sum(delta_snr > 0.0)),
        "pesq_noisy_mean": pesq_n_stats["mean"],
        "pesq_noisy_std": pesq_n_stats["std"],
        "pesq_noisy_median": pesq_n_stats["median"],
        "pesq_noisy_min": pesq_n_stats["min"],
        "pesq_noisy_max": pesq_n_stats["max"],
        "pesq_denoised_mean": pesq_d_stats["mean"],
        "pesq_denoised_std": pesq_d_stats["std"],
        "pesq_denoised_median": pesq_d_stats["median"],
        "pesq_denoised_min": pesq_d_stats["min"],
        "pesq_denoised_max": pesq_d_stats["max"],
        "delta_pesq_mean": dpesq_stats["mean"],
        "delta_pesq_std": dpesq_stats["std"],
        "delta_pesq_median": dpesq_stats["median"],
        "delta_pesq_min": dpesq_stats["min"],
        "delta_pesq_max": dpesq_stats["max"],
        "pesq_improved_cases": int(np.sum(delta_pesq > 0.0)),
    }

    if include_resunet:
        snr_resunet = _to_float_array(rows, "snr_resunet")
        pesq_resunet = _to_float_array(rows, "pesq_resunet")
        delta_snr_resunet = _to_float_array(rows, "delta_snr_resunet")
        delta_pesq_resunet = _to_float_array(rows, "delta_pesq_resunet")

        snr_r_stats = _finite_stats(snr_resunet)
        pesq_r_stats = _finite_stats(pesq_resunet)
        dsnr_r_stats = _finite_stats(delta_snr_resunet)
        dpesq_r_stats = _finite_stats(delta_pesq_resunet)

        row.update(
            {
                "snr_resunet_mean": snr_r_stats["mean"],
                "snr_resunet_std": snr_r_stats["std"],
                "snr_resunet_median": snr_r_stats["median"],
                "snr_resunet_min": snr_r_stats["min"],
                "snr_resunet_max": snr_r_stats["max"],
                "delta_snr_resunet_mean": dsnr_r_stats["mean"],
                "delta_snr_resunet_std": dsnr_r_stats["std"],
                "delta_snr_resunet_median": dsnr_r_stats["median"],
                "delta_snr_resunet_min": dsnr_r_stats["min"],
                "delta_snr_resunet_max": dsnr_r_stats["max"],
                "snr_resunet_improved_cases": int(np.sum(delta_snr_resunet > 0.0)),
                "pesq_resunet_mean": pesq_r_stats["mean"],
                "pesq_resunet_std": pesq_r_stats["std"],
                "pesq_resunet_median": pesq_r_stats["median"],
                "pesq_resunet_min": pesq_r_stats["min"],
                "pesq_resunet_max": pesq_r_stats["max"],
                "delta_pesq_resunet_mean": dpesq_r_stats["mean"],
                "delta_pesq_resunet_std": dpesq_r_stats["std"],
                "delta_pesq_resunet_median": dpesq_r_stats["median"],
                "delta_pesq_resunet_min": dpesq_r_stats["min"],
                "delta_pesq_resunet_max": dpesq_r_stats["max"],
                "pesq_resunet_improved_cases": int(np.sum(delta_pesq_resunet > 0.0)),
            }
        )

    with save_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_metric_plots(rows: list[dict[str, float | str]], plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    include_resunet = "snr_resunet" in rows[0]
    case_idx = np.arange(1, len(rows) + 1)

    snr_noisy = _to_float_array(rows, "snr_noisy")
    snr_denoised = _to_float_array(rows, "snr_denoised")
    pesq_noisy = _to_float_array(rows, "pesq_noisy")
    pesq_denoised = _to_float_array(rows, "pesq_denoised")
    delta_snr = _to_float_array(rows, "delta_snr")
    delta_pesq = _to_float_array(rows, "delta_pesq")

    if include_resunet:
        snr_resunet = _to_float_array(rows, "snr_resunet")
        pesq_resunet = _to_float_array(rows, "pesq_resunet")
        delta_snr_resunet = _to_float_array(rows, "delta_snr_resunet")
        delta_pesq_resunet = _to_float_array(rows, "delta_pesq_resunet")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(case_idx, snr_noisy, "o-", label="Noisy", color="#D62728")
    ax.plot(case_idx, snr_denoised, "o-", label="PCA", color="#1F77B4")
    if include_resunet:
        ax.plot(case_idx, snr_resunet, "o-", label="ResUNet", color="#FF7F0E")
    ax.set_title("SNR Per Case")
    ax.set_xlabel("Case")
    ax.set_ylabel("SNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "snr_per_case.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(case_idx, pesq_noisy, "o-", label="Noisy", color="#D62728")
    ax.plot(case_idx, pesq_denoised, "o-", label="PCA", color="#1F77B4")
    if include_resunet:
        ax.plot(case_idx, pesq_resunet, "o-", label="ResUNet", color="#FF7F0E")
    ax.set_title("PESQ Per Case")
    ax.set_xlabel("Case")
    ax.set_ylabel("PESQ")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "pesq_per_case.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    snr_colors = np.where(delta_snr >= 0.0, "#2CA02C", "#D62728")
    pesq_colors = np.where(delta_pesq >= 0.0, "#2CA02C", "#D62728")

    if include_resunet:
        width = 0.38
        axes[0].bar(case_idx - width / 2, delta_snr, width=width, color="#1F77B4", label="PCA")
        axes[0].bar(case_idx + width / 2, delta_snr_resunet, width=width, color="#FF7F0E", label="ResUNet")
        axes[0].legend()
    else:
        axes[0].bar(case_idx, delta_snr, color=snr_colors)
    axes[0].axhline(0.0, color="black", linewidth=1.0)
    axes[0].set_title("Delta SNR Per Case (Model - Noisy)")
    axes[0].set_ylabel("dSNR (dB)")
    axes[0].grid(alpha=0.3, axis="y")

    if include_resunet:
        width = 0.38
        axes[1].bar(case_idx - width / 2, delta_pesq, width=width, color="#1F77B4", label="PCA")
        axes[1].bar(case_idx + width / 2, delta_pesq_resunet, width=width, color="#FF7F0E", label="ResUNet")
        axes[1].legend()
    else:
        axes[1].bar(case_idx, delta_pesq, color=pesq_colors)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_title("Delta PESQ Per Case (Model - Noisy)")
    axes[1].set_xlabel("Case")
    axes[1].set_ylabel("dPESQ")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "metric_deltas_per_case.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights_path}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    all_pairs = collect_signal_noise_pairs(args.input_dir, args.signal_subdir, args.noise_subdir)
    pairs = select_pairs(all_pairs, args.num_samples, args.random_seed)
    if not pairs:
        raise RuntimeError("No signal/noise pairs found in the input folders.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    v_s, mean = load_pca_weights(args.weights_path)
    frame_size = v_s.shape[0]

    if args.hop_size <= 0 or args.hop_size > frame_size:
        raise ValueError(f"hop-size must be in [1, {frame_size}]")

    ffmpeg_bin = None
    if not args.skip_video:
        ffmpeg_bin = resolve_ffmpeg_binary()

    resunet_model = None
    resunet_device = None
    if args.compare_resunet:
        if not args.resunet_weights_path.exists():
            raise FileNotFoundError(f"ResUNet checkpoint not found: {args.resunet_weights_path}")
        if args.resunet_frame_size <= 0:
            raise ValueError("resunet-frame-size must be positive")
        if args.resunet_hop_size <= 0 or args.resunet_hop_size > args.resunet_frame_size:
            raise ValueError(f"resunet-hop-size must be in [1, {args.resunet_frame_size}]")
        resunet_model, resunet_device = load_resunet_model(args.resunet_weights_path)

    print(f"Using weights: {args.weights_path}")
    print(f"Input signal/noise pairs: {len(pairs)}")
    print(f"Target SNR for mixing: {args.mix_snr_db} dB")
    if args.compare_resunet:
        print(f"Comparing with ResUNet checkpoint: {args.resunet_weights_path}")

    metrics_rows: list[dict[str, float | str]] = []

    for idx, (signal_path, noise_path) in enumerate(pairs, start=1):
        clean = load_audio(signal_path, target_sr=args.target_sr)
        noise = load_audio(noise_path, target_sr=args.target_sr)
        noisy, alpha = mix_at_snr(clean, noise, args.mix_snr_db)
        denoised = pca_denoise_signal(noisy, v_s=v_s, mean=mean, hop_size=args.hop_size)

        denoised_resunet = None
        if resunet_model is not None and resunet_device is not None:
            denoised_resunet = resunet_denoise_signal(
                noisy,
                model=resunet_model,
                device=resunet_device,
                frame_size=args.resunet_frame_size,
                hop_size=args.resunet_hop_size,
            )

        if denoised_resunet is None:
            n = min(len(clean), len(noisy), len(denoised))
        else:
            n = min(len(clean), len(noisy), len(denoised), len(denoised_resunet))

        clean = clean[:n]
        noisy = noisy[:n]
        denoised = denoised[:n]
        if denoised_resunet is not None:
            denoised_resunet = denoised_resunet[:n]

        snr_noisy = compute_snr(clean, noisy)
        snr_denoised = compute_snr(clean, denoised)
        pesq_noisy = compute_pesq(clean, noisy, args.target_sr)
        pesq_denoised = compute_pesq(clean, denoised, args.target_sr)

        snr_resunet = None
        pesq_resunet = None
        if denoised_resunet is not None:
            snr_resunet = compute_snr(clean, denoised_resunet)
            pesq_resunet = compute_pesq(clean, denoised_resunet, args.target_sr)

        print_case_metrics(
            case=idx,
            signal_name=signal_path.name,
            noise_name=noise_path.name,
            snr_noisy=snr_noisy,
            snr_denoised=snr_denoised,
            pesq_noisy=pesq_noisy,
            pesq_denoised=pesq_denoised,
            snr_resunet=snr_resunet,
            pesq_resunet=pesq_resunet,
        )

        stem = f"case_{idx:02d}_{_safe_stem(signal_path)}"
        clean_wav = args.output_dir / f"{stem}_signal.wav"
        noisy_wav = args.output_dir / f"{stem}_noisy.wav"
        denoised_wav = args.output_dir / f"{stem}_denoised.wav"
        denoised_resunet_wav = args.output_dir / f"{stem}_resunet_denoised.wav"
        denoised_video = args.output_dir / f"{stem}_denoised.mp4"

        sf.write(clean_wav, clean, args.target_sr)
        sf.write(noisy_wav, noisy, args.target_sr)
        sf.write(denoised_wav, denoised, args.target_sr)
        if denoised_resunet is not None:
            sf.write(denoised_resunet_wav, denoised_resunet, args.target_sr)

        if ffmpeg_bin is not None:
            with tempfile.TemporaryDirectory() as tmp:
                cover = Path(tmp) / "waveform.png"
                create_waveform_image(
                    cleaned=denoised,
                    sr=args.target_sr,
                    image_path=cover,
                    title=f"Denoised Output (PCA) - {stem}",
                )
                make_video_from_audio(ffmpeg_bin, cover, denoised_wav, denoised_video)

        row: dict[str, float | str] = {
            "case": idx,
            "signal_file": signal_path.name,
            "noise_file": noise_path.name,
            "alpha": alpha,
            "snr_noisy": snr_noisy,
            "snr_denoised": snr_denoised,
            "delta_snr": snr_denoised - snr_noisy,
            "pesq_noisy": pesq_noisy,
            "pesq_denoised": pesq_denoised,
            "delta_pesq": pesq_denoised - pesq_noisy,
        }
        if snr_resunet is not None and pesq_resunet is not None:
            row["snr_resunet"] = snr_resunet
            row["delta_snr_resunet"] = snr_resunet - snr_noisy
            row["pesq_resunet"] = pesq_resunet
            row["delta_pesq_resunet"] = pesq_resunet - pesq_noisy
        metrics_rows.append(row)

        if denoised_resunet is not None:
            print(f"Saved: {clean_wav.name}, {noisy_wav.name}, {denoised_wav.name}, {denoised_resunet_wav.name}")
        else:
            print(f"Saved: {clean_wav.name}, {noisy_wav.name}, {denoised_wav.name}")

    metrics_path = args.output_dir / "metrics_summary.csv"
    with metrics_path.open("w", newline="") as f:
        fieldnames = [
            "case",
            "signal_file",
            "noise_file",
            "alpha",
            "snr_noisy",
            "snr_denoised",
            "delta_snr",
            "pesq_noisy",
            "pesq_denoised",
            "delta_pesq",
        ]
        if args.compare_resunet:
            fieldnames += [
                "snr_resunet",
                "delta_snr_resunet",
                "pesq_resunet",
                "delta_pesq_resunet",
            ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    aggregate_metrics_path = args.output_dir / "metrics_aggregate.csv"
    save_aggregate_metrics(metrics_rows, aggregate_metrics_path)
    plots_dir = args.output_dir / "plots"
    save_metric_plots(metrics_rows, plots_dir)

    summarize_metrics(metrics_rows)
    print(f"\nMetrics CSV: {metrics_path}")
    print(f"Aggregate metrics CSV: {aggregate_metrics_path}")
    print(f"Metric plots: {plots_dir}")


if __name__ == "__main__":
    main()
