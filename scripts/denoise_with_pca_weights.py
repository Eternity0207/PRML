from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from prml_denoise.data_io import load_audio
from prml_denoise.dsp import frame_signal, overlap_add_column_major


INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
WEIGHTS_PATH = Path("pca_weights.npz")
TARGET_SR = 16_000
HOP_SIZE = 256


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


def make_video_from_audio(image_path: Path, audio_path: Path, video_path: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError(
            "ffmpeg is required to create video output. Install ffmpeg and run again."
        )

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


def main() -> None:
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights file not found: {WEIGHTS_PATH}")
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    audio_files = sorted(INPUT_DIR.glob("*.wav"))
    if not audio_files:
        raise FileNotFoundError("No .wav files found in input/")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    v_s, mean = load_pca_weights(WEIGHTS_PATH)
    frame_size = v_s.shape[0]

    if HOP_SIZE <= 0 or HOP_SIZE > frame_size:
        raise ValueError(f"HOP_SIZE must be in [1, {frame_size}]")

    print(f"Found {len(audio_files)} input files in {INPUT_DIR}")
    print(f"Using weights: {WEIGHTS_PATH}")

    for input_audio in audio_files:
        noisy = load_audio(input_audio, target_sr=TARGET_SR)
        denoised = pca_denoise_signal(noisy, v_s=v_s, mean=mean, hop_size=HOP_SIZE)

        stem = input_audio.stem
        denoised_wav = OUTPUT_DIR / f"{stem}_denoised.wav"
        denoised_video = OUTPUT_DIR / f"{stem}_denoised.mp4"

        sf.write(denoised_wav, denoised, TARGET_SR)

        with tempfile.TemporaryDirectory() as tmp:
            cover = Path(tmp) / "waveform.png"
            create_waveform_image(
                cleaned=denoised,
                sr=TARGET_SR,
                image_path=cover,
                title=f"Denoised Output (PCA) - {stem}",
            )
            make_video_from_audio(cover, denoised_wav, denoised_video)

        print(f"Done: {input_audio.name}")
        print(f"  Audio -> {denoised_wav}")
        print(f"  Video -> {denoised_video}")


if __name__ == "__main__":
    main()
