from pathlib import Path
from typing import Iterable

import kagglehub
import librosa
import numpy as np


AUDIO_EXTENSIONS = (".flac", ".wav", ".mp3", ".ogg")


def download_datasets() -> tuple[Path, Path]:
    librispeech_root = Path(kagglehub.dataset_download("yasiashpot/librispeech"))
    urbansound_root = Path(kagglehub.dataset_download("chrisfilo/urbansound8k"))
    return librispeech_root, urbansound_root


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    peak = float(np.max(np.abs(y)))
    if peak > 1e-9:
        y = y / peak
    return y.astype(np.float32)


def list_audio_files(root: Path, extensions: Iterable[str] = AUDIO_EXTENSIONS) -> list[Path]:
    ext = tuple(e.lower() for e in extensions)
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in ext]
    if not files:
        raise FileNotFoundError(f"No audio files {ext} found under {root}")
    return files
