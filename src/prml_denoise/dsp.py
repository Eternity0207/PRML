import numpy as np
from pesq import pesq as pesq_score


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / max(len(noise), 1)))
        noise = np.tile(noise, repeats)
    noise = noise[: len(speech)]

    ps = float(np.mean(speech ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    alpha = float(np.sqrt(ps / (pn * 10.0 ** (snr_db / 10.0))))
    noisy = speech + alpha * noise

    peak = float(np.max(np.abs(noisy)))
    if peak > 1e-9:
        noisy = noisy / peak
    return noisy.astype(np.float32), alpha


def frame_signal(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(x) < frame_size:
        return np.zeros((frame_size, 0), dtype=np.float32)
    n_frames = 1 + (len(x) - frame_size) // hop_size
    frames = np.zeros((frame_size, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        frames[:, i] = x[start : start + frame_size]
    return frames


def frame_signal_row_major(x: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(x) < frame_size:
        return np.zeros((0, frame_size), dtype=np.float32)
    return np.asarray(
        [x[i : i + frame_size] for i in range(0, len(x) - frame_size + 1, hop_size)],
        dtype=np.float32,
    )


def overlap_add_column_major(frames: np.ndarray, hop_size: int, signal_length: int) -> np.ndarray:
    frame_size, n_frames = frames.shape
    out = np.zeros(signal_length, dtype=np.float64)
    cnt = np.zeros(signal_length, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        if end > signal_length:
            break
        out[start:end] += frames[:, i].astype(np.float64)
        cnt[start:end] += 1.0
    cnt = np.where(cnt < 1.0, 1.0, cnt)
    return (out / cnt).astype(np.float32)


def overlap_add_row_major(frames: np.ndarray, hop_size: int, signal_length: int) -> np.ndarray:
    n_frames, frame_size = frames.shape
    out = np.zeros(signal_length, dtype=np.float32)
    cnt = np.zeros(signal_length, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        if end > signal_length:
            break
        out[start:end] += frames[i]
        cnt[start:end] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return (out / cnt).astype(np.float32)


def compute_snr(clean: np.ndarray, degraded: np.ndarray) -> float:
    ps = float(np.mean(clean ** 2)) + 1e-12
    pn = float(np.mean((clean - degraded) ** 2)) + 1e-12
    return 10.0 * np.log10(ps / pn)


def compute_pesq(clean: np.ndarray, degraded: np.ndarray, sr: int) -> float:
    n = min(len(clean), len(degraded))
    try:
        return float(pesq_score(sr, clean[:n], degraded[:n], "wb"))
    except Exception:
        return float("nan")
