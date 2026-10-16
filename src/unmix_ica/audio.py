"""Audio I/O and mixing helpers for the ICA project."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_wav(path: str | Path, target_sample_rate: int | None = None) -> tuple[int, np.ndarray]:
    """Read a WAV file as mono float64 audio in [-1, 1] (approximately)."""
    sample_rate, raw = wavfile.read(str(path))
    if raw.ndim == 2:
        raw = raw.astype(np.float64).mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = max(abs(info.min), abs(info.max))
        audio = raw.astype(np.float64) / scale
    else:
        audio = raw.astype(np.float64)
    audio = np.nan_to_num(audio)
    if target_sample_rate and target_sample_rate != sample_rate:
        from math import gcd
        common = gcd(int(target_sample_rate), int(sample_rate))
        audio = resample_poly(audio, target_sample_rate // common, sample_rate // common)
        sample_rate = target_sample_rate
    return int(sample_rate), audio


def write_wav(path: str | Path, sample_rate: int, audio: np.ndarray) -> None:
    """Write floating-point audio as clipped 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float64)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.999:
        audio = audio * (0.999 / peak)
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(path), int(sample_rate), pcm)


def normalize_source(audio: np.ndarray, target_rms: float = 0.2) -> np.ndarray:
    """Remove DC and set RMS to target_rms while avoiding division by zero."""
    x = np.asarray(audio, dtype=np.float64).copy()
    x -= np.mean(x)
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-12:
        raise ValueError("Cannot normalize a silent source.")
    return x * (target_rms / rms)


def mix_sources(sources: np.ndarray, mixing_matrix: np.ndarray) -> np.ndarray:
    """Mix sources using X = A @ S; input/output arrays are channels x samples."""
    sources = np.asarray(sources, dtype=np.float64)
    A = np.asarray(mixing_matrix, dtype=np.float64)
    if sources.ndim != 2 or A.ndim != 2:
        raise ValueError("sources and mixing_matrix must both be 2-D.")
    if A.shape[1] != sources.shape[0]:
        raise ValueError("mixing_matrix columns must equal the number of sources.")
    if A.shape[0] < sources.shape[0]:
        raise ValueError("Need at least as many mixtures/channels as sources.")
    if np.linalg.matrix_rank(A) < sources.shape[0]:
        raise ValueError("mixing_matrix must have full column rank.")
    return A @ sources
