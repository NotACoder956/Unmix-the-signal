"""Real-world multidevice recording workflow for the Unmix the Signal project.

This module handles recordings captured independently by multiple phones or
microphones. It aligns their starts by cross-correlation, runs the project's
own FastICA implementation, saves unlabeled component audio, and writes
quality diagnostics. It never invents reference-based metrics when clean
source recordings are unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly
from math import gcd

from .audio import read_wav, write_wav
from .ica import FastICA


def _resample(audio: np.ndarray, old_rate: int, new_rate: int) -> np.ndarray:
    if old_rate == new_rate:
        return np.asarray(audio, dtype=np.float64)
    common = gcd(int(old_rate), int(new_rate))
    return resample_poly(audio, new_rate // common, old_rate // common).astype(np.float64)


def estimate_relative_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    max_lag_seconds: float = 3.0,
    sync_window_seconds: float = 12.0,
    alignment_rate: int = 1000,
) -> int:
    """Estimate candidate's signal delay relative to reference in target-rate samples.

    A positive return means the shared sound appears later in candidate than in
    reference. A sharp clap near the beginning helps. Signals are downsampled
    for alignment so the correlation is inexpensive.
    """
    if max_lag_seconds < 0 or sync_window_seconds <= 0:
        raise ValueError("Lag must be non-negative and sync window must be positive.")
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if len(ref) < 2 or len(cand) < 2:
        raise ValueError("Each recording must contain at least two samples.")

    # Downsample the same initial section for fast coarse alignment.
    ref = ref[: min(len(ref), int(sync_window_seconds * sample_rate))]
    cand = cand[: min(len(cand), int(sync_window_seconds * sample_rate))]
    ref_small = _resample(ref, sample_rate, alignment_rate)
    cand_small = _resample(cand, sample_rate, alignment_rate)
    common_length = min(len(ref_small), len(cand_small))
    ref_small = ref_small[:common_length]
    cand_small = cand_small[:common_length]
    ref_small = ref_small - np.mean(ref_small)
    cand_small = cand_small - np.mean(cand_small)
    ref_scale = float(np.std(ref_small))
    cand_scale = float(np.std(cand_small))
    if ref_scale < 1e-10 or cand_scale < 1e-10:
        raise ValueError("A recording is silent or nearly silent; cannot synchronize it.")
    ref_small /= ref_scale
    cand_small /= cand_scale

    corr = correlate(cand_small, ref_small, mode="full", method="fft")
    lags = correlation_lags(len(cand_small), len(ref_small), mode="full")
    max_lag = int(round(max_lag_seconds * alignment_rate))
    allowed = np.abs(lags) <= max_lag
    if not np.any(allowed):
        raise ValueError("No lags fall inside the requested search window.")
    local = np.argmax(np.abs(corr[allowed]))
    lag_small = int(lags[allowed][local])
    return int(round(lag_small * sample_rate / alignment_rate))


def align_recordings(
    recordings: Sequence[np.ndarray],
    sample_rate: int,
    max_lag_seconds: float = 3.0,
    sync_window_seconds: float = 12.0,
    auto_align: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Align recordings and return a (devices, samples) matrix and lags in samples.

    Relative lags are estimated against the first recording. The routine keeps
    only the overlapping section after alignment; it does not stretch audio to
    compensate for hardware sample-clock drift.
    """
    if len(recordings) < 2:
        raise ValueError("ICA requires at least two independent mixture recordings.")
    arrays = [np.asarray(x, dtype=np.float64).reshape(-1) for x in recordings]
    if any(len(x) < 2 for x in arrays):
        raise ValueError("Each recording must contain at least two samples.")
    if any(not np.all(np.isfinite(x)) for x in arrays):
        raise ValueError("A recording contains NaN or infinite samples.")

    if auto_align:
        lags = [0] + [
            estimate_relative_lag(
                arrays[0], x, sample_rate,
                max_lag_seconds=max_lag_seconds,
                sync_window_seconds=sync_window_seconds,
            )
            for x in arrays[1:]
        ]
    else:
        lags = [0] * len(arrays)

    # If lag_i > lag_min, remove that many samples from the start of recording i.
    # This aligns positive and negative offsets while retaining common overlap.
    min_lag = min(lags)
    start_indices = [lag - min_lag for lag in lags]
    lengths = [len(x) - start for x, start in zip(arrays, start_indices)]
    common_length = min(lengths)
    if common_length < max(2, int(0.25 * sample_rate)):
        raise ValueError(
            "Too little overlapping audio remains after synchronization. "
            "Check the recordings and increase --max-lag-seconds if needed."
        )
    aligned = np.vstack([
        x[start:start + common_length]
        for x, start in zip(arrays, start_indices)
    ])
    aligned -= np.mean(aligned, axis=1, keepdims=True)
    if np.any(np.std(aligned, axis=1) < 1e-10):
        raise ValueError("At least one aligned device recording is silent.")
    return aligned, lags


def _correlation_matrix(x: np.ndarray) -> np.ndarray:
    return np.corrcoef(x)


def _excess_kurtosis(x: np.ndarray) -> list[float]:
    centered = x - np.mean(x, axis=1, keepdims=True)
    std = np.std(centered, axis=1, keepdims=True)
    z = centered / np.maximum(std, 1e-12)
    return [float(np.mean(row**4) - 3.0) for row in z]


def _plot_waveforms(mixed: np.ndarray, components: np.ndarray, sample_rate: int, output_dir: Path) -> None:
    duration = min(3.0, mixed.shape[1] / sample_rate)
    n = max(1, int(duration * sample_rate))
    t = np.arange(n) / sample_rate
    fig, axes = plt.subplots(mixed.shape[0], 1, figsize=(11, max(4, 2.0 * mixed.shape[0])), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(t, mixed[i, :n], linewidth=0.65)
        ax.set_ylabel(f"Device {i+1}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Real-world recordings after synchronization")
    fig.tight_layout()
    fig.savefig(output_dir / "01_device_recordings.png", dpi=150)
    plt.close(fig)

    duration = min(3.0, components.shape[1] / sample_rate)
    n = max(1, int(duration * sample_rate))
    t = np.arange(n) / sample_rate
    fig, axes = plt.subplots(components.shape[0], 1, figsize=(11, max(4, 2.0 * components.shape[0])), sharex=True)
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.plot(t, components[i, :n], linewidth=0.65)
        ax.set_ylabel(f"Component {i+1}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("ICA component estimates (order and scale are arbitrary)")
    fig.tight_layout()
    fig.savefig(output_dir / "02_ica_components.png", dpi=150)
    plt.close(fig)

    corr_in = _correlation_matrix(mixed)
    corr_out = _correlation_matrix(components)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, corr, title in zip(axes, [corr_in, corr_out], ["Input channel correlations", "Output component correlations"]):
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")
        ax.set_xticks(range(corr.shape[0]))
        ax.set_yticks(range(corr.shape[0]))
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Pearson correlation")
    fig.suptitle("Correlation diagnostics (independence is stronger than zero correlation)")
    fig.savefig(output_dir / "03_correlation_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_real_world_pipeline(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    target_sample_rate: int = 16000,
    max_lag_seconds: float = 3.0,
    sync_window_seconds: float = 12.0,
    auto_align: bool = True,
    max_iter: int = 500,
    random_state: int = 42,
) -> dict:
    """Load multi-device recordings, synchronize, run ICA, and save diagnostics."""
    paths = [Path(p) for p in input_paths]
    if len(paths) < 2:
        raise ValueError("Provide at least two device recordings; three or more is preferred.")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing input recordings: " + ", ".join(missing))

    recordings: list[np.ndarray] = []
    source_rates: list[int] = []
    source_lengths_native: list[int] = []
    source_lengths_resampled: list[int] = []
    for path in paths:
        original_rate, audio = read_wav(path)
        source_rates.append(original_rate)
        source_lengths_native.append(len(audio))
        audio = _resample(audio, original_rate, target_sample_rate)
        recordings.append(audio)
        source_lengths_resampled.append(len(audio))

    mixed, lags = align_recordings(
        recordings, target_sample_rate,
        max_lag_seconds=max_lag_seconds,
        sync_window_seconds=sync_window_seconds,
        auto_align=auto_align,
    )
    model = FastICA(max_iter=max_iter, random_state=random_state)
    components = model.fit_transform(mixed)
    result = model.result_
    if result is None:
        raise RuntimeError("ICA did not produce diagnostics.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for i, component in enumerate(components, start=1):
        write_wav(output / f"component_{i:02d}_raw.wav", target_sample_rate, component)
    np.save(output / "components_raw.npy", components)
    np.save(output / "aligned_device_recordings.npy", mixed)
    _plot_waveforms(mixed, components, target_sample_rate, output)

    lag_rows = [
        {
            "file": str(path),
            "estimated_lag_samples_vs_first_device": int(lag),
            "estimated_lag_ms_vs_first_device": float(1000 * lag / target_sample_rate),
            "input_sample_rate_hz": int(rate),
            "input_samples_at_original_rate": int(native_length),
            "input_samples_after_resampling": int(resampled_length),
        }
        for path, lag, rate, native_length, resampled_length in zip(
            paths, lags, source_rates, source_lengths_native, source_lengths_resampled
        )
    ]
    summary = {
        "experiment": "Real-world multi-device ICA",
        "inputs": lag_rows,
        "target_sample_rate_hz": int(target_sample_rate),
        "aligned_duration_seconds": float(mixed.shape[1] / target_sample_rate),
        "number_of_device_mixtures": int(mixed.shape[0]),
        "number_of_ica_components": int(components.shape[0]),
        "auto_alignment_enabled": bool(auto_align),
        "alignment_method": "cross-correlation against first recording" if auto_align else "none; inputs assumed pre-aligned",
        "ica_algorithm": "from-scratch symmetric FastICA with tanh non-linearity",
        "ica_converged": bool(result.converged),
        "ica_iterations": int(result.n_iter),
        "final_convergence_delta": float(result.convergence_history[-1]),
        "input_correlation_matrix": _correlation_matrix(mixed).tolist(),
        "output_component_correlation_matrix": _correlation_matrix(components).tolist(),
        "output_component_excess_kurtosis": _excess_kurtosis(components),
        "output_files": [f"component_{i:02d}_raw.wav" for i in range(1, components.shape[0] + 1)],
        "limitations": [
            "Real-world source identities are unknown unless clean reference tracks are separately available.",
            "SI-SDR and reference correlation are not reported because this workflow does not assume clean ground-truth sources.",
            "ICA component order, sign, and scale are inherently ambiguous; component files are unlabeled.",
            "Cross-correlation estimates relative start offsets; it does not correct sample-clock drift or fully model reverberant/convolutive mixing.",
            "The number of independent mixtures should be at least the number of sources, and the mixing matrix must have full rank.",
        ],
    }
    (output / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.txt").write_text(
        "REAL-WORLD MULTI-DEVICE ICA OUTPUTS\n\n"
        "component_XX_raw.wav: unlabelled output components from the from-scratch ICA implementation.\n"
        "components_raw.npy: exact float component matrix before WAV conversion/peak protection.\n"
        "aligned_device_recordings.npy: aligned inputs used by ICA.\n"
        "diagnostics.json: input offsets, convergence, correlations, kurtosis and known limitations.\n"
        "PNG figures: input/output waveforms and correlation diagnostics.\n\n"
        "Component names are intentionally not guessed. Listen to them and identify them manually.\n"
        "No reference-based score is reported because no clean isolated target sources were supplied.\n",
        encoding="utf-8",
    )
    return summary
