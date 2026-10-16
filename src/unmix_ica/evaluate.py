"""Source matching and scale-invariant evaluation metrics."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def correlation_matrix(reference: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    """Return signed Pearson correlations (reference rows x estimate rows)."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimated, dtype=np.float64)
    if ref.ndim != 2 or est.ndim != 2 or ref.shape[1] != est.shape[1]:
        raise ValueError("Signals must be 2-D with a common sample count.")
    ref = ref - ref.mean(axis=1, keepdims=True)
    est = est - est.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(ref, axis=1)[:, None] * np.linalg.norm(est, axis=1)[None, :]
    return (ref @ est.T) / np.maximum(denom, 1e-15)


def align_sources(reference: np.ndarray, estimated: np.ndarray) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Align estimate rows to reference rows; correct sign and least-squares scale.

    Matching maximizes absolute correlation using the Hungarian assignment. The
    returned estimates are rescaled to the references, which is appropriate for
    demonstrations where clean originals are known. Never use originals to label
    blindly separated production audio; here they are used only for evaluation.
    """
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimated, dtype=np.float64)
    if ref.shape[1] != est.shape[1]:
        raise ValueError("Reference and estimate must have the same sample count.")
    corr = correlation_matrix(ref, est)
    rows, cols = linear_sum_assignment(-np.abs(corr))
    aligned = np.zeros_like(ref)
    mapping = [-1] * ref.shape[0]
    chosen_corr = np.zeros(ref.shape[0], dtype=np.float64)
    for r, c in zip(rows, cols):
        candidate = est[c] - np.mean(est[c])
        target = ref[r] - np.mean(ref[r])
        signed_corr = corr[r, c]
        if signed_corr < 0:
            candidate = -candidate
        scale = float(np.dot(target, candidate) / max(np.dot(candidate, candidate), 1e-15))
        aligned[r] = candidate * scale
        mapping[r] = int(c)
        chosen_corr[r] = abs(float(signed_corr))
    return aligned, mapping, chosen_corr


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-15) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (higher is better)."""
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    denom = float(np.dot(ref, ref))
    if denom < eps:
        return float("nan")
    target = (float(np.dot(est, ref)) / denom) * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def evaluate_sources(reference: np.ndarray, estimated: np.ndarray) -> tuple[list[dict[str, float | int]], dict[str, float], np.ndarray]:
    """Evaluate estimates after permutation/sign/scale matching to known sources."""
    aligned, mapping, _ = align_sources(reference, estimated)
    corr = correlation_matrix(reference, aligned)
    rows: list[dict[str, float | int]] = []
    for i in range(reference.shape[0]):
        rows.append({
            "source_index": i + 1,
            "estimated_component": mapping[i] + 1,
            "correlation": float(corr[i, i]),
            "si_sdr_db": si_sdr(reference[i], aligned[i]),
        })
    summary = {
        "mean_abs_correlation": float(np.mean([abs(r["correlation"]) for r in rows])),
        "mean_si_sdr_db": float(np.mean([r["si_sdr_db"] for r in rows])),
    }
    return rows, summary, aligned
