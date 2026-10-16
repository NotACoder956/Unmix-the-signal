"""End-to-end experiment runner: mix, center, whiten, separate, evaluate, plot."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .audio import read_wav, write_wav, normalize_source, mix_sources
from .ica import FastICA, pca_whiten
from .evaluate import evaluate_sources, align_sources, correlation_matrix

SOURCE_FILES = [
    ("Speech (eSpeak TTS)", "speech.wav"),
    ("Synthetic instrumental music", "music.wav"),
    ("Synthetic rain ambience", "ambient_rain.wav"),
]
MIXING_MATRIX = np.array([
    [1.00, 0.45, 0.20],
    [0.25, 1.00, 0.50],
    [0.40, 0.20, 1.00],
], dtype=np.float64)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_waveforms(sources: np.ndarray, mixed: np.ndarray, separated: np.ndarray,
                    source_names: list[str], sample_rate: int, path: Path) -> None:
    seconds_to_show = min(1.25, sources.shape[1] / sample_rate)
    n = max(1, int(seconds_to_show * sample_rate))
    t = np.arange(n) / sample_rate
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), sharex=True)
    row_data = [sources, mixed, separated]
    for r, data in enumerate(row_data):
        for c in range(3):
            axes[r, c].plot(t, data[c, :n], linewidth=0.7)
            axes[r, c].grid(alpha=0.25)
            if r == 0:
                title = f"Original source — {source_names[c]}"
            elif r == 1:
                title = f"Mixed observation channel {c + 1}"
            else:
                title = f"ICA estimate matched to — {source_names[c]}"
            axes[r, c].set_title(title, fontsize=9)
            if c == 0:
                axes[r, c].set_ylabel("Amplitude")
            if r == 2:
                axes[r, c].set_xlabel("Time (s)")
    fig.suptitle("Blind audio source separation: time-domain comparison", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_correlations(sources: np.ndarray, mixed: np.ndarray, estimated: np.ndarray,
                       source_names: list[str], path: Path) -> None:
    # For mixtures, each channel has one row, so the left panel shows raw source-to-mixture correlation.
    matrices = [correlation_matrix(sources, mixed), correlation_matrix(sources, estimated)]
    titles = ["Original source vs. mixed channels", "Original source vs. ICA components"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, matrix, title in zip(axes, matrices, titles):
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Mixture / component index")
        ax.set_ylabel("Original source")
        ax.set_xticks(range(matrix.shape[1]), [f"{i+1}" for i in range(matrix.shape[1])])
        ax.set_yticks(range(matrix.shape[0]), source_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes[1], shrink=0.78, label="Pearson correlation")
    fig.suptitle("Correlation structure before and after ICA")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_metrics(baseline_rows: list[dict], ica_rows: list[dict], path: Path) -> None:
    labels = ["PCA whitening only", "PCA whitening + ICA"]
    corr = [np.mean([abs(float(r["correlation"])) for r in baseline_rows]),
            np.mean([abs(float(r["correlation"])) for r in ica_rows])]
    sdr = [np.mean([float(r["si_sdr_db"]) for r in baseline_rows]),
           np.mean([float(r["si_sdr_db"]) for r in ica_rows])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, corr)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean absolute correlation")
    axes[0].set_title("Source similarity")
    axes[1].bar(labels, sdr)
    axes[1].set_ylabel("Mean SI-SDR (dB)")
    axes[1].set_title("Scale-invariant separation quality")
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelrotation=10)
    fig.suptitle("Baseline comparison (matched against known clean sources)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence(history: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sign-insensitive convergence delta (log scale)")
    ax.set_title("FastICA convergence history")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def _plot_snr_robustness(rows: list[dict], path: Path) -> None:
    snr_values = [float(row["snr_db"]) for row in rows]
    corr = [float(row["mean_abs_correlation"]) for row in rows]
    sdr = [float(row["mean_si_sdr_db"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(snr_values, corr, marker="o")
    axes[0].set_xlabel("Input mixture SNR (dB; higher is cleaner)")
    axes[0].set_ylabel("Mean absolute correlation")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("Source similarity vs. noise")
    axes[1].plot(snr_values, sdr, marker="o")
    axes[1].set_xlabel("Input mixture SNR (dB; higher is cleaner)")
    axes[1].set_ylabel("Mean SI-SDR (dB)")
    axes[1].set_title("Separation quality vs. noise")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Bonus experiment: robustness to additive white noise")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment(root: str | Path, max_iter: int = 500, random_state: int = 42) -> dict:
    """Run the complete experiment in a project root and return its metrics."""
    root = Path(root)
    originals_dir = root / "data" / "originals"
    mixed_dir = root / "data" / "mixed"
    separated_dir = root / "outputs" / "separated"
    figures_dir = root / "reports" / "figures"
    metrics_dir = root / "reports" / "metrics"
    for directory in (mixed_dir, separated_dir, figures_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loaded = []
    sample_rates = []
    for _, filename in SOURCE_FILES:
        sr, audio = read_wav(originals_dir / filename)
        loaded.append(audio)
        sample_rates.append(sr)
    if len(set(sample_rates)) != 1:
        raise ValueError("All original sources must use the same sample rate.")
    sample_rate = sample_rates[0]
    min_length = min(map(len, loaded))
    # Equal length is required for the matrix mixing equation.
    sources = np.vstack([normalize_source(audio[:min_length], target_rms=0.2) for audio in loaded])
    source_names = [x[0] for x in SOURCE_FILES]

    mixed = mix_sources(sources, MIXING_MATRIX)
    # Keep relative channel scales intact: apply one global scale to the whole matrix.
    global_scale = min(1.0, 0.92 / max(float(np.max(np.abs(mixed))), 1e-12))
    mixed = mixed * global_scale
    for idx, row in enumerate(mixed):
        write_wav(mixed_dir / f"mixture_{idx + 1}.wav", sample_rate, row)

    # PCA whitening is implemented in pca_whiten; it forms the explicit baseline.
    whitened, whitening_matrix, _, eigenvalues = pca_whiten(mixed)
    model = FastICA(n_components=sources.shape[0], max_iter=max_iter,
                    tol=1e-6, random_state=random_state, alpha=1.0)
    estimates = model.fit_transform(mixed)
    if model.result_ is None:
        raise RuntimeError("ICA did not produce a result.")

    baseline_rows, baseline_summary, _ = evaluate_sources(sources, whitened)
    ica_rows, ica_summary, separated_aligned = evaluate_sources(sources, estimates)

    # Bonus experiment: add white noise at defined input SNRs and rerun ICA.
    snr_rng = np.random.default_rng(random_state + 1000)
    snr_rows: list[dict[str, float | int]] = []
    for input_snr_db in (30, 20, 10, 5):
        noisy = mixed.copy()
        for channel in range(noisy.shape[0]):
            signal_power = float(np.mean(mixed[channel] ** 2))
            noise_power = signal_power / (10.0 ** (input_snr_db / 10.0))
            noisy[channel] += snr_rng.normal(0.0, np.sqrt(noise_power), size=noisy.shape[1])
        noisy_model = FastICA(n_components=sources.shape[0], max_iter=max_iter,
                              tol=1e-6, random_state=random_state, alpha=1.0)
        noisy_estimates = noisy_model.fit_transform(noisy)
        _, noisy_summary, _ = evaluate_sources(sources, noisy_estimates)
        snr_rows.append({
            "snr_db": float(input_snr_db),
            "mean_abs_correlation": noisy_summary["mean_abs_correlation"],
            "mean_si_sdr_db": noisy_summary["mean_si_sdr_db"],
            "ica_iterations": int(noisy_model.result_.n_iter if noisy_model.result_ else 0),
            "ica_converged": bool(noisy_model.result_.converged if noisy_model.result_ else False),
        })

    # Save playable WAVs after source matching and least-squares scale alignment.
    # The ordering uses known originals only for this reproducible evaluation demo.
    for idx, row in enumerate(separated_aligned):
        write_wav(separated_dir / f"separated_{idx + 1}.wav", sample_rate, row)

    # Preserve the exact numeric components plus playable WAVs in algorithmic order.
    np.save(separated_dir / "components_raw.npy", estimates)
    for idx, row in enumerate(estimates):
        # WAV files are peak-scaled only if required to avoid integer PCM clipping.
        write_wav(separated_dir / f"component_{idx + 1}_raw.wav", sample_rate, row)

    with (metrics_dir / "source_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_index", "source_name", "estimated_component", "correlation", "si_sdr_db"])
        writer.writeheader()
        for i, row in enumerate(ica_rows):
            writer.writerow({"source_name": source_names[i], **row})

    with (metrics_dir / "baseline_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source_index", "source_name", "estimated_component", "correlation", "si_sdr_db"])
        writer.writeheader()
        for i, row in enumerate(baseline_rows):
            writer.writerow({"source_name": source_names[i], **row})

    summary = {
        "project": "Unmix the Signal (ICA)",
        "sample_rate_hz": sample_rate,
        "duration_seconds": min_length / sample_rate,
        "number_of_sources": int(sources.shape[0]),
        "number_of_mixtures": int(mixed.shape[0]),
        "mixing_matrix": MIXING_MATRIX.tolist(),
        "mixture_global_scale": float(global_scale),
        "algorithm": "Symmetric FastICA with tanh non-linearity, implemented from scratch",
        "whitening_eigenvalues": eigenvalues.tolist(),
        "ica_iterations": model.result_.n_iter,
        "ica_converged": model.result_.converged,
        "final_convergence_delta": model.result_.convergence_history[-1],
        "baseline_pca_whitening_only": baseline_summary,
        "ica": ica_summary,
        "source_metrics": ica_rows,
        "baseline_source_metrics": baseline_rows,
        "snr_robustness": snr_rows,
        "source_names": source_names,
        "notes": [
            "Sources are demonstration audio: eSpeak-synthesized speech, procedural instrumental music, and procedural rain ambience.",
            "Metrics use known originals for source matching and scale alignment; the algorithm itself receives only mixed signals.",
            "ICA source order, sign, and scale are intrinsically ambiguous.",
        ],
    }
    _save_json(metrics_dir / "summary.json", summary)

    np.savetxt(metrics_dir / "mixing_matrix.csv", MIXING_MATRIX, delimiter=",")
    np.savetxt(metrics_dir / "effective_mixing_matrix.csv", global_scale * MIXING_MATRIX, delimiter=",")
    with (metrics_dir / "snr_robustness.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["snr_db", "mean_abs_correlation", "mean_si_sdr_db", "ica_iterations", "ica_converged"])
        writer.writeheader()
        writer.writerows(snr_rows)
    _plot_snr_robustness(snr_rows, figures_dir / "05_snr_robustness.png")
    _plot_waveforms(sources, mixed, separated_aligned, source_names, sample_rate,
                    figures_dir / "01_waveform_comparison.png")
    _plot_correlations(sources, mixed, estimates, source_names,
                       figures_dir / "02_correlation_matrices.png")
    _plot_metrics(baseline_rows, ica_rows, figures_dir / "03_metrics_comparison.png")
    _plot_convergence(model.result_.convergence_history, figures_dir / "04_ica_convergence.png")

    # A plain-text summary is useful in terminals and in the README.
    lines = [
        "UNMIX THE SIGNAL — EXPERIMENT SUMMARY",
        f"Sample rate: {sample_rate} Hz | Duration: {min_length / sample_rate:.2f} s",
        f"FastICA converged: {model.result_.converged} (iterations: {model.result_.n_iter})",
        f"Mean absolute correlation, PCA whitening only: {baseline_summary['mean_abs_correlation']:.4f}",
        f"Mean absolute correlation, ICA: {ica_summary['mean_abs_correlation']:.4f}",
        f"Mean SI-SDR, PCA whitening only: {baseline_summary['mean_si_sdr_db']:.2f} dB",
        f"Mean SI-SDR, ICA: {ica_summary['mean_si_sdr_db']:.2f} dB",
        "",
        "Per-source ICA metrics (after permutation/sign/scale matching):",
    ]
    for i, row in enumerate(ica_rows):
        lines.append(f"  {source_names[i]}: correlation={row['correlation']:.4f}, SI-SDR={row['si_sdr_db']:.2f} dB")
    (root / "reports" / "EXPERIMENT_SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
