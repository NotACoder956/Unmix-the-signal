#!/usr/bin/env python3
"""Run from-scratch ICA on recordings captured by separate devices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unmix_ica.realworld import run_real_world_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize WAV recordings of the same sound scene captured by "
            "different phones/microphones, run the project's own FastICA, and "
            "save unlabeled components plus diagnostics."
        )
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="At least two WAV files from separate devices; three or more is preferred.")
    parser.add_argument("--output-dir", default="outputs/real_world", help="Where components, plots and diagnostics are written.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Common processing sample rate (default: 16000 Hz).")
    parser.add_argument("--max-lag-seconds", type=float, default=3.0, help="Maximum automatic start-offset search (default: 3 seconds).")
    parser.add_argument("--sync-window-seconds", type=float, default=12.0, help="Initial audio used for synchronization (default: 12 seconds).")
    parser.add_argument("--no-auto-align", action="store_true", help="Use only if recordings are already sample-aligned.")
    parser.add_argument("--max-iter", type=int, default=500, help="Maximum FastICA iterations.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed for reproducible ICA initialization.")
    args = parser.parse_args()

    summary = run_real_world_pipeline(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        target_sample_rate=args.sample_rate,
        max_lag_seconds=args.max_lag_seconds,
        sync_window_seconds=args.sync_window_seconds,
        auto_align=not args.no_auto_align,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    print(json.dumps(summary, indent=2))
    print(f"\nResults written to: {Path(args.output_dir).resolve()}")
    if not summary["ica_converged"]:
        print("WARNING: FastICA reached max_iter before convergence; inspect diagnostics and input quality.")


if __name__ == "__main__":
    main()
