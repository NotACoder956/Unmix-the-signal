#!/usr/bin/env python3
"""Run the complete Unmix the Signal experiment from the repository root."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from unmix_ica.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_experiment(ROOT, max_iter=args.max_iter, random_state=args.seed)
    print((ROOT / "reports" / "EXPERIMENT_SUMMARY.txt").read_text(encoding="utf-8"))
    print(f"\nFull metrics: {ROOT / 'reports' / 'metrics' / 'summary.json'}")
    if not summary["ica_converged"]:
        print("WARNING: FastICA did not meet the convergence threshold; inspect the convergence plot.")

if __name__ == "__main__":
    main()
