"""Train the PRISM model family on CICIDS2018 raw CSV data.

This wrapper first preprocesses the raw CICIDS2018 CSV files into states,
then runs the existing training and evaluation entrypoints for:
- transformer
- lstm
- latent
- gnn
- LR/RF baselines

It expects the raw CSVs at data/raw/cicids2018 and writes the derived state
artifact to data/processed/states.npz.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
STATES = ROOT / "data" / "processed" / "states.npz"


def run_step(command: list[str]) -> None:
    print("\n=== Running: {} ===".format(" ".join(command)))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PRISM model family")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_step([
        PYTHON,
        str(ROOT / "scripts" / "download_data.py"),
        "--preprocess",
        "--dataset", "cicids2018",
        "--output", str(ROOT / "data" / "raw"),
        "--processed", str(ROOT / "data" / "processed"),
    ])

    if not STATES.exists():
        raise FileNotFoundError(f"Missing state artifact after preprocessing: {STATES}")

    runs = [
        ("transformer", ROOT / "weights" / "cicids2018_transformer_raw"),
        ("lstm", ROOT / "weights" / "cicids2018_lstm_raw"),
        ("latent", ROOT / "weights" / "cicids2018_latent_raw"),
        ("gnn", ROOT / "weights" / "cicids2018_gnn_raw"),
    ]

    for arch, out_dir in runs:
        run_step([
            PYTHON,
            str(ROOT / "scripts" / "train.py"),
            "--states", str(STATES),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--arch", arch,
            "--output-dir", str(out_dir),
        ])

    run_step([
        PYTHON,
        str(ROOT / "scripts" / "evaluate.py"),
        "--train-baselines",
        "--train-npz", str(ROOT / "data" / "splits" / "train.npz"),
        "--test-npz", str(ROOT / "data" / "splits" / "test.npz"),
        "--output-dir", str(ROOT / "results" / "benchmark_cicids2018_raw"),
    ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
