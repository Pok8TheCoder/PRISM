"""
PRISM 200-Epoch Master Training Engine (scripts/train_all_200epochs.py)
Trains all 6 model architectures on real CSE-CIC-IDS2018 telemetry for 200 epochs
accelerated by the NVIDIA GeForce RTX 3050 GPU.

Models Trained:
1. StateTransformerWorldModel (Causal Attention, 200 epochs, CUDA)
2. LSTMWorldModel (Recurrent Dynamics, 200 epochs, CUDA)
3. GraphWorldModel (GNN Topology, 200 epochs, CUDA)
4. LatentDynamicsWorldModel (VAE Latent Dynamics, 200 epochs, CUDA)
5. LogisticRegressionBaseline (200 iterations)
6. RandomForestBaseline (200 estimators)

Usage:
    py -3.11 scripts/train_all_200epochs.py
"""

from __future__ import annotations

import os
import sys
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def train_command(cmd_args: list[str], model_name: str):
    print("\n" + "=" * 80)
    print(f"  STARTING 200-EPOCH TRAINING: {model_name}")
    print("=" * 80)
    start_t = time.time()
    res = subprocess.run(cmd_args, cwd=str(ROOT_DIR))
    elapsed = time.time() - start_t
    status = "SUCCESS" if res.returncode == 0 else f"FAILED (code {res.returncode})"
    print(f"\n[DONE] {model_name} -> {status} in {elapsed/60:.1f} minutes.")
    return res.returncode == 0


def main():
    py_exec = sys.executable
    print("\n" + "#" * 80)
    print("  PRISM 200-EPOCH FULL SUITE GPU TRAINING PIPELINE")
    print(f"  Python Interpreter: {py_exec}")
    print(f"  Target Dataset    : Real CSE-CIC-IDS2018 (data/splits/)")
    print(f"  Device            : NVIDIA GeForce RTX 3050 (CUDA 12.1)")
    print("#" * 80)

    models_to_train = [
        ("Transformer World Model", [
            py_exec, "scripts/train.py",
            "--arch", "transformer",
            "--device", "cuda",
            "--epochs", "200",
            "--patience", "40",
            "--batch-size", "64",
            "--output-dir", "weights",
        ]),
        ("LSTM World Model", [
            py_exec, "scripts/train.py",
            "--arch", "lstm",
            "--device", "cuda",
            "--epochs", "200",
            "--patience", "40",
            "--batch-size", "64",
            "--output-dir", "weights/lstm",
        ]),
        ("Graph World Model", [
            py_exec, "scripts/train.py",
            "--arch", "gnn",
            "--device", "cuda",
            "--epochs", "200",
            "--patience", "40",
            "--batch-size", "64",
            "--output-dir", "weights/gnn",
        ]),
        ("Latent Dynamics World Model", [
            py_exec, "scripts/train.py",
            "--arch", "latent",
            "--device", "cuda",
            "--epochs", "200",
            "--patience", "40",
            "--batch-size", "64",
            "--output-dir", "weights/latent",
        ]),
    ]

    for name, cmd in models_to_train:
        train_command(cmd, name)

    # Re-train static baselines with 200 iterations / 200 estimators
    print("\n" + "=" * 80)
    print("  TRAINING STATIC BASELINES (200 iterations / 200 estimators)")
    print("=" * 80)
    from src.data.dataset import load_states_from_npz
    from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline, flatten_state_sequences

    train_p = ROOT_DIR / "data" / "splits" / "train.npz"
    train_data = load_states_from_npz(str(train_p))
    X_train, y_train, _ = flatten_state_sequences(
        train_data["states"], train_data["labels_binary"], train_data["labels_mitre"], lookback=1
    )

    print("Fitting Logistic Regression (max_iter=200)...")
    lr = LogisticRegressionBaseline(max_iter=200)
    lr.fit(X_train, y_train)
    lr.save(str(ROOT_DIR / "weights" / "lr_baseline.pkl"))

    print("Fitting Random Forest (n_estimators=200)...")
    rf = RandomForestBaseline(n_estimators=200)
    rf.fit(X_train, y_train)
    rf.save(str(ROOT_DIR / "weights" / "rf_baseline.pkl"))

    print("\n" + "=" * 80)
    print("  GENERATING UPDATED 200-EPOCH BENCHMARK & VISUAL DASHBOARD")
    print("=" * 80)
    from src.models.model_comparison_viz import evaluate_all_prism_models
    evaluate_all_prism_models()


if __name__ == "__main__":
    main()
