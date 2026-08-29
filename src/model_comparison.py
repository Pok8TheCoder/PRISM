"""
PRISM Model Comparison Engine (src/model_comparison.py)
Direct entrypoint within `src/` to compare the World Model against
static baselines (Logistic Regression and Random Forest) on real held-out telemetry.

Usage:
    python src/model_comparison.py
    python -m src.model_comparison
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from src.models.world_model import build_world_model, load_checkpoint
from src.models.baseline import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    flatten_state_sequences,
)
from src.evaluation.benchmark import BenchmarkRunner
from src.data.dataset import load_states_from_npz
from src.utils.config import load_config


def run_model_comparison(
    test_path: str = "data/splits/test.npz",
    train_path: str = "data/splits/train.npz",
    model_path: str = "weights/world_model_best.pt",
    output_dir: str = "results/benchmark",
    plot: bool = True,
) -> pd.DataFrame:
    """
    Evaluates PRISM World Model against static baselines on the same test set.

    Returns:
        pd.DataFrame containing side-by-side metric comparison.
    """
    test_p = ROOT_DIR / test_path
    train_p = ROOT_DIR / train_path
    ckpt_p = ROOT_DIR / model_path
    out_d = ROOT_DIR / output_dir
    out_d.mkdir(parents=True, exist_ok=True)

    if not test_p.exists():
        raise FileNotFoundError(f"Test split not found at {test_p}. Run split_dataset.py first.")

    print("\n" + "=" * 80)
    print("  PRISM MODEL COMPARISON BENCHMARK (src/model_comparison.py)")
    print("=" * 80)
    print(f"  Test Dataset  : {test_p}")
    print(f"  Model Weights : {ckpt_p}")
    print(f"  Output Dir    : {out_d}\n")

    # 1. Load test data to determine dimensions
    test_data = load_states_from_npz(str(test_p))
    d_state = test_data["states"].shape[1]
    n_test = len(test_data["states"])
    n_attacks = int(np.sum(test_data["labels_binary"]))
    print(f"Loaded {n_test} test windows ({d_state} features each) | Attack Windows: {n_attacks} ({100 * n_attacks / n_test:.1f}%)")

    # 2. Load World Model
    cfg = load_config(ROOT_DIR / "configs" / "model.yaml")
    cfg.model.d_state = d_state
    device = "cpu"

    print("Loading PRISM StateTransformerWorldModel...")
    world_model = build_world_model(cfg.model).to(device)
    if ckpt_p.exists():
        load_checkpoint(world_model, str(ckpt_p), device=device)
        print(f"  -> Checkpoint successfully loaded: {ckpt_p}")
    else:
        print(f"  -> Warning: Checkpoint {ckpt_p} not found. Using initialized weights.")
    world_model.eval()

    # 3. Load or train Baselines
    baselines = {}
    lr_ckpt = ROOT_DIR / "weights" / "lr_baseline.pkl"
    rf_ckpt = ROOT_DIR / "weights" / "rf_baseline.pkl"

    if lr_ckpt.exists() and rf_ckpt.exists():
        print("Loading pre-trained static baselines (LR & RF)...")
        lr = LogisticRegressionBaseline(task="binary")
        lr.load(str(lr_ckpt))
        baselines["Logistic Regression"] = lr

        rf = RandomForestBaseline(task="binary")
        rf.load(str(rf_ckpt))
        baselines["Random Forest"] = rf
    elif train_p.exists():
        print(f"Training static baselines from {train_p}...")
        train_data = load_states_from_npz(str(train_p))
        X_train, y_bin_train, _ = flatten_state_sequences(
            train_data["states"],
            train_data["labels_binary"],
            train_data["labels_mitre"],
            lookback=1,
        )
        lr = LogisticRegressionBaseline(task="binary")
        lr.fit(X_train, y_bin_train)
        lr.save(str(lr_ckpt))
        baselines["Logistic Regression"] = lr

        rf = RandomForestBaseline(task="binary")
        rf.fit(X_train, y_bin_train)
        rf.save(str(rf_ckpt))
        baselines["Random Forest"] = rf

    # 4. Run Benchmark
    print("Running side-by-side evaluation...")
    runner = BenchmarkRunner(
        world_model=world_model,
        baselines=baselines,
        device=device,
    )

    results = runner.run(
        test_states_npz=str(test_p),
        lookback=20,
        batch_size=32,
    )

    df = runner.compare(results)
    
    # Format display names
    if "World Model" in df.index:
        df = df.rename(index={"World Model": "PRISM World Model (Ours)"})
    if "LR Baseline" in df.index:
        df = df.rename(index={"LR Baseline": "Logistic Regression Baseline"})
    if "RF Baseline" in df.index:
        df = df.rename(index={"RF Baseline": "Random Forest Baseline"})

    print("\n" + "=" * 90)
    print("  EVALUATION RESULTS COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string())
    print("=" * 90)

    # Save results
    csv_path = out_d / "benchmark_table.csv"
    df.to_csv(csv_path)
    print(f"\n[Saved] Comparison table CSV -> {csv_path}")

    if plot:
        plot_path = out_d / "benchmark_comparison.png"
        runner.plot_comparison(df, save_path=str(plot_path))
        print(f"[Saved] Comparison chart PNG -> {plot_path}")

    return df


if __name__ == "__main__":
    run_model_comparison()
