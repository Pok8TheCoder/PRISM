"""
PRISM - Evaluation Script
Runs benchmark comparison between world model and baselines.

Usage:
    python scripts/evaluate.py --model weights/world_model_best.pt
    python scripts/evaluate.py --model weights/world_model_best.pt --train-baselines
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_device
from src.utils.logger import setup_logger
from src.data.dataset import load_states_from_npz, create_dataloaders
from src.models.world_model import build_world_model, load_checkpoint
from src.models.baseline import (
    LogisticRegressionBaseline,
    RandomForestBaseline,
    flatten_state_sequences,
)
from src.evaluation.benchmark import BenchmarkRunner

logger = setup_logger("prism.evaluate", log_dir="results/logs")


def main():
    parser = argparse.ArgumentParser(description="PRISM Benchmark Evaluation")
    parser.add_argument("--model", default="weights/world_model_best.pt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--test-npz", default="data/splits/test.npz")
    parser.add_argument("--train-npz", default="data/splits/train.npz",
                        help="Train data for fitting baselines")
    parser.add_argument("--train-baselines", action="store_true",
                        help="Fit LR and RF baselines from scratch")
    parser.add_argument("--lr-model", default=None, help="Pre-trained LR baseline path")
    parser.add_argument("--rf-model", default=None, help="Pre-trained RF baseline path")
    parser.add_argument("--output-dir", default="results/benchmark")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.train)

    # ---------------------------------------------------------------
    # Load test data
    # ---------------------------------------------------------------
    logger.info("Loading test data from %s", args.test_npz)
    test_data = load_states_from_npz(args.test_npz)
    d_state = test_data["states"].shape[1]
    cfg.model.d_state = d_state

    # ---------------------------------------------------------------
    # Load world model
    # ---------------------------------------------------------------
    logger.info("Loading world model from %s", args.model)
    world_model = build_world_model(cfg.model).to(device)
    load_checkpoint(world_model, args.model, device=device)
    world_model.eval()

    # ---------------------------------------------------------------
    # Baselines
    # ---------------------------------------------------------------
    baselines = {}

    if args.train_baselines:
        logger.info("Training LR and RF baselines on %s", args.train_npz)
        train_data = load_states_from_npz(args.train_npz)
        X_train, y_bin_train, y_mit_train = flatten_state_sequences(
            train_data["states"],
            train_data["labels_binary"],
            train_data["labels_mitre"],
            lookback=1,
        )

        lr = LogisticRegressionBaseline(task="binary")
        lr.fit(X_train, y_bin_train)
        lr.save("weights/lr_baseline.pkl")
        baselines["LR Baseline"] = lr

        rf = RandomForestBaseline(task="binary")
        rf.fit(X_train, y_bin_train)
        rf.save("weights/rf_baseline.pkl")
        baselines["RF Baseline"] = rf

    else:
        if args.lr_model:
            lr = LogisticRegressionBaseline()
            lr.load(args.lr_model)
            baselines["LR Baseline"] = lr

        if args.rf_model:
            rf = RandomForestBaseline()
            rf.load(args.rf_model)
            baselines["RF Baseline"] = rf

    # ---------------------------------------------------------------
    # Run benchmark
    # ---------------------------------------------------------------
    runner = BenchmarkRunner(
        world_model=world_model,
        baselines=baselines,
        device=device,
    )

    results = runner.run(
        test_states_npz=args.test_npz,
        lookback=cfg.data.lookback,
        batch_size=cfg.train.batch_size,
    )

    df = runner.compare(results)
    runner.print_table(df)
    runner.save_results(results, df, output_dir=args.output_dir)

    if args.plot:
        plot_path = f"{args.output_dir}/benchmark_comparison.png"
        runner.plot_comparison(df, save_path=plot_path)
        logger.info("Comparison plot saved -> %s", plot_path)


if __name__ == "__main__":
    main()
