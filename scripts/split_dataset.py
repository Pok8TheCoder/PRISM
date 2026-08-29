"""
PRISM — Real Dataset Temporal Splitting
Generates temporal train (70%), val (15%), and test (15%) splits from real
network state representations without future temporal leakage.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prism.split")


def create_scenario_temporal_splits(
    states: np.ndarray,
    labels_binary: np.ndarray,
    labels_mitre: np.ndarray,
    window_ids: np.ndarray | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    num_blocks: int = 4,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Splits sequential states into train, validation, and test partitions
    using block-temporal splitting.

    Divides the sequence into `num_blocks` contiguous temporal chunks (e.g.
    corresponding to monitoring periods or attack days). Within each block,
    the first 70% goes to train, next 15% to val, and final 15% to test.
    This guarantees:
      1. Zero future leakage: within any time block, test observations occur strictly AFTER train/val.
      2. Comprehensive class representation in val and test.
    """
    n_total = len(states)
    block_size = int(np.ceil(n_total / num_blocks))

    train_idx, val_idx, test_idx = [], [], []

    for b in range(num_blocks):
        b_start = b * block_size
        b_end = min((b + 1) * block_size, n_total)
        if b_start >= n_total:
            break

        b_len = b_end - b_start
        b_train_end = b_start + int(b_len * train_ratio)
        b_val_end = b_start + int(b_len * (train_ratio + val_ratio))

        train_idx.extend(range(b_start, b_train_end))
        val_idx.extend(range(b_train_end, b_val_end))
        test_idx.extend(range(b_val_end, b_end))

    def make_split(indices: list[int]) -> Dict[str, np.ndarray]:
        idx = np.array(indices, dtype=np.int64)
        res = {
            "states": states[idx],
            "labels_binary": labels_binary[idx],
            "labels_mitre": labels_mitre[idx],
        }
        if window_ids is not None:
            res["window_ids"] = window_ids[idx]
        return res

    splits = {
        "train": make_split(train_idx),
        "val": make_split(val_idx),
        "test": make_split(test_idx),
    }

    for name, split in splits.items():
        n_att = int(np.sum(split["labels_binary"]))
        stages, counts = np.unique(split["labels_mitre"], return_counts=True)
        stage_dist = dict(zip(stages.tolist(), counts.tolist()))
        logger.info(
            "Split '%s': %d windows | Attack: %d (%.1f%%) | MITRE stages: %s",
            name,
            len(split["states"]),
            n_att,
            100.0 * n_att / max(len(split["states"]), 1),
            stage_dist,
        )

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM Dataset Splitter")
    parser.add_argument(
        "--input",
        default="data/processed/cicids2018_states.npz",
        help="Path to preprocessed states .npz",
    )
    parser.add_argument(
        "--output-dir",
        default="data/splits",
        help="Directory to save train.npz, val.npz, test.npz",
    )
    parser.add_argument("--blocks", type=int, default=4, help="Number of temporal blocks")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        fallback = Path("data/raw/cicids_train.npz")
        if fallback.exists():
            input_path = fallback
            logger.info("Using fallback input: %s", fallback)
        else:
            raise FileNotFoundError(f"Input file not found: {args.input}")

    logger.info("Loading states from %s", input_path)
    data = np.load(input_path)
    states = data["states"]
    labels_binary = data["labels_binary"]
    labels_mitre = data["labels_mitre"]
    window_ids = data.get("window_ids", None)

    splits = create_scenario_temporal_splits(
        states,
        labels_binary,
        labels_mitre,
        window_ids=window_ids,
        num_blocks=args.blocks,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, split in splits.items():
        save_path = out_dir / f"{name}.npz"
        np.savez(save_path, **split)
        logger.info("Saved %s -> %s", name, save_path)

    logger.info("Splitting complete! All splits saved to %s", out_dir)


if __name__ == "__main__":
    main()
