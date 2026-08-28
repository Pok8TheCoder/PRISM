"""
PRISM — Data Download & Multi-Dataset Preprocessing Engine
Downloads and pre-processes public cybersecurity datasets into PRISM world model state sequences.

Supported Datasets:
    1. CSE-CIC-IDS2018
    2. CTU-13
    3. UNSW-NB15
    4. CICIoT2023
    5. LANL Authentication Dataset
    6. DARPA Intrusion Detection Dataset

Attribution & References:
    NCIIPC (National Critical Information Infrastructure Protection Centre)
    Web: https://nciipc.gov.in | Helpdesk: helpdesk1@nciipc.gov.in

Usage:
    python scripts/download_data.py --demo
    python scripts/download_data.py --dataset unsw-nb15 --output data/raw
    python scripts/download_data.py --dataset ciciot2023 --preprocess
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.utils.constants import NCIIPC_INFO, SUPPORTED_DATASETS
from src.data.registry import (
    DATASET_CATALOG,
    get_dataset_extractor,
    get_dataset_metadata,
    list_supported_datasets,
)

logger = setup_logger("prism.download", log_dir="results/logs")


# ---------------------------------------------------------------------------
# Demo / synthetic data generator
# ---------------------------------------------------------------------------
def generate_demo_data(output_dir: str = "data/raw", n_windows: int = 500):
    """
    Generate synthetic labelled network state data for demo purposes.
    Simulates a realistic attack scenario across MITRE stages.
    """
    from src.utils.constants import MITRE_STAGES

    np.random.seed(42)
    T = n_windows
    D = 50

    states = np.zeros((T, D), dtype=np.float32)
    labels_binary = np.zeros(T, dtype=np.int64)
    labels_mitre = np.zeros(T, dtype=np.int64)

    states[:] = np.random.randn(T, D) * 0.3

    def simulate_attack(t_start, t_end, stage_name, feature_pattern):
        stage_id = MITRE_STAGES[stage_name]
        for t in range(t_start, t_end):
            states[t] += feature_pattern * np.random.uniform(0.5, 1.5)
            labels_binary[t] = 1
            labels_mitre[t] = stage_id

    # Reconnaissance
    recon_pattern = np.zeros(D)
    recon_pattern[0] = 3.0
    recon_pattern[4] = 2.5
    recon_pattern[8] = -1.5
    simulate_attack(200, 250, "Reconnaissance", recon_pattern)

    # Initial Access
    init_access_pattern = np.zeros(D)
    init_access_pattern[5] = 2.0
    init_access_pattern[10] = 3.0
    init_access_pattern[3] = 1.5
    simulate_attack(250, 300, "Initial Access", init_access_pattern)

    # Lateral Movement
    lateral_pattern = np.zeros(D)
    lateral_pattern[2] = 2.5
    lateral_pattern[11] = 3.0
    lateral_pattern[6] = 1.5
    simulate_attack(300, 350, "Lateral Movement", lateral_pattern)

    # C2
    c2_pattern = np.zeros(D)
    c2_pattern[15] = -2.0
    c2_pattern[20] = 2.0
    c2_pattern[1] = 1.5
    simulate_attack(350, 400, "Command & Control", c2_pattern)

    # Impact
    impact_pattern = np.zeros(D)
    impact_pattern[25] = 4.0
    impact_pattern[4] = 4.0
    impact_pattern[0] = 2.0
    simulate_attack(400, T, "Impact", impact_pattern)

    from scipy.ndimage import gaussian_filter1d
    for d in range(D):
        states[:, d] = gaussian_filter1d(states[:, d], sigma=3)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "demo_states.npz")
    np.savez(
        output_path,
        states=states,
        labels_binary=labels_binary,
        labels_mitre=labels_mitre,
        window_ids=np.arange(T),
    )
    logger.info(
        "Demo data generated: %d windows, %d features -> %s",
        T, D, output_path,
    )

    from src.utils.constants import MITRE_STAGES_INV
    summary = pd.DataFrame({
        "window_id": np.arange(T),
        "label_binary": labels_binary,
        "label_mitre": labels_mitre,
        "mitre_stage": [MITRE_STAGES_INV.get(int(l), "Benign") for l in labels_mitre],
    })
    summary_path = os.path.join(output_dir, "demo_labels.csv")
    summary.to_csv(summary_path, index=False)

    return output_path


def print_download_instructions(dataset_name: str, output_dir: str):
    """Print dataset download and registration instructions."""
    meta = get_dataset_metadata(dataset_name)
    logger.info("=" * 70)
    logger.info(f"Dataset Download Guide: {meta.display_name}")
    logger.info("=" * 70)
    logger.info(f"Description: {meta.description}")
    logger.info(f"Source URL: {meta.source_url}")
    logger.info(f"Citation:   {meta.citation}")
    logger.info(f"\nSteps to download & prepare:")
    logger.info(f"  1. Download raw CSV/log files from {meta.source_url}")
    logger.info(f"  2. Place CSV files in directory: {output_dir}/{dataset_name}/")
    logger.info(f"  3. Execute preprocessing:\n     python scripts/download_data.py --preprocess --dataset {dataset_name}\n")


def preprocess_dataset(dataset: str, raw_dir: str, output_dir: str):
    """Run feature extraction and temporal state building for any registered dataset."""
    from src.data.state_builder import StateBuilder
    from src.data.dataset import create_temporal_splits, save_splits
    from src.utils.config import load_config

    cfg = load_config("configs/data.yaml")
    meta = get_dataset_metadata(dataset)

    logger.info(f"Starting PRISM preprocessing pipeline for dataset: {meta.display_name}")
    logger.info(f"NCIIPC Reference: {NCIIPC_INFO['agency']} ({NCIIPC_INFO['website']})")

    # Instantiate extractor via registry factory
    extractor = get_dataset_extractor(dataset, raw_dir=raw_dir)

    dataset_dir = os.path.join(raw_dir, dataset)
    if os.path.exists(dataset_dir):
        df = extractor.extract(directory=dataset_dir)
    else:
        logger.info(f"Directory '{dataset_dir}' not found. Executing extractor on single file or synthetic fallback...")
        df = extractor.extract()

    os.makedirs(output_dir, exist_ok=True)
    try:
        processed_path = os.path.join(output_dir, f"{dataset}_processed.parquet")
        df.to_parquet(processed_path, index=False)
    except Exception:
        processed_path = os.path.join(output_dir, f"{dataset}_processed.csv")
        df.to_csv(processed_path, index=False)
    logger.info("Saved processed features -> %s (%d rows)", processed_path, len(df))

    builder = StateBuilder(window_size_seconds=cfg.data.window_size_seconds)
    result = builder.build_states(df)

    states_path = os.path.join(output_dir, "states.npz")
    np.savez(
        states_path,
        states=result["states"],
        labels_binary=result["labels_binary"],
        labels_mitre=result["labels_mitre"],
        window_ids=result["window_ids"],
    )
    logger.info("Saved state sequences -> %s (%d windows)", states_path, len(result["states"]))

    splits = create_temporal_splits(
        result["states"], result["labels_binary"], result["labels_mitre"]
    )
    save_splits(splits, "data/splits")
    logger.info("Preprocessing complete! Train model with: python scripts/train.py --states data/processed/states.npz")


def main():
    parser = argparse.ArgumentParser(description="PRISM Data Download & Multi-Dataset Preprocessing")
    parser.add_argument(
        "--dataset",
        choices=["cicids2018", "ctu13", "unsw-nb15", "ciciot2023", "lanl", "darpa"],
        default="cicids2018",
        help="Target dataset key",
    )
    parser.add_argument("--output", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed", default="data/processed", help="Processed output directory")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo state dataset")
    parser.add_argument("--preprocess", action="store_true", help="Execute extraction and state building")
    parser.add_argument("--list-datasets", action="store_true", help="List all registered public datasets")
    args = parser.parse_args()

    if args.list_datasets:
        print("\nSupported Open-Source Cybersecurity Datasets (NCIIPC Compliant):")
        print("=" * 75)
        for ds in list_supported_datasets():
            print(f"[{ds['id']}] {ds['display_name']} ({ds['format']}) - {ds['source_url']}")
        return

    if args.demo:
        path = generate_demo_data(args.output)
        print(f"\nDemo state dataset generated: {path}")
        print("Run training with: python scripts/train.py --states data/raw/demo_states.npz")
        return

    if args.preprocess:
        preprocess_dataset(args.dataset, args.output, args.processed)
        return

    print_download_instructions(args.dataset, args.output)


if __name__ == "__main__":
    main()
