"""
PRISM V2 Training Pipeline (286 Dimensions, 15-Second Windows, L=30 Lookback):
End-to-end training entry point for Multi-Dataset World Model across
CICIOT23, CIC-IDS2018, CIC-IDS2017, and UNSW-NB15.
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Force UTF-8 stdout & append project root
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.schema_aligner import SchemaAligner
from src.data.state_builder import StateBuilder
from src.data.dataset import StateSequenceDataset, multi_file_chronological_split
from src.models.world_model import StateTransformerWorldModel
from src.models.baseline import BaselineClassifier
from src.evaluation.benchmark import run_benchmark
from src.utils.logger import setup_logger

logger = setup_logger("TrainV2")

PROCESSED_DIR = os.path.join("data", "processed")
WEIGHTS_DIR = "weights"
RESULTS_DIR = "results"

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_and_preprocess_traffic(
    file_list: list,
    window_size: int = 15,
    chunksize: int = 100_000,
    max_rows_per_file: int = 250_000
):
    """Loads CSV files in chunks, aligns features into 64 dimensions, and builds 15-second 286-dim state vectors."""
    aligner = SchemaAligner()
    builder = StateBuilder(window_size_seconds=window_size)

    file_state_tuples = []
    all_states = []
    all_attacks = []
    all_mitre = []
    all_fractions = []

    for filepath in file_list:
        filename = os.path.basename(filepath)
        logger.info(f"Processing {filename} in memory-safe chunks (max {max_rows_per_file:,} rows)...")
        
        chunks = []
        rows_read = 0
        try:
            for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False, encoding='cp1252'):
                aligned_chunk = aligner.align_dataframe(chunk)
                chunks.append(aligned_chunk)
                rows_read += len(chunk)
                if rows_read >= max_rows_per_file:
                    break
        except Exception as e:
            logger.warning(f"Fallback reading for {filename}: {e}")
            for chunk in pd.read_csv(filepath, chunksize=chunksize, low_memory=False, encoding_errors='replace'):
                aligned_chunk = aligner.align_dataframe(chunk)
                chunks.append(aligned_chunk)
                rows_read += len(chunk)
                if rows_read >= max_rows_per_file:
                    break

        if not chunks:
            continue

        full_aligned = pd.concat(chunks, ignore_index=True)
        full_aligned = full_aligned.sort_values(by="timestamp").reset_index(drop=True)

        states, atks, mitres, fracs, _ = builder.build_states_from_dataframe(full_aligned)
        logger.info(f"Built {len(states)} 15-second window states (286 dims) from {filename}")

        if len(states) > 0:
            file_state_tuples.append((states, atks, mitres, fracs))
            all_states.append(states)
            all_attacks.append(atks)
            all_mitre.append(mitres)
            all_fractions.append(fracs)

    if not all_states:
        raise ValueError("No states could be extracted from input files.")

    combined_states = np.concatenate(all_states, axis=0)
    combined_atks = np.concatenate(all_attacks, axis=0)
    combined_mitres = np.concatenate(all_mitre, axis=0)
    combined_fracs = np.concatenate(all_fractions, axis=0)

    # Save compact preprocessed cache
    np.save(os.path.join(PROCESSED_DIR, "states.npy"), combined_states)
    np.save(os.path.join(PROCESSED_DIR, "attack_labels.npy"), combined_atks)
    np.save(os.path.join(PROCESSED_DIR, "mitre_labels.npy"), combined_mitres)
    np.save(os.path.join(PROCESSED_DIR, "attack_fractions.npy"), combined_fracs)
    logger.info(f"Saved processed dataset cache: {len(combined_states)} total 15-second time windows.")

    return file_state_tuples, combined_states, combined_atks, combined_mitres, combined_fracs


def train_pipeline(epochs: int = 50, batch_size: int = 64, lr: float = 1e-4, lookback: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")

    # Load preprocessed scaled dataset with leak-free stage-stratified split
    split_indices_path = os.path.join(PROCESSED_DIR, "split_indices.npz")
    if not os.path.exists(split_indices_path):
        from scripts.prepare_scaled_dataset import build_scaled_dataset
        logger.info("Scaled preprocessed dataset not found. Generating now...")
        build_scaled_dataset(window_size=15)

    logger.info("Loading preprocessed scaled dataset and stage-stratified chronological splits...")
    states = np.load(os.path.join(PROCESSED_DIR, "states.npy"))
    atks = np.load(os.path.join(PROCESSED_DIR, "attack_labels.npy"))
    mitres = np.load(os.path.join(PROCESSED_DIR, "mitre_labels.npy"))
    fracs = np.load(os.path.join(PROCESSED_DIR, "attack_fractions.npy"))
    scaler_mean = np.load(os.path.join(PROCESSED_DIR, "scaler_mean.npy"))
    scaler_std = np.load(os.path.join(PROCESSED_DIR, "scaler_std.npy"))

    splits = np.load(split_indices_path)
    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]
    test_idx = splits["test_indices"]

    train_ds = StateSequenceDataset(
        states[train_idx], atks[train_idx], mitres[train_idx], fracs[train_idx],
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )
    val_ds = StateSequenceDataset(
        states[val_idx], atks[val_idx], mitres[val_idx], fracs[val_idx],
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )
    test_ds = StateSequenceDataset(
        states[test_idx], atks[test_idx], mitres[test_idx], fracs[test_idx],
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    logger.info(f"Scaled Dataset Loaded: {len(states):,} total windows (d_state={states.shape[1]})")
    logger.info(f"Sequences -> Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")

    # 1. Train Baseline Classifier
    X_train_flat = train_ds.raw_states[train_ds.lookback:]
    y_train_atk = train_ds.attack_labels[train_ds.lookback:]
    y_train_mitre = train_ds.mitre_labels[train_ds.lookback:]

    baseline = BaselineClassifier(model_type="logistic_regression")
    baseline.fit(X_train_flat, y_train_atk, y_train_mitre)
    joblib.dump(baseline, os.path.join(WEIGHTS_DIR, "baseline.joblib"))

    # 2. Train StateTransformerWorldModel (d_state=286, lookback=30)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = StateTransformerWorldModel(
        d_state=292,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=lookback + 20
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    class_weights = torch.tensor([1.0, 3.5], dtype=torch.float32).to(device)

    # Balanced MITRE multi-class weights (smoothed to prevent minority class over-prediction)
    mitre_counts = np.bincount(train_ds.mitre_labels, minlength=7)
    total_mitre = len(train_ds.mitre_labels)
    raw_mitre_w = total_mitre / (7.0 * np.maximum(mitre_counts, 1).astype(np.float32))
    clipped_mitre_w = np.clip(raw_mitre_w, 0.35, 2.5)
    mitre_weights = torch.from_numpy(clipped_mitre_w).float().to(device)
    logger.info(f"Balanced MITRE class weights: {np.round(clipped_mitre_w, 2).tolist()}")

    best_val_loss = float("inf")
    best_checkpoint_path = os.path.join(WEIGHTS_DIR, "world_model.pt")
    start_epoch = 1
    patience = 8
    patience_counter = 0

    if os.path.exists(best_checkpoint_path):
        try:
            chk = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            if "model_state_dict" in chk:
                model.load_state_dict(chk["model_state_dict"])
                start_epoch = chk.get("epoch", 0) + 1
                best_val_loss = chk.get("val_loss", float("inf"))
                if "optimizer_state_dict" in chk:
                    optimizer.load_state_dict(chk["optimizer_state_dict"])
                logger.info(f"Resuming training from checkpoint: Start Epoch = {start_epoch}, Best Val Loss = {best_val_loss:.4f}")
        except Exception as e:
            logger.warning(f"Could not resume from checkpoint, starting fresh: {e}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            seq = batch["state_seq"].to(device)
            next_state = batch["next_state"].to(device)
            next_atk = batch["next_attack"].to(device)
            next_mitre = batch["next_mitre"].to(device)
            next_frac = batch["next_fraction"].to(device)

            pred_mean, pred_logvar, pred_atk, pred_mit, pred_fr, _ = model(seq)

            loss, _ = model.compute_loss(
                pred_mean, pred_logvar, pred_atk, pred_mit, pred_fr,
                next_state, next_atk, next_mitre, next_frac,
                class_weights=class_weights,
                mitre_weights=mitre_weights
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, train_batches)

        # Validation loop
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["state_seq"].to(device)
                next_state = batch["next_state"].to(device)
                next_atk = batch["next_attack"].to(device)
                next_mitre = batch["next_mitre"].to(device)
                next_frac = batch["next_fraction"].to(device)

                pred_mean, pred_logvar, pred_atk, pred_mit, pred_fr, _ = model(seq)
                loss, _ = model.compute_loss(
                    pred_mean, pred_logvar, pred_atk, pred_mit, pred_fr,
                    next_state, next_atk, next_mitre, next_frac,
                    class_weights=class_weights,
                    mitre_weights=mitre_weights
                )
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / max(1, val_batches) if val_batches > 0 else avg_train_loss

        is_best = avg_val_loss <= best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            patience_counter = 0
            tmp_ckpt = best_checkpoint_path + ".tmp"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_mean": train_ds.scaler_mean,
                "scaler_std": train_ds.scaler_std,
                "val_loss": best_val_loss
            }, tmp_ckpt)
            if os.path.exists(best_checkpoint_path):
                try:
                    os.remove(best_checkpoint_path)
                except Exception:
                    pass
            os.replace(tmp_ckpt, best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (no validation improvement for {patience} epochs).")
                break

        logger.info(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {'[BEST]' if is_best else ''}")

    logger.info(f"Training complete! Best checkpoint saved to {best_checkpoint_path} (Val Loss: {best_val_loss:.4f})")

    # 3. Benchmark Evaluation on Test Set
    logger.info("Running test set benchmark comparison with dynamic threshold calibration...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    benchmark_df = run_benchmark(model, baseline, test_ds, val_dataset=val_ds, device=device)
    print("\n" + "=" * 60)
    print("           PRISM V2 FINAL BENCHMARK RESULTS")
    print("=" * 60)
    print(benchmark_df.to_string(index=False))
    print("=" * 60 + "\n")

    benchmark_df.to_json(os.path.join(RESULTS_DIR, "benchmark_results.json"), orient="records", indent=2)
    try:
        benchmark_df.to_markdown(os.path.join(RESULTS_DIR, "benchmark_results.md"), index=False)
    except Exception:
        with open(os.path.join(RESULTS_DIR, "benchmark_results.md"), "w", encoding="utf-8") as f:
            f.write(benchmark_df.to_string(index=False))
    logger.info("Benchmark evaluation saved to results/.")


if __name__ == "__main__":
    train_pipeline(epochs=50, batch_size=256, lookback=30)
