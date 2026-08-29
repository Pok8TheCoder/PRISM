"""
PRISM - Training Script
Full training pipeline: data loading, model building, multi-task training,
checkpointing, and final evaluation.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml model.architecture=lstm
    python scripts/train.py --states data/splits/train.npz --epochs 30
"""

import argparse
import logging
import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_device
from src.utils.logger import setup_logger
from src.models.world_model import build_world_model, save_checkpoint, load_checkpoint
from src.models.components import MultiTaskLoss
from src.data.dataset import (
    load_states_from_npz,
    create_temporal_splits,
    create_dataloaders,
    save_splits,
    compute_class_weights,
)
from src.evaluation.metrics import evaluate_model

logger = setup_logger("prism.train", log_dir="results/logs")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    if cfg.train.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=cfg.train.epochs, eta_min=1e-6
        )
    elif cfg.train.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=True
        )
    else:
        return None


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        state_seq = batch["state_seq"].to(device)
        next_state = batch["next_state"].to(device)
        y_bin = batch["label_binary"].to(device)
        y_mit = batch["label_mitre"].to(device)

        optimizer.zero_grad()
        out = model(state_seq)

        losses = loss_fn(
            out["pred_state_mean"],
            out["pred_state_logvar"],
            next_state,
            out["pred_binary"],
            y_bin,
            out["pred_mitre"],
            y_mit,
        )

        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += losses["total"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            state_seq = batch["state_seq"].to(device)
            next_state = batch["next_state"].to(device)
            y_bin = batch["label_binary"].to(device)
            y_mit = batch["label_mitre"].to(device)

            out = model(state_seq)
            losses = loss_fn(
                out["pred_state_mean"], out["pred_state_logvar"], next_state,
                out["pred_binary"], y_bin,
                out["pred_mitre"], y_mit,
            )
            total_loss += losses["total"].item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="PRISM World Model Training")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--states", default=None, help="Path to states.npz (skips split creation)")
    parser.add_argument("--splits-dir", default=None, help="Path to pre-split directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--arch", default=None, choices=["transformer", "lstm", "gnn", "latent"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="weights")
    args = parser.parse_args()

    # Load config
    overrides = {}
    if args.epochs: overrides["train.epochs"] = args.epochs
    if args.patience: overrides["train.patience"] = args.patience
    if args.batch_size: overrides["train.batch_size"] = args.batch_size
    if args.lr: overrides["train.learning_rate"] = args.lr
    if args.arch: overrides["model.architecture"] = args.arch
    if args.device: overrides["train.device"] = args.device

    cfg = load_config(args.config, overrides)
    device = get_device(cfg.train)
    set_seed(cfg.train.seed)

    logger.info("=" * 60)
    logger.info("PRISM Training | arch=%s | device=%s | epochs=%d",
                cfg.model.architecture, device, cfg.train.epochs)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    splits_dir = args.splits_dir or cfg.data.splits_dir

    if args.states:
        logger.info("Loading states from %s", args.states)
        data = load_states_from_npz(args.states)
        splits = create_temporal_splits(
            data["states"], data["labels_binary"], data["labels_mitre"],
            train_ratio=cfg.data.train_ratio,
            val_ratio=cfg.data.val_ratio,
        )
        save_splits(splits, splits_dir)
    else:
        logger.info("Loading pre-split data from %s", splits_dir)
        splits = {}
        for split_name in ["train", "val", "test"]:
            npz_path = os.path.join(splits_dir, f"{split_name}.npz")
            if not os.path.exists(npz_path):
                raise FileNotFoundError(
                    f"Split file not found: {npz_path}. "
                    f"Run feature extraction first or provide --states."
                )
            splits[split_name] = load_states_from_npz(npz_path)

    # Infer state dimension from data
    d_state = splits["train"]["states"].shape[1]
    logger.info("State dimension: %d", d_state)
    cfg.model.d_state = d_state
    if cfg.model.architecture == "gnn" and cfg.model.d_graph != d_state:
        logger.info(
            "Aligning GNN graph dimension to state dimension: d_graph=%d -> %d",
            cfg.model.d_graph,
            d_state,
        )
        cfg.model.d_graph = d_state

    balanced_sampling = getattr(cfg.train, "balanced_sampling", True)
    use_focal = getattr(cfg.train, "use_focal", True)

    loaders = create_dataloaders(
        splits,
        lookback=cfg.data.lookback,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        balanced_sampling=balanced_sampling,
    )

    # Class weights for imbalanced data
    if balanced_sampling:
        # Since WeightedRandomSampler already provides 50/50 batches, use balanced weights
        binary_weights = torch.ones(2, dtype=torch.float32, device=device)
    else:
        binary_weights = compute_class_weights(
            splits["train"]["labels_binary"], num_classes=2
        ).to(device)
        binary_weights[1] = binary_weights[1] * 1.5
        binary_weights = binary_weights / binary_weights.sum() * 2.0

    from src.utils.constants import NUM_MITRE_STAGES
    mitre_weights = compute_class_weights(
        splits["train"]["labels_mitre"], num_classes=NUM_MITRE_STAGES
    ).to(device)
    logger.info("Binary class weights: %s", binary_weights.cpu().tolist())
    logger.info("MITRE class weights: %s", mitre_weights.cpu().tolist())

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model = build_world_model(cfg.model).to(device)
    loss_fn = MultiTaskLoss(
        lambda_dynamics=cfg.train.lambda_dynamics,
        lambda_infiltration=cfg.train.lambda_infiltration,
        lambda_mitre=cfg.train.lambda_mitre,
        binary_class_weights=binary_weights,
        mitre_class_weights=mitre_weights,
        use_focal=use_focal,
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_score = -999.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "world_model_best.pt")

    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_epoch(
            model, loaders["train"], loss_fn, optimizer, device, cfg.train.grad_clip
        )
        val_loss = validate(model, loaders["val"], loss_fn, device)

        # Quick val metrics
        val_metrics = evaluate_model(model, loaders["val"], device)
        val_f1 = val_metrics["binary"]["f1"]
        val_prec = val_metrics["binary"]["precision"]
        val_rec = val_metrics["binary"]["recall"]
        val_fpr = val_metrics["binary"]["fpr"]
        val_mitre_macro = val_metrics["mitre"]["f1_macro"]

        # Balanced validation score: Maximize F1 and Recall, minimize False Alarm Rate (FPR), reward MITRE progression
        val_score = val_f1 + 0.4 * val_rec - 0.6 * val_fpr + 0.3 * val_mitre_macro

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        logger.info(
            "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
            "val_F1=%.4f | val_MITRE=%.4f | val_FPR=%.4f",
            epoch, cfg.train.epochs, train_loss, val_loss,
            val_f1, val_mitre_macro, val_metrics["binary"]["fpr"],
        )

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        # Checkpoint best model
        if val_score > best_val_score:
            best_val_score = val_score
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch,
                {"val_score": val_score, "val_f1": val_f1, "val_mitre": val_mitre_macro, "val_loss": val_loss},
                best_ckpt_path,
            )
            # Sync to weights/transformer if training primary transformer
            if args.output_dir == "weights" and cfg.model.architecture in ("transformer", "temporal_transformer"):
                tf_dir = os.path.join(args.output_dir, "transformer")
                os.makedirs(tf_dir, exist_ok=True)
                save_checkpoint(
                    model, optimizer, epoch,
                    {"val_score": val_score, "val_f1": val_f1, "val_loss": val_loss},
                    os.path.join(tf_dir, "world_model_best.pt"),
                )
            logger.info("  -> New best model saved (val_score=%.4f, val_F1=%.4f, val_MITRE=%.4f)", val_score, val_f1, val_mitre_macro)
        else:
            patience_counter += 1

        # Periodic checkpoint
        if epoch % cfg.train.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"world_model_epoch{epoch}.pt")
            save_checkpoint(model, optimizer, epoch, {}, ckpt_path)

        # Early stopping
        if patience_counter >= cfg.train.patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d exhausted)",
                epoch, cfg.train.patience,
            )
            break

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading best checkpoint for test evaluation...")
    load_checkpoint(model, best_ckpt_path, device=device)

    test_metrics = evaluate_model(model, loaders["test"], device)

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS")
    logger.info("  Binary F1      : %.4f", test_metrics["binary"]["f1"])
    logger.info("  Precision      : %.4f", test_metrics["binary"]["precision"])
    logger.info("  Recall         : %.4f", test_metrics["binary"]["recall"])
    logger.info("  FPR            : %.4f", test_metrics["binary"]["fpr"])
    logger.info("  MITRE F1_macro : %.4f", test_metrics["mitre"]["f1_macro"])
    logger.info("  Dynamics MSE   : %.6f", test_metrics["dynamics"]["mse"])
    logger.info("  ROC AUC        : %.4f", test_metrics["binary"].get("roc_auc", 0))
    logger.info("=" * 60)

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save final test metrics
    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "binary": {k: v for k, v in test_metrics["binary"].items()
                       if k not in ("report", "confusion_matrix")},
            "mitre": {k: v for k, v in test_metrics["mitre"].items()
                      if k not in ("report", "confusion_matrix")},
            "dynamics": test_metrics["dynamics"],
        }, f, indent=2)

    logger.info("Training complete. Best model -> %s", best_ckpt_path)
    print(test_metrics["binary"]["report"])


if __name__ == "__main__":
    main()
