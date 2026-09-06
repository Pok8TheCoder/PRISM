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
from src.evaluation.metrics import evaluate_model, evaluate_with_calibration

logger = setup_logger("prism.train", log_dir="results/logs")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


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


def train_epoch(model, loader, loss_fn, optimizer, device, grad_clip, scaler=None):
    model.train()
    total_loss = 0.0
    total_dyn = 0.0
    total_inf = 0.0
    total_mit = 0.0
    n_batches = 0
    use_amp = (device.type == "cuda" and scaler is not None)

    for batch in loader:
        state_seq = batch["state_seq"].to(device)
        next_state = batch["next_state"].to(device)
        y_bin = batch["label_binary"].to(device)
        y_mit = batch["label_mitre"].to(device)

        # GPU-vectorized minority augmentation (Stages 1, 2, 3, 5)
        # Runs in <1ms on CUDA without any CPU DataLoader overhead
        if model.training:
            minority_mask = (y_mit == 1) | (y_mit == 2) | (y_mit == 3) | (y_mit == 5)
            idx_m = torch.where(minority_mask)[0]
            if len(idx_m) > 1 and torch.rand(1, device=device).item() < 0.70:
                perm = idx_m[torch.randperm(len(idx_m), device=device)]
                lam = torch.empty(len(idx_m), 1, 1, device=device).uniform_(0.3, 0.7)
                lam_s = lam.squeeze(-1)
                state_seq[idx_m] = lam * state_seq[idx_m] + (1.0 - lam) * state_seq[perm]
                next_state[idx_m] = lam_s * next_state[idx_m] + (1.0 - lam_s) * next_state[perm]
                # Subtle feature jitter for manifold smoothness
                state_seq[idx_m] = state_seq[idx_m] + torch.randn_like(state_seq[idx_m]) * 0.01
                next_state[idx_m] = next_state[idx_m] + torch.randn_like(next_state[idx_m]) * 0.01

        optimizer.zero_grad()
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(state_seq)
                losses = loss_fn(
                    out["pred_state_mean"],
                    out["pred_state_logvar"],
                    next_state,
                    out["pred_binary"],
                    y_bin,
                    out["pred_mitre"],
                    y_mit,
                    contrastive_z=out.get("contrastive_z", None),
                )
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(state_seq)
            losses = loss_fn(
                out["pred_state_mean"],
                out["pred_state_logvar"],
                next_state,
                out["pred_binary"],
                y_bin,
                out["pred_mitre"],
                y_mit,
                contrastive_z=out.get("contrastive_z", None),
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += losses["total"].item()
        total_dyn += losses["dynamics"].item()
        total_inf += losses["infiltration"].item()
        total_mit += losses["mitre"].item()
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "total": total_loss / denom,
        "dynamics": total_dyn / denom,
        "infiltration": total_inf / denom,
        "mitre": total_mit / denom,
    }


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in loader:
            state_seq = batch["state_seq"].to(device)
            next_state = batch["next_state"].to(device)
            y_bin = batch["label_binary"].to(device)
            y_mit = batch["label_mitre"].to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(state_seq)
                    losses = loss_fn(
                        out["pred_state_mean"], out["pred_state_logvar"], next_state,
                        out["pred_binary"], y_bin,
                        out["pred_mitre"], y_mit,
                        contrastive_z=out.get("contrastive_z", None),
                    )
            else:
                out = model(state_seq)
                losses = loss_fn(
                    out["pred_state_mean"], out["pred_state_logvar"], next_state,
                    out["pred_binary"], y_bin,
                    out["pred_mitre"], y_mit,
                    contrastive_z=out.get("contrastive_z", None),
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
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--arch", default=None, choices=["gen7", "transformer", "timesfm", "gen6", "gen5", "gen4", "gen3", "gen2", "gen1", "lstm", "gnn", "latent"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="weights")
    args = parser.parse_args()

    # Load config
    overrides = {}
    if args.epochs: overrides["train.epochs"] = args.epochs
    if args.patience: overrides["train.patience"] = args.patience
    if args.batch_size: overrides["train.batch_size"] = args.batch_size
    if args.lr: overrides["train.learning_rate"] = args.lr
    if args.lookback:
        overrides["data.lookback"] = args.lookback
        overrides["model.lookback"] = args.lookback
    if args.arch: overrides["model.architecture"] = args.arch
    if args.device: overrides["train.device"] = args.device

    cfg = load_config(args.config, overrides)
    device = torch.device(get_device(cfg.train))
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
        device=device,
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
    asymmetric_fn_weight = getattr(cfg.train, "asymmetric_fn_weight", 3.5)
    max_sq_err = getattr(cfg.train, "max_sq_err", 5.0)
    label_smoothing = getattr(cfg.train, "label_smoothing", 0.01)
    lambda_contrastive = getattr(cfg.train, "lambda_contrastive", 0.0)
    contrastive_temp = getattr(cfg.train, "contrastive_temp", 0.07)

    loss_fn = MultiTaskLoss(
        lambda_dynamics=cfg.train.lambda_dynamics,
        lambda_infiltration=cfg.train.lambda_infiltration,
        lambda_mitre=cfg.train.lambda_mitre,
        lambda_contrastive=lambda_contrastive,
        contrastive_temp=contrastive_temp,
        binary_class_weights=binary_weights,
        mitre_class_weights=mitre_weights,
        use_focal=use_focal,
        asymmetric_fn_weight=asymmetric_fn_weight,
        label_smoothing=label_smoothing,
        max_sq_err=max_sq_err,
    ).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(loaders["train"]))

    # ------------------------------------------------------------------
    # Fix G: two-phase training setup
    # Phase 1: train all parameters (dynamics + classifiers together)
    # Phase 2: freeze dynamics/backbone, train only classifier heads
    # ------------------------------------------------------------------
    phase2_start = getattr(cfg.train, "phase2_start_epoch", int(cfg.train.epochs * 0.6))
    phase2_started = False

    def _freeze_dynamics(model):
        """Freeze all parameters except MITRE head, binary head, contrastive head."""
        classifier_prefixes = (
            "head_binary", "head_mitre", "contrastive_head",
            "mitre_transformer", "mitre_pooler",    # Path C (Fix F)
            "threat_pooler",                        # fused representation
        )
        frozen, active = 0, 0
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in classifier_prefixes):
                param.requires_grad = True
                active += param.numel()
            else:
                param.requires_grad = False
                frozen += param.numel()
        logger.info(
            "Fix-G Phase 2: froze %.2fM params, kept %.2fM params active",
            frozen / 1e6, active / 1e6,
        )
        return frozen, active

    def _rebuild_optimizer_phase2(model, cfg):
        """New optimizer over only the active (unfrozen) parameters."""
        active_params = [p for p in model.parameters() if p.requires_grad]
        # Slightly higher LR for Phase 2 so the head adapts faster
        lr_phase2 = cfg.train.learning_rate * 1.5
        return torch.optim.AdamW(active_params, lr=lr_phase2, weight_decay=cfg.train.weight_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_score = -999.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_mitre": []}

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "world_model_best.pt")

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if scaler is not None:
        logger.info("AMP Tensor Cores Active: RTX 3050 hardware acceleration enabled")

    for epoch in range(1, cfg.train.epochs + 1):
        # ------------------------------------------------------------------
        # Fix G: transition to Phase 2 when epoch reaches phase2_start
        # ------------------------------------------------------------------
        if not phase2_started and epoch >= phase2_start:
            logger.info(
                "Fix-G: entering Phase 2 at epoch %d (classifier-only training)", epoch
            )
            _freeze_dynamics(model)
            optimizer = _rebuild_optimizer_phase2(model, cfg)
            # Restart cosine schedule over remaining epochs
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, cfg.train.epochs - epoch + 1),
                eta_min=1e-6,
            )
            phase2_started = True

        train_loss_dict = train_epoch(
            model, loaders["train"], loss_fn, optimizer, device, cfg.train.grad_clip, scaler=scaler
        )
        train_loss = train_loss_dict["total"]
        val_loss = validate(model, loaders["val"], loss_fn, device)

        # Quick val metrics
        val_metrics = evaluate_model(model, loaders["val"], device)
        val_f1 = val_metrics["binary"]["f1"]
        val_prec = val_metrics["binary"]["precision"]
        val_rec = val_metrics["binary"]["recall"]
        val_fpr = val_metrics["binary"]["fpr"]
        val_mitre_macro = val_metrics["mitre"]["f1_macro"]

        # High-impact MITRE validation score: Target 90%+ MITRE F1, maximize Recall (low FNR), penalize False Alarms (FPR)
        val_score = 1.5 * val_mitre_macro + 0.8 * val_f1 + 0.5 * val_rec - 0.7 * val_fpr

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_mitre"].append(val_mitre_macro)

        logger.info(
            "Epoch %3d/%d | loss=%.4f (dyn=%.4f, inf=%.4f, mit=%.4f) | "
            "val_F1=%.4f | val_MITRE=%.4f | val_FPR=%.4f",
            epoch, cfg.train.epochs, train_loss,
            train_loss_dict["dynamics"], train_loss_dict["infiltration"], train_loss_dict["mitre"],
            val_f1, val_mitre_macro, val_metrics["binary"]["fpr"],
        )
        for _h in logger.handlers:
            _h.flush()

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
            # Sync named checkpoints
            if cfg.model.architecture in ("gen6", "gen6_timesfm", "timesfm"):
                gen6_dir = os.path.join(args.output_dir, "gen6")
                os.makedirs(gen6_dir, exist_ok=True)
                save_checkpoint(
                    model, optimizer, epoch,
                    {"val_score": val_score, "val_f1": val_f1, "val_mitre": val_mitre_macro, "val_loss": val_loss},
                    os.path.join(args.output_dir, "gen6_timesfm_best.pt"),
                )
                save_checkpoint(
                    model, optimizer, epoch,
                    {"val_score": val_score, "val_f1": val_f1, "val_mitre": val_mitre_macro, "val_loss": val_loss},
                    os.path.join(gen6_dir, "world_model_best.pt"),
                )
            elif args.output_dir == "weights" and cfg.model.architecture in ("transformer", "temporal_transformer", "gen5"):
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

    # Fix E: calibrate per-stage MITRE thresholds on val, then evaluate test
    logger.info("Running Fix-E calibrated evaluation on test set ...")
    test_metrics, mitre_thresholds = evaluate_with_calibration(
        model,
        val_loader=loaders["val"],
        test_loader=loaders["test"],
        device=device,
    )
    # Also run uncalibrated for comparison
    test_metrics_raw = evaluate_model(model, loaders["test"], device)

    logger.info("=" * 60)
    logger.info("FINAL TEST RESULTS (Fix-E calibrated)")
    logger.info("  Binary F1      : %.4f", test_metrics["binary"]["f1"])
    logger.info("  Precision      : %.4f", test_metrics["binary"]["precision"])
    logger.info("  Recall         : %.4f", test_metrics["binary"]["recall"])
    logger.info("  FPR            : %.4f", test_metrics["binary"]["fpr"])
    logger.info("  MITRE F1_macro : %.4f  (raw argmax: %.4f)",
                test_metrics["mitre"]["f1_macro"],
                test_metrics_raw["mitre"]["f1_macro"])
    logger.info("  Dynamics MSE   : %.6f", test_metrics["dynamics"]["mse"])
    logger.info("  ROC AUC        : %.4f", test_metrics["binary"].get("roc_auc", 0))
    logger.info("  MITRE thresholds: %s", np.round(mitre_thresholds, 3).tolist())
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
                       if k != "report"},
            "mitre": {k: v for k, v in test_metrics["mitre"].items()
                      if k != "report"},
            "mitre_raw": {k: v for k, v in test_metrics_raw["mitre"].items()
                          if k != "report"},
            "dynamics": test_metrics["dynamics"],
            "mitre_thresholds": mitre_thresholds.tolist(),
        }, f, indent=2)

    logger.info("Training complete. Best model -> %s", best_ckpt_path)
    logger.info("\n--- Binary Classification Report ---\n%s", test_metrics["binary"]["report"])
    logger.info("\n--- MITRE ATT&CK Classification Report ---\n%s", test_metrics["mitre"]["report"])
    cm_str = "\n".join([" ".join([f"{val:6d}" for val in row]) for row in test_metrics["mitre"]["confusion_matrix"]])
    logger.info("\n--- MITRE ATT&CK Confusion Matrix ---\n%s", cm_str)



if __name__ == "__main__":
    main()
