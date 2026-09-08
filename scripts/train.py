"""
PRISM - High-Performance Training Pipeline
Gen 8 Decoupled Temporal Transformer + Deep Residual MITRE MLP Expert
With Discriminative Telemetry Highway & Infiltration Disambiguation Losses
"""

import argparse
import logging
import os
import sys
import time
import random
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, get_device
from src.utils.logger import setup_logger
from src.utils.constants import MITRE_STAGES_INV, NUM_MITRE_STAGES
from src.models.world_model import build_world_model, save_checkpoint, load_checkpoint
from src.data.dataset import load_states_from_npz, create_dataloaders


class CleanFocalLoss(nn.Module):
    """Clean focal loss with class weighting and smooth label smoothing (gamma=1.5)."""
    def __init__(self, gamma: float = 1.5, label_smoothing: float = 0.01, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0)
        focal_mod = (1.0 - p_t) ** self.gamma
        ce = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.label_smoothing, reduction="none")
        return (focal_mod * ce).mean()


class RobustDynamicsMSE(nn.Module):
    """Robust squared error dynamics loss with outlier clamping."""
    def __init__(self, max_sq_err: float = 5.0):
        super().__init__()
        self.max_sq_err = max_sq_err

    def forward(self, pred_mean: torch.Tensor, pred_logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = (target - pred_mean) ** 2
        return sq_err.clamp(max=self.max_sq_err).mean()


def compute_infil_pair_and_repulsion_loss(
    out: Dict[str, torch.Tensor],
    y_mit: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise disambiguation cross-entropy and latent contrastive repulsion
    for infiltration minority stages (S1, S2, S3, S5).
    """
    loss_pair = torch.tensor(0.0, device=y_mit.device)
    n_pairs = 0

    # 1. Infiltration Pair: Recon (1) vs. Initial Access (2)
    m_12 = (y_mit == 1) | (y_mit == 2)
    if m_12.sum() > 4 and "logits_infil_pair" in out:
        target_12 = (y_mit[m_12] == 2).long()
        loss_pair = loss_pair + F.cross_entropy(out["logits_infil_pair"][m_12], target_12)
        n_pairs += 1

    # 2. Egress Pair: Lateral Movement (3) vs. Exfiltration (5)
    m_35 = (y_mit == 3) | (y_mit == 5)
    if m_35.sum() > 4 and "logits_egress_pair" in out:
        target_35 = (y_mit[m_35] == 5).long()
        loss_pair = loss_pair + F.cross_entropy(out["logits_egress_pair"][m_35], target_35)
        n_pairs += 1

    # 3. Recon vs Lateral: Recon (1) vs. Lateral Movement (3)
    m_13 = (y_mit == 1) | (y_mit == 3)
    if m_13.sum() > 4 and "logits_recon_lateral_pair" in out:
        target_13 = (y_mit[m_13] == 3).long()
        loss_pair = loss_pair + F.cross_entropy(out["logits_recon_lateral_pair"][m_13], target_13)
        n_pairs += 1

    if n_pairs > 0:
        loss_pair = loss_pair / n_pairs

    # 4. Supervised Contrastive Repulsion on Unit Sphere S^127
    loss_repulse = torch.tensor(0.0, device=y_mit.device)
    if "contrastive_z" in out:
        z = out["contrastive_z"]  # (B, 128) l2-normalized
        centroids = {}
        for s in [1, 2, 3, 5]:
            mask_s = (y_mit == s)
            if mask_s.sum() >= 2:
                c = z[mask_s].mean(dim=0)
                centroids[s] = F.normalize(c, p=2, dim=0)

        # Repulse S1 from S3 (Recon vs Lateral)
        if 1 in centroids and 3 in centroids:
            cos_13 = (centroids[1] * centroids[3]).sum()
            loss_repulse = loss_repulse + F.relu(cos_13 - 0.20)

        # Repulse S1 from S2 (Recon vs Init Access)
        if 1 in centroids and 2 in centroids:
            cos_12 = (centroids[1] * centroids[2]).sum()
            loss_repulse = loss_repulse + F.relu(cos_12 - 0.20)

        # Repulse S3 from S5 (Lateral vs Exfil)
        if 3 in centroids and 5 in centroids:
            cos_35 = (centroids[3] * centroids[5]).sum()
            loss_repulse = loss_repulse + F.relu(cos_35 - 0.20)

    return loss_pair, loss_repulse


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def evaluate_fast(model, loader, device, compute_report: bool = False):
    """High-speed GPU evaluation in <1.5s."""
    model.eval()
    all_mitre_probs, all_mitre_true = [], []
    all_bin_probs, all_bin_true = [], []
    total_sq_err = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            state_seq = batch["state_seq"].to(device)
            next_state = batch["next_state"].to(device)
            with torch.amp.autocast("cuda"):
                out = model(state_seq)
            
            all_mitre_probs.append(torch.softmax(out["pred_mitre"], dim=-1).cpu())
            all_mitre_true.append(batch["label_mitre"].cpu())
            all_bin_probs.append(torch.softmax(out["pred_binary"], dim=-1)[:, 1].cpu())
            all_bin_true.append(batch["label_binary"].cpu())
            
            sq_err = ((next_state - out["pred_state_mean"]) ** 2).sum().item()
            total_sq_err += sq_err
            n_samples += next_state.numel()

    mitre_prob = torch.cat(all_mitre_probs).numpy()
    y_true = torch.cat(all_mitre_true).numpy()
    bin_prob = torch.cat(all_bin_probs).numpy()
    y_bin = torch.cat(all_bin_true).numpy()

    # Binary metrics
    bin_preds = (bin_prob >= 0.5).astype(int)
    bin_f1 = float(f1_score(y_bin, bin_preds, zero_division=0))
    bin_prec = float(precision_score(y_bin, bin_preds, zero_division=0))
    bin_rec = float(recall_score(y_bin, bin_preds, zero_division=0))
    
    cm_bin = confusion_matrix(y_bin, bin_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel() if cm_bin.size == 4 else (0, 0, 0, 0)
    bin_fpr = float(fp / max(fp + tn, 1))

    # MITRE metrics
    mitre_preds = np.argmax(mitre_prob, axis=1)
    macro_f1 = float(f1_score(y_true, mitre_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, mitre_preds, average="weighted", zero_division=0))
    per_stage_f1 = f1_score(y_true, mitre_preds, average=None, zero_division=0).tolist()
    
    dyn_mse = float(total_sq_err / max(n_samples, 1))

    res = {
        "binary_f1": bin_f1,
        "binary_precision": bin_prec,
        "binary_recall": bin_rec,
        "binary_fpr": bin_fpr,
        "mitre_macro_f1": macro_f1,
        "mitre_weighted_f1": weighted_f1,
        "per_stage_f1": [float(x) for x in per_stage_f1],
        "dynamics_mse": dyn_mse,
    }

    if compute_report:
        stage_names = [MITRE_STAGES_INV.get(i, f"Stage_{i}") for i in range(NUM_MITRE_STAGES)]
        res["binary_report"] = classification_report(y_bin, bin_preds, zero_division=0)
        res["mitre_report"] = classification_report(y_true, mitre_preds, target_names=stage_names, zero_division=0)
        res["mitre_cm"] = confusion_matrix(y_true, mitre_preds).tolist()

    return res


def main():
    parser = argparse.ArgumentParser(description="PRISM High-Throughput World Model Training")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="weights/universal_gen8")
    args = parser.parse_args()

    overrides = {}
    if args.epochs: overrides["train.epochs"] = args.epochs
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

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    
    logger = setup_logger("prism.train", log_dir=cfg.train.log_dir)

    logger.info("=" * 65)
    logger.info("PRISM Gen 8 Training | arch=%s | device=%s | epochs=%d",
                cfg.model.architecture, device, cfg.train.epochs)
    logger.info("=" * 65)

    splits_dir = args.splits_dir or cfg.data.splits_dir
    logger.info("Loading pre-split data from: %s", splits_dir)
    
    splits = {
        s: load_states_from_npz(os.path.join(splits_dir, f"{s}.npz"))
        for s in ["train", "val", "test"]
    }
    
    d_state = splits["train"]["states"].shape[1]
    cfg.model.d_state = d_state
    logger.info("State dimension: %d | Train windows: %d | Val: %d | Test: %d",
                d_state, len(splits["train"]["states"]), len(splits["val"]["states"]), len(splits["test"]["states"]))

    lookback = getattr(cfg.data, "lookback", 20)
    batch_size = getattr(cfg.train, "batch_size", 2048)
    stride = getattr(cfg.data, "stride", 2)

    loaders = create_dataloaders(
        splits,
        lookback=lookback,
        batch_size=batch_size,
        stride=stride,
        balanced_sampling=True,
        device=device,
    )

    model = build_world_model(cfg.model).to(device)
    logger.info("Model parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    dyn_loss_fn = RobustDynamicsMSE(max_sq_err=getattr(cfg.train, "max_sq_err", 5.0))
    bin_loss_fn = CleanFocalLoss(gamma=1.5, label_smoothing=getattr(cfg.train, "label_smoothing", 0.01))
    
    # Class-weighted MITRE Focal Loss: prioritize S1, S2, S3, S5
    mitre_weights = torch.tensor([0.5, 2.5, 2.5, 2.5, 0.8, 3.0, 0.8], device=device, dtype=torch.float32)
    mit_loss_fn = CleanFocalLoss(gamma=1.5, label_smoothing=getattr(cfg.train, "label_smoothing", 0.01), weight=mitre_weights)

    lr = getattr(cfg.train, "learning_rate", 5e-4)
    weight_decay = getattr(cfg.train, "weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_score = -999.0
    best_ckpt_path = os.path.join(args.output_dir, "world_model_best.pt")
    history = {"train_loss": [], "val_f1": [], "val_mitre": [], "val_fpr": []}

    lambda_d = getattr(cfg.train, "lambda_dynamics", 0.1)
    lambda_i = getattr(cfg.train, "lambda_infiltration", 2.0)
    lambda_m = getattr(cfg.train, "lambda_mitre", 5.0)
    lambda_pair = getattr(cfg.train, "lambda_pair", 1.5)
    lambda_repulse = getattr(cfg.train, "lambda_repulse", 1.0)

    logger.info("Loss weights: Dynamics=%.2f, Binary=%.2f, MITRE=%.2f, Pair=%.2f, Repulse=%.2f | Focal Gamma=1.5",
                lambda_d, lambda_i, lambda_m, lambda_pair, lambda_repulse)
    logger.info("Starting high-throughput training (%d batches/epoch)...", len(loaders["train"]))

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_mit = 0.0, 0.0

        for batch in loaders["train"]:
            state_seq = batch["state_seq"].to(device)
            next_state = batch["next_state"].to(device)
            y_bin = batch["label_binary"].to(device)
            y_mit = batch["label_mitre"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(state_seq)
                l_dyn = dyn_loss_fn(out["pred_state_mean"], out["pred_state_logvar"], next_state)
                l_bin = bin_loss_fn(out["pred_binary"], y_bin)
                l_mit = mit_loss_fn(out["pred_mitre"], y_mit)
                l_pair, l_repulse = compute_infil_pair_and_repulsion_loss(out, y_mit)

                total = (
                    lambda_d * l_dyn
                    + lambda_i * l_bin
                    + lambda_m * l_mit
                    + lambda_pair * l_pair
                    + lambda_repulse * l_repulse
                )

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += total.item()
            total_mit += l_mit.item()

        scheduler.step()
        n_b = len(loaders["train"])
        avg_loss = total_loss / n_b
        avg_mit = total_mit / n_b

        # High-speed validation evaluation (~1.4s)
        val_res = evaluate_fast(model, loaders["val"], device)
        elapsed = time.time() - t0

        stages_str = ", ".join([f"S{i}:{f1:.3f}" for i, f1 in enumerate(val_res["per_stage_f1"])])
        
        # Validation score
        val_score = 1.5 * val_res["mitre_macro_f1"] + 0.8 * val_res["binary_f1"] + 0.5 * val_res["binary_recall"] - 0.7 * val_res["binary_fpr"]

        history["train_loss"].append(avg_loss)
        history["val_f1"].append(val_res["binary_f1"])
        history["val_mitre"].append(val_res["mitre_macro_f1"])
        history["val_fpr"].append(val_res["binary_fpr"])

        logger.info(
            "Epoch %2d/%d (%4.1fs) | Loss=%.3f (MIT=%.3f) | Bin_F1=%.4f (FPR=%.4f) | MITRE_Macro_F1=%.4f | [%s]",
            epoch, cfg.train.epochs, elapsed, avg_loss, avg_mit,
            val_res["binary_f1"], val_res["binary_fpr"], val_res["mitre_macro_f1"], stages_str,
        )

        if val_score > best_val_score:
            best_val_score = val_score
            save_checkpoint(
                model, optimizer, epoch,
                {"val_score": val_score, "metrics": val_res},
                best_ckpt_path,
            )

    # Final evaluation on test set
    logger.info("=" * 65)
    logger.info("Training complete. Loading best checkpoint from %s for test evaluation...", best_ckpt_path)
    load_checkpoint(model, best_ckpt_path, device=device)
    
    test_res = evaluate_fast(model, loaders["test"], device, compute_report=True)
    
    logger.info("=" * 65)
    logger.info("FINAL TEST RESULTS (Universal 4-Dataset Benchmark)")
    logger.info("  Binary Detection F1 : %.4f  (Recall: %.4f, FPR: %.4f)",
                test_res["binary_f1"], test_res["binary_recall"], test_res["binary_fpr"])
    logger.info("  MITRE Macro F1      : %.4f  (Weighted F1: %.4f)",
                test_res["mitre_macro_f1"], test_res["mitre_weighted_f1"])
    logger.info("  Dynamics MSE        : %.6f", test_res["dynamics_mse"])
    stages_test_str = ", ".join([f"S{i}:{f1:.3f}" for i, f1 in enumerate(test_res["per_stage_f1"])])
    logger.info("  Per-Stage MITRE F1  : [%s]", stages_test_str)
    logger.info("=" * 65)

    # Save metrics and history
    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump({k: v for k, v in test_res.items() if not k.endswith("_report") and k != "mitre_cm"}, f, indent=2)

    logger.info("\n--- Binary Detection Report ---\n%s", test_res["binary_report"])
    logger.info("\n--- MITRE ATT&CK Classification Report ---\n%s", test_res["mitre_report"])


if __name__ == "__main__":
    main()
