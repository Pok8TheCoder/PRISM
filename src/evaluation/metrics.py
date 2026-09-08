"""
PRISM - Evaluation Metrics
F1, Precision, Recall, FPR, Early Warning Time, State Prediction MSE.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score,
    classification_report, average_precision_score,
)

logger = logging.getLogger("prism.evaluation.metrics")

# Number of candidate thresholds to sweep per MITRE stage (Fix E)
_THRESHOLD_GRID_STEPS = 50


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : (N,) ground-truth binary labels (0=benign, 1=attack)
    y_pred : (N,) predicted binary labels
    y_prob : (N,) optional predicted probabilities for P(attack)

    Returns
    -------
    dict with f1, precision, recall, fpr, tpr, roc_auc, avg_precision
    """
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_true, y_pred, average="binary", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)

    metrics = {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["avg_precision"] = float(
                average_precision_score(y_true, y_prob)
            )
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["avg_precision"] = 0.0

    return metrics


def compute_mitre_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stage_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute MITRE stage classification metrics (multi-class).

    Parameters
    ----------
    y_true       : (N,) integer stage ids
    y_pred       : (N,) predicted stage ids
    stage_names  : optional list of class names for the report
    """
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_stage = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    acc = float((y_true == y_pred).mean())

    return {
        "accuracy": acc,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_stage": [float(x) for x in f1_per_stage],
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred,
            labels=list(range(len(stage_names))) if stage_names else None,
            target_names=stage_names,
            zero_division=0,
        ),
    }


def compute_dynamics_metrics(
    pred_states: np.ndarray,
    true_states: np.ndarray,
) -> dict:
    """
    Evaluate quality of the world model's state prediction (dynamics head).

    Parameters
    ----------
    pred_states : (N, D_state) — predicted next states
    true_states : (N, D_state) — actual next states

    Returns
    -------
    dict with mse, rmse, mae, per-feature mse
    """
    diff = pred_states - true_states
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    per_feature_mse = np.mean(diff ** 2, axis=0).tolist()

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "per_feature_mse": per_feature_mse,
    }


def compute_early_warning_score(
    rollout_probs: np.ndarray,
    actual_attack_step: int,
    threshold: float = 0.40,
) -> dict:
    """
    Measure how many steps before the true attack the model raises an alert.

    Parameters
    ----------
    rollout_probs     : (K,) — infiltration prob per rollout step
    actual_attack_step: the ground-truth step at which attack occurs
    threshold         : alert threshold

    Returns
    -------
    dict with early_warning_steps, alerted_before, detection_delay
    """
    first_alert = None
    for i, p in enumerate(rollout_probs):
        if p >= threshold:
            first_alert = i
            break

    if first_alert is None:
        return {
            "early_warning_steps": 0,
            "alerted_before": False,
            "detection_delay": None,
            "missed_attack": True,
        }

    delta = actual_attack_step - first_alert
    return {
        "early_warning_steps": max(0, delta),
        "alerted_before": first_alert < actual_attack_step,
        "detection_delay": max(0, -delta),
        "missed_attack": False,
        "alert_step": first_alert,
    }


def _collect_logits(
    model,
    dataloader,
    device: str,
):
    """Collect raw model outputs (logits/probs) without making predictions.
    Returns (binary_prob, mitre_prob, y_binary, y_mitre, pred_states, true_states).
    """
    import torch

    model.eval()
    all_binary_prob, all_mitre_prob = [], []
    all_binary_true, all_mitre_true = [], []
    all_pred_states, all_true_states = [], []
    dev = torch.device(device) if isinstance(device, str) else device
    use_amp = (dev.type == "cuda")

    with torch.no_grad():
        for batch in dataloader:
            state_seq = batch["state_seq"].to(dev)
            next_state = batch["next_state"].to(dev)
            y_binary = batch["label_binary"].to(dev)
            y_mitre = batch["label_mitre"].to(dev)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(state_seq)
            else:
                out = model(state_seq)

            binary_probs = torch.softmax(out["pred_binary"], dim=-1)[:, 1]   # P(attack)
            mitre_probs = torch.softmax(out["pred_mitre"], dim=-1)           # (B, 7)

            all_binary_prob.append(binary_probs)
            all_mitre_prob.append(mitre_probs)
            all_binary_true.append(y_binary)
            all_mitre_true.append(y_mitre)
            all_pred_states.append(out["pred_state_mean"])
            all_true_states.append(next_state)

    return (
        torch.cat(all_binary_prob).cpu().numpy(),
        torch.cat(all_mitre_prob, dim=0).cpu().numpy(),
        torch.cat(all_binary_true).cpu().numpy(),
        torch.cat(all_mitre_true).cpu().numpy(),
        torch.cat(all_pred_states, dim=0).cpu().numpy(),
        torch.cat(all_true_states, dim=0).cpu().numpy(),
    )


def calibrate_temperature(
    mitre_prob: np.ndarray,
    y_mitre: np.ndarray,
    num_stages: int = 7,
) -> tuple:
    """Temperature Scaling calibration for MITRE head.

    Learns a single temperature T on the validation set that maximizes
    MITRE Macro F1.  Temperature scaling preserves the ranking of classes
    (so argmax is unchanged at T=1) but can sharpen or soften the softmax
    distribution to improve macro-averaged metrics on imbalanced classes.

    Parameters
    ----------
    mitre_prob  : (N, num_stages) softmax probabilities
    y_mitre     : (N,) ground-truth MITRE stage labels
    num_stages  : number of MITRE stages (default 7)

    Returns
    -------
    (best_T, best_f1) : optimal temperature and corresponding val F1
    """
    # Recover log-probabilities (inverse softmax up to constant)
    mitre_logprob = np.log(mitre_prob + 1e-8)

    best_f1, best_T = -1.0, 1.0
    for T in np.linspace(0.1, 5.0, 100):
        scaled = mitre_logprob / T
        # Recompute softmax from scaled logits
        exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs_scaled = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
        preds = np.argmax(probs_scaled, axis=1)
        f1 = f1_score(y_mitre, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_T = f1, T

    logger.info(
        "Temperature calibration: best T=%.3f (val MITRE Macro F1=%.4f)",
        best_T, best_f1,
    )
    return best_T, best_f1


def calibrate_mitre_thresholds(
    mitre_prob: np.ndarray,
    y_mitre: np.ndarray,
    binary_prob: np.ndarray,
    binary_threshold: float = 0.5,
    num_stages: int = 7,
) -> np.ndarray:
    """Fix E: Per-stage threshold calibration.

    For each MITRE stage c, sweep candidate thresholds over [0, 1] and
    select the one that maximises F1 for that stage on the calibration
    (val) set.  At inference, a sample is assigned stage c when
    ``mitre_prob[:, c] >= thresholds[c]`` (with ties broken by argmax).

    Parameters
    ----------
    mitre_prob      : (N, num_stages) softmax probabilities from the MITRE head
    y_mitre         : (N,) ground-truth MITRE stage labels
    binary_prob     : (N,) P(attack) from the binary head
    binary_threshold: threshold applied to binary head first (hierarchical)
    num_stages      : number of MITRE stages (default 7)

    Returns
    -------
    thresholds : (num_stages,) float array, one per stage
    """
    is_attack = binary_prob >= binary_threshold
    thresholds = np.full(num_stages, 1.0 / num_stages)  # default = uniform

    grid = np.linspace(0.05, 0.95, _THRESHOLD_GRID_STEPS)

    for c in range(num_stages):
        # Only tune on samples the binary head calls "attack" (hierarchical)
        if c == 0:
            # Benign stage: among samples binary says benign, all go to stage 0
            # No per-stage threshold needed for Benign; keep default.
            continue

        best_f1 = -1.0
        best_thr = thresholds[c]
        y_binary_c = (y_mitre == c).astype(int)  # one-vs-rest for stage c

        for thr in grid:
            # Predict stage c if: binary says attack AND stage-c prob >= thr
            pred_c = ((mitre_prob[:, c] >= thr) & is_attack).astype(int)
            f1 = f1_score(y_binary_c, pred_c, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr

        thresholds[c] = best_thr
        logger.info(
            "Fix-E calibration | Stage %d: threshold=%.3f (val F1=%.4f)",
            c, best_thr, best_f1,
        )

    return thresholds


def _predict_with_thresholds(
    binary_prob: np.ndarray,
    mitre_prob: np.ndarray,
    binary_threshold: float,
    mitre_thresholds: Optional[np.ndarray],
) -> tuple:
    """Apply binary + MITRE thresholds to produce final predictions."""
    binary_preds = (binary_prob >= binary_threshold).astype(int)
    is_attack = binary_preds.astype(bool)

    if mitre_thresholds is not None:
        # Per-stage threshold: pick the attack stage whose prob exceeds its
        # calibrated threshold by the largest margin (confidence-weighted).
        # Falls back to argmax if no stage exceeds its threshold.
        num_stages = mitre_prob.shape[1]
        margins = mitre_prob - mitre_thresholds[np.newaxis, :]   # (N, 7)
        margins[:, 0] = -np.inf                                  # ignore benign column
        best_attack_stage = np.argmax(margins, axis=1)           # (N,)
        # If no attack stage exceeds its threshold, fall back to plain argmax
        no_winner = (margins.max(axis=1) < 0)
        fallback = 1 + mitre_prob[:, 1:].argmax(axis=1)
        best_attack_stage[no_winner] = fallback[no_winner]
    else:
        best_attack_stage = 1 + mitre_prob[:, 1:].argmax(axis=1)

    mitre_preds = np.where(is_attack, best_attack_stage, 0)
    return binary_preds, mitre_preds


def evaluate_model(
    model,
    dataloader,
    device: str = "cpu",
    threshold: float = 0.5,
    mitre_thresholds: Optional[np.ndarray] = None,
) -> dict:
    """
    Full evaluation of a world model over a DataLoader.

    Parameters
    ----------
    mitre_thresholds : optional (num_stages,) array from calibrate_mitre_thresholds().
        When provided, per-stage thresholds are applied instead of plain argmax
        (Fix E).  Pass None to use the original argmax behaviour.

    Returns all metrics: binary, MITRE, dynamics.
    Also returns ``_logits`` key with raw probs for downstream calibration.
    """
    binary_prob, mitre_prob, y_binary, y_mitre, pred_states, true_states = \
        _collect_logits(model, dataloader, device)

    binary_preds, mitre_preds = _predict_with_thresholds(
        binary_prob, mitre_prob, threshold, mitre_thresholds
    )

    from src.utils.constants import MITRE_STAGES_INV

    binary_metrics = compute_binary_metrics(y_binary, binary_preds, binary_prob)
    mitre_metrics = compute_mitre_metrics(
        y_mitre, mitre_preds,
        stage_names=list(MITRE_STAGES_INV.values()),
    )
    dynamics_metrics = compute_dynamics_metrics(pred_states, true_states)

    logger.info(
        "Evaluation | Binary F1=%.4f, Prec=%.4f, Rec=%.4f, FPR=%.4f | "
        "MITRE F1_macro=%.4f | Dynamics MSE=%.6f",
        binary_metrics["f1"], binary_metrics["precision"],
        binary_metrics["recall"], binary_metrics["fpr"],
        mitre_metrics["f1_macro"], dynamics_metrics["mse"],
    )

    return {
        "binary": binary_metrics,
        "mitre": mitre_metrics,
        "dynamics": dynamics_metrics,
        # Raw probs stored so callers can run calibration without a second pass
        "_logits": {"binary_prob": binary_prob, "mitre_prob": mitre_prob},
    }


def evaluate_with_calibration(
    model,
    val_loader,
    test_loader,
    device: str = "cpu",
    binary_threshold: float = 0.5,
) -> tuple:
    """Best-of-three calibration: raw argmax vs temperature scaling vs per-stage thresholds.

    1. Runs a forward pass over ``val_loader`` to collect logits.
    2. Evaluates three MITRE prediction strategies on val:
       a) Raw argmax (no calibration)
       b) Temperature scaling (single scalar T)
       c) Per-stage threshold sweep (Fix-E legacy)
    3. Picks whichever strategy gives best val MITRE Macro F1.
    4. Evaluates test set with the winning strategy.

    Returns
    -------
    (test_metrics_dict, mitre_thresholds_array_or_None)
    """
    logger.info("Calibration: collecting val set predictions ...")
    binary_prob_val, mitre_prob_val, y_binary_val, y_mitre_val, _, _ = \
        _collect_logits(model, val_loader, device)

    # --- Strategy A: Raw argmax ---
    raw_preds = np.argmax(mitre_prob_val, axis=1)
    raw_f1 = f1_score(y_mitre_val, raw_preds, average="macro", zero_division=0)
    logger.info("Strategy A (raw argmax): val MITRE Macro F1 = %.4f", raw_f1)

    # --- Strategy B: Temperature scaling ---
    best_T, temp_f1 = calibrate_temperature(mitre_prob_val, y_mitre_val)
    logger.info("Strategy B (temperature T=%.3f): val MITRE Macro F1 = %.4f", best_T, temp_f1)

    # --- Strategy C: Per-stage thresholds (legacy Fix-E) ---
    mitre_thresholds = calibrate_mitre_thresholds(
        mitre_prob_val, y_mitre_val, binary_prob_val,
        binary_threshold=binary_threshold,
    )
    # Evaluate per-stage on val
    _, thresh_preds = _predict_with_thresholds(
        binary_prob_val, mitre_prob_val, binary_threshold, mitre_thresholds,
    )
    thresh_f1 = f1_score(y_mitre_val, thresh_preds, average="macro", zero_division=0)
    logger.info("Strategy C (per-stage thresholds): val MITRE Macro F1 = %.4f", thresh_f1)

    # --- Pick best strategy ---
    strategies = {
        "raw_argmax": raw_f1,
        "temperature": temp_f1,
        "per_stage_threshold": thresh_f1,
    }
    best_strategy = max(strategies, key=strategies.get)
    logger.info(
        "Best calibration strategy: %s (F1=%.4f) | all: %s",
        best_strategy, strategies[best_strategy],
        {k: f"{v:.4f}" for k, v in strategies.items()},
    )

    # --- Evaluate test set with winning strategy ---
    if best_strategy == "per_stage_threshold":
        logger.info("Evaluating test set with per-stage thresholds ...")
        test_metrics = evaluate_model(
            model, test_loader, device,
            threshold=binary_threshold,
            mitre_thresholds=mitre_thresholds,
        )
    elif best_strategy == "temperature":
        logger.info("Evaluating test set with temperature scaling T=%.3f ...", best_T)
        test_metrics = evaluate_model_with_temperature(
            model, test_loader, device,
            temperature=best_T,
            binary_threshold=binary_threshold,
        )
        # Store temperature in thresholds for logging compatibility
        mitre_thresholds = np.full(7, best_T)  # placeholder
    else:
        logger.info("Evaluating test set with raw argmax (no calibration) ...")
        test_metrics = evaluate_model(
            model, test_loader, device,
            threshold=binary_threshold,
            mitre_thresholds=None,
        )
        mitre_thresholds = np.full(7, 1.0 / 7)  # default

    return test_metrics, mitre_thresholds


def evaluate_model_with_temperature(
    model,
    dataloader,
    device: str = "cpu",
    temperature: float = 1.0,
    binary_threshold: float = 0.5,
) -> dict:
    """Evaluate model using temperature-scaled MITRE softmax predictions.

    Same as evaluate_model but applies temperature scaling to MITRE logits
    before argmax. This can improve macro F1 on imbalanced classes by
    adjusting confidence calibration.
    """
    import torch

    binary_prob, mitre_prob, y_binary, y_mitre, pred_states, true_states = \
        _collect_logits(model, dataloader, device)

    # Apply temperature scaling
    mitre_logprob = np.log(mitre_prob + 1e-8)
    scaled = mitre_logprob / temperature
    exp_scaled = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    mitre_prob_scaled = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)

    # Binary + argmax on scaled probs
    binary_preds = (binary_prob >= binary_threshold).astype(int)
    is_attack = binary_preds.astype(bool)
    best_attack_stage = np.argmax(mitre_prob_scaled[:, 1:], axis=1) + 1
    mitre_preds = np.where(is_attack, best_attack_stage, 0)

    from src.utils.constants import MITRE_STAGES_INV

    binary_metrics = compute_binary_metrics(y_binary, binary_preds, binary_prob)
    mitre_metrics = compute_mitre_metrics(
        y_mitre, mitre_preds,
        stage_names=list(MITRE_STAGES_INV.values()),
    )
    dynamics_metrics = compute_dynamics_metrics(pred_states, true_states)

    logger.info(
        "Evaluation (T=%.2f) | Binary F1=%.4f, Prec=%.4f, Rec=%.4f, FPR=%.4f | "
        "MITRE F1_macro=%.4f | Dynamics MSE=%.6f",
        temperature,
        binary_metrics["f1"], binary_metrics["precision"],
        binary_metrics["recall"], binary_metrics["fpr"],
        mitre_metrics["f1_macro"], dynamics_metrics["mse"],
    )

    return {
        "binary": binary_metrics,
        "mitre": mitre_metrics,
        "dynamics": dynamics_metrics,
        "_logits": {"binary_prob": binary_prob, "mitre_prob": mitre_prob},
    }
