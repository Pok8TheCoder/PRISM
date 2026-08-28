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
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    acc = float((y_true == y_pred).mean())

    return {
        "accuracy": acc,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
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


def evaluate_model(
    model,
    dataloader,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:
    """
    Full evaluation of a world model over a DataLoader.

    Returns all metrics: binary, MITRE, dynamics.
    """
    import torch

    model.eval()
    all_binary_true, all_binary_pred, all_binary_prob = [], [], []
    all_mitre_true, all_mitre_pred = [], []
    all_pred_states, all_true_states = [], []

    with torch.no_grad():
        for batch in dataloader:
            state_seq = batch["state_seq"].to(device)
            next_state = batch["next_state"].to(device)
            y_binary = batch["label_binary"].cpu().numpy()
            y_mitre = batch["label_mitre"].cpu().numpy()

            out = model(state_seq)

            # Binary predictions
            binary_probs = torch.softmax(out["pred_binary"], dim=-1)[:, 1]
            binary_preds = (binary_probs > threshold).long()

            # MITRE predictions
            mitre_preds = out["pred_mitre"].argmax(dim=-1)

            # State predictions
            pred_states = out["pred_state_mean"].cpu().numpy()
            true_states = next_state.cpu().numpy()

            all_binary_true.extend(y_binary.tolist())
            all_binary_pred.extend(binary_preds.cpu().numpy().tolist())
            all_binary_prob.extend(binary_probs.cpu().numpy().tolist())
            all_mitre_true.extend(y_mitre.tolist())
            all_mitre_pred.extend(mitre_preds.cpu().numpy().tolist())
            all_pred_states.append(pred_states)
            all_true_states.append(true_states)

    from src.utils.constants import MITRE_STAGES_INV

    binary_metrics = compute_binary_metrics(
        np.array(all_binary_true),
        np.array(all_binary_pred),
        np.array(all_binary_prob),
    )
    mitre_metrics = compute_mitre_metrics(
        np.array(all_mitre_true),
        np.array(all_mitre_pred),
        stage_names=list(MITRE_STAGES_INV.values()),
    )
    dynamics_metrics = compute_dynamics_metrics(
        np.vstack(all_pred_states),
        np.vstack(all_true_states),
    )

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
    }
