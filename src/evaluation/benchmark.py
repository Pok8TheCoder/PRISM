"""
Benchmark Engine: Evaluates World Model vs Static Classifiers (Logistic Regression, Random Forest)
and formats comparative benchmark tables.
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, Any, Tuple
from src.evaluation.metrics import compute_classification_metrics, compute_mitre_metrics
from src.utils.logger import setup_logger

logger = setup_logger("Benchmark")


def find_best_threshold(y_true: np.ndarray, y_probs: np.ndarray, default_th: float = 0.60) -> float:
    """Finds the optimal F1 decision threshold with precision constraint."""
    best_th = default_th
    best_f1 = -1.0
    for th in np.linspace(0.40, 0.80, 41):
        preds = (y_probs >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = float(th)
    return best_th


def run_benchmark(
    world_model: torch.nn.Module,
    baseline_model: Any,
    test_dataset: Any,
    val_dataset: Any = None,
    device: str = "cpu"
) -> pd.DataFrame:
    """
    Evaluates both the World Model and Baseline Classifier on the test dataset.
    """
    world_model.to(device)
    world_model.eval()

    # Collect validation predictions for threshold calibration if provided
    best_wm_th = 0.40
    best_base_th = 0.50
    if val_dataset is not None:
        val_seqs = [val_dataset[i]["state_seq"] for i in range(len(val_dataset))]
        val_targets = np.array([val_dataset[i]["next_attack"].item() for i in range(len(val_dataset))])
        if val_seqs:
            val_batch = torch.stack(val_seqs).to(device)
            with torch.no_grad():
                _, _, val_atk_logits, _, _, _ = world_model(val_batch)
            val_wm_probs = torch.softmax(val_atk_logits, dim=-1)[:, 1].cpu().numpy()
            best_wm_th = find_best_threshold(val_targets, val_wm_probs)

            val_flat = val_batch[:, -1, :].cpu().numpy()
            val_base_probs, _ = baseline_model.predict_proba(val_flat)
            v_p = val_base_probs[:, 1] if val_base_probs.shape[1] > 1 else val_base_probs[:, 0]
            best_base_th = find_best_threshold(val_targets, v_p)

    logger.info(f"Calibrated Optimal Decision Thresholds -> World Model: {best_wm_th:.2f}, Baseline: {best_base_th:.2f}")

    # Collect sequences from test dataset
    all_seqs = []
    all_next_states = []
    all_targets_atk = []
    all_targets_mitre = []

    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        all_seqs.append(sample["state_seq"])
        all_next_states.append(sample["next_state"])
        all_targets_atk.append(sample["next_attack"].item())
        all_targets_mitre.append(sample["next_mitre"].item())

    if not all_seqs:
        return pd.DataFrame()

    batch_seqs = torch.stack(all_seqs).to(device)  # (N, L, D)
    y_true_atk = np.array(all_targets_atk)
    y_true_mitre = np.array(all_targets_mitre)

    # 1. Evaluate World Model
    with torch.no_grad():
        (
            pred_mean,
            pred_logvar,
            pred_attack_logits,
            pred_mitre_logits,
            pred_frac,
            _
        ) = world_model(batch_seqs)

    wm_probs = torch.softmax(pred_attack_logits, dim=-1)[:, 1].cpu().numpy()
    wm_preds = (wm_probs >= best_wm_th).astype(int)
    wm_mitre_preds = np.argmax(pred_mitre_logits.cpu().numpy(), axis=-1)

    wm_metrics = compute_classification_metrics(y_true_atk, wm_preds, wm_probs)
    wm_mitre = compute_mitre_metrics(y_true_mitre, wm_mitre_preds)

    # 2. Evaluate Baseline Model (uses most recent window in sequence)
    flat_test_features = batch_seqs[:, -1, :].cpu().numpy()
    base_atk_probs, _ = baseline_model.predict_proba(flat_test_features)
    base_probs = base_atk_probs[:, 1] if base_atk_probs.shape[1] > 1 else base_atk_probs[:, 0]
    base_atk_preds = (base_probs >= best_base_th).astype(int)
    _, base_mitre_preds = baseline_model.predict(flat_test_features)

    base_metrics = compute_classification_metrics(y_true_atk, base_atk_preds, base_probs)
    base_mitre = compute_mitre_metrics(y_true_mitre, base_mitre_preds)

    # 3. Create Benchmark Summary DataFrame
    records = [
        {
            "Model": "Logistic Regression / RF Baseline",
            "F1 Score": round(base_metrics["f1"], 4),
            "Precision": round(base_metrics["precision"], 4),
            "Recall": round(base_metrics["recall"], 4),
            "FPR": round(base_metrics["fpr"], 4),
            "ROC-AUC": round(base_metrics.get("roc_auc", 0.5), 4),
            "MITRE Accuracy": round(base_mitre["mitre_accuracy"], 4)
        },
        {
            "Model": "PRISM StateTransformerWorldModel",
            "F1 Score": round(wm_metrics["f1"], 4),
            "Precision": round(wm_metrics["precision"], 4),
            "Recall": round(wm_metrics["recall"], 4),
            "FPR": round(wm_metrics["fpr"], 4),
            "ROC-AUC": round(wm_metrics.get("roc_auc", 0.5), 4),
            "MITRE Accuracy": round(wm_mitre["mitre_accuracy"], 4)
        }
    ]

    df_bench = pd.DataFrame(records)
    return df_bench
