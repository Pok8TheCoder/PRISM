"""
Evaluation Metrics: F1, Precision, Recall, FPR, MITRE Stage Accuracy,
Dynamics MSE, and Early Warning Lead-Time calculation.
"""

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error
)
from typing import Dict, Any, List


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict[str, float]:
    """Computes binary intrusion detection metrics."""
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix for False Positive Rate (FPR = FP / (FP + TN))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5

    return metrics


def compute_mitre_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes multiclass MITRE ATT&CK stage metrics."""
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "mitre_accuracy": acc,
        "mitre_macro_f1": macro_f1
    }


def compute_early_warning_metrics(
    actual_sequence: np.ndarray,
    predicted_probs: np.ndarray,
    alert_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Measures how many time windows ahead of an actual attack onset the model crosses the alert threshold.
    """
    lead_times = []
    T = len(actual_sequence)

    for i in range(1, T):
        # Identify attack onset (0 -> 1)
        if actual_sequence[i - 1] == 0 and actual_sequence[i] == 1:
            # Check backwards when prediction first exceeded threshold
            lead = 0
            for look_back in range(1, min(10, i + 1)):
                if predicted_probs[i - look_back] >= alert_threshold:
                    lead = look_back
                else:
                    break
            lead_times.append(lead)

    avg_lead = float(np.mean(lead_times)) if lead_times else 0.0
    return {
        "attack_onsets_detected": len(lead_times),
        "avg_early_warning_windows": avg_lead,
        "lead_time_distribution": lead_times
    }
