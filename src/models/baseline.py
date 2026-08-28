"""
PRISM - Logistic Regression & Static ML Baselines
Treats each flow/window in isolation (no temporal context).
Used as benchmark comparison against the world model.
"""

import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

logger = logging.getLogger("prism.models.baseline")


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline that classifies each time window
    independently — no temporal context, no state transition learning.

    Used to benchmark the world model's temporal dynamics advantage.
    """

    def __init__(
        self,
        task: str = "binary",   # "binary" | "mitre"
        max_iter: int = 1000,
        C: float = 1.0,
        class_weight: str = "balanced",
    ):
        self.task = task
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=max_iter,
                C=C,
                class_weight=class_weight,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])
        self._is_fitted = False
        logger.info("LogisticRegressionBaseline: task=%s, C=%.3f", task, C)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        """
        Train on flat feature matrix.

        Parameters
        ----------
        X : (N, D) — one row per time window, features flattened
        y : (N,)   — labels (0/1 for binary, 0-6 for MITRE)
        """
        logger.info(
            "Training LR baseline: %d samples, %d features, "
            "classes=%s", len(X), X.shape[1], np.unique(y).tolist()
        )
        self.model.fit(X, y)
        self._is_fitted = True
        logger.info("LR baseline training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return dict of evaluation metrics."""
        self._check_fitted()
        preds = self.predict(X)
        avg = "binary" if self.task == "binary" else "macro"

        metrics = {
            "f1": f1_score(y, preds, average=avg, zero_division=0),
            "precision": precision_score(y, preds, average=avg, zero_division=0),
            "recall": recall_score(y, preds, average=avg, zero_division=0),
            "report": classification_report(y, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
        }
        if self.task == "binary":
            tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
            metrics["fpr"] = fp / max(fp + tn, 1)
            metrics["tpr"] = tp / max(tp + fn, 1)

        logger.info(
            "LR Baseline | F1=%.4f | Prec=%.4f | Recall=%.4f",
            metrics["f1"], metrics["precision"], metrics["recall"],
        )
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Saved LR baseline -> %s", path)

    def load(self, path: str) -> "LogisticRegressionBaseline":
        self.model = joblib.load(path)
        self._is_fitted = True
        logger.info("Loaded LR baseline from %s", path)
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")


class RandomForestBaseline:
    """Random Forest baseline — stronger static baseline for fair comparison."""

    def __init__(
        self,
        task: str = "binary",
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced",
        n_jobs: int = -1,
    ):
        self.task = task
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=42,
        )
        self._scaler = StandardScaler()
        self._is_fitted = False
        logger.info(
            "RandomForestBaseline: task=%s, n_estimators=%d",
            task, n_estimators,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        X_scaled = self._scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._is_fitted = True
        logger.info("RF baseline training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(self._scaler.transform(X))

    def feature_importances(self, feature_names: list[str]) -> list[tuple]:
        """Return sorted (feature, importance) pairs."""
        self._check_fitted()
        importances = self.model.feature_importances_
        return sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        self._check_fitted()
        preds = self.predict(X)
        avg = "binary" if self.task == "binary" else "macro"
        metrics = {
            "f1": f1_score(y, preds, average=avg, zero_division=0),
            "precision": precision_score(y, preds, average=avg, zero_division=0),
            "recall": recall_score(y, preds, average=avg, zero_division=0),
            "report": classification_report(y, preds, zero_division=0),
            "confusion_matrix": confusion_matrix(y, preds).tolist(),
        }
        if self.task == "binary":
            tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
            metrics["fpr"] = fp / max(fp + tn, 1)
            metrics["tpr"] = tp / max(tp + fn, 1)
        logger.info(
            "RF Baseline | F1=%.4f | Prec=%.4f | Recall=%.4f",
            metrics["f1"], metrics["precision"], metrics["recall"],
        )
        return metrics

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self._scaler}, path)
        logger.info("Saved RF baseline -> %s", path)

    def load(self, path: str) -> "RandomForestBaseline":
        obj = joblib.load(path)
        self.model = obj["model"]
        self._scaler = obj["scaler"]
        self._is_fitted = True
        return self

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")


def flatten_state_sequences(
    states: np.ndarray,
    labels_binary: np.ndarray,
    labels_mitre: np.ndarray,
    lookback: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (T, D) state array to flat (N, D*lookback) for static baselines.
    No temporal context when lookback=1 (pure static classifier).

    Returns X, y_binary, y_mitre
    """
    T, D = states.shape
    X_list, yb_list, ym_list = [], [], []

    for t in range(lookback - 1, T):
        window = states[t - lookback + 1 : t + 1]  # (lookback, D)
        X_list.append(window.flatten())
        yb_list.append(labels_binary[t])
        ym_list.append(labels_mitre[t])

    X = np.array(X_list, dtype=np.float32)
    yb = np.array(yb_list, dtype=np.int64)
    ym = np.array(ym_list, dtype=np.int64)
    logger.info(
        "Flattened states: %d samples, %d features (lookback=%d)",
        len(X), X.shape[1], lookback,
    )
    return X, yb, ym
