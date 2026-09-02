"""
Baseline Classifiers (Logistic Regression and Random Forest) for static benchmark comparison.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
from src.utils.logger import setup_logger

logger = setup_logger("Baseline")


class BaselineClassifier:
    """
    Static (non-temporal) baseline classifier for intrusion detection and MITRE stages.
    """

    def __init__(self, model_type: str = "logistic_regression"):
        self.model_type = model_type
        if model_type == "logistic_regression":
            self.binary_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ])
            self.mitre_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ])
        elif model_type == "random_forest":
            self.binary_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
            ])
            self.mitre_model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, X_train: np.ndarray, y_attack: np.ndarray, y_mitre: np.ndarray):
        """Fits both binary attack and multiclass MITRE stage classifiers."""
        logger.info(f"Fitting {self.model_type} baseline on {len(X_train)} samples...")
        self.binary_model.fit(X_train, y_attack)
        self.mitre_model.fit(X_train, y_mitre)
        logger.info("Baseline fitting complete.")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns binary attack predictions and predicted MITRE stages."""
        pred_attack = self.binary_model.predict(X)
        pred_mitre = self.mitre_model.predict(X)
        return pred_attack, pred_mitre

    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns attack probabilities and MITRE class probabilities."""
        prob_attack = self.binary_model.predict_proba(X)
        prob_mitre = self.mitre_model.predict_proba(X)
        return prob_attack, prob_mitre
