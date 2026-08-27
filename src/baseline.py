"""Baseline: Logistic Regression for flow classification (same features as the world model)."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from src.pipeline.extract import csv_to_matrix
from src.pipeline.features import FEATURE_COLS

DATA_DIR = Path("data/raw")
FILES = {
    "thursday": DATA_DIR / "thursday_01_03_2018.csv",
    "wednesday": DATA_DIR / "wednesday_28_02_2018.csv",
}


def load_data():
    present = [p for p in FILES.values() if p.exists()]
    if not present:
        raise FileNotFoundError(
            "CIC-IDS-2018 CSVs not found in data/raw. "
            "Run: python scripts/download_data.py"
        )
    Xs, ys = [], []
    for path in present:
        X, y = csv_to_matrix(path)
        Xs.append(X)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def train_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn + 1e-9)
    print(f"False positive rate: {fpr:.4f}")


if __name__ == "__main__":
    print("Loading Data...")
    X, y = load_data()
    print(f"Dataset: {len(X)} rows | Benign={(y == 0).sum()} | Attack={(y == 1).sum()}")
    print("Training Baseline Model...")
    train_baseline(X, y)
