"""Train an XGBoost one-step-ahead forecaster, same protocol as train_forecast_models.py's RF.

Tests the hypothesis: "XGBoost will show the same recursive-rollout collapse
as Random Forest." XGBoost's leaves are also constants (sum of boosted trees'
leaf values), so the same piecewise-constant argument applies -- but boosting
combines *hundreds* of shallow trees additively rather than averaging deep
ones, so the effective step function has far more, finer-grained plateaus.
Worth testing empirically rather than assuming.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
SEED = 0


def gather_pairs(captures, split, key):
    Xs, Ys = [], []
    for c in captures:
        if c["split"] != split:
            continue
        traj = c[key]
        if len(traj) < 2:
            continue
        Xs.append(traj[:-1])
        Ys.append(traj[1:])
    return np.concatenate(Xs), np.concatenate(Ys)


def main():
    with open(DATA, "rb") as f:
        captures = pickle.load(f)

    for schema, key in (("YMT (v2, 64f)", "v2"), ("AMT (aryan, 82f)", "amt")):
        print(f"\n=== {schema} : XGBoost ===")
        Xtr, Ytr = gather_pairs(captures, "train", key)
        Xte, Yte = gather_pairs(captures, "test", key)
        print(f"  pairs: train={len(Xtr)} test={len(Xte)}  dim={Xtr.shape[1]}")

        sc = StandardScaler().fit(Xtr)
        s = lambda A: sc.transform(A).astype(np.float32)  # noqa: E731
        Xtr_s, Ytr_s = s(Xtr), s(Ytr)
        Xte_s, Yte_s = s(Xte), s(Yte)

        t0 = time.time()
        base = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
        )
        model = MultiOutputRegressor(base, n_jobs=-1)
        model.fit(Xtr_s, Ytr_s)
        fit_time = time.time() - t0
        pred = model.predict(Xte_s)
        test_mse = float(np.mean((pred - Yte_s) ** 2))
        print(f"  -> test MSE {test_mse:.4f}  ({fit_time:.1f}s)")

        tag = "v2" if key == "v2" else "amt"
        joblib.dump({"model": model, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_},
                    CKPT_DIR / f"forecast_{tag}_xgb.joblib")


if __name__ == "__main__":
    main()
