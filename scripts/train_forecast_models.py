"""Train one-step-ahead forecasters: state[t] -> state[t+1], for real this time.

The existing ZMT/YMT/AMT checkpoints train their `state_head` against the
*current* last timestep (`state_loss = MSE(ns, bx[:, -1, :])`), which is
reconstruction, not forecasting. This trains a genuine forecaster for both
schemas:

  YAE  — small encoder/latent/decoder MLP ("autoencoder") on the 64-d v2 state
  AAE  — same architecture on the 82-d AMT state
  RF   — RandomForestRegressor, multi-output, on both schemas (fast baseline)

Split is capture-level (inherited from build_forecast_dataset.py), so no
capture contributes pairs to both train and test.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "results" / "forecast" / "v1_onestep_ae_rf"
EPOCHS = 60
BATCH_SIZE = 256
LR = 1e-3
LATENT = 32
HIDDEN = 96
SEED = 0


class MLPForecaster(nn.Module):
    """Encoder -> latent bottleneck -> decoder, predicting state[t+1] from state[t]."""

    def __init__(self, dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, LATENT), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def gather_pairs(captures: list[dict], split: str, key: str) -> tuple[np.ndarray, np.ndarray]:
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


def train_mlp(name: str, Xtr, Ytr, Xva, Yva, Xte, Yte, device) -> dict:
    torch.manual_seed(SEED)
    dim = Xtr.shape[1]
    model = MLPForecaster(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    crit = nn.MSELoss()

    Xt = torch.from_numpy(Xtr).to(device)
    Yt = torch.from_numpy(Ytr).to(device)
    Xv = torch.from_numpy(Xva).to(device)
    Yv = torch.from_numpy(Yva).to(device)
    n = len(Xt)

    best_val, best_state = float("inf"), None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), Yv).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"    [{name}] epoch {ep+1:>2}/{EPOCHS}  val_mse={val_loss:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_mse = crit(model(torch.from_numpy(Xte).to(device)),
                        torch.from_numpy(Yte).to(device)).item()
    fit_time = time.time() - t0
    print(f"    -> test MSE {test_mse:.4f}  ({fit_time:.1f}s)")
    return {"model": model, "test_mse": test_mse, "fit_time": fit_time, "val_mse": best_val}


def train_rf(name: str, Xtr, Ytr, Xte, Yte) -> dict:
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=150, max_depth=14, n_jobs=-1, random_state=SEED)
    rf.fit(Xtr, Ytr)
    fit_time = time.time() - t0
    pred = rf.predict(Xte)
    test_mse = float(np.mean((pred - Yte) ** 2))
    print(f"    [{name} RF] -> test MSE {test_mse:.4f}  ({fit_time:.1f}s)")
    return {"model": rf, "test_mse": test_mse, "fit_time": fit_time}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for schema, key in (("YMT (v2, 64f)", "v2"), ("AMT (aryan, 82f)", "amt")):
        print(f"\n=== {schema} ===")
        Xtr, Ytr = gather_pairs(captures, "train", key)
        Xva, Yva = gather_pairs(captures, "val", key)
        Xte, Yte = gather_pairs(captures, "test", key)
        print(f"  pairs: train={len(Xtr)} val={len(Xva)} test={len(Xte)}  dim={Xtr.shape[1]}")

        sc = StandardScaler().fit(Xtr)
        s = lambda A: sc.transform(A).astype(np.float32)  # noqa: E731
        Xtr_s, Ytr_s = s(Xtr), s(Ytr)
        Xva_s, Yva_s = s(Xva), s(Yva)
        Xte_s, Yte_s = s(Xte), s(Yte)

        mlp_res = train_mlp(f"{key}-AE", Xtr_s, Ytr_s, Xva_s, Yva_s, Xte_s, Yte_s, device)
        rf_res = train_rf(f"{key}-RF", Xtr_s, Ytr_s, Xte_s, Yte_s)

        # Naive baseline: predict "nothing changes" (state[t+1] = state[t]).
        naive_mse = float(np.mean((Xte_s - Yte_s) ** 2))
        print(f"    [naive persistence] test MSE {naive_mse:.4f}")

        tag = "v2" if key == "v2" else "amt"
        torch.save({"model": mlp_res["model"].state_dict(), "dim": Xtr.shape[1],
                   "scaler_mean": sc.mean_, "scaler_scale": sc.scale_},
                  CKPT_DIR / f"forecast_{tag}_ae.pth")
        joblib.dump({"model": rf_res["model"], "scaler_mean": sc.mean_, "scaler_scale": sc.scale_},
                    CKPT_DIR / f"forecast_{tag}_rf.joblib")

        results[schema] = {
            "dim": int(Xtr.shape[1]), "n_train_pairs": int(len(Xtr)),
            "n_test_pairs": int(len(Xte)), "naive_persistence_mse": naive_mse,
            "autoencoder_mlp": {"test_mse": mlp_res["test_mse"], "fit_time_sec": mlp_res["fit_time"]},
            "random_forest": {"test_mse": rf_res["test_mse"], "fit_time_sec": rf_res["fit_time"]},
        }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'Schema':<20}{'Naive':>10}{'AE-MLP':>10}{'RF':>10}")
    print("-" * 70)
    for schema, r in results.items():
        print(f"{schema:<20}{r['naive_persistence_mse']:>10.4f}"
              f"{r['autoencoder_mlp']['test_mse']:>10.4f}{r['random_forest']['test_mse']:>10.4f}")
    print("=" * 70)
    print(f"Saved -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
