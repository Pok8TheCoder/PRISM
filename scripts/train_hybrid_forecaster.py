"""Train a 'middle ground' forecaster: temporal context + bounded delta step.

Why this design, specifically responding to the RF-flatline / MLP-explosion
failure modes found in test_recursive_rollout.py:

  RF failure:  output is a piecewise-constant average of training leaves ->
               recursion contracts hard onto a tiny 2-leaf loop near the
               "typical" training region. Bounded, but loses all momentum.

  AE-MLP failure: output is an unconstrained absolute next-state -> small
               per-step errors compound multiplicatively over a long
               recursive rollout and can blow up to physically nonsensical
               values (we saw AMT duration diverge to +-millions of us).

Hybrid fix, two changes:
  1. TEMPORAL CONTEXT: instead of only seeing state[t], the model sees a
     short window [state[t-2], state[t-1], state[t]] concatenated, so it can
     infer velocity/trend directly (this is the 'temporal' part -- closer to
     what a real forecaster needs than a memoryless map).
  2. BOUNDED DELTA: instead of predicting the absolute next state, the model
     predicts a *delta* which is squashed through tanh and scaled by a
     per-feature max-step learned from the training set's actual step sizes
     (99th percentile of |state[t+1]-state[t]|, standardized units). The
     final prediction is state[t] + delta. This can never explode (tanh is
     bounded) and is not restricted to a finite set of leaf values (so it
     doesn't need to collapse onto a discrete loop either) -- it can keep
     drifting smoothly in one direction step after step, right up to its
     per-step speed limit.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"

WINDOW = 3          # how many past states the model gets to see (incl. current)
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-3
HIDDEN = 96
LATENT = 32
STEP_PCTL = 99.0    # percentile of observed |delta| used as the per-feature speed limit
SEED = 0


class HybridForecaster(nn.Module):
    """window of states -> bounded delta on top of the most recent state."""

    def __init__(self, dim: int, window: int, max_step: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.window = window
        self.register_buffer("max_step", max_step)  # (dim,)
        self.net = nn.Sequential(
            nn.Linear(dim * window, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, LATENT), nn.GELU(),
            nn.Linear(LATENT, HIDDEN), nn.GELU(),
            nn.Linear(HIDDEN, dim),
        )

    def forward(self, window_x: torch.Tensor) -> torch.Tensor:
        """window_x: (B, window*dim), most recent state is the LAST `dim` slice."""
        cur = window_x[:, -self.dim:]
        raw_delta = self.net(window_x)
        delta = torch.tanh(raw_delta) * self.max_step
        return cur + delta


def gather_windowed_pairs(captures: list[dict], split: str, key: str, window: int):
    Xs, Ys = [], []
    for c in captures:
        if c["split"] != split:
            continue
        traj = c[key]
        if len(traj) < window + 1:
            continue
        for t in range(window - 1, len(traj) - 1):
            Xs.append(traj[t - window + 1:t + 1].reshape(-1))  # flattened window
            Ys.append(traj[t + 1])
    return np.stack(Xs), np.stack(Ys)


def train_one_schema(name: str, captures, key: str, device) -> dict:
    Xtr, Ytr = gather_windowed_pairs(captures, "train", key, WINDOW)
    Xva, Yva = gather_windowed_pairs(captures, "val", key, WINDOW)
    Xte, Yte = gather_windowed_pairs(captures, "test", key, WINDOW)
    dim = Ytr.shape[1]
    print(f"  pairs: train={len(Xtr)} val={len(Xva)} test={len(Xte)}  dim={dim}  window={WINDOW}")

    # Scale using the "current state" slice statistics (same scaler for every
    # position in the window, since it's the same feature schema repeated).
    sc = StandardScaler().fit(Xtr[:, -dim:])

    def scale_window(A):
        B = A.reshape(len(A), WINDOW, dim)
        Bs = (B - sc.mean_) / sc.scale_
        return Bs.reshape(len(A), WINDOW * dim).astype(np.float32)

    def scale_flat(A):
        return ((A - sc.mean_) / sc.scale_).astype(np.float32)

    Xtr_s, Ytr_s = scale_window(Xtr), scale_flat(Ytr)
    Xva_s, Yva_s = scale_window(Xva), scale_flat(Yva)
    Xte_s, Yte_s = scale_window(Xte), scale_flat(Yte)

    cur_tr = Xtr_s[:, -dim:]
    abs_delta = np.abs(Ytr_s - cur_tr)
    max_step = np.percentile(abs_delta, STEP_PCTL, axis=0)
    max_step = np.clip(max_step, 1e-3, None)
    print(f"  per-feature speed limit (99th pctl |delta|, std units): "
          f"mean={max_step.mean():.3f} min={max_step.min():.3f} max={max_step.max():.3f}")

    torch.manual_seed(SEED)
    model = HybridForecaster(dim, WINDOW, torch.tensor(max_step, dtype=torch.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    crit = nn.MSELoss()

    Xt, Yt = torch.from_numpy(Xtr_s).to(device), torch.from_numpy(Ytr_s).to(device)
    Xv, Yv = torch.from_numpy(Xva_s).to(device), torch.from_numpy(Yva_s).to(device)
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
        test_mse = crit(model(torch.from_numpy(Xte_s).to(device)),
                        torch.from_numpy(Yte_s).to(device)).item()
    fit_time = time.time() - t0
    print(f"  -> test MSE {test_mse:.4f}  ({fit_time:.1f}s)")

    tag = "v2" if key == "v2" else "amt"
    torch.save({
        "model": model.state_dict(), "dim": dim, "window": WINDOW,
        "max_step": max_step, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
    }, CKPT_DIR / f"forecast_{tag}_hybrid.pth")
    return {"test_mse": test_mse, "fit_time": fit_time}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)
    for schema, key in (("YMT (v2, 64f)", "v2"), ("AMT (aryan, 82f)", "amt")):
        print(f"\n=== {schema} : hybrid delta forecaster ===")
        train_one_schema(schema, captures, key, device)


if __name__ == "__main__":
    main()
