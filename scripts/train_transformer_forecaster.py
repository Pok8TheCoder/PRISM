"""Train a genuine self-attention Transformer forecaster (TFT.01 — Temporal
Forecast Transformer), to finally test "GRU vs Transformer" for this task
head-to-head rather than debate it.

Same protocol, same data, same output parameterization as the GRU
`TemporalForecaster` (see `train_temporal_forecaster_ctx60.py`):
  - context=60 steps in, horizon=40-step-ahead single-step target during
    training (recursive rollout is used at eval time, same as the GRU)
  - tanh-bounded delta on top of the last known state (`last + tanh(d) * max_step`)
  - trained on the same concatenated same-class chains, same
    train/val split, same feature schemas ("v2" and "amt")

The ONLY thing that changes vs. `TemporalForecaster` is the sequence
backbone: a small `nn.TransformerEncoder` (learned positional embedding +
self-attention) instead of a GRU. Kept deliberately small (d_model=64,
2 layers, narrow feedforward) so this is a fair "recurrence vs. attention"
comparison at a similar parameter budget -- NOT the much bigger
d_model=256/16-head/3-layer config used by the ZMT/YMT/AMT *classifiers*.
Throwing a much bigger model at this small dataset would just prove
"bigger wins", not "attention wins".
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

from scripts.train_temporal_forecaster_ctx60 import (  # noqa: E402
    build_class_chains,
    gather_windows_no_crossing,
)

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"

CONTEXT = 60
HORIZON = 40
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
STEP_PCTL = 99.0
SEED = 0

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


class TransformerForecaster(nn.Module):
    """Self-attention encoder over a state sequence -> bounded delta on top
    of the last state. Same output convention as the GRU TemporalForecaster
    so the two are drop-in interchangeable everywhere (rollout, RAM.01's
    OnlineAdaptive/EpisodicMemoryBank, plotting, etc.)."""

    def __init__(self, dim: int, max_step: torch.Tensor, context: int = CONTEXT):
        super().__init__()
        self.dim = dim
        self.context = context
        self.register_buffer("max_step", max_step)
        self.input_proj = nn.Linear(dim, D_MODEL)
        self.pos_embed = nn.Parameter(torch.zeros(1, context, D_MODEL))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
            nn.Linear(D_MODEL, dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, dim), T <= context, most recent state is seq[:, -1, :]."""
        b, t, _ = seq.shape
        x = self.input_proj(seq) + self.pos_embed[:, :t, :]
        enc = self.encoder(x)
        raw_delta = self.head(enc[:, -1, :])
        delta = torch.tanh(raw_delta) * self.max_step
        return seq[:, -1, :] + delta


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_one_schema(name: str, captures, key: str, device) -> dict:
    train_chains = build_class_chains(captures, "train", key)
    val_chains = build_class_chains(captures, "val", key)

    Xtr, Ytr = gather_windows_no_crossing(train_chains, CONTEXT, HORIZON)
    Xva, Yva = gather_windows_no_crossing(val_chains, CONTEXT, HORIZON)
    dim = Ytr.shape[-1]
    print(f"  windows: train={len(Xtr)} val={len(Xva)}  dim={dim}  context={CONTEXT}")

    sc = StandardScaler().fit(Xtr.reshape(-1, dim))
    scale = lambda A: ((A - sc.mean_) / sc.scale_).astype(np.float32)  # noqa: E731

    Xtr_s, Ytr_s = scale(Xtr), scale(Ytr)
    Xva_s, Yva_s = scale(Xva), scale(Yva)

    cur_tr = Xtr_s[:, -1, :]
    abs_delta = np.abs(Ytr_s - cur_tr)
    max_step = np.clip(np.percentile(abs_delta, STEP_PCTL, axis=0), 1e-3, None)

    torch.manual_seed(SEED)
    model = TransformerForecaster(dim, torch.tensor(max_step, dtype=torch.float32), context=CONTEXT).to(device)
    print(f"  TransformerForecaster params: {n_params(model):,}  "
          f"(d_model={D_MODEL}, nhead={NHEAD}, layers={NUM_LAYERS}, ff={DIM_FEEDFORWARD})")
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
        if (ep + 1) % 25 == 0:
            print(f"    [{name}] epoch {ep+1:>3}/{EPOCHS}  val_mse={val_loss:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    fit_time = time.time() - t0
    print(f"  -> best val MSE {best_val:.4f}  ({fit_time:.1f}s)")

    tag = "v2" if key == "v2" else "amt"
    torch.save({
        "model": model.state_dict(), "dim": dim, "context": CONTEXT,
        "max_step": max_step, "scaler_mean": sc.mean_, "scaler_scale": sc.scale_,
    }, CKPT_DIR / f"forecast_{tag}_transformer_ctx60.pth")
    return {"val_mse": best_val, "fit_time": fit_time}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)
    for schema, key in (("YMT (v2, 64f)", "v2"), ("AMT (aryan, 82f)", "amt")):
        print(f"\n=== {schema} : Transformer (TFT.01) forecaster, context=60/horizon=40 ===")
        train_one_schema(schema, captures, key, device)


if __name__ == "__main__":
    main()
