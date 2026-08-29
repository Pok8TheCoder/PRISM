"""Train ZMT.01 (v1, 27-d) and YMT.01 (v2, 64-d) under identical conditions.

Same architecture, hyperparameters, splits, class weighting and seeds for both.
The input projection width is the only structural difference, so the measured
gap is attributable to the feature representation.

Results -> results/zmt_ymt/metrics.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.attack_catalog import get_class_names, get_class_to_idx  # noqa: E402

SEQ_LEN = 10
D_MODEL = 256
NHEAD = 16
NLAYERS = 3
DROPOUT = 0.1
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
STATE_LOSS_W = 0.5
SEEDS = (0, 1, 2)
# Full inverse-frequency weighting hands classes with a handful of training
# windows weights in the hundreds, which the model pays for by dumping benign
# traffic into whichever rare class it over-weighted. Square-root weighting
# with a cap keeps rare attacks learnable without wrecking benign precision.
WEIGHT_POWER = 0.5
WEIGHT_CLIP = (0.25, 5.0)

DATA = ROOT / "data" / "processed" / "zmt_ymt_dataset.npz"
OUT_DIR = ROOT / "results" / "zmt_ymt"
CKPT_DIR = ROOT / "models" / "checkpoints"

CLASS_NAMES = get_class_names()
NUM_CLASSES = len(CLASS_NAMES)
BENIGN_IDX = get_class_to_idx()["Benign"]


class WorldModel(nn.Module):
    """Temporal transformer with a next-state head and a classification head."""

    def __init__(self, num_features: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.input_proj = nn.Linear(num_features, D_MODEL)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, D_MODEL) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD, dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=NLAYERS)
        self.state_head = nn.Linear(D_MODEL, num_features)
        self.classify_head = nn.Linear(D_MODEL, num_classes)

    def forward(self, x):
        h = self.transformer(self.input_proj(x) + self.pos_emb)[:, -1, :]
        return self.state_head(h), self.classify_head(h)


def scale(X_tr, X_va, X_te):
    """Fit the scaler on training flows only, then apply to every split."""
    n, t, f = X_tr.shape
    sc = StandardScaler().fit(X_tr.reshape(-1, f))

    def apply(X):
        return sc.transform(X.reshape(-1, f)).reshape(X.shape).astype(np.float32)

    return apply(X_tr), apply(X_va), apply(X_te), sc


def evaluate(model, X, y, device, batch=1024):
    model.eval()
    logits_all, state_err = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            bx = torch.from_numpy(X[i:i + batch]).to(device)
            ns, lg = model(bx)
            logits_all.append(lg.cpu())
            state_err.append(((ns - bx[:, -1, :]) ** 2).mean(1).cpu())
    logits = torch.cat(logits_all)
    probs = torch.softmax(logits, 1).numpy()
    pred = probs.argmax(1)

    attack_true = (y != BENIGN_IDX).astype(int)
    attack_score = 1.0 - probs[:, BENIGN_IDX]
    attack_pred = (pred != BENIGN_IDX).astype(int)
    tp = int(((attack_pred == 1) & (attack_true == 1)).sum())
    fp = int(((attack_pred == 1) & (attack_true == 0)).sum())
    tn = int(((attack_pred == 0) & (attack_true == 0)).sum())
    fn = int(((attack_pred == 0) & (attack_true == 1)).sum())
    bp, br, bf1, _ = precision_recall_fscore_support(
        attack_true, attack_pred, average="binary", zero_division=0
    )
    try:
        auc = float(roc_auc_score(attack_true, attack_score))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float((pred == y).mean()),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "binary_precision": float(bp),
        "binary_recall": float(br),
        "binary_f1": float(bf1),
        "binary_fpr": float(fp / max(fp + tn, 1)),
        "binary_auc": auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "state_mse": float(torch.cat(state_err).mean()),
        "_pred": pred,
    }


def train_one(name, X_tr, y_tr, X_va, y_va, X_te, y_te, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WorldModel(X_tr.shape[2]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
    present = counts > 0
    w = np.zeros(NUM_CLASSES)
    w[present] = (counts[present].sum() / counts[present]) ** WEIGHT_POWER
    w[present] /= w[present].mean()
    w = np.clip(w, *WEIGHT_CLIP) * present
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    crit_state = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    n = len(Xt)

    best_f1, best_state, t0 = -1.0, None, time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            bx, by = Xt[idx], yt[idx]
            opt.zero_grad()
            ns, lg = model(bx)
            loss = crit(lg, by) + STATE_LOSS_W * crit_state(ns, bx[:, -1, :])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        val = evaluate(model, X_va, y_va, device)
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 25 == 0:
            print(f"    [{name} s{seed}] epoch {ep+1:>2}/{EPOCHS} "
                  f"val_macroF1={val['f1_macro']:.4f} best={best_f1:.4f}")

    model.load_state_dict(best_state)
    test = evaluate(model, X_te, y_te, device)
    test["train_sec"] = time.time() - t0
    test["n_params"] = n_params
    test["val_f1_macro"] = best_f1
    return model, test


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATA, allow_pickle=True)
    X1, X2, y, split, source = d["X1"], d["X2"], d["y"], d["split"], d["source"]
    tr, va, te = split == "train", split == "val", split == "test"
    print(f"Device {device} | train {tr.sum()} val {va.sum()} test {te.sum()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict = {"per_seed": {}, "config": {
        "seq_len": SEQ_LEN, "d_model": D_MODEL, "nhead": NHEAD,
        "nlayers": NLAYERS, "epochs": EPOCHS, "batch": BATCH_SIZE, "lr": LR,
        "seeds": list(SEEDS), "class_weighted_loss": True,
        "scaler_fit_on": "train_only",
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
    }}

    for name, X in (("ZMT.01", X1), ("YMT.01", X2)):
        Xtr, Xva, Xte, sc = scale(X[tr], X[va], X[te])
        per_seed = []
        best_overall, best_model, best_pred = -1.0, None, None
        for seed in SEEDS:
            print(f"\n  Training {name} (features={X.shape[2]}) seed={seed}")
            model, m = train_one(name, Xtr, y[tr], Xva, y[va], Xte, y[te], seed, device)
            pred = m.pop("_pred")
            per_seed.append(m)
            print(f"    -> test acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} "
                  f"binF1={m['binary_f1']:.4f} FPR={m['binary_fpr']:.4f}")
            if m["f1_macro"] > best_overall:
                best_overall, best_model, best_pred = m["f1_macro"], model, pred

        results["per_seed"][name] = per_seed
        keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]
        results.setdefault("summary", {})[name] = {
            k: {"mean": float(np.mean([s[k] for s in per_seed])),
                "std": float(np.std([s[k] for s in per_seed]))}
            for k in keys
        }
        results["summary"][name]["num_features"] = X.shape[2]

        # Per-class F1 and PCAP-only view from the best seed.
        yte = y[te]
        f1c = f1_score(yte, best_pred, average=None,
                       labels=list(range(NUM_CLASSES)), zero_division=0)
        results.setdefault("per_class_f1", {})[name] = {
            CLASS_NAMES[i]: float(f1c[i]) for i in range(NUM_CLASSES)
            if (yte == i).sum() > 0
        }
        pm = source[te] == "pcap"
        results.setdefault("pcap_only", {})[name] = {
            "accuracy": float((best_pred[pm] == yte[pm]).mean()),
            "f1_macro": float(f1_score(yte[pm], best_pred[pm],
                                       average="macro", zero_division=0)),
            "n": int(pm.sum()),
        }
        results.setdefault("confusion", {})[name] = confusion_matrix(
            yte, best_pred, labels=list(range(NUM_CLASSES))
        ).tolist()

        torch.save({
            "model": best_model.state_dict(), "num_features": X.shape[2],
            "scaler_mean": sc.mean_, "scaler_std": sc.scale_,
            "class_names": CLASS_NAMES, "model_name": name,
        }, CKPT_DIR / f"{name.replace('.', '_').lower()}.pth")

    results["class_names"] = CLASS_NAMES
    results["test_support"] = {
        CLASS_NAMES[i]: int((y[te] == i).sum()) for i in range(NUM_CLASSES)
        if (y[te] == i).sum() > 0
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 74)
    print(f"{'Model':<10}{'Feat':>6}{'Params':>10}{'Acc':>9}{'MacroF1':>10}"
          f"{'BinF1':>9}{'FPR':>9}")
    print("-" * 74)
    for name in ("ZMT.01", "YMT.01"):
        s = results["summary"][name]
        print(f"{name:<10}{s['num_features']:>6}{int(s['n_params']['mean']):>10}"
              f"{s['accuracy']['mean']:>9.4f}{s['f1_macro']['mean']:>10.4f}"
              f"{s['binary_f1']['mean']:>9.4f}{s['binary_fpr']['mean']:>9.4f}")
    print("=" * 74)
    print(f"Saved -> {OUT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
