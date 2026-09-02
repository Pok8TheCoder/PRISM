"""Train ARY TemporalTransformerWorldModel on 242-d (or subset) CIC splits.

Used by ``ablate_ary242_blocks.py`` and local/Colab ablation sweeps.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.components import MultiTaskLoss  # noqa: E402
from src.aryan.dataset import (  # noqa: E402
    AryanSequenceDataset,
    LOOKBACK,
    SPLITS_DIR,
    fit_scaler,
    load_all_splits,
    transform_states,
)
from src.aryan.feature_schema242 import column_indices, describe_subset  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402

CKPT_DIR = ROOT / "models" / "checkpoints" / "ary242_ablation"
RESULTS_DIR = ROOT / "results" / "ary242_ablation"

# ARY.01 recipe (epoch-13 checkpoint on origin/aryan @ 1ec06bc)
DEFAULT_CFG = dict(
    d_model=256,
    n_layers=4,
    n_heads=8,
    batch_size=64,
    epochs=80,
    patience=15,
    lr=2e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    lambda_dynamics=0.5,
    lambda_infiltration=1.2,
    lambda_mitre=1.0,
    use_focal=False,
    attack_weight_boost=1.6,
    lookback=LOOKBACK,
)


def class_weights(labels: np.ndarray, n_classes: int, boost_idx: int | None = None) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float32)
    present = counts > 0
    w[present] = counts[present].sum() / counts[present]
    if boost_idx is not None and present[boost_idx]:
        w[boost_idx] *= DEFAULT_CFG["attack_weight_boost"]
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    bin_true, bin_pred, mit_true, mit_pred = [], [], [], []
    dyn_errs = []
    for seq, nxt, yb, ym in loader:
        seq, nxt = seq.to(device), nxt.to(device)
        out = model(seq)
        pb = out["pred_binary"].argmax(1).cpu().numpy()
        pm = out["pred_mitre"].argmax(1).cpu().numpy()
        bin_true.extend(yb.numpy())
        bin_pred.extend(pb)
        mit_true.extend(ym.numpy())
        mit_pred.extend(pm)
        dyn_errs.append(((out["pred_state_mean"] - nxt) ** 2).mean(1).cpu().numpy())
    bin_true = np.array(bin_true)
    bin_pred = np.array(bin_pred)
    mit_true = np.array(mit_true)
    mit_pred = np.array(mit_pred)
    bp, br, bf1, _ = precision_recall_fscore_support(
        bin_true, bin_pred, average="binary", pos_label=1, zero_division=0
    )
    tn = int(((bin_pred == 0) & (bin_true == 0)).sum())
    fp = int(((bin_pred == 1) & (bin_true == 0)).sum())
    mit_f1 = f1_score(mit_true, mit_pred, average="macro", zero_division=0)
    val_score = bf1 + 0.5 * mit_f1
    return {
        "binary_f1": float(bf1),
        "binary_precision": float(bp),
        "binary_recall": float(br),
        "binary_fpr": float(fp / max(fp + tn, 1)),
        "mitre_f1_macro": float(mit_f1),
        "val_score": float(val_score),
        "dynamics_mse": float(np.concatenate(dyn_errs).mean()),
    }


def train_one(
    variant: str,
    col_idx: list[int],
    seed: int,
    device: torch.device,
    cfg: dict | None = None,
    max_epochs: int | None = None,
    splits_dir: Path | None = None,
) -> tuple[TemporalTransformerWorldModel, dict]:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    if max_epochs is not None:
        cfg["epochs"] = max_epochs

    torch.manual_seed(seed)
    np.random.seed(seed)

    splits = load_all_splits(splits_dir)
    tr_s, tr_b, tr_m = splits["train"]
    va_s, va_b, va_m = splits["val"]
    te_s, te_b, te_m = splits["test"]

    tr_s = tr_s[:, col_idx]
    va_s = va_s[:, col_idx]
    te_s = te_s[:, col_idx]
    mean, std = fit_scaler(tr_s)
    tr_s = transform_states(tr_s, mean, std)
    va_s = transform_states(va_s, mean, std)
    te_s = transform_states(te_s, mean, std)

    d_state = len(col_idx)
    train_ds = AryanSequenceDataset(tr_s, tr_b, tr_m, cfg["lookback"])
    val_ds = AryanSequenceDataset(va_s, va_b, va_m, cfg["lookback"])
    test_ds = AryanSequenceDataset(te_s, te_b, te_m, cfg["lookback"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = TemporalTransformerWorldModel(
        d_state=d_state,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        lookback=cfg["lookback"],
    ).to(device)

    bin_w = class_weights(tr_b, 2, boost_idx=1).to(device)
    mit_w = class_weights(tr_m, 7).to(device)
    loss_fn = MultiTaskLoss(
        lambda_dynamics=cfg["lambda_dynamics"],
        lambda_infiltration=cfg["lambda_infiltration"],
        lambda_mitre=cfg["lambda_mitre"],
        binary_class_weights=bin_w,
        mitre_class_weights=mit_w,
        use_focal=cfg["use_focal"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best_score, best_state, bad = -1.0, None, 0
    t0 = time.time()
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for seq, nxt, yb, ym in train_loader:
            seq, nxt = seq.to(device), nxt.to(device)
            yb, ym = yb.to(device), ym.to(device)
            out = model(seq)
            ldict = loss_fn(
                out["pred_state_mean"], out["pred_state_logvar"], nxt,
                out["pred_binary"], yb, out["pred_mitre"], ym,
            )
            opt.zero_grad()
            ldict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            losses.append(float(ldict["total"].detach().cpu()))
        sched.step()

        val_m = evaluate(model, val_loader, device)
        val_m["train_loss"] = float(np.mean(losses))
        val_m["epoch"] = epoch
        history.append(val_m)

        if val_m["val_score"] > best_score:
            best_score = val_m["val_score"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)

    test_m = evaluate(model, test_loader, device)
    elapsed = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())

    meta = {
        "variant": variant,
        "seed": seed,
        "num_features": d_state,
        "column_indices": col_idx,
        "feature_subset": describe_subset(col_idx),
        "n_params": n_params,
        "best_val_score": best_score,
        "epochs_run": len(history),
        "train_sec": elapsed,
        "history_last": history[-1] if history else {},
        **{f"test_{k}": v for k, v in test_m.items()},
        "cfg": cfg,
        "splits_dir": str(splits_dir or SPLITS_DIR),
    }
    return model, meta


def save_checkpoint(model, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "metrics": meta,
        "d_state": meta["num_features"],
        "column_indices": meta["column_indices"],
    }, path)


def resolve_columns(spec: dict) -> list[int]:
    return column_indices(
        drop_blocks=spec.get("drop_blocks"),
        keep_blocks=spec.get("keep_blocks"),
        drop_cols=spec.get("drop_cols"),
    )


def composite_score(m: dict) -> float:
    return m["test_binary_f1"] + 0.5 * m["test_mitre_f1_macro"] - 0.35 * m["test_binary_fpr"]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="full_242")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--splits-dir", type=Path, default=None, help="NPZ split dir (default: data/aryan_splits)")
    p.add_argument("--out", type=Path, default=None, help="Checkpoint path")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = column_indices()
    cfg = {}
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size

    splits_dir = args.splits_dir
    model, meta = train_one(args.variant, cols, args.seed, device, cfg, splits_dir=splits_dir)
    if args.out:
        out = args.out
    elif splits_dir and "xmt" in str(splits_dir).lower():
        out = ROOT / "models" / "checkpoints" / "xmt_world_model_best.pt"
    else:
        out = CKPT_DIR / f"{args.variant}_seed{args.seed}.pt"
    save_checkpoint(model, meta, out)
    print(json.dumps({k: meta[k] for k in (
        "variant", "num_features", "n_params", "train_sec", "epochs_run",
        "best_val_score", "test_binary_f1", "test_binary_fpr", "test_mitre_f1_macro",
    )}, indent=2))
    print(f"Saved -> {out}")
