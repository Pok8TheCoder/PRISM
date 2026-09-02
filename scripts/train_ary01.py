#!/usr/bin/env python3
"""Train ARY TemporalTransformerWorldModel with the ARY.01 recipe.

Matches ``origin/aryan`` @ ``1ec06bc`` / ``configs/train.yaml``:
  seed 42, AdamW lr=2e-4, cosine schedule, patience 200, no focal, no balanced sampling,
  inverse-sqrt class weights, val_score = bin_f1 + 0.5 * mitre_f1,
  **raw states from NPZ** (no extra StandardScaler).

Usage:
  python scripts/train_ary01.py --splits-dir data/aryan_splits_5s --tag ary5s
  python scripts/train_ary01.py --init models/checkpoints/aryan_world_model_best.pt  # fine-tune
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.components import MultiTaskLoss  # noqa: E402
from src.aryan.dataset import AryanSequenceDataset, LOOKBACK, load_all_splits  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402

DEFAULT = dict(
    seed=42,
    batch_size=64,
    epochs=200,
    patience=200,
    lr=2e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    lambda_dynamics=0.5,
    lambda_infiltration=1.2,
    lambda_mitre=1.0,
    lookback=LOOKBACK,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse sqrt frequency — same as origin/aryan ``compute_class_weights``."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if present.any():
        inv = 1.0 / np.sqrt(counts[present])
        weights[present] = inv / inv.sum() * present.sum()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    bin_true, bin_pred, mit_true, mit_pred = [], [], [], []
    dyn_errs = []
    for seq, nxt, yb, ym in loader:
        seq = seq.to(device, non_blocking=True)
        nxt = nxt.to(device, non_blocking=True)
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
        bin_true, bin_pred, average="binary", pos_label=1, zero_division=0,
    )
    tn = int(((bin_pred == 0) & (bin_true == 0)).sum())
    fp = int(((bin_pred == 1) & (bin_true == 0)).sum())
    mit_f1 = f1_score(mit_true, mit_pred, average="macro", zero_division=0)
    return {
        "binary_f1": float(bf1),
        "binary_precision": float(bp),
        "binary_recall": float(br),
        "binary_fpr": float(fp / max(fp + tn, 1)),
        "mitre_f1_macro": float(mit_f1),
        "val_score": float(bf1 + 0.5 * mit_f1),
        "dynamics_mse": float(np.concatenate(dyn_errs).mean()),
    }


def train(
    splits_dir: Path,
    out_path: Path,
    init_ckpt: Path | None = None,
    cfg: dict | None = None,
) -> dict:
    cfg = {**DEFAULT, **(cfg or {})}
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("amp") and device.type == "cuda")
    fast_io = bool(cfg.get("fast_io"))
    if fast_io and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    splits = load_all_splits(splits_dir)
    tr_s, tr_b, tr_m = splits["train"]
    va_s, va_b, va_m = splits["val"]
    te_s, te_b, te_m = splits["test"]

    loader_kw: dict = {"batch_size": cfg["batch_size"], "pin_memory": device.type == "cuda"}
    if fast_io:
        loader_kw.update(num_workers=4, persistent_workers=True, prefetch_factor=2)

    train_loader = DataLoader(
        AryanSequenceDataset(tr_s, tr_b, tr_m, cfg["lookback"]),
        shuffle=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        AryanSequenceDataset(va_s, va_b, va_m, cfg["lookback"]),
        shuffle=False,
        **loader_kw,
    )
    test_loader = DataLoader(
        AryanSequenceDataset(te_s, te_b, te_m, cfg["lookback"]),
        shuffle=False,
        **loader_kw,
    )

    d_state = tr_s.shape[1]
    model = TemporalTransformerWorldModel(
        d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=cfg["lookback"],
    ).to(device)

    if init_ckpt and init_ckpt.exists():
        ck = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model_state_dict", ck)
        if sd["embedding.proj.weight"].shape[1] != d_state:
            print(f"WARNING: init ckpt d_state mismatch; training from scratch")
        else:
            model.load_state_dict(sd, strict=True)
            print(f"Loaded init weights from {init_ckpt}")

    bin_w = compute_class_weights(tr_b, 2).to(device)
    mit_w = compute_class_weights(tr_m, 7).to(device)
    loss_fn = MultiTaskLoss(
        lambda_dynamics=cfg["lambda_dynamics"],
        lambda_infiltration=cfg["lambda_infiltration"],
        lambda_mitre=cfg["lambda_mitre"],
        binary_class_weights=bin_w,
        mitre_class_weights=mit_w,
        use_focal=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_state, best_epoch, bad = -1.0, None, 0, 0
    t0 = time.time()
    nb = cfg["batch_size"]
    print(f"  device={device}  batch={nb}  amp={use_amp}  fast_io={fast_io}  train_samples={len(train_loader.dataset)}")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for seq, nxt, yb, ym in train_loader:
            seq = seq.to(device, non_blocking=True)
            nxt = nxt.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ym = ym.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(seq)
                ld = loss_fn(
                    out["pred_state_mean"], out["pred_state_logvar"], nxt,
                    out["pred_binary"], yb, out["pred_mitre"], ym,
                )
            scaler.scale(ld["total"]).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
        sched.step()

        val_m = evaluate(model, val_loader, device)
        if val_m["val_score"] > best_score:
            best_score = val_m["val_score"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"Early stop epoch {epoch} (best={best_epoch}, score={best_score:.4f})")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep {epoch:3d}  val_score={val_m['val_score']:.4f}  "
                  f"bin_f1={val_m['binary_f1']:.3f}  mitre_f1={val_m['mitre_f1_macro']:.3f}")

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, device)
    elapsed = time.time() - t0

    meta = {
        "tag": out_path.stem,
        "splits_dir": str(splits_dir),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "train_sec": round(elapsed, 1),
        "cfg": cfg,
        **{f"test_{k}": v for k, v in test_m.items()},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "metrics": meta,
        "model_class": "TemporalTransformerWorldModel",
    }, out_path)
    print(json.dumps(meta, indent=2))
    print(f"Saved -> {out_path}")
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "aryan_splits_5s")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tag", default="ary5s_v01")
    p.add_argument("--init", type=Path, default=None, help="Optional checkpoint to fine-tune from")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Override learning rate (e.g. 5e-5 for fine-tune)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--amp", action="store_true", help="Mixed precision on CUDA")
    p.add_argument("--fast", action="store_true", help="cudnn.benchmark + DataLoader workers")
    args = p.parse_args()

    out = args.out or (ROOT / "models" / "checkpoints" / f"{args.tag}.pt")
    cfg = {}
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.patience:
        cfg["patience"] = args.patience
    if args.amp:
        cfg["amp"] = True
    if args.fast:
        cfg["fast_io"] = True
    train(args.splits_dir, out, init_ckpt=args.init, cfg=cfg)


if __name__ == "__main__":
    main()
