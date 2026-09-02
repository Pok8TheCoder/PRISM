"""Estimate GPU/RAM needs and smoke-test training configs for ARY242 ablations.

Run before Colab/local sweeps:
  python scripts/estimate_ary242_resources.py
  python scripts/estimate_ary242_resources.py --smoke-train --d-state 82
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_aryan_wm import DEFAULT_CFG, train_one  # noqa: E402
from src.aryan.feature_schema242 import column_indices, describe_subset  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
        "sm_count": props.multi_processor_count,
    }


def measure_peak_vram(d_state: int, batch_size: int, lookback: int = 20) -> dict:
    if not torch.cuda.is_available():
        return {"peak_vram_mb": None, "ok": False, "note": "no cuda"}
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = TemporalTransformerWorldModel(d_state=d_state, lookback=lookback).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    x = torch.randn(batch_size, lookback, d_state, device=device)
    nxt = torch.randn(batch_size, d_state, device=device)
    yb = torch.randint(0, 2, (batch_size,), device=device)
    ym = torch.randint(0, 7, (batch_size,), device=device)

    out = model(x)
    loss = (
        ((out["pred_state_mean"] - nxt) ** 2).mean()
        + torch.nn.functional.cross_entropy(out["pred_binary"], yb)
        + torch.nn.functional.cross_entropy(out["pred_mitre"], ym)
    )
    opt.zero_grad()
    loss.backward()
    opt.step()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt, x, nxt, yb, ym, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "d_state": d_state,
        "batch_size": batch_size,
        "lookback": lookback,
        "peak_vram_mb": round(peak_mb, 1),
        "n_params_m": round(n_params / 1e6, 3),
        "ok": True,
    }


def recommend_batch(total_vram_gb: float, d_state: int, target_util: float = 0.75) -> int:
    """Binary search max batch that fits ~target_util of VRAM."""
    if not torch.cuda.is_available():
        return DEFAULT_CFG["batch_size"]
    budget_mb = total_vram_gb * 1024 * target_util
    lo, hi, best = 8, 512, 8
    for _ in range(8):
        mid = (lo + hi) // 2
        m = measure_peak_vram(d_state, mid)
        if m["peak_vram_mb"] <= budget_mb:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(8, min(best, 128))


def estimate_epoch_sec(d_state: int, batch_size: int, n_train: int = 1937) -> float:
    """One short timed epoch on real data (subset if CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cols = column_indices()[:d_state] if d_state < 242 else column_indices()
    cfg = {"epochs": 1, "patience": 1, "batch_size": batch_size}
    t0 = time.time()
    _, meta = train_one("_estimate_", cols, 0, device, cfg, max_epochs=1)
    return meta["train_sec"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-train", action="store_true", help="Run 1-epoch real-data smoke test")
    ap.add_argument("--d-state", type=int, default=242)
    ap.add_argument("--batch-size", type=int, default=0)
    args = ap.parse_args()

    info = gpu_info()
    d_state = args.d_state
    print("=== GPU ===")
    print(json.dumps(info, indent=2))

    if not info.get("cuda"):
        print("\nNo CUDA — estimates are CPU-only.")
        return

    rec_batch = recommend_batch(info["total_vram_gb"], d_state)
    batch = args.batch_size or rec_batch
    vram = measure_peak_vram(d_state, batch)

    # Reference configs for Colab T4 (15GB) vs local 3060 (12GB)
    profiles = {}
    for label, vram_gb in [("rtx3060_12gb", 12.0), ("colab_t4_15gb", 15.0)]:
        b = recommend_batch(vram_gb, d_state)
        profiles[label] = {
            "recommended_batch_size": b,
            **measure_peak_vram(d_state, b),
            "estimated_full_train_min": None,
        }

    print("\n=== VRAM probe (current GPU) ===")
    print(json.dumps(vram, indent=2))
    print(f"\nRecommended batch for this GPU @ d_state={d_state}: {rec_batch}")

    print("\n=== Profile table ===")
    for label, p in profiles.items():
        print(f"  {label}: batch={p['recommended_batch_size']}  peak={p['peak_vram_mb']} MB")

    if args.smoke_train:
        print(f"\n=== Smoke train (1 epoch, d_state={d_state}, batch={batch}) ===")
        cols = column_indices()
        if d_state < len(cols):
            cols = cols[:d_state]
        _, meta = train_one("smoke", cols, 0, torch.device("cuda"),
                            {"epochs": 1, "patience": 1, "batch_size": batch}, max_epochs=1)
        print(json.dumps({
            k: meta[k] for k in (
                "num_features", "n_params", "train_sec", "test_binary_f1",
                "test_binary_fpr", "test_mitre_f1_macro", "best_val_score",
            )
        }, indent=2))
        # Extrapolate full sweep time
        epochs_typical = 25
        n_variants = 12
        est_min = meta["train_sec"] / 60 * epochs_typical * n_variants
        print(f"\nRough sweep estimate ({n_variants} variants × ~{epochs_typical} epochs): {est_min:.1f} min")

    out = ROOT / "results" / "ary242_ablation" / "resource_estimate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "gpu": info,
        "probe": vram,
        "recommended_batch": rec_batch,
        "profiles": profiles,
        "subset": describe_subset(column_indices()[:d_state] if d_state < 242 else column_indices()),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
