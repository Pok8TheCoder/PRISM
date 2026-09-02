#!/usr/bin/env python3
"""Falloff: overlapping dynamic tiers — solo vs blend vs fixed 5s/30s.

10 min context, 20 min forecast (1s fine grid).

Usage:
  python scripts/falloff_dynamic_tiers.py
  python scripts/plot_falloff_dynamic.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.falloff_ary30_vs_ary5 import (  # noqa: E402
    CKPT_30,
    CKPT_5,
    SPLITS_5,
    build_timeline_safe,
    detect_falloff,
    load_ckpt,
    per_step_abs_err,
    pick_anchor,
)
from scripts.timesfm_ram_compare import labels_to_ids  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.dynamic_windows import (  # noqa: E402
    DEFAULT_OVERLAP_TIERS,
    CombineMode,
    FINE_SEC,
    run_overlap_forecast,
    upsample_states_5s_to_1s,
)
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402
from scripts.timesfm_ram_compare import ary_model_window  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "results" / "falloff_dynamic"
CONTEXT_SEC = 600.0
HORIZON_SEC = 1200.0
TIMELINE_LEN = 800
TOP_FEATURES = 48


def openloop_fixed(model, states, t_end, horizon_steps):
    preds = np.zeros((horizon_steps, states.shape[1]), dtype=np.float32)
    buf = ary_model_window(states, t_end)
    with torch.no_grad():
        for i in range(horizon_steps):
            x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
            nxt = model(x)["pred_state_mean"].squeeze(0).numpy()
            preds[i] = nxt
            buf = buf[1:] + [nxt]
    return preds


def falloff_summary(pred, actual, feature_indices, baseline_len=10) -> dict:
    rows, secs = [], []
    for fidx in feature_indices:
        err = per_step_abs_err(pred[:, fidx], actual[:, fidx])
        fo = detect_falloff(err, baseline_len=baseline_len)
        fo_sec = fo * FINE_SEC if fo else None
        if fo_sec is not None:
            secs.append(fo_sec)
        rows.append({
            "feature_idx": fidx,
            "feature_name": FEATURE_COLS_242[fidx],
            "falloff_sec": fo_sec,
            "falloff_min": fo_sec / 60.0 if fo_sec else None,
            "per_step_err": err.tolist(),
        })
    arr = np.array(secs) if secs else np.array([])
    return {
        "features": rows,
        "n_features": len(feature_indices),
        "n_never_falloff": sum(1 for r in rows if r["falloff_sec"] is None),
        "falloff_sec_median": float(np.median(arr)) if len(arr) else None,
        "falloff_min_median": float(np.median(arr) / 60.0) if len(arr) else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--top-features", type=int, default=TOP_FEATURES)
    args = p.parse_args()

    print("Overlap tier schedule:")
    for t in DEFAULT_OVERLAP_TIERS:
        print(f"  {t.window_sec:>2}s ws  solo [{t.solo_start},{t.solo_end}]  steps from {t.step_start}s")

    m30, m5 = load_ckpt(CKPT_30), load_ckpt(CKPT_5)
    models = {"ary30": m30, "ary5": m5}

    splits = load_all_splits(SPLITS_5)
    full_5s, labels, _ = build_timeline_safe(splits["train"], splits["val"], splits["test"], TIMELINE_LEN)
    full_1s = upsample_states_5s_to_1s(full_5s)
    bin_labels, _ = labels_to_ids(labels)

    ctx_n = int(CONTEXT_SEC / FINE_SEC)
    hz_n = int(HORIZON_SEC / FINE_SEC)
    t_end_5s = pick_anchor(full_5s, bin_labels, max(20, int(CONTEXT_SEC / 5)), int(HORIZON_SEC / 5))
    t_end_fine = min(t_end_5s * 5 + 4, len(full_1s) - hz_n - 2)

    ctx = full_1s[t_end_fine - ctx_n + 1 : t_end_fine + 1]
    actual = full_1s[t_end_fine + 1 : t_end_fine + 1 + hz_n]

    order = np.argsort(splits["train"][0].var(axis=0))[::-1]
    fidx = [int(order[i]) for i in range(min(args.top_features, len(order)))]

    print(f"\nContext {CONTEXT_SEC/60:.0f} min -> forecast {HORIZON_SEC/60:.0f} min ({hz_n} fine steps)")
    print(f"Evaluating {len(fidx)} features...\n")

    pred_solo, _ = run_overlap_forecast(models, ctx, HORIZON_SEC, DEFAULT_OVERLAP_TIERS, CombineMode.SOLO)
    print("  solo done")
    pred_blend, _ = run_overlap_forecast(models, ctx, HORIZON_SEC, DEFAULT_OVERLAP_TIERS, CombineMode.BLEND)
    print("  blend done")

    n = min(len(pred_solo), len(actual))
    pred_solo, pred_blend, actual = pred_solo[:n], pred_blend[:n], actual[:n]

    horizon_5s = int(HORIZON_SEC / 5)
    pred_5 = openloop_fixed(m5, full_5s, t_end_5s, horizon_5s)
    actual_5 = full_5s[t_end_5s + 1 : t_end_5s + 1 + horizon_5s]

    t_end_30 = pick_anchor(full_5s, bin_labels, max(20, int(CONTEXT_SEC / 30)), int(HORIZON_SEC / 30))
    horizon_30 = int(HORIZON_SEC / 30)
    pred_30 = openloop_fixed(m30, full_5s, t_end_30, horizon_30)
    actual_30 = full_5s[t_end_30 + 1 : t_end_30 + 1 + horizon_30]

    payload = {
        "context_sec": CONTEXT_SEC,
        "horizon_sec": HORIZON_SEC,
        "tiers": [
            {"window_sec": t.window_sec, "solo_start": t.solo_start, "solo_end": t.solo_end, "step_start": t.step_start}
            for t in DEFAULT_OVERLAP_TIERS
        ],
        "overlap_solo": falloff_summary(pred_solo, actual, fidx),
        "overlap_blend": falloff_summary(pred_blend, actual, fidx),
        "fixed_5s": falloff_summary(pred_5, actual_5, fidx),
        "fixed_30s": falloff_summary(pred_30, actual_30, fidx),
    }
    payload["overlap_solo"]["mode"] = "overlap_solo"
    payload["overlap_blend"]["mode"] = "overlap_blend"

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "falloff_dynamic.json").write_text(json.dumps(payload, indent=2))

    print("\n=== Falloff (min into 20-min forecast) ===")
    for key, label in [
        ("overlap_solo", "Overlap SOLO"),
        ("overlap_blend", "Overlap BLEND"),
        ("fixed_5s", "Fixed 5s"),
        ("fixed_30s", "Fixed 30s"),
    ]:
        r = payload[key]
        med = r["falloff_min_median"]
        print(f"  {label:<16} median={med:.1f} min  never={r['n_never_falloff']}/{r['n_features']}")
    print(f"\nJSON -> {args.out / 'falloff_dynamic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
