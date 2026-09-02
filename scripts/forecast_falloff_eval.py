"""Open-loop falloff test: 500 steps of history → 500-step forecast.

Shows *where* each model's prediction diverges from ground truth (not aggregate
MSE alone). Models:

  ary_base  — frozen ARY.01, 20-step lookback, autoregressive rollout
  ary_V10   — same open-loop rollout (V10 TTT/memory need revealed labels)
  timesfm   — TimesFM-3 zero-shot, full 500-step context, horizon=500

Outputs JSON only (use ``scripts/falloff_viewer.py`` for interactive plots):
  results/ram_improve/falloff_500/{feat}/falloff.json
  results/ram_improve/falloff_500/run_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ram_improve_eval import load_model  # noqa: E402
from scripts.timesfm_ram_compare import (  # noqa: E402
    ary_model_window,
    build_timeline,
    labels_to_ids,
    load_all_splits,
)
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402

CONTEXT = 500
HORIZON = 500
TIMELINE_LEN = 1500
OUT = ROOT / "results" / "ram_improve" / "falloff_500"
FALLOFF_MULT = 2.0
TIMESFM_BATCH = 32


def ary_openloop_forecast(model, states: np.ndarray, t_end: int, horizon: int) -> np.ndarray:
    """Autoregressive open-loop rollout for `horizon` steps after index t_end."""
    d = states.shape[1]
    preds = np.zeros((horizon, d), dtype=np.float32)
    buf = ary_model_window(states, t_end)
    model.eval()
    with torch.no_grad():
        for i in range(horizon):
            x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
            out = model(x)
            nxt = out["pred_state_mean"].squeeze(0).numpy()
            preds[i] = nxt
            buf = buf[1:] + [nxt]
    return preds


def timesfm_forecast_batch(
    forecaster, series_list: list[np.ndarray], t_end: int, horizon: int
) -> list[np.ndarray]:
    contexts = [s[t_end - CONTEXT + 1 : t_end + 1].astype(np.float32) for s in series_list]
    outs = forecaster.predict_batch(
        contexts, horizon=horizon, return_quantiles=False, use_symmetric_averaging=False
    )
    return [np.asarray(o.forecast, dtype=np.float32) for o in outs]


def per_step_errors(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if pred.ndim == 1:
        return np.abs(pred - actual)
    return np.mean((pred - actual) ** 2, axis=1) ** 0.5


def detect_falloff(errors: np.ndarray, mult: float = FALLOFF_MULT) -> int | None:
    if len(errors) < 20:
        return None
    baseline = float(np.median(errors[:50]))
    if baseline <= 1e-9:
        baseline = float(np.median(errors[:50]) + 1e-6)
    thresh = mult * baseline
    for i, e in enumerate(errors):
        if e > thresh:
            return i + 1
    return None


def pick_anchor(states: np.ndarray, bin_labels: np.ndarray, context: int, horizon: int) -> int:
    n = len(states)
    lo, hi = context - 1, n - horizon - 1
    best, best_score = lo, -1
    for t in range(lo, hi + 1):
        future_attack = int(bin_labels[t + 1 : t + 1 + horizon].sum())
        if future_attack > best_score:
            best_score, best = future_attack, t
    return best


def model_stats(name: str, pred_1d: np.ndarray, actual_1d: np.ndarray) -> dict:
    err = per_step_errors(pred_1d, actual_1d)
    falloff = detect_falloff(err)
    return {
        "per_step_abs_err": err.tolist(),
        "mean_abs_err": float(np.mean(err)),
        "falloff_step": falloff,
        "falloff_threshold": float(FALLOFF_MULT * np.median(err[:50])),
        "forecast": pred_1d.tolist(),
    }


def run_all_features(
    full, bin_labels, base_model, forecaster, feature_indices: list[int] | None = None
) -> list[dict]:
    t_end = pick_anchor(full, bin_labels, CONTEXT, HORIZON)
    attack_steps = int(bin_labels[t_end + 1 : t_end + 1 + HORIZON].sum())
    print(
        f"anchor t_end={t_end}  context [{t_end - CONTEXT + 1},{t_end}]  "
        f"forecast [{t_end + 1},{t_end + HORIZON}]  attack_steps_in_future={attack_steps}"
    )

    print("Running ARY open-loop (all 242 dims)...")
    ary_pred = ary_openloop_forecast(base_model, full, t_end, HORIZON)
    actual_full = full[t_end + 1 : t_end + 1 + HORIZON]

    n_feat = full.shape[1]
    if feature_indices is None:
        feature_indices = list(range(n_feat))
    tfm_preds: dict[int, np.ndarray] = {}
    print(f"Running TimesFM on {len(feature_indices)} features (batch={TIMESFM_BATCH})...")
    for start in range(0, len(feature_indices), TIMESFM_BATCH):
        chunk = feature_indices[start : start + TIMESFM_BATCH]
        batch_series = [full[:, j] for j in chunk]
        batch_out = timesfm_forecast_batch(forecaster, batch_series, t_end, HORIZON)
        for j, pred in zip(chunk, batch_out):
            tfm_preds[j] = pred
        print(f"  features {chunk[0]}..{chunk[-1]} done")

    summaries = []
    for fidx in feature_indices:
        feat_name = FEATURE_COLS_242[fidx]
        actual_1d = actual_full[:, fidx]
        context_1d = full[t_end - CONTEXT + 1 : t_end + 1, fidx]
        ary_1d = ary_pred[:, fidx]
        tfm_1d = tfm_preds[fidx]

        results = {
            "ary_base": model_stats("ary_base", ary_1d, actual_1d),
            "ary_V10": model_stats("ary_V10", ary_1d, actual_1d),
            "timesfm": model_stats("timesfm", tfm_1d, actual_1d),
        }

        fo_ary = results["ary_base"]["falloff_step"]
        fo_tfm = results["timesfm"]["falloff_step"]
        fs_ary = f"step {fo_ary}" if fo_ary else "never"
        fs_tfm = f"step {fo_tfm}" if fo_tfm else "never"
        delta = (fo_ary - fo_tfm) if (fo_ary and fo_tfm) else None
        print(
            f"  [{fidx:03d}] {feat_name:<28} "
            f"ary@{fs_ary:<10} tfm@{fs_tfm:<10} "
            f"delta={delta if delta is not None else 'n/a'}"
        )

        out_dir = OUT / f"feat{fidx:03d}_{feat_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "context": CONTEXT,
            "horizon": HORIZON,
            "anchor_t_end": t_end,
            "feature_idx": fidx,
            "feature_name": feat_name,
            "attack_steps_in_future": attack_steps,
            "note": "ary_V10 open-loop equals ary_base (TTT/memory require revealed future)",
            "context_series": context_1d.tolist(),
            "actual_series": actual_1d.tolist(),
            "models": results,
        }
        (out_dir / "falloff.json").write_text(json.dumps(payload, indent=2))
        summaries.append(
            {
                "feature_idx": fidx,
                "feature_name": feat_name,
                "anchor_t_end": t_end,
                "ary_falloff": fo_ary,
                "timesfm_falloff": fo_tfm,
                "falloff_delta_ary_minus_tfm": delta,
                "ary_mean_abs_err": results["ary_base"]["mean_abs_err"],
                "timesfm_mean_abs_err": results["timesfm"]["mean_abs_err"],
            }
        )
    return summaries


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--feature-idx", type=int, default=None)
    p.add_argument("--feature-ranks", type=str, default=None, help="e.g. 1,2 to run by variance rank")
    p.add_argument("--all-features", action="store_true", help="Run all 242 features (default if no idx)")
    args = p.parse_args()

    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    full, labels, _ = build_timeline(train, val, test, TIMELINE_LEN)
    bin_labels, _ = labels_to_ids(labels)

    run_all = args.all_features or (
        args.feature_idx is None and args.feature_ranks is None
    )
    if args.feature_ranks:
        order = np.argsort(train[0].var(axis=0))[::-1]
        ranks = [int(x) for x in args.feature_ranks.split(",")]
        indices = [int(order[r - 1]) for r in ranks]
        run_all = False
    elif args.feature_idx is not None:
        indices = [args.feature_idx]
        run_all = False
    else:
        indices = []

    print(f"Timeline len={len(full)}  context={CONTEXT}  horizon={HORIZON}\n")
    base_model = load_model()

    print("Loading TimesFM-3...")
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=TIMESFM_BATCH, device="cuda")
    )

    OUT.mkdir(parents=True, exist_ok=True)
    feature_indices = None if run_all else indices
    summaries = run_all_features(full, bin_labels, base_model, forecaster, feature_indices)

    wins = {"ary": 0, "timesfm": 0, "tie": 0, "unknown": 0}
    for s in summaries:
        a, t = s["ary_falloff"], s["timesfm_falloff"]
        if a is None and t is None:
            wins["unknown"] += 1
        elif a is None:
            wins["timesfm"] += 1
        elif t is None:
            wins["ary"] += 1
        elif a > t:
            wins["ary"] += 1
        elif t > a:
            wins["timesfm"] += 1
        else:
            wins["tie"] += 1

    summary_doc = {
        "context": CONTEXT,
        "horizon": HORIZON,
        "n_features": len(summaries),
        "falloff_wins": wins,
        "features": summaries,
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary_doc, indent=2))
    print(f"\nFalloff wins (later = better): ARY {wins['ary']}  TimesFM {wins['timesfm']}  tie {wins['tie']}")
    print(f"Summary -> {OUT / 'run_summary.json'}")
    print("Open viewer: python scripts/falloff_viewer.py")


if __name__ == "__main__":
    main()
