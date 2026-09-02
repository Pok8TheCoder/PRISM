#!/usr/bin/env python3
"""Open-loop forecast falloff: ARY.01 (30s) vs ARY-5sV01 (5s).

Scenario: **10 min context** (real observed windows) → **20 min open-loop forecast**.
Falloff = first forecast step where |error| exceeds 2x median of early forecast steps.

Usage:
  python scripts/falloff_ary30_vs_ary5.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.timesfm_ram_compare import ary_model_window, labels_to_ids  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402
from src.aryan.timeline import build_timeline, _longest_run  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402

CKPT_30 = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"
CKPT_5 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
SPLITS_30 = ROOT / "data" / "aryan_splits"
SPLITS_5 = ROOT / "data" / "aryan_splits_5s"
OUT = ROOT / "results" / "falloff_ary30_vs_ary5"
CONTEXT_SEC = 600.0   # 10 minutes observed context
HORIZON_SEC = 1200.0  # 20 minutes open-loop forecast
TIMELINE_LEN = 800
FALLOFF_MULT = 2.0
LOOKBACK = 20


def steps_for_seconds(window_sec: float, seconds: float) -> int:
    return max(LOOKBACK, int(round(seconds / window_sec)))


def baseline_steps(horizon: int) -> int:
    """Early-forecast window for error baseline (~25% of horizon, capped)."""
    return min(50, max(5, horizon // 4))


def _segment_for_stage(states, mits, stage: int, fallbacks: tuple[int, ...] = ()):
    run = _longest_run(mits, stage)
    if run is None:
        for fb in fallbacks:
            run = _longest_run(mits, fb)
            if run is not None:
                break
    if run is None:
        atk = np.where(mits > 0)[0]
        if len(atk) == 0:
            return states[: min(32, len(states))]
        mid = len(atk) // 2
        half = max(8, len(atk) // 4)
        lo = max(0, mid - half)
        hi = min(len(states), mid + half)
        return states[lo:hi]
    lo, hi = run
    return states[lo:hi]


def build_timeline_safe(train, val, test, target_len: int):
    """Like ``build_timeline`` but tolerates missing MITRE stages in a split."""
    try:
        return build_timeline(train, val, test, target_len)
    except TypeError:
        from src.aryan.constants import KILLCHAIN_ATTACKS

        tr_s, _, tr_m = train
        va_s, _, va_m = val
        te_s, _, te_m = test
        seg_initial_access = _segment_for_stage(va_s, va_m, 2, (6,))
        seg_lateral_move = _segment_for_stage(te_s, te_m, 3, (6, 2))
        seg_impact = _segment_for_stage(tr_s, tr_m, 6, (2, 3))

        benign_pool = np.concatenate([tr_s[tr_m == 0], va_s[va_m == 0], te_s[te_m == 0]])
        attack_total = len(seg_initial_access) + len(seg_lateral_move) + len(seg_impact)
        remaining = max(target_len - attack_total, 40)
        gap_len = remaining // 4
        n_needed = gap_len * 4
        reps = int(np.ceil(n_needed / max(len(benign_pool), 1)))
        benign_tiled = np.tile(benign_pool, (reps, 1))[:n_needed]
        gaps = [benign_tiled[i * gap_len:(i + 1) * gap_len] for i in range(4)]

        pieces = [
            ("Benign", gaps[0]),
            (KILLCHAIN_ATTACKS[2], seg_initial_access),
            ("Benign", gaps[1]),
            (KILLCHAIN_ATTACKS[3], seg_lateral_move),
            ("Benign", gaps[2]),
            (KILLCHAIN_ATTACKS[6], seg_impact),
            ("Benign", gaps[3]),
        ]
        arrs, labels, segments = [], [], []
        pos = 0
        for lbl, arr in pieces:
            arrs.append(arr)
            labels.extend([lbl] * len(arr))
            segments.append((lbl, pos, pos + len(arr)))
            pos += len(arr)
        return np.concatenate(arrs), labels, segments


def load_ckpt(path: Path) -> TemporalTransformerWorldModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(
        d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=LOOKBACK,
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def ary_openloop_forecast(model, states: np.ndarray, t_end: int, horizon: int) -> np.ndarray:
    d = states.shape[1]
    preds = np.zeros((horizon, d), dtype=np.float32)
    buf = ary_model_window(states, t_end)
    with torch.no_grad():
        for i in range(horizon):
            x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
            out = model(x)
            nxt = out["pred_state_mean"].squeeze(0).numpy()
            preds[i] = nxt
            buf = buf[1:] + [nxt]
    return preds


def per_step_abs_err(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    if pred.ndim == 1:
        return np.abs(pred - actual)
    return np.mean(np.abs(pred - actual), axis=1)


def detect_falloff(errors: np.ndarray, mult: float = FALLOFF_MULT, baseline_len: int = 10) -> int | None:
    if len(errors) < max(5, baseline_len):
        return None
    n = min(baseline_len, len(errors))
    baseline = float(np.median(errors[:n]))
    if baseline <= 1e-9:
        baseline = 1e-6
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


def run_model_falloff(
    name: str,
    window_sec: float,
    model,
    splits_dir: Path,
    feature_indices: list[int],
) -> dict:
    splits = load_all_splits(splits_dir)
    full, labels, segments = build_timeline_safe(splits["train"], splits["val"], splits["test"], TIMELINE_LEN)
    bin_labels, _ = labels_to_ids(labels)
    context = steps_for_seconds(window_sec, CONTEXT_SEC)
    horizon = steps_for_seconds(window_sec, HORIZON_SEC)
    blen = baseline_steps(horizon)
    t_end = pick_anchor(full, bin_labels, context, horizon)
    attack_steps = int(bin_labels[t_end + 1 : t_end + 1 + horizon].sum())

    print(f"  [{name}] {context} ctx steps ({CONTEXT_SEC/60:.0f} min) + {horizon} forecast steps "
          f"({HORIZON_SEC/60:.0f} min) @ {window_sec}s/window  anchor={t_end}  attacks_in_horizon={attack_steps}")
    pred = ary_openloop_forecast(model, full, t_end, horizon)
    actual = full[t_end + 1 : t_end + 1 + horizon]

    feat_rows = []
    falloff_steps = []
    falloff_secs = []
    never = 0

    for fidx in feature_indices:
        err = per_step_abs_err(pred[:, fidx], actual[:, fidx])
        fo = detect_falloff(err, baseline_len=blen)
        fo_sec = fo * window_sec if fo else None
        fo_min = fo_sec / 60.0 if fo_sec is not None else None
        if fo is None:
            never += 1
        else:
            falloff_steps.append(fo)
            falloff_secs.append(fo_sec)
        feat_rows.append({
            "feature_idx": fidx,
            "feature_name": FEATURE_COLS_242[fidx],
            "falloff_step": fo,
            "falloff_sec": fo_sec,
            "falloff_min_into_forecast": fo_min,
            "mean_abs_err": float(np.mean(err)),
            "baseline_err_median": float(np.median(err[:blen])),
            "threshold": float(FALLOFF_MULT * np.median(err[:blen])),
            "per_step_err": err.tolist(),
        })

    steps_arr = np.array(falloff_steps) if falloff_steps else np.array([])
    secs_arr = np.array(falloff_secs) if falloff_secs else np.array([])

    return {
        "model": name,
        "window_sec": window_sec,
        "splits_dir": str(splits_dir),
        "timeline_len": len(full),
        "anchor_t_end": t_end,
        "attack_steps_in_horizon": attack_steps,
        "context_sec": CONTEXT_SEC,
        "horizon_sec": HORIZON_SEC,
        "context_steps": context,
        "horizon_steps": horizon,
        "baseline_steps": blen,
        "falloff_mult": FALLOFF_MULT,
        "n_features": len(feature_indices),
        "n_never_falloff": never,
        "falloff_step_mean": float(steps_arr.mean()) if len(steps_arr) else None,
        "falloff_step_median": float(np.median(steps_arr)) if len(steps_arr) else None,
        "falloff_step_p25": float(np.percentile(steps_arr, 25)) if len(steps_arr) else None,
        "falloff_step_p75": float(np.percentile(steps_arr, 75)) if len(steps_arr) else None,
        "falloff_sec_mean": float(secs_arr.mean()) if len(secs_arr) else None,
        "falloff_sec_median": float(np.median(secs_arr)) if len(secs_arr) else None,
        "falloff_sec_p25": float(np.percentile(secs_arr, 25)) if len(secs_arr) else None,
        "falloff_sec_p75": float(np.percentile(secs_arr, 75)) if len(secs_arr) else None,
        "features": feat_rows,
        "segments": [(lbl, a, b) for lbl, a, b in segments],
    }


def compare_reports(r30: dict, r5: dict) -> dict:
    by_f = {f["feature_idx"]: f for f in r5["features"]}
    earlier_5s = later_5s = tie = only30 = only5 = 0
    detail = []
    for f30 in r30["features"]:
        fidx = f30["feature_idx"]
        f5 = by_f[fidx]
        s30, s5 = f30["falloff_step"], f5["falloff_step"]
        if s30 is None and s5 is None:
            tie += 1
            winner = "tie_never"
        elif s30 is None:
            only5 += 1
            winner = "ary5_only"
        elif s5 is None:
            only30 += 1
            winner = "ary30_only"
        elif s5 < s30:
            earlier_5s += 1
            winner = "ary5_earlier"
        elif s30 < s5:
            later_5s += 1
            winner = "ary30_earlier"
        else:
            tie += 1
            winner = "tie_step"
        detail.append({
            "feature_idx": fidx,
            "feature_name": f30["feature_name"],
            "ary30_falloff_step": s30,
            "ary5_falloff_step": s5,
            "ary30_falloff_sec": f30["falloff_sec"],
            "ary5_falloff_sec": f5["falloff_sec"],
            "winner": winner,
            "delta_sec_5_minus_30": (
                (f5["falloff_sec"] - f30["falloff_sec"])
                if f30["falloff_sec"] is not None and f5["falloff_sec"] is not None else None
            ),
        })
    return {
        "ary5_earlier_steps": earlier_5s,
        "ary30_earlier_steps": later_5s,
        "tie": tie,
        "ary5_only_falloff": only5,
        "ary30_only_falloff": only30,
        "ary5_earlier_wall_clock": sum(
            1 for d in detail if d["delta_sec_5_minus_30"] is not None and d["delta_sec_5_minus_30"] < 0
        ),
        "ary30_earlier_wall_clock": sum(
            1 for d in detail if d["delta_sec_5_minus_30"] is not None and d["delta_sec_5_minus_30"] > 0
        ),
        "features": detail,
    }


def write_report_md(payload: dict, path: Path) -> None:
    r30 = payload["ary30"]
    r5 = payload["ary5"]
    cmp = payload["comparison"]
    lines = [
        "# ARY.01 vs ARY-5sV01 — Open-Loop Forecast Falloff",
        "",
        f"- **ARY.01**: 30s windows, checkpoint `{CKPT_30.name}`",
        f"- **ARY-5sV01**: 5s windows, checkpoint `{CKPT_5.name}`",
        f"- **Scenario**: {CONTEXT_SEC/60:.0f} min context → {HORIZON_SEC/60:.0f} min open-loop forecast",
        f"- **Falloff rule**: first forecast step where abs error > {FALLOFF_MULT}x median of early steps",
        f"- **ARY.01**: {r30['context_steps']} ctx + {r30['horizon_steps']} forecast steps @ 30s",
        f"- **ARY-5sV01**: {r5['context_steps']} ctx + {r5['horizon_steps']} forecast steps @ 5s",
        f"- **Features evaluated**: {r30['n_features']}",
        "",
        "## Summary",
        "",
        "| Metric | ARY.01 (30s) | ARY-5sV01 (5s) |",
        "|---|---:|---:|",
        f"| Median falloff step | {r30['falloff_step_median']} | {r5['falloff_step_median']} |",
        f"| Mean falloff step | {r30['falloff_step_mean']:.1f} | {r5['falloff_step_mean']:.1f} |" if r30['falloff_step_mean'] and r5['falloff_step_mean'] else "",
        f"| **Median falloff (wall-clock sec)** | **{r30['falloff_sec_median']}** | **{r5['falloff_sec_median']}** |",
        f"| Mean falloff (wall-clock sec) | {r30['falloff_sec_mean']:.0f} | {r5['falloff_sec_mean']:.0f} |" if r30['falloff_sec_mean'] and r5['falloff_sec_mean'] else "",
        f"| Never falloff (of {r30['n_features']} feats) | {r30['n_never_falloff']} | {r5['n_never_falloff']} |",
        "",
        "## Head-to-head (by forecast step)",
        "",
        f"- ARY-5s falls off **earlier** (fewer steps): **{cmp['ary5_earlier_steps']}** features",
        f"- ARY.01 falls off earlier: **{cmp['ary30_earlier_steps']}** features",
        f"- Same step / both never: **{cmp['tie']}**",
        "",
        "## Head-to-head (by wall-clock seconds)",
        "",
        f"- ARY-5s falls off earlier in real time: **{cmp['ary5_earlier_wall_clock']}** features",
        f"- ARY.01 falls off earlier in real time: **{cmp['ary30_earlier_wall_clock']}** features",
        "",
        "## Interpretation",
        "",
    ]
    if r5["falloff_sec_median"] and r30["falloff_sec_median"]:
        if r5["falloff_sec_median"] < r30["falloff_sec_median"]:
            lines.append(
                f"ARY-5sV01 median falloff occurs at **{r5['falloff_sec_median']:.0f}s** vs ARY.01 **{r30['falloff_sec_median']:.0f}s** "
                f"— shorter windows do **not** buy longer trustworthy forecast horizon in wall-clock time; "
                f"5s falls off **sooner in real seconds**."
            )
        else:
            lines.append(
                f"ARY-5sV01 median falloff at **{r5['falloff_sec_median']:.0f}s** vs ARY.01 **{r30['falloff_sec_median']:.0f}s** "
                f"— 5s maintains forecast quality longer in wall-clock time despite more steps."
            )
    lines.extend([
        "",
        "Note: one forecast step = one window (30s or 5s). More steps at 5s ≠ longer real-time horizon per step.",
        "",
        "## Per-feature falloff (earliest 15 where 5s loses in wall-clock time)",
        "",
        "| Feature | ARY.01 step (sec) | ARY-5s step (sec) | Δ sec (5−30) |",
        "|---|---:|---:|---:|",
    ])
    worse = sorted(
        [d for d in cmp["features"] if d["delta_sec_5_minus_30"] is not None],
        key=lambda d: d["delta_sec_5_minus_30"],
    )[:15]
    for d in worse:
        s30 = f"{d['ary30_falloff_step']} ({d['ary30_falloff_sec']:.0f}s)" if d["ary30_falloff_sec"] else "never"
        s5 = f"{d['ary5_falloff_step']} ({d['ary5_falloff_sec']:.0f}s)" if d["ary5_falloff_sec"] else "never"
        lines.append(f"| {d['feature_name']} | {s30} | {s5} | {d['delta_sec_5_minus_30']:.0f} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--top-features", type=int, default=242, help="Top-N by train variance (default all 242)")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()

    if not CKPT_5.exists():
        print(f"Missing {CKPT_5}. Run: python scripts/save_ary5sv01.py")
        return 1

    print("Loading ARY.01 (30s)...")
    m30 = load_ckpt(CKPT_30)
    print("Loading ARY-5sV01...")
    m5 = load_ckpt(CKPT_5)

    tr_s, _, _ = load_all_splits(SPLITS_30)["train"]
    order = np.argsort(tr_s.var(axis=0))[::-1]
    n = min(args.top_features, len(order))
    feature_indices = [int(order[i]) for i in range(n)]
    print(f"Evaluating {n} features (variance rank 1..{n})\n")

    print("=== ARY.01 30s falloff ===")
    r30 = run_model_falloff("ARY.01", 30.0, m30, SPLITS_30, feature_indices)
    print("=== ARY-5sV01 falloff ===")
    r5 = run_model_falloff("ARY-5sV01", 5.0, m5, SPLITS_5, feature_indices)

    comparison = compare_reports(r30, r5)
    payload = {"ary30": r30, "ary5": r5, "comparison": comparison}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "falloff_report.json").write_text(json.dumps(payload, indent=2))
    write_report_md(payload, args.out / "FALLOFF_REPORT.md")

    print("\n=== FALLOFF SUMMARY ===")
    print(f"  ARY.01  median step={r30['falloff_step_median']}  ({r30['falloff_sec_median']}s wall-clock)")
    print(f"  ARY-5s  median step={r5['falloff_step_median']}  ({r5['falloff_sec_median']}s wall-clock)")
    print(f"  5s earlier (steps): {comparison['ary5_earlier_steps']}  30s earlier: {comparison['ary30_earlier_steps']}")
    print(f"  5s earlier (sec):   {comparison['ary5_earlier_wall_clock']}  30s earlier: {comparison['ary30_earlier_wall_clock']}")
    print(f"\nReport -> {args.out / 'FALLOFF_REPORT.md'}")
    print(f"JSON   -> {args.out / 'falloff_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
