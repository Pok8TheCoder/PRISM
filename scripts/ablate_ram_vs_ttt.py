#!/usr/bin/env python3
"""Offline ablation: does V10's classify-blend memory (RAM) add anything
beyond its Test-Time-Training (TTT), or vice versa?

`ary_v10` = adapt (TTT) + use_memory (hidden-key classify-blend, k=3).
Both mechanisms only ever act on *already-captured* traffic (the round's
own `state`/`true_bin`/`true_mit` sequence, saved verbatim in
`round.json::trace`), so we can replay that exact sequence -- no Docker,
no re-capture -- through 3 configurations of `StreamingARY` and diff the
scores:

  v10_full      -- adapt=True,  use_memory=True   (what the live lab ran)
  v10_ttt_only  -- adapt=True,  use_memory=False  (TTT's own contribution)
  v10_mem_only  -- adapt=False, use_memory=True   (RAM's own contribution,
                   same hidden-key/k=3 config V10 uses, just without TTT)

This isolates whether V10's live-lab detections came from the gradient
update, the memory blend, or both -- and, unlike the live orchestrator,
this replay is deterministic and can be re-run cheaply for every round.

Usage:
  python scripts/ablate_ram_vs_ttt.py [--results-dir results/ram_improve/live_lab]
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

from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.lab_scoring import compute_round_scores  # noqa: E402
from src.aryan.streaming_variants import (  # noqa: E402
    ADAPT_LR,
    StreamingARY,
    calibrate_thresholds,
    load_model,
)


def make_variant(
    adapt: bool,
    use_memory: bool,
    base_model,
    raw_thresh: float,
    hidden_thresh: float,
) -> StreamingARY:
    """Build a StreamingARY with an arbitrary (adapt, use_memory) combo,
    always using V10's hidden-key/k=3 memory config when memory is on, so
    the *only* thing that varies across variants is adapt/use_memory."""
    sys_obj = StreamingARY("ary_v10" if adapt else "ary_base", base_model=base_model,
                            raw_thresh=raw_thresh, hidden_thresh=hidden_thresh)
    sys_obj.adapt = adapt
    sys_obj.use_memory = use_memory
    sys_obj.key_space = "hidden"
    sys_obj.knn_k = 3
    sys_obj.match_thresh = hidden_thresh
    if adapt and sys_obj.opt is None:
        sys_obj.opt = torch.optim.SGD(sys_obj.online.parameters(), lr=ADAPT_LR)
    return sys_obj


def replay_round(round_path: Path, base_model, raw_thresh: float, hidden_thresh: float) -> dict:
    data = json.loads(round_path.read_text())
    trace = data["trace"]
    events = data["events"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    true_bins = [int(t["true_bin"]) for t in trace]
    true_mits = [int(t["true_mit"]) for t in trace]

    variants = {
        "v10_full (TTT+RAM)": make_variant(True, True, base_model, raw_thresh, hidden_thresh),
        "v10_ttt_only (-RAM)": make_variant(True, False, base_model, raw_thresh, hidden_thresh),
        "v10_mem_only (-TTT)": make_variant(False, True, base_model, raw_thresh, hidden_thresh),
    }

    replay_trace = [
        {"window_idx": t["window_idx"], "t_start": t["t_start"], "true_bin": tb, "systems": {}}
        for t, tb in zip(trace, true_bins)
    ]
    for name, sys_obj in variants.items():
        for i, (s, tb, tm) in enumerate(zip(states, true_bins, true_mits)):
            out = sys_obj.step(s, true_bin=tb, true_mit=tm)
            replay_trace[i]["systems"][name] = {"p_att": out["p_att"]}

    scores = compute_round_scores(replay_trace, events, list(variants.keys()))

    # Sanity check: v10_full should reproduce the live orchestrator's own
    # recorded ary_v10 trace almost exactly (same code path/hyperparams).
    live_p_att = [float(t["systems"].get("ary_v10", {}).get("p_att", 0.0)) for t in trace]
    replay_p_att = [replay_trace[i]["systems"]["v10_full (TTT+RAM)"]["p_att"] for i in range(len(trace))]
    max_diff = float(np.max(np.abs(np.array(live_p_att) - np.array(replay_p_att)))) if live_p_att else None

    return {
        "objective": data.get("objective"),
        "evasion": data.get("evasion"),
        "scores": scores,
        "replay_vs_live_max_abs_diff": max_diff,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(ROOT / "results" / "ram_improve" / "live_lab"))
    args = p.parse_args()

    round_files = sorted(Path(args.results_dir).rglob("round.json"))
    if not round_files:
        print(f"No round.json files found under {args.results_dir}", file=sys.stderr)
        return 1

    print("Loading ARY.01 checkpoint...")
    base_model = load_model()
    print("Calibrating thresholds on aryan val split...")
    splits = load_all_splits()
    va_s, va_b, _ = splits["val"]
    raw_thresh, hidden_thresh = calibrate_thresholds(base_model, va_s, va_b)
    print(f"  raw={raw_thresh:.2f} hidden={hidden_thresh:.2f}\n")

    header = f"{'objective/evasion':<28} {'variant':<22} {'F1':>6} {'TTD':>5} {'lost':>6} {'harm':>5}"
    print(header)
    print("-" * len(header))

    agg: dict[str, list[float]] = {}
    for rf in round_files:
        result = replay_round(rf, base_model, raw_thresh, hidden_thresh)
        label = f"{result['objective']}/{result['evasion']}"
        for variant, sc in result["scores"].items():
            agg.setdefault(variant, []).append(sc["binary_f1"])
            lost = "LOST" if sc["lost_to_attacker"] is True else ("held" if sc["lost_to_attacker"] is False else "-")
            print(
                f"{label:<28} {variant:<22} {sc['binary_f1']:>6.3f} "
                f"{str(sc['ttd_windows']):>5} {lost:>6} {sc['benign_harm_count']:>5}"
            )
        diff = result["replay_vs_live_max_abs_diff"]
        if diff is not None and diff > 1e-4:
            print(f"  ! replay-vs-live p_att max abs diff = {diff:.6f} (expected ~0 for v10_full)")
        print()

    print("=" * len(header))
    print("Mean F1 across all replayed rounds:")
    for variant, f1s in agg.items():
        print(f"  {variant:<22} mean_f1={np.mean(f1s):.3f}  (n={len(f1s)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
