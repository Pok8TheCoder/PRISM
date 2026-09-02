#!/usr/bin/env python3
"""Test `StreamingARYTieredV10` (suspicion-gated, two-tier memory) against
the old flat `ary_v10` memory, by replaying already-captured live-lab
`round.json` traffic -- no Docker, no re-capture needed.

Compares:
  v10_flat_mem    -- the original ary_v10 (writes every window forever)
  v10_tiered_mem  -- new: 20-slot frozen benign baseline + a dynamic pool
                     (20 slots pre-attack, grows to 40 once a confirmed
                     attack is seen) that only ever stores suspicious/
                     confirmed-attack windows, FIFO-evicted once full

Usage:
  python scripts/test_tiered_ram.py [--results-dir DIR ...] [--suspicious-thresh 0.01]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.lab_scoring import compute_round_scores  # noqa: E402
from src.aryan.streaming_variants import (  # noqa: E402
    StreamingARY,
    StreamingARYTieredV10,
    calibrate_thresholds,
    load_model,
)


def replay_round(round_path: Path, base_model, hidden_thresh: float, suspicious_thresh: float) -> dict:
    data = json.loads(round_path.read_text())
    trace = data["trace"]
    events = data["events"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    true_bins = [int(t["true_bin"]) for t in trace]
    true_mits = [int(t["true_mit"]) for t in trace]

    variants = {
        "v10_flat_mem": StreamingARY("ary_v10", base_model=base_model, hidden_thresh=hidden_thresh),
        "v10_tiered_mem": StreamingARYTieredV10(
            base_model=base_model, hidden_thresh=hidden_thresh, suspicious_thresh=suspicious_thresh,
        ),
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
    tiered_bank = variants["v10_tiered_mem"].bank
    return {
        "objective": data.get("objective"),
        "evasion": data.get("evasion"),
        "scores": scores,
        "final_bank_size": len(tiered_bank),
        "final_dynamic_size": len(tiered_bank.dynamic_keys),
        "final_dynamic_capacity": tiered_bank._dynamic_capacity,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir", nargs="+",
        default=[
            str(ROOT / "results" / "ram_improve" / "live_lab"),
            str(ROOT / "results" / "ram_improve" / "live_lab_offset_test"),
        ],
    )
    p.add_argument("--suspicious-thresh", type=float, default=0.01)
    args = p.parse_args()

    round_files: list[Path] = []
    for d in args.results_dir:
        round_files.extend(sorted(Path(d).rglob("round.json")))
    if not round_files:
        print("No round.json files found.", file=sys.stderr)
        return 1

    print("Loading ARY.01 checkpoint...")
    base_model = load_model()
    print("Calibrating hidden-key threshold on aryan val split...")
    splits = load_all_splits()
    va_s, va_b, _ = splits["val"]
    _, hidden_thresh = calibrate_thresholds(base_model, va_s, va_b)
    print(f"  hidden={hidden_thresh:.2f}  suspicious_thresh={args.suspicious_thresh}\n")

    header = f"{'objective/evasion':<32} {'variant':<16} {'F1':>6} {'TTD':>5} {'lost':>6} {'harm':>5} {'bank(dyn/cap)':>14}"
    print(header)
    print("-" * len(header))

    agg: dict[str, list[float]] = {}
    for rf in round_files:
        tag = rf.parent.parent.name + "/" + rf.parent.name
        if "offset_test" in str(rf):
            tag += " [offset warmup=30]"
        result = replay_round(rf, base_model, hidden_thresh, args.suspicious_thresh)
        for variant, sc in result["scores"].items():
            agg.setdefault(variant, []).append(sc["binary_f1"])
            lost = "LOST" if sc["lost_to_attacker"] is True else ("held" if sc["lost_to_attacker"] is False else "-")
            bank_info = ""
            if variant == "v10_tiered_mem":
                bank_info = f"{result['final_dynamic_size']}/{result['final_dynamic_capacity']}"
            print(
                f"{tag:<32} {variant:<16} {sc['binary_f1']:>6.3f} "
                f"{str(sc['ttd_windows']):>5} {lost:>6} {sc['benign_harm_count']:>5} {bank_info:>14}"
            )
        print()

    print("=" * len(header))
    print("Mean F1 across all replayed rounds:")
    for variant, f1s in agg.items():
        print(f"  {variant:<16} mean_f1={np.mean(f1s):.3f}  (n={len(f1s)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
