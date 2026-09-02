#!/usr/bin/env python3
"""Benchmark RAMX_V.01 on captured live-lab traffic (realistic replay).

Compares against flat-memory baselines on the same ``round.json`` traces:
  ary_v10           -- flat hidden-key RAM (current V10)
  ary_ramx_v01      -- ARY + RAMX_V.01 rolling baseline
  timesfm_ary_cls   -- TimesFM+ARY classifier, no RAM
  fmary_ram_ungated -- TimesFM+ARY + flat ungated RAM
  fmary_ramx_v01    -- TimesFM+ARY + RAMX_V.01

Also runs a long-run stress replay: repeats real benign warm-up windows
before the same real attack to simulate 100+ step benign runs where
baseline rotation matters.

Usage:
  python scripts/test_ramx_v01.py [--results-dir DIR ...]
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
    StreamingARYRamxV01,
    StreamingTimesFMHybrid,
    StreamingTimesFMHybridRamx,
    calibrate_thresholds,
    load_model,
)

SYSTEMS = [
    "ary_v10",
    "ary_ramx_v01",
    "timesfm_ary_cls",
    "fmary_ram_ungated",
    "fmary_ramx_v01",
]


def make_systems(base_model, raw_thresh: float, hidden_thresh: float) -> dict:
    return {
        "ary_v10": StreamingARY("ary_v10", base_model=base_model, hidden_thresh=hidden_thresh),
        "ary_ramx_v01": StreamingARYRamxV01(base_model=base_model, hidden_thresh=hidden_thresh),
        "timesfm_ary_cls": StreamingTimesFMHybrid(
            "none", base_model=base_model, forecaster=None, raw_thresh=raw_thresh,
        ),
        "fmary_ram_ungated": StreamingTimesFMHybrid(
            "ungated", base_model=base_model, forecaster=None, raw_thresh=raw_thresh,
        ),
        "fmary_ramx_v01": StreamingTimesFMHybridRamx(
            base_model=base_model, forecaster=None, raw_thresh=raw_thresh,
        ),
    }


def replay_sequence(
    systems: dict,
    states: list[np.ndarray],
    true_bins: list[int],
    true_mits: list[int],
    trace_template: list[dict],
) -> list[dict]:
    replay_trace = [
        {"window_idx": t["window_idx"], "t_start": t["t_start"], "true_bin": tb, "systems": {}}
        for t, tb in zip(trace_template, true_bins)
    ]
    for sid, sys_obj in systems.items():
        for i, (s, tb, tm) in enumerate(zip(states, true_bins, true_mits)):
            out = sys_obj.step(s, true_bin=tb, true_mit=tm)
            replay_trace[i]["systems"][sid] = {"p_att": out["p_att"]}
    return replay_trace


def replay_round(round_path: Path, base_model, raw_thresh: float, hidden_thresh: float) -> dict:
    data = json.loads(round_path.read_text())
    trace = data["trace"]
    events = data["events"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    true_bins = [int(t["true_bin"]) for t in trace]
    true_mits = [int(t["true_mit"]) for t in trace]

    systems = make_systems(base_model, raw_thresh, hidden_thresh)
    replay_trace = replay_sequence(systems, states, true_bins, true_mits, trace)
    scores = compute_round_scores(replay_trace, events, SYSTEMS)

    ramx = systems["ary_ramx_v01"].bank
    return {
        "objective": data.get("objective"),
        "evasion": data.get("evasion"),
        "scores": scores,
        "ramx_baseline": ramx.baseline_occupied,
        "ramx_dynamic": len(ramx.dynamic_keys),
        "ramx_rotations": ramx.rotation_count,
    }


def long_run_stress(base_model, raw_thresh: float, hidden_thresh: float, extra_cycles: int = 4) -> dict:
    """Splice extra real-benign windows before the attack from defacement/none."""
    d = json.loads(
        (ROOT / "results" / "ram_improve" / "live_lab" / "defacement" / "none" / "round.json").read_text()
    )
    trace = d["trace"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    tbins = [int(t["true_bin"]) for t in trace]
    tmits = [int(t["true_mit"]) for t in trace]

    benign_pool = list(zip(states[:20], tbins[:20], tmits[:20]))
    attack_tail = list(zip(states[20:], tbins[20:], tmits[20:]))
    long_run = benign_pool + benign_pool * extra_cycles + attack_tail

    states_lr = [x[0] for x in long_run]
    tbins_lr = [x[1] for x in long_run]
    tmits_lr = [x[2] for x in long_run]
    template = [{"window_idx": i, "t_start": float(i * 30)} for i in range(len(long_run))]

    systems = make_systems(base_model, raw_thresh, hidden_thresh)
    replay_trace = replay_sequence(systems, states_lr, tbins_lr, tmits_lr, template)
    attack_idx = len(benign_pool) * (1 + extra_cycles)

    out = {}
    for sid in SYSTEMS:
        p_atts = [float(w["systems"][sid]["p_att"]) for w in replay_trace]
        out[sid] = {
            "p_att_at_attack": p_atts[attack_idx],
            "detected_at_attack": p_atts[attack_idx] >= 0.5,
            "max_p_att_post_attack": max(p_atts[attack_idx:attack_idx + 5]),
        }
    out["attack_idx"] = attack_idx
    out["total_windows"] = len(long_run)
    out["ramx_rotations"] = systems["ary_ramx_v01"].bank.rotation_count
    out["ramx_baseline"] = systems["ary_ramx_v01"].bank.baseline_occupied
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir", nargs="+",
        default=[str(ROOT / "results" / "ram_improve" / "live_lab")],
    )
    args = p.parse_args()

    round_files: list[Path] = []
    for d in args.results_dir:
        round_files.extend(sorted(Path(d).rglob("round.json")))
    # de-dupe by objective/evasion path
    seen: set[str] = set()
    unique_rounds: list[Path] = []
    for rf in round_files:
        key = str(rf.parent)
        if key not in seen and "smoketest" not in key and "fixtest" not in key and "viewertest" not in key:
            seen.add(key)
            unique_rounds.append(rf)

    print("Loading ARY.01 checkpoint...")
    base_model = load_model()
    splits = load_all_splits()
    va_s, va_b, _ = splits["val"]
    raw_thresh, hidden_thresh = calibrate_thresholds(base_model, va_s, va_b)
    print(f"  raw={raw_thresh:.2f}  hidden={hidden_thresh:.2f}\n")

    header = (
        f"{'round':<28} {'system':<18} {'F1':>6} {'TTD':>5} {'lost':>6} "
        f"{'harm':>5} {'ramx(base/dyn/rot)':>18}"
    )
    print("=== Live-lab replay (realistic captured traffic) ===")
    print(header)
    print("-" * len(header))

    agg: dict[str, list[float]] = {s: [] for s in SYSTEMS}
    for rf in unique_rounds:
        tag = rf.parent.name + "/" + rf.parent.parent.name.replace("live_lab", "").strip("/") or "main"
        if tag.startswith("/"):
            tag = tag[1:]
        result = replay_round(rf, base_model, raw_thresh, hidden_thresh)
        for sid, sc in result["scores"].items():
            agg[sid].append(sc["binary_f1"])
            lost = "LOST" if sc["lost_to_attacker"] is True else ("held" if sc["lost_to_attacker"] is False else "-")
            bank_info = ""
            if sid == "ary_ramx_v01":
                bank_info = f"{result['ramx_baseline']}/{result['ramx_dynamic']}/{result['ramx_rotations']}"
            print(
                f"{tag:<28} {sid:<18} {sc['binary_f1']:>6.3f} "
                f"{str(sc['ttd_windows']):>5} {lost:>6} {sc['benign_harm_count']:>5} {bank_info:>18}"
            )
        print()

    print("=" * len(header))
    print("Mean F1 across rounds:")
    for sid, f1s in agg.items():
        print(f"  {sid:<18} mean_f1={np.mean(f1s):.3f}  (n={len(f1s)})")

    print("\n=== Long-run stress (100-window benign before same real attack) ===")
    stress = long_run_stress(base_model, raw_thresh, hidden_thresh, extra_cycles=4)
    print(f"  total_windows={stress['total_windows']}  attack_at_idx={stress['attack_idx']}")
    print(f"  RAMX rotations during run: {stress['ramx_rotations']}  baseline_occupied={stress['ramx_baseline']}")
    print(f"\n  {'system':<18} {'P(att)@attack':>14} {'detected':>10} {'max_post_attack':>16}")
    print("  " + "-" * 60)
    for sid in SYSTEMS:
        s = stress[sid]
        print(
            f"  {sid:<18} {s['p_att_at_attack']:>14.5f} "
            f"{str(s['detected_at_attack']):>10} {s['max_p_att_post_attack']:>16.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
