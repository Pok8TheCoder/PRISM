#!/usr/bin/env python3
"""Benchmark SHNV.01 (base vs RAMX) on:
  1. Built-in realistic attack scenarios (A/B/C)
  2. Processed chronological test split
  3. Live adversarial lab rounds (242-d -> 110-d adapter)

Usage:
  python scripts/test_shnv01.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FRIEND_ROOT = SHNV_PKG_ROOT
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FRIEND_ROOT))

from src.shnv01.streaming import (  # noqa: E402
    DETECT_THRESHOLD,
    SHNV_PKG_ROOT,
    StreamingSHNV01,
    calibrate_shnv01_hidden_thresh,
    lab242_windows_to_friend110_minutes,
    load_shnv01_bundle,
)

LAB_DIR = ROOT / "results" / "ram_improve" / "live_lab"
DETECT_TH = DETECT_THRESHOLD
SHNV_SYSTEMS = ("shnv_01_base", "shnv_01_ramx")


def _friend_import(module: str):
    import importlib
    saved = sys.path[:]
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            if key.startswith("src.shnv01"):
                continue
            del sys.modules[key]
    filtered = [p for p in saved if p and Path(p).resolve() != ROOT.resolve()]
    sys.path[:] = [str(FRIEND_ROOT)] + filtered
    try:
        return importlib.import_module(module)
    finally:
        sys.path[:] = saved


def score_trace(p_atts: list[float], true_bins: list[int], detect_th: float = DETECT_TH) -> dict:
    pred = [1 if p >= detect_th else 0 for p in p_atts]
    tp = sum(1 for p, t in zip(pred, true_bins) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, true_bins) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(pred, true_bins) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(pred, true_bins) if p == 0 and t == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    harm = sum(1 for p, t in zip(pred, true_bins) if p == 1 and t == 0)
    ttd = None
    atk_idxs = [i for i, t in enumerate(true_bins) if t == 1]
    if atk_idxs:
        start = atk_idxs[0]
        for i in range(start, len(true_bins)):
            if pred[i] == 1 and true_bins[i] == 1:
                ttd = i - start
                break
    return {"f1": f1, "precision": prec, "recall": rec, "harm": harm, "ttd": ttd}


def run_sequence(
    sys_obj: StreamingSHNV01,
    states: list[np.ndarray],
    true_bins: list[int],
    true_mits: list[int],
) -> list[float]:
    p_atts = []
    for s, tb, tm in zip(states, true_bins, true_mits):
        out = sys_obj.step(s, true_bin=tb, true_mit=tm)
        p_atts.append(out["p_att"])
    return p_atts


def eval_scenarios(bundle: dict, hidden_thresh: float) -> None:
    states = np.load(FRIEND_ROOT / "data" / "processed" / "states.npy")
    atks = np.load(FRIEND_ROOT / "data" / "processed" / "attack_labels.npy")
    mitres = np.load(FRIEND_ROOT / "data" / "processed" / "mitre_labels.npy")
    gen = _friend_import("src.prediction.interactive_stream").ScenarioGenerator()

    print("\n=== Friend realistic attack scenarios (110-d native) ===")
    header = f"{'scenario':<42} {'system':<14} {'F1':>6} {'TTD':>5} {'harm':>5} {'P@1st_atk':>10}"
    print(header)
    print("-" * len(header))

    for scen in ["Scenario A", "Scenario B", "Scenario C"]:
        s, a, m, _meta = gen.generate_scenario_states(
            f"⚡ {scen}: placeholder", states, atks, mitres,
        )
        s_arr = np.asarray(s, dtype=np.float32)
        tb, tm = a.tolist(), m.tolist()
        for name, use_ram in [("shnv_01_base", False), ("shnv_01_ramx", True)]:
            sys_obj = StreamingSHNV01(bundle, match_thresh=hidden_thresh, use_ram=use_ram)
            p_atts = run_sequence(sys_obj, [s_arr[i] for i in range(len(s_arr))], tb, tm)
            sc = score_trace(p_atts, tb)
            atk_i = next((i for i, t in enumerate(tb) if t == 1), None)
            p_atk = p_atts[atk_i] if atk_i is not None else 0.0
            label = scen.replace("Scenario ", "Scen ")
            print(
                f"{label:<42} {name:<14} {sc['f1']:>6.3f} {str(sc['ttd']):>5} "
                f"{sc['harm']:>5} {p_atk:>10.3f}"
            )


def eval_friend_test_split(bundle: dict, hidden_thresh: float) -> None:
    states = np.load(FRIEND_ROOT / "data" / "processed" / "states.npy")
    atks = np.load(FRIEND_ROOT / "data" / "processed" / "attack_labels.npy")
    mitres = np.load(FRIEND_ROOT / "data" / "processed" / "mitre_labels.npy")
    val_end = int(len(states) * 0.85)
    lookback = bundle["lookback"]

    # Score only the held-out test tail, but warm up the lookback buffer from prior minutes.
    replay_states = states[val_end - lookback :].tolist()
    replay_tb = atks[val_end - lookback :].tolist()
    replay_tm = mitres[val_end - lookback :].tolist()
    score_offset = lookback  # first `lookback` steps are warm-up only

    print("\n=== Friend processed test split (chronological, native 110-d) ===")
    for name, use_ram in [("friend_base", False), ("friend_ramx", True)]:
        sys_obj = StreamingFriendPRISM(bundle, match_thresh=hidden_thresh, use_ram=use_ram)
        p_all = run_sequence(sys_obj, replay_states, replay_tb, replay_tm)
        p_atts = p_all[score_offset:]
        tb = replay_tb[score_offset:]
        sc = score_trace(p_atts, tb)
        print(f"  {name:<14} F1={sc['f1']:.3f}  prec={sc['precision']:.3f}  rec={sc['recall']:.3f}  harm={sc['harm']}")


def eval_live_lab(bundle: dict, hidden_thresh: float) -> None:
    round_files = sorted(LAB_DIR.rglob("round.json"))
    if not round_files:
        print("\n(no live_lab round.json files found — skipping lab eval)")
        return

    print("\n=== Live adversarial lab (242-d -> 110-d adapter, 15s->60s buckets) ===")
    header = f"{'objective/evasion':<28} {'system':<14} {'F1':>6} {'TTD':>5} {'lost':>6} {'harm':>5}"
    print(header)
    print("-" * len(header))

    agg: dict[str, list[float]] = {"shnv_01_base": [], "shnv_01_ramx": []}
    for rf in round_files:
        data = json.loads(rf.read_text())
        trace = data["trace"]
        events = data["events"]
        states242 = [np.array(t["state"], dtype=np.float32) for t in trace]
        tb242 = [int(t["true_bin"]) for t in trace]
        tm242 = [int(t["true_mit"]) for t in trace]
        states, tb, tm = lab242_windows_to_friend110_minutes(states242, tb242, tm242)

        tag = f"{data.get('objective')}/{data.get('evasion')}"
        for name, use_ram in [("shnv_01_base", False), ("shnv_01_ramx", True)]:
            sys_obj = StreamingSHNV01(bundle, match_thresh=hidden_thresh, use_ram=use_ram)
            p_atts = run_sequence(sys_obj, states, tb, tm)
            sc = score_trace(p_atts, tb)
            agg[name].append(sc["f1"])

            success = [e for e in events if e.get("stage") == "post_exploit" and e.get("success")]
            lost = "-"
            if success and p_atts:
                atk_i = next((i for i, t in enumerate(tb) if t == 1), len(p_atts) - 1)
                lost = "LOST" if p_atts[min(atk_i, len(p_atts) - 1)] < DETECT_TH else "held"

            print(
                f"{tag:<28} {name:<14} {sc['f1']:>6.3f} {str(sc['ttd']):>5} {lost:>6} {sc['harm']:>5}"
            )

    print("-" * len(header))
    for name, f1s in agg.items():
        print(f"  {name:<14} mean_f1={np.mean(f1s):.3f}  (n={len(f1s)})")


def main() -> int:
    if not (FRIEND_ROOT / "weights" / "world_model.pt").exists():
        print("Missing _friend_pkg/weights/world_model.pt — extract PRISM_MODEL_PACKAGE.zip first.", file=sys.stderr)
        return 1

    print("Loading SHNV.01 checkpoint...")
    bundle = load_shnv01_bundle()
    print(f"  lookback={bundle['lookback']}  d_state=110")

    states = np.load(FRIEND_ROOT / "data" / "processed" / "states.npy")
    atks = np.load(FRIEND_ROOT / "data" / "processed" / "attack_labels.npy")
    mitres = np.load(FRIEND_ROOT / "data" / "processed" / "mitre_labels.npy")
    chrono = _friend_import("src.data.dataset").chronological_split
    _tr, va, _te = chrono(states, atks, mitres, lookback=bundle["lookback"])
    va_start = int(len(states) * 0.70)
    va_raw = states[va_start : va_start + len(va)]
    va_bins = atks[va_start : va_start + len(va)]
    hidden_thresh = calibrate_shnv01_hidden_thresh(bundle, va_raw, va_bins)
    print(f"  calibrated RAM hidden-key threshold={hidden_thresh:.2f}")

    eval_scenarios(bundle, hidden_thresh)
    eval_friend_test_split(bundle, hidden_thresh)
    eval_live_lab(bundle, hidden_thresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
