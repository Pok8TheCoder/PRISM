#!/usr/bin/env python3
"""Live Docker lab spoofability test for ARY-5sV01 + RAMX.

1. Run live_attack_lab rounds (fresh traffic from Docker).
2. Replay captured states through StreamingARYRamxV01 in two modes:
   - oracle: true_bin passed (current eval — memory gets labels)
   - live: true_bin=None (deployment-realistic — memory via suspicious gate only)

Usage:
  python scripts/test_ramx_spoofability_live.py
  python scripts/test_ramx_spoofability_live.py --skip-capture  # replay existing rounds only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_ary5_vs_ary01 import CKPT_5, load_ckpt, score_seq  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.streaming_variants import StreamingARYRamxV01, calibrate_thresholds  # noqa: E402

THRESH = 0.5
DEFAULT_OUT = ROOT / "results" / "ramx_spoof_live"


def replay_mode(
    factory,
    states: list[np.ndarray],
    bins: list[int],
    mits: list[int],
    *,
    oracle: bool,
) -> list[float]:
    sys_obj = factory()
    p_atts = []
    for s, tb, tm in zip(states, bins, mits):
        if oracle:
            out = sys_obj.step(s, true_bin=tb, true_mit=tm)
        else:
            out = sys_obj.step(s, true_bin=None, true_mit=None)
        p_atts.append(out["p_att"])
    return p_atts


def warmup_stats(p_atts: list[float], bins: list[int], attack_start: int) -> dict:
    warm_p = p_atts[:attack_start]
    warm_b = bins[:attack_start]
    fp = sum(1 for p, b in zip(warm_p, warm_b) if p >= THRESH and b == 0)
    return {
        "n_warmup": len(warm_p),
        "false_alarms": fp,
        "max_p_warmup": max(warm_p) if warm_p else 0.0,
    }


def eval_round(round_path: Path, factory) -> dict:
    data = json.loads(round_path.read_text())
    trace = data["trace"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    bins = [int(t["true_bin"]) for t in trace]
    mits = [int(t["true_mit"]) for t in trace]
    attack_start = next((i for i, b in enumerate(bins) if b == 1), len(bins))

    p_oracle = replay_mode(factory, states, bins, mits, oracle=True)
    p_live = replay_mode(factory, states, bins, mits, oracle=False)

    sc_o = score_seq(p_oracle, bins, attack_start)
    sc_l = score_seq(p_live, bins, attack_start)
    w_o = warmup_stats(p_oracle, bins, attack_start)
    w_l = warmup_stats(p_live, bins, attack_start)

    return {
        "round": str(round_path.relative_to(ROOT)),
        "objective": data.get("objective"),
        "evasion": data.get("evasion"),
        "class_id": data.get("class_id"),
        "n_windows": len(trace),
        "attack_start": attack_start,
        "window_sec": data.get("window_sec"),
        "oracle": {
            "detected": sc_o["detected"],
            "f1": sc_o["f1"],
            "max_p_attack": sc_o["max_p_attack"],
            "warmup_false_alarms": w_o["false_alarms"],
            "max_p_warmup": w_o["max_p_warmup"],
        },
        "live_realistic": {
            "detected": sc_l["detected"],
            "f1": sc_l["f1"],
            "max_p_attack": sc_l["max_p_attack"],
            "warmup_false_alarms": w_l["false_alarms"],
            "max_p_warmup": w_l["max_p_warmup"],
        },
    }


def run_live_capture(out_dir: Path, objective: str, evasion: str, speed: float, warmup: int) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "live_attack_lab.py"),
        "--objective", objective,
        "--evasion", evasion,
        "--speed", str(speed),
        "--warmup-windows", str(warmup),
        "--out-dir", str(out_dir),
        "--device", "cpu",
    ]
    print(f"\n>>> {' '.join(cmd)}\n")
    return subprocess.call(cmd, cwd=str(ROOT))


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "n_rounds": n,
        "oracle_det_rate": sum(1 for r in rows if r["oracle"]["detected"]) / n,
        "live_det_rate": sum(1 for r in rows if r["live_realistic"]["detected"]) / n,
        "oracle_warmup_harm_total": sum(r["oracle"]["warmup_false_alarms"] for r in rows),
        "live_warmup_harm_total": sum(r["live_realistic"]["warmup_false_alarms"] for r in rows),
        "oracle_mean_f1": float(np.mean([r["oracle"]["f1"] for r in rows])),
        "live_mean_f1": float(np.mean([r["live_realistic"]["f1"] for r in rows])),
        "regressions": [
            r["round"] for r in rows
            if r["oracle"]["detected"] and not r["live_realistic"]["detected"]
        ],
        "live_misses": [r["round"] for r in rows if not r["live_realistic"]["detected"]],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--json", type=Path, default=DEFAULT_OUT / "spoof_live_report.json")
    p.add_argument("--skip-capture", action="store_true")
    p.add_argument("--objective", default="all")
    p.add_argument("--evasion", default="all")
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--warmup-windows", type=int, default=20)
    args = p.parse_args()

    if not args.skip_capture:
        print("=== Live Docker capture (live_attack_lab) ===")
        rc = run_live_capture(
            args.out_dir, args.objective, args.evasion, args.speed, args.warmup_windows,
        )
        if rc != 0:
            print(f"live_attack_lab failed with exit {rc}", file=sys.stderr)
            return rc

    round_files = sorted(args.out_dir.rglob("round.json"))
    if not round_files:
        print(f"No round.json under {args.out_dir}", file=sys.stderr)
        return 1

    print(f"\n=== ARY-5sV01 + RAMX replay on {len(round_files)} live round(s) ===")
    model = load_ckpt(CKPT_5)
    va_s, va_b, _ = load_all_splits(ROOT / "data" / "aryan_splits_5s")["val"]
    _, h5 = calibrate_thresholds(model, va_s, va_b)
    factory = lambda: StreamingARYRamxV01(base_model=model, hidden_thresh=h5)

    rows = [eval_round(rf, factory) for rf in round_files]
    summary = summarize(rows)

    report = {
        "checkpoint": str(CKPT_5),
        "out_dir": str(args.out_dir),
        "threshold": THRESH,
        "summary": summary,
        "rounds": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))

    print(f"\n{'round':<45} {'oracle':>8} {'live':>8}  warm_fp(o/l)")
    print("-" * 75)
    for r in rows:
        tag = f"{r['objective']}/{r['evasion']}"
        o = "DET" if r["oracle"]["detected"] else "miss"
        l = "DET" if r["live_realistic"]["detected"] else "miss"
        wf = f"{r['oracle']['warmup_false_alarms']}/{r['live_realistic']['warmup_false_alarms']}"
        print(
            f"{tag:<45} {o:>8} {l:>8}  {wf:>10}  "
            f"max_p o={r['oracle']['max_p_attack']:.3f} l={r['live_realistic']['max_p_attack']:.3f}"
        )

    print("\n=== SUMMARY ===")
    print(f"  Rounds: {summary.get('n_rounds', 0)}")
    print(f"  Oracle det rate:  {summary.get('oracle_det_rate', 0):.1%}  mean F1={summary.get('oracle_mean_f1', 0):.3f}")
    print(f"  Live det rate:    {summary.get('live_det_rate', 0):.1%}  mean F1={summary.get('live_mean_f1', 0):.3f}")
    print(f"  Warmup false alarms (oracle/live total): "
          f"{summary.get('oracle_warmup_harm_total', 0)}/{summary.get('live_warmup_harm_total', 0)}")
    print(f"  JSON -> {args.json}")

    if summary.get("live_warmup_harm_total", 0) == 0:
        print("  Benign spoof (false alarms on warmup): NOT easy on live traffic")
    else:
        print("  Benign spoof: some warmup false alarms on live traffic")

    if summary.get("live_det_rate", 1) < 0.5:
        print("  Attack evasion (live-realistic): EASY — most attacks missed without oracle labels")
    elif summary.get("live_det_rate", 0) >= 0.8:
        print("  Attack evasion (live-realistic): HARD — most attacks still detected")
    else:
        print("  Attack evasion (live-realistic): MIXED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
