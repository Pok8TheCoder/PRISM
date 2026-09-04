#!/usr/bin/env python3
"""Rescore stored live round.json traces with sn2rx3 alert-suppression logic.

Uses stored sn2rx raw_p_att + context_gated flags; applies v3 cap (0.49) during
context gate. Full state replay requires PCAPs (not stored in round dirs).

Usage:
  python scripts/rescore_live_sn2rx3.py
  python scripts/rescore_live_sn2rx3.py --round results/zeroday_live_sweep/live_rescore/cred_theft/round.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.lab_scoring import compute_round_scores  # noqa: E402

ALERT_CAP = 0.49
DEFAULT_ROUNDS = [
    ROOT / "results/zeroday_live_sweep/live_rescore/cred_theft/round.json",
    ROOT / "results/zeroday_live_sweep/live_rescore/key_theft/round.json",
    ROOT / "results/zeroday_live_sweep/live_rescore/defacement/round.json",
]
OUT = ROOT / "results" / "ramx_v3_lab_bench" / "live_rescore_sn2rx3.json"


def apply_v3_cap(trace: list[dict]) -> list[float]:
    """Simulate sn2rx3 p_att from stored sn2rx fields."""
    out: list[float] = []
    for w in trace:
        sn = w.get("systems", {}).get("sn2rx", {})
        raw = float(sn.get("raw_p_att", sn.get("p_att", 0.0)))
        gated = bool(sn.get("context_gated", False))
        if gated and raw >= 0.5:
            out.append(min(raw, ALERT_CAP))
        else:
            out.append(float(sn.get("p_att", raw)))
    return out


def inject_series(trace: list[dict], p_atts: list[float]) -> list[dict]:
    patched = []
    for w, p in zip(trace, p_atts):
        w2 = json.loads(json.dumps(w))
        w2.setdefault("systems", {})["sn2rx3_sim"] = {
            "p_att": p,
            "raw_p_att": w["systems"].get("sn2rx", {}).get("raw_p_att", p),
            "context_gated": w["systems"].get("sn2rx", {}).get("context_gated", False),
            "alert_suppressed": p < w["systems"].get("sn2rx", {}).get("p_att", p),
            "ramx_version": "3.0-sim",
        }
        patched.append(w2)
    return patched


def summarize(round_path: Path) -> dict:
    data = json.loads(round_path.read_text(encoding="utf-8"))
    trace = data["trace"]
    events = data.get("events", [])
    sn2rx_scores = data.get("scores", {}).get("sn2rx", {})

    p_v3 = apply_v3_cap(trace)
    patched = inject_series(trace, p_v3)
    v3_scores = compute_round_scores(patched, events, ["sn2rx3_sim"])["sn2rx3_sim"]

    warmup_idx = [i for i, w in enumerate(trace) if w.get("phase") == "warmup"]
    warmup_fp_v2 = sum(
        1 for i in warmup_idx
        if float(trace[i]["systems"].get("sn2rx", {}).get("p_att", 0)) >= 0.5
    )
    warmup_fp_v3 = sum(1 for i in warmup_idx if p_v3[i] >= 0.5)

    return {
        "objective": data.get("objective"),
        "class_id": data.get("class_id"),
        "n_windows": len(trace),
        "warmup_windows": data.get("warmup_windows"),
        "sn2rx": {
            "binary_f1": sn2rx_scores.get("binary_f1"),
            "benign_harm_count": sn2rx_scores.get("benign_harm_count"),
            "fp": sn2rx_scores.get("fp"),
            "warmup_fp": warmup_fp_v2,
        },
        "sn2rx3_sim": {
            "binary_f1": v3_scores.get("binary_f1"),
            "benign_harm_count": v3_scores.get("benign_harm_count"),
            "fp": v3_scores.get("fp"),
            "warmup_fp": warmup_fp_v3,
        },
        "delta_harm": v3_scores.get("benign_harm_count", 0) - sn2rx_scores.get("benign_harm_count", 0),
        "delta_f1": (v3_scores.get("binary_f1") or 0) - (sn2rx_scores.get("binary_f1") or 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", action="append", type=Path, default=[])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    rounds = args.round or DEFAULT_ROUNDS

    rows = []
    for rp in rounds:
        if not rp.exists():
            rows.append({"path": str(rp), "error": "missing"})
            continue
        rows.append({"path": str(rp.relative_to(ROOT)), **summarize(rp)})

    payload = {"rounds": rows, "note": "sn2rx3_sim uses stored raw_p_att + alert cap during context_gated windows"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
