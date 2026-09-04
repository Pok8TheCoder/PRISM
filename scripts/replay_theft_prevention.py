#!/usr/bin/env python3
"""Simulate theft-prevention policy on a stored IPS round.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.ips_scoring import compute_ips_scores  # noqa: E402


def first_ts(events: list[dict], stage: str) -> float | None:
    ok = [float(e["ts"]) for e in events if e.get("stage") == stage and e.get("success")]
    return min(ok) if ok else None


def simulate(
    data: dict,
    *,
    block_stages: set[str],
    ml_block: bool,
    block_id: str,
) -> dict:
    trace = data["trace"]
    events = data.get("events", [])
    t0 = float(trace[0]["t_start"]) if trace else 0.0

    stolen_at = first_ts(events, "post_exploit")
    stolen_at_sec = (stolen_at - t0) if stolen_at is not None else None

    blocked_at_sec = None
    blocked_at_window = None
    block_reason = ""

    for w in trace:
        if w.get("phase") != "attack":
            continue
        idx = int(w["t_start"])
        wi = int(w["window_idx"])
        p = float(w.get("systems", {}).get(block_id, {}).get("p_att", 0))

        for stage in block_stages:
            ts = first_ts(events, stage)
            if ts is None:
                continue
            if t0 <= ts <= float(w["t_end"]) and blocked_at_sec is None:
                blocked_at_sec = ts - t0
                blocked_at_window = wi
                block_reason = f"event:{stage}"
                break

        if ml_block and p >= 0.5 and blocked_at_sec is None:
            blocked_at_sec = float(w["t_end"]) - t0
            blocked_at_window = wi
            block_reason = "ml_score"

        if blocked_at_sec is not None:
            break

    ips_blocked = blocked_at_sec is not None
    creds_stolen = stolen_at is not None
    if creds_stolen and ips_blocked and stolen_at_sec is not None and blocked_at_sec is not None:
        theft_prevented = blocked_at_sec < stolen_at_sec
        outcome = "prevented" if theft_prevented else "too_late"
    elif creds_stolen:
        outcome = "undetected_theft" if not ips_blocked else "too_late"
        theft_prevented = False
    elif ips_blocked:
        outcome = "prevented"
        theft_prevented = True
    else:
        outcome = "no_theft_attempted"
        theft_prevented = False

    scores = compute_ips_scores(
        trace, events, system_id=block_id,
        ips_blocked=ips_blocked,
        blocked_at_sec=blocked_at_sec,
        blocked_at_window=blocked_at_window,
    )
    return {
        "source": data.get("path", ""),
        "policy": {"ml_block": ml_block, "event_block_stages": sorted(block_stages)},
        "stolen_at_sec": stolen_at_sec,
        "blocked_at_sec": blocked_at_sec,
        "block_reason": block_reason,
        "theft_prevented": theft_prevented,
        "outcome": outcome,
        "original_outcome": data.get("outcome") or data.get("scores", {}).get("outcome"),
        "scores": scores,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=Path, required=True)
    ap.add_argument("--block-id", default="sn2rx")
    ap.add_argument("--event-stages", default="recon")
    ap.add_argument("--no-ml", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.round.read_text(encoding="utf-8"))
    data["path"] = str(args.round)
    block_id = data.get("block_system_id", args.block_id)
    stages = {s.strip() for s in args.event_stages.split(",") if s.strip()}

    result = simulate(data, block_stages=stages, ml_block=not args.no_ml, block_id=block_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
