"""IPS outcome scoring for live red-team experiments.

Extends IDS proxy metrics with real prevention outcomes: whether IPS blocked
the red-team IP before credential theft succeeded.
"""

from __future__ import annotations

from typing import Any

from src.aryan.lab_scoring import DETECT_THRESHOLD, compute_round_scores

STAGE_POST_EXPLOIT = "post_exploit"


def _first_success_ts(events: list[dict], stage: str = STAGE_POST_EXPLOIT) -> float | None:
    ok = [float(e["ts"]) for e in events if e.get("stage") == stage and e.get("success")]
    return min(ok) if ok else None


def _experiment_start(trace: list[dict]) -> float | None:
    if not trace:
        return None
    return float(trace[0]["t_start"])


def compute_ips_scores(
    trace: list[dict],
    events: list[dict],
    *,
    system_id: str = "ary5_ramx",
    ips_blocked: bool,
    blocked_at_sec: float | None,
    blocked_at_window: int | None,
    detect_threshold: float = DETECT_THRESHOLD,
) -> dict[str, Any]:
    """Merge IDS proxy scores with IPS prevention outcome fields."""
    base = compute_round_scores(trace, events, [system_id], detect_threshold=detect_threshold)
    det = base.get(system_id, {})

    t0 = _experiment_start(trace)
    stolen_at = _first_success_ts(events)
    stolen_at_sec = (stolen_at - t0) if (stolen_at is not None and t0 is not None) else None

    creds_stolen = stolen_at is not None
    if creds_stolen and ips_blocked and stolen_at_sec is not None and blocked_at_sec is not None:
        theft_prevented = blocked_at_sec < stolen_at_sec
        outcome = "prevented" if theft_prevented else "too_late"
    elif creds_stolen:
        theft_prevented = False
        outcome = "undetected_theft" if not ips_blocked else "too_late"
    elif ips_blocked:
        theft_prevented = True
        outcome = "prevented"
    else:
        theft_prevented = False
        outcome = "no_theft_attempted"

    return {
        **det,
        "system_id": system_id,
        "creds_stolen": creds_stolen,
        "theft_succeeded": creds_stolen,
        "stolen_at_sec": stolen_at_sec,
        "ips_blocked": ips_blocked,
        "blocked_at_sec": blocked_at_sec,
        "blocked_at_window": blocked_at_window,
        "theft_prevented": theft_prevented,
        "prevented_theft": theft_prevented,
        "time_to_block_sec": blocked_at_sec,
        "time_to_detect_sec": det.get("ttd_sec"),
        "outcome": outcome,
        "benign_harm": det.get("benign_harm_count", 0),
    }
