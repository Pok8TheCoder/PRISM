"""Session state builder for Lab Session tab."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MOCKS = ROOT / "apps" / "lab-console" / "mocks"
EVENTS = ROOT / "data" / "lab_events"
STATE_PATH = EVENTS / "demo_forecast.json"
PID_PATH = EVENTS / "demo_forecast.pid"

_POLICY_MODE = "ids"


def get_policy_mode() -> str:
    return _POLICY_MODE


def set_policy_mode(mode: str) -> str:
    global _POLICY_MODE
    _POLICY_MODE = "ips" if mode == "ips" else "ids"
    return _POLICY_MODE


def scorer_running() -> bool:
    if not PID_PATH.exists():
        return STATE_PATH.exists()
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return False
    if pid <= 0:
        return False
    import os
    import signal

    try:
        os.kill(pid, 0)
        return True
    except (OSError, AttributeError):
        # Windows: os.kill(pid, 0) works on py3
        try:
            import subprocess

            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in (r.stdout or "")
        except Exception:
            return STATE_PATH.exists()


def _read_forecast_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _series(n: int, base: float, amp: float, phase: float = 0) -> list[dict[str, float]]:
    return [
        {
            "t": float(i),
            "y": base + math.sin((i + phase) / 6) * amp + (max(0, i - n * 0.55) * 0.008),
        }
        for i in range(n)
    ]


def _model_regions(offset: int) -> list[dict[str, Any]]:
    return [
        {"start": 19 + offset, "end": 27 + offset, "kind": "suspicious", "label": "T1046_service_scan", "classId": "T1046"},
        {
            "start": 32 + offset,
            "end": 40 + offset,
            "kind": "attack",
            "label": "T1190_web_exploit",
            "classId": "T1190",
            "resolved": True,
            "correct": True,
        },
        {
            "start": 46 + offset,
            "end": 52 + offset,
            "kind": "suspicious",
            "label": "T1110_ssh_bruteforce",
            "classId": "T1110",
            "resolved": True,
            "correct": False,
        },
    ]


def _recorded_fallback() -> dict[str, Any]:
    path = MOCKS / "session_full.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("models"):
                return data
        except Exception:
            pass

    duration = 70
    models = [
        {
            "id": "shaun_v3",
            "name": "Shaun v3",
            "predicted": _series(duration, 0.12, 0.04, 0),
            "regions": _model_regions(0),
            "accuracy": {"lineMae": 0.042, "regionPrecision": 0.78, "regionRecall": 0.71},
        },
        {
            "id": "hx_c",
            "name": "HX-C",
            "predicted": _series(duration, 0.10, 0.05, 2),
            "regions": _model_regions(1),
            "accuracy": {"lineMae": 0.038, "regionPrecision": 0.82, "regionRecall": 0.75},
        },
        {
            "id": "gen10_world_model",
            "name": "PRISM Gen 10 World Model",
            "predicted": _series(duration, 0.08, 0.03, 1),
            "regions": _model_regions(0),
            "accuracy": {"lineMae": 0.018, "regionPrecision": 0.992, "regionRecall": 0.988},
        },
    ]
    return {
        "id": "demo-001",
        "playheadSec": 38,
        "durationSec": duration,
        "mode": "recorded",
        "policyMode": get_policy_mode(),
        "ips": {"armed": True, "streaks": {"shaun_v3": 1, "hx_c": 0}},
        "actual": _series(duration, 0.11, 0.035, 1),
        "models": models,
        "groundTruthRegions": [
            {"start": 18, "end": 28, "kind": "ground_truth", "label": "recon", "classId": "T1046"},
            {"start": 30, "end": 42, "kind": "ground_truth", "label": "enum", "classId": "T1190"},
            {"start": 44, "end": 55, "kind": "ground_truth", "label": "spray", "classId": "T1110"},
        ],
    }


def _timeline_to_series(timeline: dict[str, Any], window_sec: float, elapsed: float) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for pt in timeline.get("retrospective") or []:
        t = elapsed + float(pt.get("t_rel", 0))
        points.append({"t": round(t, 2), "y": float(pt.get("p", 0))})
    for pt in timeline.get("prospective") or []:
        t = elapsed + float(pt.get("t_rel", 0))
        points.append({"t": round(t, 2), "y": float(pt.get("p", 0))})
    if points:
        points.sort(key=lambda p: p["t"])
    return points


def _regions_from_points(
    points: list[dict[str, float]],
    *,
    suspicious_thresh: float = 0.35,
    attack_thresh: float = 0.5,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if not points:
        return regions
    current: dict[str, Any] | None = None
    for p in points:
        y = float(p["y"])
        if y >= attack_thresh:
            kind = "attack"
        elif y >= suspicious_thresh:
            kind = "suspicious"
        else:
            if current:
                regions.append(current)
                current = None
            continue
        t = float(p["t"])
        if current and current["kind"] == kind:
            current["end"] = t
        else:
            if current:
                regions.append(current)
            current = {"start": t, "end": t, "kind": kind, "label": kind}
    if current:
        regions.append(current)
    return regions


def _gt_regions(phase_regions: list[dict[str, Any]], window_sec: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in phase_regions:
        start = float(r.get("start_w", 0)) * window_sec
        end = (float(r.get("end_w", 0)) + 1) * window_sec
        out.append(
            {
                "start": start,
                "end": end,
                "kind": "ground_truth",
                "label": r.get("label") or r.get("phase", "phase"),
                "classId": r.get("phase"),
            }
        )
    return out


def _live_from_forecast(state: dict[str, Any]) -> dict[str, Any]:
    window_sec = float(state.get("window_sec", 1.0))
    elapsed = float(state.get("elapsed_sec", state.get("window", 0) * window_sec))
    duration = max(elapsed + window_sec * float(state.get("horizon", 6)), elapsed + 10)
    models_raw = state.get("models") or {}

    model_defs = [
        ("laplace_world_model", "PRISM Laplace Model"),
        ("shaun_v3", "Shaun v3"),
        ("hx_c", "HX-C"),
    ]
    models: list[dict[str, Any]] = []
    actual: list[dict[str, float]] = []

    for mid, name in model_defs:
        timeline = models_raw.get(mid) or models_raw.get("gen10_world_model") or {}
        predicted = _timeline_to_series(timeline, window_sec, elapsed)
        if mid == "shaun_v3" and not actual:
            actual = [{"t": p["t"], "y": p["y"]} for p in predicted if p["t"] <= elapsed + 0.01]
        models.append(
            {
                "id": mid,
                "name": name,
                "predicted": predicted,
                "regions": _regions_from_points(predicted),
                "accuracy": {"lineMae": 0.04, "regionPrecision": 0.75, "regionRecall": 0.70},
            }
        )

    streaks = {
        "shaun_v3": int(state.get("pending_sn2rx3", 0)),
        "hx_c": int(state.get("pending_hx_c", 0)),
    }
    attack_phase = state.get("attack_phase", "idle")
    ips_armed = bool(state.get("ips_armed")) or attack_phase in {"enum", "spray", "loot"}

    return {
        "id": "live",
        "playheadSec": elapsed,
        "durationSec": duration,
        "mode": "live",
        "policyMode": get_policy_mode(),
        "ips": {
            "armed": ips_armed,
            "blocker": state.get("first_blocker"),
            "blockAtSec": state.get("block_at_sec"),
            "streaks": streaks,
        },
        "actual": actual or _series(int(duration), 0.11, 0.03),
        "models": models,
        "groundTruthRegions": _gt_regions(state.get("phase_regions") or [], window_sec),
        "phase": state.get("phase"),
        "attackPhase": attack_phase,
        "scorerRunning": True,
    }


def build_session(session_id: str, mode: str = "recorded") -> dict[str, Any]:
    forecast = _read_forecast_state()
    live_ok = scorer_running() and forecast is not None

    if mode == "live" and live_ok:
        session = _live_from_forecast(forecast)
    elif live_ok and mode != "recorded":
        session = _live_from_forecast(forecast)
    else:
        session = _recorded_fallback()
        session["mode"] = "recorded"

    session["id"] = session_id
    session["policyMode"] = get_policy_mode()
    session["scorerRunning"] = live_ok
    return session
