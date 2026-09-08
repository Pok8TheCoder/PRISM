"""Session state builder for Lab Session tab."""

from __future__ import annotations

import json
import math
import time
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
        return False
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return False
    if pid <= 0:
        return False
    import os

    if os.name == "nt":
        try:
            import subprocess

            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = (r.stdout or "").strip().lower()
            return bool(line) and str(pid) in line and "python" in line
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_forecast_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    for _ in range(4):
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            time.sleep(0.05)
        except Exception:
            return None
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
            "id": "ary_5s",
            "name": "ARY 5s + RAMX",
            "predicted": _series(duration, 0.15, 0.06, 4),
            "regions": _model_regions(-1),
            "accuracy": {"lineMae": 0.051, "regionPrecision": 0.65, "regionRecall": 0.68},
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


def _timeline_observed(timeline: dict[str, Any], window_sec: float) -> list[dict[str, float]]:
    """Retrospective (scored) points on the session timeline."""
    points: list[dict[str, float]] = []
    for pt in timeline.get("retrospective") or []:
        w = pt.get("w")
        if w is not None:
            t = float(w) * window_sec
        else:
            t = float(pt.get("t_rel", 0))
        points.append({"t": round(t, 3), "y": float(pt.get("p", 0))})
    if points:
        points.sort(key=lambda p: p["t"])
    return points


def _timeline_forecast(timeline: dict[str, Any], window_sec: float, data_sec: float) -> list[dict[str, float]]:
    """Prospective (forecast) points ahead of the scored playhead."""
    points: list[dict[str, float]] = []
    for pt in timeline.get("prospective") or []:
        w = pt.get("w")
        if w is not None:
            t = float(w) * window_sec
        else:
            t = data_sec + float(pt.get("t_rel", 0))
        if t > data_sec + 0.001:
            points.append({"t": round(t, 3), "y": float(pt.get("p", 0))})
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


def _memory_regions(regions: list[dict[str, Any]], window_sec: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in regions:
        start = float(r.get("start_w", 0)) * window_sec
        end = (float(r.get("end_w", 0)) + 1) * window_sec
        out.append({
            "start": start,
            "end": end,
            "kind": "episodic",
            "label": r.get("label") or "RAMX episodic",
            "classId": r.get("system"),
        })
    return out


def _merge_gt_regions(
    scorer_regions: list[dict[str, Any]],
    ui_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(scorer_regions)
    for ui in ui_regions:
        duplicate = any(
            abs(float(ui.get("start", 0)) - float(g.get("start", 0))) < 1.0
            and str(ui.get("label", "")).upper() == str(g.get("label", "")).upper()
            for g in merged
        )
        if not duplicate:
            merged.append(ui)
    return merged


def _gt_p_attack_at(t: float, regions: list[dict[str, Any]]) -> float:
    """Synthetic P(attack) from scheduled / known ground-truth regions (lab generator)."""
    for r in regions:
        start = float(r.get("start", 0))
        end = float(r.get("end", 0))
        if start <= t <= end:
            label = str(r.get("label") or r.get("classId") or "").upper()
            if "RECON" in label or r.get("kind") == "recon":
                return 0.28
            return 0.92
    return 0.06


def _build_actual_series(
    observed: list[dict[str, float]],
    ground_truth: list[dict[str, Any]],
    data_sec: float,
    window_sec: float,
    horizon: int,
) -> list[dict[str, float]]:
    """Observed retrospective + known/scheduled future (lab controls traffic)."""
    past = [p for p in observed if p["t"] <= data_sec + 0.01]
    baseline = past[-1]["y"] if past else 0.06
    # Cover the chart's visible future half-window (~13s) plus scorer horizon.
    future_end = data_sec + max(float(horizon) * window_sec, 15.0, window_sec * 14)
    step = max(window_sec * 0.5, 0.25)
    future: list[dict[str, float]] = []
    t = data_sec + step
    while t <= future_end + 0.01:
        future.append({"t": round(t, 3), "y": baseline})
        t += step
    if past and future and past[-1]["t"] < data_sec + 0.01:
        past = past + [{"t": round(data_sec, 3), "y": past[-1]["y"]}]
    return past + future


def _live_from_forecast(state: dict[str, Any]) -> dict[str, Any]:
    window_sec = float(state.get("window_sec", 1.0))
    window_idx = int(state.get("window", 0))
    data_sec = window_idx * window_sec
    elapsed = float(state.get("elapsed_sec", data_sec))
    horizon = int(state.get("horizon", 6))
    duration = max(data_sec + window_sec * horizon, data_sec + window_sec, 10.0)
    models_raw = state.get("models") or {}

    model_defs = [
        ("shaun_v3", "Shaun v3"),
        ("hx_c", "HX-C"),
    ]
    models: list[dict[str, Any]] = []
    actual: list[dict[str, float]] = []

    scorer_gt = _gt_regions(state.get("phase_regions") or [], window_sec)
    try:
        from services.attack_scheduler import ui_attack_regions

        ground_truth = _merge_gt_regions(scorer_gt, ui_attack_regions())
    except Exception:
        ground_truth = scorer_gt

    for mid, name in model_defs:
        timeline = models_raw.get(mid) or {}
        observed = _timeline_observed(timeline, window_sec)
        forecast = _timeline_forecast(timeline, window_sec, data_sec)
        if mid == "shaun_v3" and not actual:
            actual = _build_actual_series(observed, ground_truth, data_sec, window_sec, horizon)
        models.append(
            {
                "id": mid,
                "name": name,
                "observed": observed,
                "predicted": forecast,
                "regions": _regions_from_points(observed + forecast),
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
        "playheadSec": data_sec,
        "elapsedSec": elapsed,
        "windowSec": window_sec,
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
        "groundTruthRegions": ground_truth,
        "memoryRegions": _memory_regions(state.get("memory_regions") or [], window_sec),
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
