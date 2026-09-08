"""Delayed kill-chain scheduler for Lab Console."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from services.lab_config_service import resolve_attack_delay

ROOT = Path(__file__).resolve().parents[4]
FORECAST_STATE = ROOT / "data" / "lab_events" / "demo_forecast.json"

_LOCK = threading.Lock()
_JOBS: list[dict[str, Any]] = []
_LOG_FN: Any = None

PHASE_META: dict[str, dict[str, Any]] = {
    "recon": {"label": "RECON", "classId": "T1046", "dur": 18.0},
    "enum": {"label": "ENUM", "classId": "T1190", "dur": 14.0},
    "spray": {"label": "SPRAY", "classId": "T1110", "dur": 16.0},
    "loot": {"label": "LOOT", "classId": "T1041", "dur": 14.0},
    "all": {"label": "KILLCHAIN", "classId": "T1190", "dur": 55.0},
}


def set_log_fn(fn: Any) -> None:
    global _LOG_FN
    _LOG_FN = fn


def _log(text: str, kind: str = "phase") -> None:
    if _LOG_FN:
        _LOG_FN(text, kind=kind)


def _forecast_data_sec() -> float:
    if not FORECAST_STATE.exists():
        return 0.0
    try:
        data = json.loads(FORECAST_STATE.read_text(encoding="utf-8"))
        ws = float(data.get("window_sec", 1.0))
        return int(data.get("window", 0)) * ws
    except Exception:
        return 0.0


def _forecast_elapsed() -> float:
    """Scored timeline position (matches chart playhead), not wall-clock elapsed."""
    return _forecast_data_sec()


def _run_phase(phase: str, job: dict[str, Any] | None = None) -> None:
    if job is not None:
        now = _forecast_elapsed()
        job["session_start_sec"] = now
        job["running"] = True
    cmd = [sys.executable, str(ROOT / "scripts" / "demo_killchain.py"), "--phase", phase]
    _log(f"[attack] running {phase}: {' '.join(cmd)}", kind="phase")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout:
        for line in proc.stdout:
            _log(line.rstrip(), kind="attack")
    rc = proc.wait()
    if job is not None:
        job["session_end_sec"] = _forecast_elapsed()
        job["running"] = False
    _log(f"[attack] {phase} exited ({rc})", kind="phase")


def schedule_attack(phase: str, delay_sec: float | None = None) -> dict[str, Any]:
    delay = resolve_attack_delay(delay_sec)
    run_at = time.time() + delay
    elapsed_now = _forecast_elapsed()
    meta = PHASE_META.get(phase, {"label": phase.upper(), "classId": phase, "dur": 12.0})
    dur = float(meta.get("dur", 12.0))
    start_sec = elapsed_now + delay
    job = {
        "id": str(uuid.uuid4())[:8],
        "phase": phase,
        "delay_sec": round(delay, 1),
        "run_at": run_at,
        "started": False,
        "done": False,
        "running": False,
        "session_start_sec": start_sec,
        "session_end_sec": start_sec + dur,
        "label": str(meta.get("label", phase.upper())),
        "classId": str(meta.get("classId", phase)),
    }
    with _LOCK:
        _JOBS.append(job)
    _log(
        f"[attack] scheduled {phase} in {delay:.0f}s (session t≈{start_sec:.0f}s)",
        kind="alert",
    )
    return job


def ui_attack_regions() -> list[dict[str, Any]]:
    """UI-only ground-truth bands for manually scheduled kill-chain phases."""
    now = _forecast_elapsed()
    with _LOCK:
        jobs = list(_JOBS[-40:])
    regions: list[dict[str, Any]] = []
    for job in jobs:
        start = float(job.get("session_start_sec", 0))
        end = float(job.get("session_end_sec", start + 10))
        if job.get("running"):
            end = max(end, now)
        elif not job.get("started"):
            end = max(end, start + float(PHASE_META.get(str(job.get("phase")), {}).get("dur", 10)))
        regions.append(
            {
                "start": round(start, 2),
                "end": round(end, 2),
                "kind": "ground_truth",
                "label": str(job.get("label") or job.get("phase", "attack")).upper(),
                "classId": job.get("classId"),
            }
        )
    return regions


def list_jobs() -> list[dict[str, Any]]:
    now = time.time()
    with _LOCK:
        out = []
        for j in _JOBS[-20:]:
            out.append({
                **j,
                "remaining_sec": max(0.0, float(j["run_at"]) - now),
            })
        return out


def _worker() -> None:
    while True:
        time.sleep(0.5)
        due: list[dict[str, Any]] = []
        with _LOCK:
            for job in _JOBS:
                if job.get("done") or job.get("started"):
                    continue
                if float(job["run_at"]) <= time.time():
                    job["started"] = True
                    due.append(job)
        for job in due:
            try:
                _run_phase(str(job["phase"]), job)
            finally:
                with _LOCK:
                    job["done"] = True
                    job["running"] = False


def start_scheduler() -> None:
    threading.Thread(target=_worker, daemon=True, name="attack-scheduler").start()
