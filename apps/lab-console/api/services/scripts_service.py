"""Script launcher + log stream for terminal rail."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

SCRIPTS: list[dict[str, str]] = [
    {"id": "killchain-recon", "label": "Kill-chain: recon", "description": "Port scan phase"},
    {"id": "killchain-enum", "label": "Kill-chain: enum", "description": "Directory bust"},
    {"id": "killchain-spray", "label": "Kill-chain: spray", "description": "Credential spray"},
    {"id": "killchain-loot", "label": "Kill-chain: loot", "description": "Payroll download"},
    {"id": "killchain-all", "label": "Kill-chain: all", "description": "Full chain"},
    {"id": "auto-attack", "label": "Auto-attack", "description": "demo_forecast --auto-attack"},
    {"id": "lab-up", "label": "Lab up", "description": "lab_ctl up"},
    {"id": "lab-down", "label": "Lab down", "description": "lab_ctl down"},
    {"id": "bench-fair-ids", "label": "Fair IDS bench", "description": "Offline PCAP scoring"},
    {"id": "forecast-record", "label": "Record session", "description": "WebM + manifest"},
    {"id": "scorer-start", "label": "Start scorer", "description": "demo_forecast.py"},
    {"id": "scorer-stop", "label": "Stop scorer", "description": "Stop demo_forecast.py"},
]

_SCRIPT_CMDS: dict[str, list[str]] = {
    "killchain-recon": [sys.executable, str(ROOT / "scripts" / "demo_killchain.py"), "--phase", "recon"],
    "killchain-enum": [sys.executable, str(ROOT / "scripts" / "demo_killchain.py"), "--phase", "enum"],
    "killchain-spray": [sys.executable, str(ROOT / "scripts" / "demo_killchain.py"), "--phase", "spray"],
    "killchain-loot": [sys.executable, str(ROOT / "scripts" / "demo_killchain.py"), "--phase", "loot"],
    "killchain-all": [sys.executable, str(ROOT / "scripts" / "demo_killchain.py"), "--phase", "all"],
    "auto-attack": [sys.executable, str(ROOT / "scripts" / "demo_forecast.py"), "--auto-attack"],
    "lab-up": [sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "up"],
    "lab-down": [sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "down"],
    "bench-fair-ids": [sys.executable, str(ROOT / "scripts" / "compare_ips_v01_v02.py")],
    "forecast-record": [sys.executable, str(ROOT / "scripts" / "demo_forecast_record.py")],
    "scorer-start": [sys.executable, str(ROOT / "scripts" / "demo_forecast.py")],
}

_LOG_LINES: list[dict[str, Any]] = [
    {"ts": 1, "kind": "info", "text": "Lab Console API ready"},
]
_LOG_LOCK = threading.Lock()
_SEQ = 1


def list_scripts() -> list[dict[str, str]]:
    return SCRIPTS


def get_logs(since: int = 0) -> list[dict[str, Any]]:
    with _LOG_LOCK:
        return [ln for ln in _LOG_LINES if int(ln.get("ts", 0)) >= since]


def _append_log(text: str, kind: str = "info") -> None:
    global _SEQ
    with _LOG_LOCK:
        _SEQ += 1
        _LOG_LINES.append({"ts": _SEQ, "kind": kind, "text": text})


def _run_subprocess(script_id: str, cmd: list[str]) -> None:
    _append_log(f"[script] starting {script_id}: {' '.join(cmd)}", kind="phase")
    try:
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
                kind = "scorer" if "FORECAST" in line or "window" in line.lower() else "info"
                if "ALERT" in line or "IPS" in line:
                    kind = "alert"
                if "[auto-attack]" in line or "phase=" in line:
                    kind = "phase"
                _append_log(line.rstrip(), kind=kind)
        proc.wait()
        _append_log(f"[script] {script_id} exited ({proc.returncode})", kind="phase")
    except Exception as exc:
        _append_log(f"[script] {script_id} failed: {exc}", kind="alert")


def _stop_scorer() -> None:
    pid_path = ROOT / "data" / "lab_events" / "demo_forecast.pid"
    if not pid_path.exists():
        _append_log("scorer not running", kind="info")
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return
    if pid <= 0:
        return
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
    _append_log(f"stopped scorer pid {pid}", kind="phase")


def run_script(script_id: str) -> dict[str, Any]:
    if script_id == "scorer-stop":
        threading.Thread(target=_stop_scorer, daemon=True).start()
        return {"job_id": f"job-{script_id}", "status": "queued"}

    cmd = _SCRIPT_CMDS.get(script_id)
    if not cmd:
        _append_log(f"unknown script: {script_id}", kind="alert")
        return {"job_id": f"job-{script_id}", "status": "error", "message": "unknown script"}

    job_id = f"job-{script_id}-{int(time.time())}"
    threading.Thread(target=_run_subprocess, args=(script_id, cmd), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}
