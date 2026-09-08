"""Script launcher + log stream for terminal rail."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
EVENTS = ROOT / "data" / "lab_events"
SCORER_LOG = EVENTS / "scorer_bff.log"
PID_PATH = EVENTS / "demo_forecast.pid"

_SCORER_SCRIPTS = frozenset({"scorer-start", "auto-attack"})

SCRIPTS: list[dict[str, str]] = [
    {"id": "killchain-recon", "label": "Kill-chain: recon", "description": "Port scan phase"},
    {"id": "killchain-enum", "label": "Kill-chain: enum", "description": "Directory bust"},
    {"id": "killchain-spray", "label": "Kill-chain: spray", "description": "Credential spray"},
    {"id": "killchain-loot", "label": "Kill-chain: loot", "description": "Payroll download"},
    {"id": "killchain-all", "label": "Kill-chain: all", "description": "Full chain"},
    {"id": "auto-attack", "label": "Auto-attack", "description": "demo_forecast --auto-attack"},
    {"id": "lab-up", "label": "Lab up", "description": "lab_ctl up"},
    {"id": "lab-down", "label": "Lab down", "description": "lab_ctl down"},
    {"id": "bench-fair-ids", "label": "Fair IDS bench", "description": "Offline ARY v01 vs v02 (log only, does not update charts)"},
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
    "auto-attack": [
        sys.executable,
        str(ROOT / "scripts" / "demo_forecast.py"),
        "--no-up",
        "--auto-attack",
        "--warmup-windows",
        "15",
    ],
    "lab-up": [sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "up"],
    "lab-down": [sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "down"],
    "bench-fair-ids": [sys.executable, str(ROOT / "scripts" / "compare_ips_v01_v02.py")],
    "forecast-record": [sys.executable, str(ROOT / "scripts" / "demo_forecast_record.py")],
    "scorer-start": [
        sys.executable,
        str(ROOT / "scripts" / "demo_forecast.py"),
        "--no-up",
        "--warmup-windows",
        "5",
    ],
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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        r = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = (r.stdout or "").strip().lower()
        return bool(line) and str(pid) in line and "python" in line
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _clear_stale_pid() -> None:
    if not PID_PATH.exists():
        return
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        PID_PATH.unlink(missing_ok=True)
        return
    if pid <= 0 or not _pid_alive(pid):
        PID_PATH.unlink(missing_ok=True)


def _tail_file(path: Path) -> None:
    """Stream scorer log lines into the API log buffer."""
    deadline = time.time() + 600
    try:
        with path.open("a+", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            while time.time() < deadline:
                line = f.readline()
                if not line:
                    if PID_PATH.exists():
                        try:
                            pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
                        except ValueError:
                            pid = 0
                        if pid > 0 and not _pid_alive(pid):
                            break
                    time.sleep(0.25)
                    continue
                text = line.rstrip()
                if not text or text.startswith("--- ["):
                    continue
                kind = "scorer" if "FORECAST" in text or " w0" in text else "info"
                if "ALERT" in text or "IPS" in text:
                    kind = "alert"
                if "[auto-attack]" in text or "phase=" in text:
                    kind = "phase"
                _append_log(text, kind=kind)
    except Exception as exc:
        _append_log(f"[scorer log] tail stopped: {exc}", kind="alert")


def _launch_scorer_detached(script_id: str, cmd: list[str]) -> None:
    _clear_stale_pid()
    if _scorer_running():
        _append_log("scorer already running — use Stop scorer first", kind="alert")
        return

    EVENTS.mkdir(parents=True, exist_ok=True)
    log_file = SCORER_LOG.open("a", encoding="utf-8", buffering=1)
    log_file.write(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] {script_id} ---\n")
    log_file.flush()

    popen_kwargs: dict[str, Any] = {
        "cwd": str(ROOT),
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(cmd, **popen_kwargs)
    except Exception as exc:
        log_file.close()
        _append_log(f"[script] {script_id} failed to launch: {exc}", kind="alert")
        return
    finally:
        log_file.close()

    _append_log(f"[script] launched {script_id} (log: {SCORER_LOG.name})", kind="phase")
    threading.Thread(target=_tail_file, args=(SCORER_LOG,), daemon=True).start()


def _run_subprocess(script_id: str, cmd: list[str]) -> None:
    _append_log(f"[script] starting {script_id}: {' '.join(cmd)}", kind="phase")
    EVENTS.mkdir(parents=True, exist_ok=True)
    log_path = EVENTS / f"script_{script_id}.log"
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(f"\n--- [{time.strftime('%Y-%m-%d %H:%M:%S')}] {script_id} ---\n")
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            proc.wait()
        _append_log(f"[script] {script_id} exited ({proc.returncode})", kind="phase")
    except Exception as exc:
        _append_log(f"[script] {script_id} failed: {exc}", kind="alert")


def _scorer_running() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return False
    if pid <= 0:
        return False
    return _pid_alive(pid)


_CHART_SCRIPTS = frozenset({
    "killchain-recon",
    "killchain-enum",
    "killchain-spray",
    "killchain-loot",
    "killchain-all",
    "bench-fair-ids",
})


def _chart_script_hint(script_id: str) -> None:
    if script_id == "bench-fair-ids":
        _append_log(
            "Fair IDS bench runs offline (ARY v01 vs v02). Lab Session charts only update "
            "when demo_forecast scorer is running — use Start scorer or Auto-attack.",
            kind="alert",
        )
        return
    if script_id in _CHART_SCRIPTS and not _scorer_running():
        _append_log(
            f"{script_id} runs attacks in Docker but will not move the charts until the "
            "forecast scorer is running. Click Start scorer or Auto-attack first.",
            kind="alert",
        )


def _stop_scorer() -> None:
    if not PID_PATH.exists():
        _append_log("scorer not running", kind="info")
        return
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        PID_PATH.unlink(missing_ok=True)
        return
    if pid <= 0:
        PID_PATH.unlink(missing_ok=True)
        return
    if not _pid_alive(pid):
        PID_PATH.unlink(missing_ok=True)
        _append_log("scorer pid file was stale — cleared", kind="info")
        return
    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
    PID_PATH.unlink(missing_ok=True)
    _append_log(f"stopped scorer pid {pid}", kind="phase")


_KILLCHAIN_PHASES = {
    "killchain-recon": "recon",
    "killchain-enum": "enum",
    "killchain-spray": "spray",
    "killchain-loot": "loot",
    "killchain-all": "all",
}


def append_log(text: str, kind: str = "info") -> None:
    _append_log(text, kind=kind)


def run_script(script_id: str, delay_sec: float | None = None) -> dict[str, Any]:
    if script_id == "scorer-stop":
        threading.Thread(target=_stop_scorer, daemon=True).start()
        return {"job_id": f"job-{script_id}", "status": "queued"}

    if script_id in _KILLCHAIN_PHASES:
        from services.attack_scheduler import schedule_attack

        phase = _KILLCHAIN_PHASES[script_id]
        _chart_script_hint(script_id)
        job = schedule_attack(phase, delay_sec)
        return {"job_id": f"job-{script_id}-{job['id']}", "status": "queued", "attack": job}

    cmd = _SCRIPT_CMDS.get(script_id)
    if not cmd:
        _append_log(f"unknown script: {script_id}", kind="alert")
        return {"job_id": f"job-{script_id}", "status": "error", "message": "unknown script"}

    job_id = f"job-{script_id}-{int(time.time())}"
    _chart_script_hint(script_id)
    if script_id in _SCORER_SCRIPTS:
        if _scorer_running():
            return {"job_id": job_id, "status": "already_running"}
        threading.Thread(target=_launch_scorer_detached, args=(script_id, cmd), daemon=True).start()
    else:
        threading.Thread(target=_run_subprocess, args=(script_id, cmd), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}
