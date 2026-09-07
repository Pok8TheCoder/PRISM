#!/usr/bin/env python3
"""Smoke test for forecast dashboard + optional auto-attack."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = "http://127.0.0.1:8788/api/state"


def fetch_state() -> dict:
    with urllib.request.urlopen(DASH, timeout=5) as r:
        return json.loads(r.read().decode())


def main() -> int:
    deadline = time.time() + 420
    saw_live = False
    saw_attack = False
    peak_hx = 0.0
    print("Polling forecast API (expect LIVE then auto-attack complete)...")
    while time.time() < deadline:
        try:
            s = fetch_state()
        except Exception as exc:
            print(f"  waiting for dashboard: {exc}")
            time.sleep(2)
            continue
        phase = s.get("phase")
        ap = s.get("attack_phase", "idle")
        hx = float(s.get("p_hx_c", 0))
        peak_hx = max(peak_hx, hx)
        models = s.get("models") or {}
        v3r = len((models.get("shaun_v3") or {}).get("retrospective") or [])
        hxr = len((models.get("hx_c") or {}).get("retrospective") or [])
        v3p = len((models.get("shaun_v3") or {}).get("prospective") or [])
        print(
            f"  phase={phase} attack={ap} w={s.get('window')} hx={hx:.3f} "
            f"retro(v3/hx)={v3r}/{hxr} pro={v3p}"
        )
        if phase == "live":
            saw_live = True
        if ap not in ("idle", ""):
            saw_attack = True
        if ap == "complete" and saw_live and v3r >= 8 and v3p >= 4:
            blocked = s.get("blocked")
            print(f"PASS — attack complete, peak_hx={peak_hx:.3f} blocked={blocked} blocker={s.get('first_blocker')}")
            return 0
        time.sleep(5)
    print("FAIL — timeout", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
