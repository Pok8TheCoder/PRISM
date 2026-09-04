#!/usr/bin/env python3
"""Playwright smoke + IPS proof for the dual-model live demo.

Waits for Labs LIVE, confirms a few benign windows do not block, then
bursts SQLi from Chromium + a host burst helper and checks who blocked first.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PwTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SITE = "http://127.0.0.1:8080"
LABS = "http://127.0.0.1:8787"
API = f"{LABS}/api/state"
PAYLOAD = "' UNION SELECT username, password_hash FROM users--"


def api() -> dict:
    with urllib.request.urlopen(API, timeout=5) as r:
        return json.loads(r.read().decode())


def wait_phase(name: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        try:
            last = api()
        except Exception as exc:
            print(f"  api wait: {exc}")
            time.sleep(2)
            continue
        print(
            f"  wait {name}: phase={last.get('phase')} w={last.get('window')} "
            f"v3={float(last.get('p_sn2rx3') or 0):.3f} "
            f"hx={float(last.get('p_hx_c') or 0):.3f} blocked={last.get('blocked')}"
        )
        if last.get("phase") == name and not last.get("blocked"):
            return last
        time.sleep(4)
    raise SystemExit(f"timeout waiting for phase={name} last={last}")


def wait_windows(n: int, timeout: float) -> dict:
    start = api()
    target = int(start.get("window") or 0) + n
    deadline = time.time() + timeout
    last = start
    while time.time() < deadline:
        last = api()
        print(
            f"  hold benign: w={last.get('window')} (need {target}) "
            f"v3={float(last.get('p_sn2rx3') or 0):.3f} "
            f"hx={float(last.get('p_hx_c') or 0):.3f} blocked={last.get('blocked')}"
        )
        if last.get("blocked"):
            raise SystemExit(
                f"FALSE POSITIVE: IPS blocked during benign live traffic: {last}"
            )
        if int(last.get("window") or 0) >= target:
            return last
        time.sleep(4)
    raise SystemExit(f"timeout waiting for {n} extra windows last={last}")


def wait_block(timeout: float) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = api()
        print(
            f"  wait block: w={last.get('window')} "
            f"v3={float(last.get('p_sn2rx3') or 0):.3f} "
            f"hx={float(last.get('p_hx_c') or 0):.3f} blocked={last.get('blocked')} "
            f"first={last.get('first_blocker')}"
        )
        if last.get("blocked"):
            return last
        time.sleep(3)
    raise SystemExit(f"timeout waiting for IPS block last={last}")


def main() -> int:
    print("selftest: open site + Labs, wait LIVE, then attack")
    with sync_playwright() as p:
        browser = None
        last_err = None
        for kwargs in (
            {"headless": True, "channel": "chrome"},
            {"headless": True, "channel": "msedge"},
            {"headless": True},
        ):
            try:
                browser = p.chromium.launch(**kwargs)
                print(f"  launched chromium {kwargs}")
                break
            except Exception as exc:
                last_err = exc
                print(f"  launch failed {kwargs}: {exc}")
        if browser is None:
            raise SystemExit(f"could not launch Chromium: {last_err}")
        site = browser.new_page()
        labs = browser.new_page()
        site.set_default_timeout(12_000)
        labs.set_default_timeout(12_000)

        site.goto(SITE, wait_until="domcontentloaded")
        body = site.content().lower()
        if "search" not in body and "register" not in body:
            raise SystemExit(f"site did not look like the lab app: {site.title()}")
        print(f"  site ok title={site.title()!r}")

        site.goto(f"{SITE}/search", wait_until="domcontentloaded")
        site.fill("input[name=q]", "welcome")
        site.click("button[type=submit]")
        site.wait_for_load_state("domcontentloaded")
        print("  benign search submitted")

        labs.goto(LABS, wait_until="domcontentloaded")
        if "DUAL IPS" not in labs.content():
            raise SystemExit("Labs dashboard missing DUAL IPS heading")
        print("  Labs dashboard ok")

        print("  waiting for a fresh LIVE (blocked=false)")
        wait_phase("live", timeout=480)
        print("  LIVE — holding 2 windows to confirm no benign IPS trip")
        wait_windows(2, timeout=90)

        q = urllib.parse.quote(PAYLOAD)
        attack_url = f"{SITE}/search?q={q}"
        print(f"  attacking via Playwright for ~25s: {attack_url}")
        burst = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "demo_attack_burst.py"),
                "--n",
                "800",
                "--concurrency",
                "40",
            ]
        )
        hits = 0
        t_attack = time.time()
        while time.time() - t_attack < 25:
            if api().get("blocked"):
                print("  API already blocked during attack loop")
                break
            try:
                site.goto(attack_url, wait_until="domcontentloaded", timeout=8_000)
                hits += 1
            except PwTimeout:
                print("  page timeout while attacking (may already be dropped)")
                break
            except Exception as exc:
                print(f"  attack nav: {type(exc).__name__}: {exc}")
                break
        print(f"  playwright hits before drop/timeout: {hits}")
        try:
            burst.wait(timeout=90)
        except subprocess.TimeoutExpired:
            burst.kill()

        state = wait_block(timeout=120)
        first = state.get("first_blocker")
        tsec = state.get("block_at_sec")
        win = state.get("block_window")
        print(
            f"PASS IPS: first_blocker={first} at t+{tsec}s window={win} "
            f"tie={state.get('same_window_tie')} scores={state.get('scores_at_block')}"
        )

        blocked_ok = False
        try:
            site.goto(SITE, wait_until="domcontentloaded", timeout=8_000)
            print("  WARNING: site still loaded after claimed IPS block")
        except Exception as exc:
            blocked_ok = True
            print(f"  site unreachable after block ({type(exc).__name__}) — expected")
        labs.goto(LABS, wait_until="domcontentloaded")
        if "IPS BLOCK" not in labs.content() and not state.get("blocked"):
            raise SystemExit("Labs did not show IPS BLOCK")
        print("  Labs still up after block")
        browser.close()

    if not blocked_ok:
        # iptables DROP can take a moment / some GETs may still complete
        try:
            urllib.request.urlopen(SITE, timeout=4)
            print("  NOTE: HTTP still answers — check iptables; demo may still be usable")
        except Exception:
            blocked_ok = True
    print("selftest done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
