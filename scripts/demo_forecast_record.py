#!/usr/bin/env python3
"""Record forecast Labs dashboard + attack terminal during auto-attack.

Uses ?view=dual for side-by-side charts and live terminal log in one frame.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LABS = "http://127.0.0.1:8788"
API = f"{LABS}/api/state"
SITE = "http://127.0.0.1:8080"
OUT_DIR = ROOT / "reports" / "lab" / "hx" / "recordings"
SCORER = [sys.executable, str(ROOT / "scripts" / "demo_forecast.py")]
PID_PATH = ROOT / "data" / "lab_events" / "demo_forecast.pid"
# Bump when dashboard/IPS/SHAP behavior changes (used in filenames).
FORECAST_DEMO_VERSION = "v2.2"


def stop_stale_scorer() -> None:
    if not PID_PATH.exists():
        return
    try:
        old = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return
    if old > 0:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(old)],
            capture_output=True,
            timeout=10,
        )


def api() -> dict:
    with urllib.request.urlopen(API, timeout=5) as r:
        return json.loads(r.read().decode())


def wait_api(timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            api()
            return
        except Exception:
            time.sleep(1.5)
    raise SystemExit(f"dashboard not up at {LABS}")


def next_take_number(layout: str, version: str) -> int:
    """Increment take counter per (layout, demo version)."""
    prefix = f"forecast_{layout}_{version}_take"
    pat = re.compile(rf"forecast_{layout}_{re.escape(version)}_take(\d+)_")
    takes = []
    for path in OUT_DIR.glob(f"{prefix}*.webm"):
        m = pat.search(path.name)
        if m:
            takes.append(int(m.group(1)))
    return (max(takes) if takes else 0) + 1


def recording_basename(layout: str, version: str, take: int, stamp: str) -> str:
    return f"forecast_{layout}_{version}_take{take:02d}_{stamp}"


def wait_recording_done(timeout: float, hold_after_shap_sec: float) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        last = api()
        ap = last.get("attack_phase", "idle")
        blocked = bool(last.get("blocked"))
        has_shap = bool(last.get("block_explanation"))
        print(
            f"  record: phase={last.get('phase')} w={last.get('window')} "
            f"hx={float(last.get('p_hx_c') or 0):.3f} attack={ap} "
            f"blocked={blocked} shap={has_shap}"
        )
        done_attack = ap == "complete" and last.get("phase") == "live"
        done_block = blocked and last.get("phase") == "live"
        if done_block:
            if not has_shap:
                shap_deadline = time.time() + 50.0
                while time.time() < shap_deadline:
                    last = api()
                    if last.get("block_explanation"):
                        break
                    time.sleep(2.0)
            time.sleep(max(0.0, hold_after_shap_sec))
            return last
        if done_attack:
            time.sleep(max(0.0, hold_after_shap_sec))
            return last
        time.sleep(3)
    raise SystemExit(f"timeout waiting for recording end last={last}")


def write_manifest(path: Path, *, layout: str, version: str, take: int, state: dict) -> None:
    payload = {
        "demo_version": version,
        "layout": layout,
        "take": take,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "blocker": state.get("first_blocker"),
        "block_at_sec": state.get("block_at_sec"),
        "block_window": state.get("block_window"),
        "blocked": state.get("blocked"),
        "scores_at_block": state.get("scores_at_block"),
        "block_explanation": state.get("block_explanation"),
        "attack_phase": state.get("attack_phase"),
        "window": state.get("window"),
        "p_hx_c": state.get("p_hx_c"),
        "p_sn2rx3": state.get("p_sn2rx3"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def wait_attack_complete(timeout: float) -> dict:
    return wait_recording_done(timeout, hold_after_shap_sec=0.0)


def launch_browser(p, headless: bool):
    for kwargs in (
        {"headless": headless, "channel": "chrome"},
        {"headless": headless, "channel": "msedge"},
        {"headless": headless},
    ):
        try:
            return p.chromium.launch(**kwargs)
        except Exception as exc:
            print(f"  launch failed {kwargs}: {exc}")
    raise SystemExit("could not launch Chromium")


def wait_site(timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(SITE, timeout=3) as r:
                if r.status < 500:
                    return
        except Exception:
            time.sleep(1.5)
    raise SystemExit(f"Harborline site not up at {SITE} — run forecast compose first")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-scorer", action="store_true", help="Scorer already running with --auto-attack")
    p.add_argument("--warmup-windows", type=int, default=30)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--hold-sec", type=float, default=18.0, help="Extra seconds after attack completes")
    p.add_argument(
        "--layout",
        choices=["dual", "tabs"],
        default="dual",
        help="dual=charts+terminal side-by-side; tabs=alternate browser tabs",
    )
    p.add_argument("--demo-version", default=FORECAST_DEMO_VERSION, help="Demo stack version tag for filenames")
    p.add_argument("--take", type=int, default=0, help="Force take number (0 = auto-increment)")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    take = args.take if args.take > 0 else next_take_number(args.layout, args.demo_version)
    base = recording_basename(args.layout, args.demo_version, take, stamp)
    video_dir = OUT_DIR / f"_session_{base}"
    video_dir.mkdir(parents=True, exist_ok=True)

    scorer_proc = None
    final_state: dict = {}
    vw, vh = (1600, 1200) if args.layout == "dual" else (1400, 900)

    try:
        from src.adversarial.ips_controller import clear_blocks

        clear_blocks()
    except Exception as exc:
        print(f"  note: could not clear IPS rules: {exc}")
    wait_site(60)

    with sync_playwright() as pw:
        browser = launch_browser(pw, headless=args.headless)
        context = browser.new_context(
            viewport={"width": vw, "height": vh},
            record_video_dir=str(video_dir),
            record_video_size={"width": vw, "height": vh},
        )

        if args.layout == "dual":
            page = context.new_page()
            page.goto("about:blank")
            print("  recording dual-pane (charts + attack terminal + SHAP)...")
        else:
            charts = context.new_page()
            term = context.new_page()
            charts.goto("about:blank")
            print("  recording tabbed (alternating forecast / terminal)...")

        if not args.no_scorer:
            stop_stale_scorer()
            time.sleep(1.0)
            scorer_proc = subprocess.Popen(
                SCORER + [
                    "--no-up",
                    "--warmup-windows", str(args.warmup_windows),
                    "--auto-attack",
                    "--window-sec", "1",
                    "--no-lab-adapt",
                ],
                cwd=str(ROOT),
            )
            print(f"  started scorer pid={scorer_proc.pid}")

        wait_api(120)

        if args.layout == "dual":
            page.goto(f"{LABS}/?view=dual", wait_until="domcontentloaded")
            page.wait_for_timeout(800)
            if "FORECAST" not in page.content():
                raise SystemExit("Labs page missing FORECAST heading")
        else:
            charts.goto(LABS, wait_until="domcontentloaded")
            term.goto(f"{LABS}/?tab=terminal", wait_until="domcontentloaded")
            charts.bring_to_front()

        final_state = wait_recording_done(600, hold_after_shap_sec=args.hold_sec)

        context.close()
        browser.close()

    videos = list(video_dir.glob("*.webm"))
    if not videos:
        raise SystemExit(f"no video saved in {video_dir}")
    final = OUT_DIR / f"{base}.webm"
    videos[0].rename(final)
    write_manifest(OUT_DIR / f"{base}.json", layout=args.layout, version=args.demo_version, take=take, state=final_state)
    try:
        video_dir.rmdir()
    except OSError:
        pass
    print(f"RECORDING SAVED: {final}")
    print(f"  manifest: {OUT_DIR / f'{base}.json'}")
    print(f"  version={args.demo_version} take={take:02d} size={final.stat().st_size / 1_048_576:.1f} MB")

    if scorer_proc is not None:
        scorer_proc.terminate()
        try:
            scorer_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            scorer_proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
