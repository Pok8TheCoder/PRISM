#!/usr/bin/env python3
"""Launch PRISM Lab Console (React UI + optional FastAPI BFF).

Replaces Streamlit dashboard (`run_dashboard.py`) as the primary lab UI.

Usage:
  python run_lab_console.py              # frontend dev server only (mock data)
  python run_lab_console.py --api        # also start FastAPI BFF on :8790
  python run_lab_console.py --build      # production build + preview
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "apps" / "lab-console" / "frontend"
API = ROOT / "apps" / "lab-console" / "api" / "main.py"
API_PORT = 8790


def _pids_on_port(port: int) -> list[int]:
    if os.name == "nt":
        r = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        pids: list[int] = []
        for line in (r.stdout or "").splitlines():
            if f":{port}" not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if parts and parts[-1].isdigit():
                pids.append(int(parts[-1]))
        return sorted(set(pids))
    r = subprocess.run(
        ["lsof", "-ti", f":{port}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    out: list[int] = []
    for tok in (r.stdout or "").split():
        if tok.isdigit():
            out.append(int(tok))
    return out


def _free_port(port: int) -> None:
    for pid in _pids_on_port(port):
        print(f"Stopping process on :{port} (pid {pid})…")
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=10)
        else:
            subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=10)
    if _pids_on_port(port):
        time.sleep(0.5)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--api", action="store_true", help="Start FastAPI BFF on port 8790")
    p.add_argument("--build", action="store_true", help="npm run build && npm run preview")
    args = p.parse_args()

    if not FRONTEND.is_dir():
        print(f"Frontend not found: {FRONTEND}", file=sys.stderr)
        return 1

    procs: list[subprocess.Popen] = []

    if args.api:
        _free_port(API_PORT)
        api_proc = subprocess.Popen(
            [sys.executable, str(API)],
            cwd=str(ROOT),
        )
        procs.append(api_proc)
        time.sleep(0.5)
        print(f"API: http://127.0.0.1:{API_PORT}")

    npm = "npm.cmd" if os.name == "nt" else "npm"
    if args.build:
        subprocess.check_call([npm, "run", "build"], cwd=str(FRONTEND))
        cmd = [npm, "run", "preview"]
    else:
        cmd = [npm, "run", "dev"]

    print("UI: http://127.0.0.1:5173")
    print("Streamlit dashboard is deprecated — use Lab Console instead.")
    vite_proc = subprocess.Popen(cmd, cwd=str(FRONTEND))
    procs.insert(0, vite_proc)
    try:
        return vite_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down Lab Console…")
        return 0
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
