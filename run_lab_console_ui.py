#!/usr/bin/env python3
"""Launch Lab Console in UI/UX demo mode (stub BFF + Vite on LAN).

No Docker, scorer, or model weights required — for design review only.

Install & run: see QUICKSTART-UI.md in repo root.

Usage:
  python run_lab_console_ui.py
  python run_lab_console_ui.py --port 5173
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
STUB_API = ROOT / "apps" / "lab-console" / "api" / "stub_main.py"
API_PORT = 8790


def _free_port(port: int) -> None:
    if os.name != "nt":
        return
    r = subprocess.run(["netstat", "-ano"], capture_output=True, text=True, timeout=15)
    for line in (r.stdout or "").splitlines():
        if f":{port}" not in line or "LISTENING" not in line:
            continue
        pid = line.split()[-1]
        if pid.isdigit():
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=10)
    time.sleep(0.3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=5173, help="Vite dev server port")
    args = p.parse_args()

    if not FRONTEND.is_dir() or not STUB_API.is_file():
        print("Lab Console frontend or stub API not found.", file=sys.stderr)
        return 1

    _free_port(API_PORT)
    _free_port(args.port)

    api_proc = subprocess.Popen([sys.executable, str(STUB_API)], cwd=str(ROOT))
    time.sleep(0.6)

    npm = "npm.cmd" if os.name == "nt" else "npm"
    env = os.environ.copy()
    env["VITE_DEV_PORT"] = str(args.port)
    vite_proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--host", "--port", str(args.port)],
        cwd=str(FRONTEND),
        env=env,
    )

    print()
    print("=" * 56)
    print("  PRISM Lab Console — UI/UX DEMO (stub backend)")
    print("=" * 56)
    print(f"  Local:   http://127.0.0.1:{args.port}/session")
    print(f"  Network: http://<your-ip>:{args.port}/session")
    print(f"  Stub API: http://127.0.0.1:{API_PORT}/api/health")
    print("  Try Live mode + kill-chain scripts for chart bands.")
    print("=" * 56)
    print()

    try:
        return vite_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down UI demo…")
        return 0
    finally:
        for proc in (vite_proc, api_proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
