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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "apps" / "lab-console" / "frontend"
API = ROOT / "apps" / "lab-console" / "api" / "main.py"


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
        procs.append(
            subprocess.Popen([sys.executable, str(API)], cwd=str(ROOT))
        )
        print("API: http://127.0.0.1:8790")

    npm = "npm.cmd" if os.name == "nt" else "npm"
    if args.build:
        subprocess.check_call([npm, "run", "build"], cwd=str(FRONTEND))
        cmd = [npm, "run", "preview"]
    else:
        cmd = [npm, "run", "dev"]

    print(f"UI: http://127.0.0.1:5173")
    print("Streamlit dashboard is deprecated — use Lab Console instead.")
    try:
        return subprocess.call(cmd, cwd=str(FRONTEND))
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
