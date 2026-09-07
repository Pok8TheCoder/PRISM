#!/usr/bin/env python3
"""Launch the PRISM Streamlit dashboard with the repo root on PYTHONPATH.

DEPRECATED: Prefer `python run_lab_console.py` for the new Lab Console UI.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "src" / "ui" / "app.py"


def main() -> int:
    if "--streamlit-legacy" not in sys.argv and "-h" not in sys.argv and "--help" not in sys.argv:
        print(
            "Note: Streamlit dashboard is deprecated. Use: python run_lab_console.py",
            file=sys.stderr,
        )
    env = os.environ.copy()
    prefix = str(ROOT)
    env["PYTHONPATH"] = prefix if not env.get("PYTHONPATH") else f"{prefix}{os.pathsep}{env['PYTHONPATH']}"

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP)]
    extra = list(sys.argv[1:])
    if not any(a.startswith("--server.address") or a.startswith("--server.address=") for a in extra):
        extra = ["--server.address", "0.0.0.0", *extra]
    if not any(a.startswith("--server.headless") or a.startswith("--server.headless=") for a in extra):
        extra = ["--server.headless", "true", *extra]
    cmd.extend(extra)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
