#!/usr/bin/env python3
"""Launch the PRISM Streamlit dashboard with the repo root on PYTHONPATH."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "src" / "ui" / "app.py"


def main() -> int:
    env = os.environ.copy()
    prefix = str(ROOT)
    env["PYTHONPATH"] = prefix if not env.get("PYTHONPATH") else f"{prefix}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [sys.executable, "-m", "streamlit", "run", str(APP), *sys.argv[1:]]
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
