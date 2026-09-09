"""Helpers to manage the isolated Docker lab lifecycle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.adversarial.lab_config import (
    ATTACKER_CONTAINER,
    BENIGN_CONTAINER,
    COMPOSE_FILE,
    TARGET_CONTAINER,
    TARGET_HOST,
)

ROOT = Path(__file__).resolve().parent.parent.parent


def _container_running(name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"
    except Exception:
        return False


def lab_status() -> dict[str, bool]:
    return {
        TARGET_CONTAINER: _container_running(TARGET_CONTAINER),
        ATTACKER_CONTAINER: _container_running(ATTACKER_CONTAINER),
        BENIGN_CONTAINER: _container_running(BENIGN_CONTAINER),
    }


def ensure_lab_running() -> bool:
    """Start compose stack if any lab container is missing."""
    status = lab_status()
    if all(status.values()):
        return True

    print("PRISM lab not fully running. Starting docker compose...")
    cmd = [sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "up"]
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def verify_lab_connectivity() -> bool:
    """Run a lightweight probe attack inside attacker-bot."""
    cmd = [
        "docker", "exec", ATTACKER_CONTAINER,
        "python3", "-m", "src.adversarial.attack_script",
        TARGET_HOST, "T1046_service_scan", "none",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    ok = result.returncode == 0 and "DONE|class=" in (result.stdout or "")
    if not ok:
        print("Lab connectivity check failed:")
        print(result.stdout or result.stderr)
    return ok
