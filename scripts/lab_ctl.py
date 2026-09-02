#!/usr/bin/env python3
"""Start/stop/status for the isolated PRISM Docker lab."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker" / "docker-compose.yml"


def _compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE), "-p", "prism", *args]


def _remove_conflicting_containers() -> None:
    """Remove legacy standalone containers that block compose names."""
    for name in ("target-server", "attacker-bot", "benign-client", "redteam"):
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.Name}}", name],
            capture_output=True,
            text=True,
        )
        if inspect.returncode != 0:
            continue
        print(f"Removing conflicting container: {name}")
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def cmd_up(_: argparse.Namespace) -> int:
    print("Building and starting isolated PRISM lab (internal network only)...")
    _remove_conflicting_containers()
    r = subprocess.run(_compose_cmd("up", "-d", "--build"), cwd=str(ROOT))
    if r.returncode != 0:
        return r.returncode
    print("\nLab containers:")
    subprocess.run(_compose_cmd("ps"), cwd=str(ROOT))
    print("\nTarget DNS: target-server  |  Attacker: attacker-bot  |  Benign: benign-client  |  Red team: redteam")
    print("No ports are published to your host.")
    return 0


def cmd_down(_: argparse.Namespace) -> int:
    print("Stopping PRISM lab...")
    return subprocess.run(_compose_cmd("down"), cwd=str(ROOT)).returncode


def cmd_status(_: argparse.Namespace) -> int:
    return subprocess.run(_compose_cmd("ps"), cwd=str(ROOT)).returncode


def cmd_logs(args: argparse.Namespace) -> int:
    tail = ["--tail", str(args.tail)]
    if args.service:
        return subprocess.run(_compose_cmd("logs", *tail, args.service), cwd=str(ROOT)).returncode
    return subprocess.run(_compose_cmd("logs", *tail), cwd=str(ROOT)).returncode


def cmd_reset(args: argparse.Namespace) -> int:
    """Force-recreate target-server for a clean per-round state.

    entrypoint.sh regenerates config/secret.key and seed_db.py reseeds
    users/posts/homepage on every boot, so recreating the container (fresh
    writable layer, no volumes) is enough to fully reset the vulnerable
    webapp between objective rounds/evasion versions -- new admin
    password, new session secret, homepage un-defaced, no leftover guest
    accounts from the previous round's exploit.
    """
    print("Resetting target-server (fresh secret/DB/homepage)...")
    r = subprocess.run(
        _compose_cmd("up", "-d", "--force-recreate", "target-server"),
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        return r.returncode

    if not args.no_wait:
        print("Waiting for the webapp to come back up...")
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            check = subprocess.run(
                [
                    "docker", "exec", "target-server",
                    "python3", "-c",
                    "import urllib.request;urllib.request.urlopen('http://localhost/api/health',timeout=2)",
                ],
                capture_output=True,
            )
            if check.returncode == 0:
                print("target-server ready.")
                break
            time.sleep(1)
        else:
            print("Warning: target-server did not report healthy within timeout.", file=sys.stderr)
    return 0


def cmd_verify(_: argparse.Namespace) -> int:
    """Ping target from attacker container over internal DNS."""
    cmd = [
        "docker", "exec", "attacker-bot",
        "python3", "-m", "src.adversarial.attack_script",
        "target-server", "T1046_service_scan", "none",
    ]
    print("Running probe attack from attacker-bot -> target-server ...")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return 0 if r.returncode == 0 and "DONE|class=" in (r.stdout or "") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="PRISM isolated lab control")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("up", help="Build and start lab").set_defaults(func=cmd_up)
    sub.add_parser("down", help="Stop and remove lab").set_defaults(func=cmd_down)
    sub.add_parser("status", help="Show container status").set_defaults(func=cmd_status)

    logs_p = sub.add_parser("logs", help="Tail container logs")
    logs_p.add_argument("service", nargs="?", default=None)
    logs_p.add_argument("--tail", type=int, default=50)
    logs_p.set_defaults(func=cmd_logs)

    sub.add_parser("verify", help="Run probe attack inside lab").set_defaults(func=cmd_verify)

    reset_p = sub.add_parser("reset", help="Force-recreate target-server for a clean per-round state")
    reset_p.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for health check")
    reset_p.add_argument("--no-wait", action="store_true", help="Don't wait/poll for health after recreate")
    reset_p.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
