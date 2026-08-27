#!/usr/bin/env python3
"""Start/stop/status for the isolated PRISM Docker lab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "docker" / "docker-compose.yml"


def _compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE), "-p", "prism", *args]


def _remove_conflicting_containers() -> None:
    """Remove legacy standalone containers that block compose names."""
    for name in ("target-server", "attacker-bot", "benign-client"):
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
    print("\nTarget DNS: target-server  |  Attacker: attacker-bot  |  Benign: benign-client")
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

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
