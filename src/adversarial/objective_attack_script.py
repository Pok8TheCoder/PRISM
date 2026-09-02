#!/usr/bin/env python3
"""Objective-bot dispatcher — runs *inside* the attacker-bot container.

Runs an objective bot's real recon -> exploit -> post-exploit HTTP kill
chain against the target webapp (optionally preceded by existing recon
bots for realistic noise), logging structured events to a container-local
JSON-lines file. The host orchestrator (scripts/live_attack_lab.py) then
`docker cp`s that file out and deletes it -- the attacker-bot bind-mount
is read-only (`docker/docker-compose.yml`), so events can't be written
directly back into the host repo tree from in here.

Usage (inside attacker-bot):
  python3 -m src.adversarial.objective_attack_script <class_id> <evasion> <round_id> \
      [--speed 10] [--recon T1595_active_scan,T1046_service_scan] \
      [--target target-server] [--event-log /tmp/lab_events.jsonl]
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path


def _setup_import_path() -> None:
    here = Path(__file__).resolve()
    candidates = [here.parent.parent.parent, Path("/app"), Path.cwd()]
    for root in candidates:
        if (root / "src" / "adversarial" / "bots").is_dir():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return


_setup_import_path()

from src.adversarial.bots import OBJECTIVE_BOT_REGISTRY, run_bot  # noqa: E402
from src.adversarial.bots.lab_objective_base import EventLogger  # noqa: E402
from src.adversarial.lab_clock import LabClock  # noqa: E402


def _run_recon_bounded(rid: str, target: str, evasion: str, budget_sec: float) -> None:
    """Run a legacy raw-socket recon bot (`RunFn -> int`, no LabClock) with a
    wall-clock time budget.

    These bots predate the live lab and were built for offline flow-count
    generation, so their own internal sleeps are real wall-clock time, not
    `LabClock`-scaled -- e.g. `T1595_active_scan`'s `slow_timing` branch
    scans *all* 1024 ports at 2-8s/port (up to ~2 hours), which would
    silently stall the whole objective kill chain despite this module's
    docstring/comment promising recon is "best-effort, never block". Run it
    on a daemon thread and move on once the budget elapses; the round
    orchestrator force-kills this whole process at round end regardless, so
    an overrunning recon thread is harmless leftover noise, not a leak.
    """
    done = threading.Event()

    def _target() -> None:
        try:
            run_bot(rid, target, evasion)
        except Exception as exc:
            print(f"RECON_ERROR|{rid}|{exc}", flush=True)
        finally:
            done.set()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    if not done.wait(timeout=budget_sec):
        print(f"RECON_TIMEOUT|{rid}|budget={budget_sec:.1f}s -- moving on, thread left running as noise", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("class_id")
    p.add_argument("evasion")
    p.add_argument("round_id")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--recon", default="", help="Comma-separated recon bot class_ids to run first")
    p.add_argument("--target", default="target-server")
    p.add_argument("--event-log", default="/tmp/lab_events.jsonl")
    args = p.parse_args()

    if args.class_id not in OBJECTIVE_BOT_REGISTRY:
        print(f"ERROR|unknown objective class_id={args.class_id}", flush=True)
        return 1

    clock = LabClock(speed=args.speed)
    event_log = EventLogger(args.event_log)

    recon_budget = max(10.0, 45.0 / args.speed)
    for rid in [r.strip() for r in args.recon.split(",") if r.strip()]:
        print(f"RECON|{rid}", flush=True)
        _run_recon_bounded(rid, args.target, args.evasion, recon_budget)

    print(f"OBJECTIVE|{args.class_id}|start", flush=True)
    result = OBJECTIVE_BOT_REGISTRY[args.class_id](
        args.target, args.evasion, clock, event_log, round_id=args.round_id,
    )
    print(f"DONE|class={args.class_id}|success={result.get('success')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
