"""Shared plumbing for the 3 real HTTP objective bots.

Unlike the raw-socket recon bots in `BOT_REGISTRY` (fire-and-forget flow
generators, `RunFn -> int`), the objective bots in
`OBJECTIVE_BOT_REGISTRY` drive a real `requests.Session` against the
vulnerable Flask target (`docker/target/webapp/app.py`), verify success by
independently re-reading server state, and emit structured per-stage
events to a shared JSON-lines log that `scripts/live_attack_lab.py` reads
to build the ground-truth timeline used for proxy scoring.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.adversarial.lab_clock import LabClock

STAGE_RECON = "recon"
STAGE_EXPLOIT = "exploit"
STAGE_POST_EXPLOIT = "post_exploit"


@dataclass
class ObjectiveEvent:
    class_id: str
    stage: str
    success: bool
    ts: float
    evidence: str = ""
    round_id: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class EventLogger:
    """Appends structured objective-bot events to a shared JSON-lines file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: ObjectiveEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_json()) + "\n")


def evasion_think_time(evasion: str, clock: LabClock) -> None:
    """Human-plausible think-time between HTTP steps, scaled by `LabClock`.

    Mirrors `bots/base.py::get_timing`'s evasion buckets but routes through
    the clock so raising `--speed` compresses this proportionally too.
    """
    if evasion == "slow_timing":
        base = random.uniform(3.0, 9.0)
    elif evasion == "random_timing":
        base = random.uniform(0.3, 4.0)
    else:
        base = random.uniform(0.2, 1.0)
    clock.sleep(base)


def new_round_id() -> str:
    return uuid.uuid4().hex[:12]


def default_event_log_path() -> Path:
    from src.adversarial.lab_config import SAVE_DIR

    return SAVE_DIR / "lab_events.jsonl"


def run_via_default_shim(objective_fn, target_ip: str, evasion: str) -> int:
    """Adapts a rich `run_objective(...) -> dict` bot to the legacy
    `RunFn -> int` contract, for ad-hoc CLI use through attack_script.py.
    Real orchestrated rounds should call `run_objective` directly with a
    shared `LabClock`/`EventLogger`/`round_id` instead of this shim.
    """
    clock = LabClock(speed=1.0)
    event_log = EventLogger(default_event_log_path())
    result = objective_fn(target_ip, evasion, clock, event_log, round_id=new_round_id())
    return 1 if result.get("success") else 0
