"""Lab Console runtime config (horizon, attack scheduling defaults)."""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "data" / "lab_events" / "lab_console_config.json"
_LOCK = threading.Lock()

_DEFAULTS: dict[str, Any] = {
    "horizonSec": 60,
    "attackDelaySec": None,
    "attackDelayMinSec": 30,
    "attackDelayMaxSec": 60,
}


def _read_file() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_config() -> dict[str, Any]:
    with _LOCK:
        return {**_DEFAULTS, **_read_file()}


def patch_config(**kwargs: Any) -> dict[str, Any]:
    with _LOCK:
        cfg = {**_DEFAULTS, **_read_file()}
        for key, val in kwargs.items():
            if key in _DEFAULTS:
                cfg[key] = val
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return cfg


def resolve_attack_delay(explicit: float | None = None) -> float:
    cfg = get_config()
    if explicit is not None and explicit >= 0:
        return float(explicit)
    fixed = cfg.get("attackDelaySec")
    if fixed is not None:
        return float(fixed)
    lo = float(cfg.get("attackDelayMinSec", 30))
    hi = float(cfg.get("attackDelayMaxSec", 60))
    return random.uniform(lo, hi)


def horizon_windows(window_sec: float = 1.0) -> int:
    cfg = get_config()
    sec = float(cfg.get("horizonSec", 60))
    return max(1, int(round(sec / max(window_sec, 0.1))))
