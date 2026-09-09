"""Wall-clock pacing control for the live adversarial attack lab.

`speed=1.0` reproduces realistic relative timing: bot think-time/evasion
jitter and the orchestrator's per-window capture cadence match the real
30s StateBuilder window used everywhere else in the pipeline
(`src/aryan/ingest.py::WINDOW_SEC`). `speed>1` compresses every sleep and
window duration proportionally, so a multi-minute kill chain can be
watched (or batch-run) in seconds without changing the *shape* of the
timeline -- recon still happens before exploit, evasion jitter still
varies the same way relative to think-time, etc.
"""

from __future__ import annotations

import time

BASE_WINDOW_SEC = 30.0


class LabClock:
    def __init__(self, speed: float = 1.0, base_window_sec: float = BASE_WINDOW_SEC):
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self.speed = speed
        self.base_window_sec = base_window_sec
        self._t0 = time.monotonic()

    def sleep(self, base_seconds: float) -> None:
        """Sleep for `base_seconds` of *simulated* lab time."""
        if base_seconds <= 0:
            return
        time.sleep(base_seconds / self.speed)

    def window_sec(self) -> float:
        """Wall-clock seconds one StateBuilder window takes at this speed."""
        return self.base_window_sec / self.speed

    def elapsed(self) -> float:
        """Real wall-clock seconds since this clock was created."""
        return time.monotonic() - self._t0

    def sim_elapsed(self) -> float:
        """Simulated lab-time seconds elapsed (elapsed() * speed)."""
        return self.elapsed() * self.speed
