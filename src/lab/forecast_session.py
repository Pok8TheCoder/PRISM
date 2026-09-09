"""Shared forecast session scorer — extract target for demo_forecast.py (phase 2).

The Lab Console BFF imports this module instead of duplicating HTTP handler logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ForecastSessionConfig:
    policy_mode: str = "ids"  # ids | ips
    ips_confirm_windows: int = 1
    window_sec: float = 1.0
    horizon: int = 6
    auto_attack: bool = False


@dataclass
class ForecastSessionState:
    phase: str = "warmup"
    window_idx: int = 0
    blocked: bool = False
    first_blocker: str | None = None
    policy_mode: str = "ids"
    models: dict[str, Any] = field(default_factory=dict)
    phase_regions: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "window_idx": self.window_idx,
            "blocked": self.blocked,
            "first_blocker": self.first_blocker,
            "policy_mode": self.policy_mode,
            "models": self.models,
            "phase_regions": self.phase_regions,
        }


class ForecastSession:
    """Placeholder session runner — wire to demo_forecast scorer loop in phase 2."""

    def __init__(self, config: ForecastSessionConfig | None = None) -> None:
        self.config = config or ForecastSessionConfig()
        self.state = ForecastSessionState(policy_mode=self.config.policy_mode)

    def set_policy(self, mode: str) -> None:
        self.config.policy_mode = mode
        self.state.policy_mode = mode

    def tick(self) -> dict[str, Any]:
        """Advance one scoring window. Returns API-shaped state fragment."""
        self.state.window_idx += 1
        return self.state.snapshot()
