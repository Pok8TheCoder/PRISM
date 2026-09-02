"""Overlapping multi-resolution forecast tiers (solo vs blend).

Schedule (wall-clock seconds from forecast start):
  1s  windows — steps each 1s,  solo owns [0, 10]
  5s  windows — steps from 11s,  solo owns [11, 120]
  10s windows — steps from 0s,   solo owns [121, 420]  (300s span)
  30s windows — steps from 0s,   solo owns [421, 1620] (1200s span)

All tiers may **predict** into overlap zones. Coarser tiers start at t=0 and
their first window covers 0–ws, overlapping finer tiers' solo zones.

Combine modes:
  solo  — at second ``s``, use finest-resolution tier whose *solo* zone contains ``s``
  blend — among tiers that issued a prediction covering ``s``, average the two
          with the **smallest window sizes** (finest pair; typically 1s+10s, 5s+10s, …)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

LOOKBACK = 20
FINE_SEC = 1.0
_SUM_EXACT = {0}


class CombineMode(str, Enum):
    SOLO = "solo"
    BLEND = "blend"


@dataclass(frozen=True)
class OverlapTier:
    window_sec: float
    solo_start: float
    solo_end: float
    step_start: float = 0.0

    @property
    def model_key(self) -> str:
        return "ary30" if self.window_sec >= 30 else "ary5"

    def solo_covers(self, t: float) -> bool:
        return self.solo_start <= t <= self.solo_end

    def steps_at(self, t: float) -> bool:
        t_int = int(round(t))
        ws = int(round(self.window_sec))
        ss = int(round(self.step_start))
        if t_int < ss:
            return False
        # 1s tier only fires within its solo window
        if ws == 1 and t_int > int(round(self.solo_end)):
            return False
        return (t_int - ss) % ws == 0


DEFAULT_OVERLAP_TIERS: tuple[OverlapTier, ...] = (
    OverlapTier(1, 0, 10, step_start=0),
    OverlapTier(5, 11, 120, step_start=11),
    OverlapTier(10, 121, 420, step_start=0),
    OverlapTier(30, 421, 1620, step_start=0),
)


def pool_states(states: np.ndarray, group: int) -> np.ndarray:
    if group <= 1 or len(states) == 0:
        return states
    n = (len(states) // group) * group
    if n == 0:
        return states[:1]
    chunks = states[:n].reshape(n // group, group, -1)
    out = []
    for c in chunks:
        row = []
        for j in range(c.shape[1]):
            if j in _SUM_EXACT:
                row.append(float(c[:, j].sum()))
            else:
                row.append(float(c[:, j].mean()))
        out.append(row)
    return np.asarray(out, dtype=np.float32)


def lookback_from_fine(buffer: np.ndarray, window_sec: float) -> np.ndarray:
    group = max(1, int(round(window_sec / FINE_SEC)))
    pooled = pool_states(buffer, group) if group > 1 else buffer
    if len(pooled) < LOOKBACK:
        pad = np.repeat(pooled[:1], LOOKBACK - len(pooled), axis=0)
        pooled = np.concatenate([pad, pooled], axis=0)
    return pooled[-LOOKBACK:].astype(np.float32)


def upsample_states_5s_to_1s(states_5s: np.ndarray) -> np.ndarray:
    if len(states_5s) == 0:
        return states_5s
    return np.repeat(states_5s, 5, axis=0).astype(np.float32)


def _combine_predictions(
    contributors: list[tuple[OverlapTier, np.ndarray]],
    mode: CombineMode,
    t_sec: float,
) -> np.ndarray:
    if not contributors:
        raise ValueError("no contributors")
    if mode == CombineMode.SOLO:
        owners = [c for c in contributors if c[0].solo_covers(t_sec)]
        if owners:
            pick = min(owners, key=lambda x: x[0].window_sec)
            return pick[1]
        return min(contributors, key=lambda x: x[0].window_sec)[1]

    # blend: two smallest window sizes among contributors
    contributors = sorted(contributors, key=lambda x: x[0].window_sec)
    if len(contributors) == 1:
        return contributors[0][1]
    a, b = contributors[0][1], contributors[1][1]
    return ((a.astype(np.float64) + b.astype(np.float64)) / 2.0).astype(np.float32)


def run_overlap_forecast(
    models: dict,
    context_fine: np.ndarray,
    horizon_sec: float,
    tiers: tuple[OverlapTier, ...] = DEFAULT_OVERLAP_TIERS,
    mode: CombineMode = CombineMode.SOLO,
) -> tuple[np.ndarray, list[dict]]:
    """Open-loop overlap forecast on 1s grid.

    Returns (pred_fine[T, D], step_log).
    """
    d = context_fine.shape[1]
    hz = int(round(horizon_sec / FINE_SEC))
    buffer = context_fine.astype(np.float32).copy()
    pred_fine = np.zeros((hz, d), dtype=np.float32)
    log: list[dict] = []

    # (tier, pred vector, covers [t0, t1))
    active_windows: list[tuple[OverlapTier, np.ndarray, float, float]] = []

    for t_idx in range(hz):
        t_sec = float(t_idx)
        stepping = [tier for tier in tiers if tier.steps_at(t_sec)]

        for tier in stepping:
            lb = lookback_from_fine(buffer, tier.window_sec)
            model = models[tier.model_key]
            with torch.no_grad():
                x = torch.from_numpy(lb).unsqueeze(0)
                nxt = model(x)["pred_state_mean"].squeeze(0).numpy().astype(np.float32)
            t0 = t_sec
            t1 = min(t_sec + tier.window_sec, horizon_sec)
            active_windows.append((tier, nxt, t0, t1))
            log.append({
                "t_sec": t_sec, "tier_ws": tier.window_sec, "model": tier.model_key,
                "covers": [t0, t1], "mode": mode.value,
            })

        contributors: list[tuple[OverlapTier, np.ndarray]] = []
        for tier, vec, t0, t1 in active_windows:
            if t0 <= t_sec < t1:
                contributors.append((tier, vec))

        if contributors:
            combined = _combine_predictions(contributors, mode, t_sec)
        else:
            combined = buffer[-1].copy()

        pred_fine[t_idx] = combined
        buffer = np.vstack([buffer, combined[None, :]])

        active_windows = [(ti, v, a, b) for ti, v, a, b in active_windows if b > t_sec + 0.5]

    return pred_fine, log
