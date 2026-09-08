"""RAMX-HX: v3 gate + self-write memory + cooldown re-baseline + confirm cap."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

RAMX_HX_VERSION = "hx-1.0"


class WarmupBaselineCalibrator:
    def __init__(self, warmup_steps: int = 15, epsilon: float = 1e-4):
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon
        self.warmup_buffer: list[np.ndarray] = []
        self.baseline_mean: np.ndarray | None = None
        self.baseline_std: np.ndarray | None = None
        self.calibrated = False

    def reset(self) -> None:
        self.warmup_buffer = []
        self.baseline_mean = None
        self.baseline_std = None
        self.calibrated = False

    def update(self, state_vector: np.ndarray) -> None:
        x = np.asarray(state_vector, dtype=np.float64).reshape(-1)
        if not self.calibrated:
            self.warmup_buffer.append(x.copy())
            if len(self.warmup_buffer) >= self.warmup_steps:
                buf = np.stack(self.warmup_buffer, axis=0)
                self.baseline_mean = np.mean(buf, axis=0)
                self.baseline_std = np.std(buf, axis=0) + self.epsilon
                self.calibrated = True
            return
        # Slow EMA so benign drift does not look like a kill chain.
        a = 0.06
        self.baseline_mean = (1.0 - a) * self.baseline_mean + a * x
        resid = np.abs(x - self.baseline_mean)
        self.baseline_std = (1.0 - a) * self.baseline_std + a * resid
        self.baseline_std = np.maximum(self.baseline_std, self.epsilon)

    def get_relative_anomaly_score(self, current_state: np.ndarray) -> float:
        if not self.calibrated or self.baseline_mean is None or self.baseline_std is None:
            return 0.0
        z_scores = np.abs((current_state - self.baseline_mean) / self.baseline_std)
        top_k = max(5, int(0.05 * len(current_state)))
        mean_top_z = float(np.mean(np.sort(z_scores)[-top_k:]))
        anomaly_prob = float(1.0 / (1.0 + np.exp(-1.05 * (mean_top_z - 3.2))))
        return float(np.clip(anomaly_prob, 0.0, 1.0))


class RawScoreCalibrator:
    def __init__(self, window: int = 20):
        self.window = window
        self.scores: list[float] = []
        self.offset: float = 0.0
        self.ready = False

    def reset(self) -> None:
        self.scores = []
        self.offset = 0.0
        self.ready = False

    def update(self, raw_p: float) -> None:
        self.scores.append(float(raw_p))
        if len(self.scores) >= self.window:
            self.offset = float(np.median(self.scores[-self.window :]))
            self.ready = True

    def adjust(self, raw_p: float) -> float:
        if not self.ready:
            return raw_p
        return float(np.clip(raw_p - self.offset, 0.0, 1.0))

    def is_saturated(self) -> bool:
        if len(self.scores) < 8:
            return False
        return float(np.std(self.scores[-min(self.window, len(self.scores)) :])) < 0.02


class ClassPriorTracker:
    """Total-variation shift vs the warmup class histogram.

    HX-C's binary attack head is often saturated (~1.0 on everything). The
    33-way catalog still moves when traffic becomes a scan/flood, so TV
    distance from the warmup prior is a usable live detector.
    """

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self._n = 0
        self.prior: np.ndarray | None = None
        self.frozen = False
        self.warmup_shifts: list[float] = []
        self.shift_offset: float = 0.0

    def reset(self) -> None:
        self._sum = None
        self._n = 0
        self.prior = None
        self.frozen = False
        self.warmup_shifts = []
        self.shift_offset = 0.0

    def observe_warmup(self, class_p: np.ndarray) -> None:
        if self.frozen:
            return
        if self.prior is not None:
            self.warmup_shifts.append(self.shift(class_p))
        p = np.asarray(class_p, dtype=np.float64).reshape(-1)
        if self._sum is None:
            self._sum = np.zeros_like(p)
        if self._sum.shape != p.shape:
            return
        self._sum += p
        self._n += 1
        self.prior = self._sum / max(self._n, 1)

    def freeze(self) -> None:
        if self._n and self._sum is not None:
            self.prior = self._sum / self._n
        if self.warmup_shifts:
            self.shift_offset = float(np.median(self.warmup_shifts))
        self.frozen = True

    def shift_adjusted(self, class_p: np.ndarray) -> float:
        return float(np.clip(self.shift(class_p) - self.shift_offset, 0.0, 1.0))

    def shift(self, class_p: np.ndarray) -> float:
        if self.prior is None:
            return 0.0
        p = np.asarray(class_p, dtype=np.float64).reshape(-1)
        if p.shape != self.prior.shape:
            return 0.0
        tv = 0.5 * float(np.abs(p - self.prior).sum())
        return float(np.clip(tv, 0.0, 1.0))


class EpisodicMemoryBank:
    def __init__(self, max_entries: int = 500):
        self.max_entries = max_entries
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict[str, Any]] = []

    def add(self, state_trajectory: np.ndarray, label: int, stage: int, info: dict | None = None) -> None:
        flat_repr = state_trajectory.flatten()
        if len(self.vectors) >= self.max_entries:
            self.vectors.pop(0)
            self.metadata.pop(0)
        self.vectors.append(flat_repr)
        self.metadata.append({"label": label, "stage": stage, "info": info or {}})

    def query(self, current_trajectory: np.ndarray, k: int = 1) -> list[tuple[dict[str, Any], float]]:
        if not self.vectors:
            return []
        M = np.stack(self.vectors, axis=0)
        target = current_trajectory.flatten()[None, :]
        if target.shape[1] != M.shape[1]:
            return []
        dists = np.linalg.norm(M - target, axis=1)
        k_nearest = min(k, len(dists))
        best = np.argsort(dists)[:k_nearest]
        return [(self.metadata[i], float(dists[i])) for i in best]


class RAMXHXPredictor:
    """Context gate, fusion, self-write memory, cooldown, optional confirm cap."""

    def __init__(
        self,
        *,
        warmup_steps: int = 15,
        context_skip_steps: int = 0,
        alert_cap_during_gate: float = 0.49,
        raw_calibrator_window: int = 20,
        cooldown_steps: int = 0,
        memory_write_thresh: float = 0.85,
        memory_boost: float = 0.65,
        memory_dist: float = 15.0,
        confirm_min: float = 0.5,
    ):
        self.warmup_steps = warmup_steps
        self.context_skip_steps = int(context_skip_steps)
        self.alert_cap_during_gate = alert_cap_during_gate
        self.cooldown_steps = cooldown_steps
        self.memory_write_thresh = memory_write_thresh
        self.memory_boost = memory_boost
        self.memory_dist = memory_dist
        self.confirm_min = confirm_min
        self.calibrator = WarmupBaselineCalibrator(warmup_steps=warmup_steps)
        self.raw_calibrator = RawScoreCalibrator(window=raw_calibrator_window)
        self.prior = ClassPriorTracker()
        self.memory_bank = EpisodicMemoryBank()
        self.step_count = 0
        self.live_step_count = 0
        self.in_live_phase = False
        self.cooldown_left = 0
        self.last_raw_pre_cap: list[float] = []

    def reset_stream(self) -> None:
        self.calibrator.reset()
        self.raw_calibrator.reset()
        self.prior.reset()
        self.step_count = 0
        self.live_step_count = 0
        self.in_live_phase = False
        self.cooldown_left = 0
        self.last_raw_pre_cap = []

    def begin_live_phase(self) -> None:
        self.in_live_phase = True
        self.live_step_count = 0
        self.calibrator.reset()
        self.raw_calibrator.reset()
        self.prior.reset()
        self.cooldown_left = 0

    def set_context_skip(self, steps: int) -> None:
        self.context_skip_steps = int(steps)
        self.prior.freeze()

    def fuse(
        self,
        *,
        raw_p: float,
        relative_anomaly: float,
        in_context_gate: bool,
        in_cooldown: bool,
        class_shift: float = 0.0,
    ) -> tuple[float, bool]:
        """Returns (fused_p, alert_suppressed) before confirm/memory."""
        shift = float(class_shift)
        not_ready = not self.raw_calibrator.ready
        if in_context_gate or not_ready:
            fused = min(max(relative_anomaly, shift), self.alert_cap_during_gate)
            return fused, True
        if in_cooldown:
            fused = min(max(raw_p, relative_anomaly, shift), self.alert_cap_during_gate)
            return fused, True
        if self.raw_calibrator.is_saturated():
            fused = float(max(relative_anomaly, shift, 0.35 * relative_anomaly + 0.65 * shift))
        elif self.calibrator.calibrated:
            fused = float(
                max(raw_p, 0.30 * raw_p + 0.40 * relative_anomaly + 0.30 * shift)
            )
        else:
            fused = float(max(raw_p, shift))
        return float(np.clip(fused, 0.0, 1.0)), False

    def maybe_memory(
        self,
        traj: np.ndarray,
        fused: float,
        confirmed: bool,
        pred_stage: int,
        in_context_gate: bool,
        in_cooldown: bool,
    ) -> tuple[float, bool]:
        if in_context_gate or in_cooldown:
            return fused, False
        written = False
        matches = self.memory_bank.query(traj, k=1)
        if matches and matches[0][1] < self.memory_dist:
            match_stage = int(matches[0][0].get("stage", 0))
            if pred_stage == 0 and match_stage > 0:
                pred_stage = match_stage
                fused = max(fused, self.memory_boost)
        if confirmed and fused >= self.memory_write_thresh and pred_stage > 0:
            self.memory_bank.add(traj, label=1, stage=pred_stage)
            written = True
        return fused, written

    def after_alert(self, confirmed: bool, fused: float) -> None:
        if self.cooldown_steps <= 0:
            return
        if confirmed and fused >= 0.5:
            self.cooldown_left = self.cooldown_steps
