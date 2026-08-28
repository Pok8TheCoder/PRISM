"""
PRISM - Infiltration Probability Scoring
Per-step risk scoring, alert generation, and trajectory analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.constants import ALERT_THRESHOLDS, MITRE_STAGES_INV

logger = logging.getLogger("prism.prediction.scoring")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StepScore:
    """Risk score for a single K-step prediction."""
    step: int
    infiltration_prob: float
    mitre_stage: str
    mitre_stage_id: int
    mitre_probs: list
    confidence: float
    alert_level: str          # "none" | "low" | "medium" | "high" | "critical"
    top_features: list = field(default_factory=list)
    attention_weights: Optional[np.ndarray] = None
    predicted_state: Optional[np.ndarray] = None


@dataclass
class RolloutResult:
    """Full K-step rollout result."""
    steps: list[StepScore]
    peak_infiltration_prob: float
    peak_step: int
    is_escalating: bool
    overall_alert: str
    trajectory_summary: str
    early_warning_steps: int   # steps ahead of threshold crossing
    ensemble_mean: Optional[np.ndarray] = None   # (K,) mean prob across rollouts
    ensemble_lower: Optional[np.ndarray] = None  # 5th percentile
    ensemble_upper: Optional[np.ndarray] = None  # 95th percentile


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------
class InfiltrationScorer:
    """
    Converts raw model outputs (infiltration probs, MITRE probs) into
    structured risk scores, alerts, and trajectory assessments.
    """

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or ALERT_THRESHOLDS

    def score_step(
        self,
        step: int,
        infiltration_prob: float,
        mitre_probs: list[float],
        confidence: float = 1.0,
        top_features: Optional[list] = None,
        attention_weights: Optional[np.ndarray] = None,
        predicted_state: Optional[np.ndarray] = None,
    ) -> StepScore:
        """Score a single prediction step."""
        stage_id = int(np.argmax(mitre_probs))
        stage_name = MITRE_STAGES_INV.get(stage_id, "Benign")
        alert_level = self._classify_alert(infiltration_prob)

        return StepScore(
            step=step,
            infiltration_prob=float(infiltration_prob),
            mitre_stage=stage_name,
            mitre_stage_id=stage_id,
            mitre_probs=[float(p) for p in mitre_probs],
            confidence=float(confidence),
            alert_level=alert_level,
            top_features=top_features or [],
            attention_weights=attention_weights,
            predicted_state=predicted_state,
        )

    def score_rollout(self, steps: list[StepScore]) -> RolloutResult:
        """
        Analyse a full K-step rollout result.

        Computes:
          - Peak infiltration probability and at which step
          - Whether trajectory is escalating
          - Overall alert level
          - Early warning: how many steps before threshold crossing
        """
        probs = np.array([s.infiltration_prob for s in steps])
        peak_idx = int(np.argmax(probs))
        peak_prob = float(probs[peak_idx])

        is_escalating = self._detect_escalation(probs)
        overall_alert = self._classify_alert(peak_prob)

        # Tighten alert if escalating
        if is_escalating and overall_alert == "medium":
            overall_alert = "high"
        elif is_escalating and overall_alert == "high":
            overall_alert = "critical"

        early_warning = self._compute_early_warning(
            probs, threshold=self.thresholds["medium"]
        )

        trajectory = self._describe_trajectory(probs, steps)

        return RolloutResult(
            steps=steps,
            peak_infiltration_prob=peak_prob,
            peak_step=peak_idx + 1,
            is_escalating=is_escalating,
            overall_alert=overall_alert,
            trajectory_summary=trajectory,
            early_warning_steps=early_warning,
        )

    def aggregate_ensemble(
        self,
        all_rollouts: list[list[StepScore]],
    ) -> RolloutResult:
        """
        Aggregate multiple stochastic rollouts into an ensemble result
        with mean, confidence intervals, and uncertainty.

        Parameters
        ----------
        all_rollouts : list of M rollout step lists, each of length K
        """
        K = len(all_rollouts[0])
        M = len(all_rollouts)

        # Stack infiltration probs: (M, K)
        prob_matrix = np.array(
            [[s.infiltration_prob for s in rollout] for rollout in all_rollouts]
        )
        mean_probs = prob_matrix.mean(axis=0)
        lower_probs = np.percentile(prob_matrix, 5, axis=0)
        upper_probs = np.percentile(prob_matrix, 95, axis=0)

        # Use mean rollout for representative steps
        mean_steps = []
        for k in range(K):
            step_probs = prob_matrix[:, k]
            mitre_probs_all = np.array(
                [all_rollouts[m][k].mitre_probs for m in range(M)]
            ).mean(axis=0)
            confidence = 1.0 - float(np.std(step_probs))

            step_score = self.score_step(
                step=k + 1,
                infiltration_prob=float(mean_probs[k]),
                mitre_probs=mitre_probs_all.tolist(),
                confidence=max(0.0, confidence),
                top_features=all_rollouts[0][k].top_features,
            )
            mean_steps.append(step_score)

        result = self.score_rollout(mean_steps)
        result.ensemble_mean = mean_probs
        result.ensemble_lower = lower_probs
        result.ensemble_upper = upper_probs
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _classify_alert(self, prob: float) -> str:
        if prob >= self.thresholds["critical"]:
            return "critical"
        elif prob >= self.thresholds["high"]:
            return "high"
        elif prob >= self.thresholds["medium"]:
            return "medium"
        elif prob >= self.thresholds["low"]:
            return "low"
        return "none"

    @staticmethod
    def _detect_escalation(probs: np.ndarray, min_increase: float = 0.10) -> bool:
        """True if probs show a consistent upward trend."""
        if len(probs) < 3:
            return False
        # Check if the last half of probabilities is higher than the first half
        mid = len(probs) // 2
        first_half_mean = probs[:mid].mean()
        second_half_mean = probs[mid:].mean()
        return float(second_half_mean - first_half_mean) > min_increase

    @staticmethod
    def _compute_early_warning(probs: np.ndarray, threshold: float) -> int:
        """
        How many steps before the probability first crosses the threshold.
        Returns 0 if never crossed.
        """
        for i, p in enumerate(probs):
            if p >= threshold:
                return len(probs) - i
        return 0

    @staticmethod
    def _describe_trajectory(probs: np.ndarray, steps: list[StepScore]) -> str:
        """Generate a human-readable trajectory description."""
        if len(probs) == 0:
            return "No prediction available."

        peak = float(probs.max())
        peak_step = int(np.argmax(probs)) + 1
        peak_stage = steps[int(np.argmax(probs))].mitre_stage

        if peak < 0.20:
            return "Network trajectory appears benign. No significant threat indicators detected."

        trend = "escalating" if probs[-1] > probs[0] else (
            "de-escalating" if probs[-1] < probs[0] else "stable"
        )

        return (
            f"Threat trajectory is {trend}. "
            f"Peak infiltration probability {peak:.1%} predicted at step {peak_step} "
            f"({peak_stage} stage). "
            + (
                "Immediate defender action recommended."
                if peak >= 0.65
                else "Continued monitoring advised."
            )
        )
