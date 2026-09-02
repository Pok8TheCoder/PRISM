"""
Predictive Scoring, Trajectory Risk Analysis, and Alert Generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from src.prediction.attack_mapper import get_stage_guidance
from src.utils.logger import setup_logger

logger = setup_logger("Scoring")

DEFAULT_THRESHOLDS = {
    "critical": 0.80,
    "high": 0.60,
    "medium": 0.35,
    "low": 0.15
}


def is_escalating(trajectory: List[Dict[str, Any]], min_steps: int = 3) -> bool:
    """Checks if infiltration probability is monotonically or consistently rising."""
    if len(trajectory) < min_steps:
        return False
    probs = [step["infiltration_prob"] for step in trajectory[:min_steps]]
    # Check if last is significantly higher than first
    return probs[-1] - probs[0] >= 0.20 and all(b >= a - 0.05 for a, b in zip(probs[:-1], probs[1:]))


def compute_trajectory_risk(trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes a multi-step forward simulation trajectory to compute overall risk indicators.
    """
    if not trajectory:
        return {"max_prob": 0.0, "risk_level": "LOW", "escalating": False, "peak_stage": "Benign"}

    probs = [s["infiltration_prob"] for s in trajectory]
    max_prob = float(np.max(probs))
    avg_prob = float(np.mean(probs))
    escalating = is_escalating(trajectory)

    # Find the stage with highest probability at the peak risk window
    peak_idx = int(np.argmax(probs))
    peak_stage = trajectory[peak_idx]["mitre_stage"]

    if max_prob >= DEFAULT_THRESHOLDS["critical"] or (max_prob >= DEFAULT_THRESHOLDS["high"] and escalating):
        risk_level = "CRITICAL"
    elif max_prob >= DEFAULT_THRESHOLDS["high"]:
        risk_level = "HIGH"
    elif max_prob >= DEFAULT_THRESHOLDS["medium"]:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "max_prob": max_prob,
        "avg_prob": avg_prob,
        "risk_level": risk_level,
        "escalating": escalating,
        "peak_step": peak_idx + 1,
        "peak_stage": peak_stage
    }


def generate_alerts(
    trajectory: List[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    Generates actionable security alerts based on forward rollout predictions.
    """
    th = thresholds or DEFAULT_THRESHOLDS
    risk_summary = compute_trajectory_risk(trajectory)
    alerts = []

    for step in trajectory:
        k = step["step"]
        prob = step["infiltration_prob"]
        stage = step["mitre_stage"]

        if prob >= th["medium"]:
            guidance = get_stage_guidance(stage)
            severity = "CRITICAL" if prob >= th["critical"] else ("HIGH" if prob >= th["high"] else "MEDIUM")
            
            alerts.append({
                "step_ahead": k,
                "severity": severity,
                "infiltration_prob": prob,
                "mitre_stage": stage,
                "description": guidance["description"],
                "action": guidance["action"]
            })

    if risk_summary["escalating"] and risk_summary["max_prob"] >= th["high"]:
        alerts.insert(0, {
            "step_ahead": risk_summary["peak_step"],
            "severity": "CRITICAL",
            "infiltration_prob": risk_summary["max_prob"],
            "mitre_stage": risk_summary["peak_stage"],
            "description": "Trajectory Escalation: Probability of network compromise increases rapidly over the forecast horizon.",
            "action": "Immediate preventative isolation recommended prior to attack culmination."
        })

    return alerts
