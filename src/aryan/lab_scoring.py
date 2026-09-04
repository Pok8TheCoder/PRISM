"""Proxy scoring for the live adversarial attack lab (plan section 5).

Computed post-hoc from each streaming system's per-window trace plus the
objective bot's real, independently-verified event log -- there is no real
blocking/mitigation in the lab, so "win/loss" and "benign harm" are proxies
derived from each system's P(attack) trace vs. ground truth:

  - Detection F1/precision/recall/FPR: window-level binary attack/benign.
  - Time-to-detect (TTD): first window at/after the attacker's first packet
    where P(attack) >= 0.5 *and* that window is genuinely attack traffic
    (a true positive), in both window count and wall-clock seconds.
  - "Lost to attacker": at the round's real exploit-success event (the
    objective bot's own verified post_exploit success), was P(attack) < 0.5
    for that system in the window covering/just before that moment?
  - "Hurt normal users": count of benign-only windows (no attacker overlap,
    including warm-up/cooldown) where P(attack) >= 0.5 -- a false positive
    that would have blocked a real user.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DETECT_THRESHOLD = 0.5


def compute_round_scores(
    trace: list[dict],
    events: list[dict],
    system_ids: list[str],
    detect_threshold: float = DETECT_THRESHOLD,
) -> dict[str, dict]:
    if not trace:
        return {sid: _empty_score() for sid in system_ids}

    true_bins = [int(w["true_bin"]) for w in trace]
    t_starts = [float(w["t_start"]) for w in trace]

    attack_idxs = [i for i, b in enumerate(true_bins) if b == 1]
    t_attack_start_idx = attack_idxs[0] if attack_idxs else None
    t_attack_start_wall = t_starts[t_attack_start_idx] if t_attack_start_idx is not None else None

    success_events = [e for e in events if e.get("stage") == "post_exploit" and e.get("success")]
    t_success = min((float(e["ts"]) for e in success_events), default=None)
    idx_before_success = None
    if t_success is not None:
        candidates = [i for i, t in enumerate(t_starts) if t <= t_success]
        idx_before_success = max(candidates) if candidates else None

    scores: dict[str, dict] = {}
    for sid in system_ids:
        p_atts = [float(w["systems"].get(sid, {}).get("p_att", 0.0)) for w in trace]
        pred = [1 if p >= detect_threshold else 0 for p in p_atts]

        tp = sum(1 for pb, tb in zip(pred, true_bins) if pb == 1 and tb == 1)
        fp = sum(1 for pb, tb in zip(pred, true_bins) if pb == 1 and tb == 0)
        tn = sum(1 for pb, tb in zip(pred, true_bins) if pb == 0 and tb == 0)
        fn = sum(1 for pb, tb in zip(pred, true_bins) if pb == 0 and tb == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        ttd_windows, ttd_sec = None, None
        if t_attack_start_idx is not None:
            for i in range(t_attack_start_idx, len(trace)):
                if pred[i] == 1 and true_bins[i] == 1:
                    ttd_windows = i - t_attack_start_idx
                    ttd_sec = t_starts[i] - t_attack_start_wall
                    break

        lost_to_attacker = None
        if idx_before_success is not None:
            lost_to_attacker = pred[idx_before_success] == 0

        benign_harm = sum(
            1 for w, pb, tb in zip(trace, pred, true_bins)
            if tb == 0 and pb == 1 and w.get("phase") != "warmup"
        )

        scores[sid] = {
            "binary_f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "ttd_windows": ttd_windows,
            "ttd_sec": ttd_sec,
            "detected": ttd_windows is not None,
            "lost_to_attacker": lost_to_attacker,
            "benign_harm_count": benign_harm,
            "n_windows": len(trace),
        }
    return scores


def _empty_score() -> dict:
    return {
        "binary_f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0,
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "ttd_windows": None, "ttd_sec": None, "detected": False,
        "lost_to_attacker": None, "benign_harm_count": 0, "n_windows": 0,
    }


def aggregate_leaderboard(round_files: list[Path]) -> dict[str, Any]:
    per_system: dict[str, dict] = defaultdict(lambda: {
        "rounds": 0, "wins": 0, "losses": 0, "detections": 0,
        "f1": [], "ttd_sec": [], "benign_harm": 0,
    })
    rounds_summary = []

    for rf in sorted(round_files):
        data = json.loads(Path(rf).read_text())
        scores = data.get("scores", {})
        round_entry = {
            "objective": data.get("objective"),
            "evasion": data.get("evasion"),
            "round_id": data.get("round_id"),
            "n_windows": data.get("n_windows"),
            "systems": scores,
        }
        for sid, sc in scores.items():
            agg = per_system[sid]
            agg["rounds"] += 1
            agg["f1"].append(sc.get("binary_f1", 0.0))
            agg["benign_harm"] += sc.get("benign_harm_count", 0)
            if sc.get("detected"):
                agg["detections"] += 1
            if sc.get("ttd_sec") is not None:
                agg["ttd_sec"].append(sc["ttd_sec"])
            if sc.get("lost_to_attacker") is True:
                agg["losses"] += 1
            elif sc.get("lost_to_attacker") is False:
                agg["wins"] += 1
        rounds_summary.append(round_entry)

    leaderboard = []
    for sid, agg in per_system.items():
        leaderboard.append({
            "system": sid,
            "rounds": agg["rounds"],
            "wins": agg["wins"],
            "losses": agg["losses"],
            "detections": agg["detections"],
            "detection_rate": agg["detections"] / agg["rounds"] if agg["rounds"] else 0.0,
            "mean_f1": float(np.mean(agg["f1"])) if agg["f1"] else 0.0,
            "mean_ttd_sec": float(np.mean(agg["ttd_sec"])) if agg["ttd_sec"] else None,
            "total_benign_harm": agg["benign_harm"],
        })
    leaderboard.sort(key=lambda r: (-r["wins"], r["losses"], -r["mean_f1"]))

    return {"leaderboard": leaderboard, "rounds": rounds_summary}
