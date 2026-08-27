"""K-step forward simulation, MITRE stage mapping, and driving features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.explain.attribution import explain_prediction
from src.model.attack_catalog import get_mitre_map
from src.model.world_model_multiclass import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    SEQ_LEN,
    predict_live,
)
from src.pipeline.extract import load_traffic_file
from src.pipeline.features import FEATURE_COLS

K_DEFAULT = 5


def k_step_rollout(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    k: int = K_DEFAULT,
) -> list[float]:
    """Autoregressive attack-probability forecast for the next K windows."""
    if len(features) == 0:
        return [0.0] * k

    window = features.astype(np.float32)
    if len(window) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(window), window.shape[1]), dtype=np.float32)
        window = np.vstack([pad, window])
    window = window[-SEQ_LEN:]
    current = ((window - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)

    benign_idx = CLASS_TO_IDX["Benign"]
    probs_out: list[float] = []
    model.eval()
    with torch.no_grad():
        for _ in range(k):
            x = torch.from_numpy(current).unsqueeze(0).to(device)
            next_state, logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            probs_out.append(float(1.0 - probs[benign_idx]))
            next_row = next_state.cpu().numpy()[0]
            current = np.vstack([current[1:], next_row.reshape(1, -1)])
    return probs_out


def observed_attack_curve(model, scaler, features: np.ndarray, device: torch.device) -> list[float]:
    """Attack probability along the observed window (one score per flow, strided)."""
    if len(features) == 0:
        return []
    scores = []
    step = max(1, len(features) // 20)
    for end in range(min(SEQ_LEN, len(features)), len(features) + 1, step):
        chunk = features[:end]
        pred_idx, probs, _ = predict_live(model, scaler, chunk, device)
        scores.append(float(1.0 - probs[CLASS_TO_IDX["Benign"]]))
    if not scores:
        pred_idx, probs, _ = predict_live(model, scaler, features, device)
        scores = [float(1.0 - probs[CLASS_TO_IDX["Benign"]])]
    return scores


def analyze_features(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    k: int = K_DEFAULT,
) -> dict[str, Any]:
    """Full PS inference output from a feature matrix."""
    if len(features) == 0:
        return {
            "ok": False,
            "message": "No flows extracted from the input file.",
        }

    pred_idx, probs, _ = predict_live(model, scaler, features, device)
    pred_class = CLASS_NAMES[pred_idx]
    mitre_map = get_mitre_map()
    tactic, technique = mitre_map.get(pred_class, ("", ""))
    attack_prob = float(1.0 - probs[CLASS_TO_IDX["Benign"]])
    forecast = k_step_rollout(model, scaler, features, device, k)
    explanation = explain_prediction(model, scaler, features, device, SEQ_LEN)
    observed = observed_attack_curve(model, scaler, features, device)

    top_classes = sorted(
        [(CLASS_NAMES[i], float(probs[i])) for i in range(len(CLASS_NAMES))],
        key=lambda x: x[1],
        reverse=True,
    )[:8]

    return {
        "ok": True,
        "flow_count": int(len(features)),
        "pred_class": pred_class,
        "pred_idx": int(pred_idx),
        "confidence": float(probs[pred_idx]),
        "attack_prob": attack_prob,
        "mitre_tactic": tactic,
        "mitre_technique": technique,
        "class_probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "top_classes": top_classes,
        "observed_curve": observed,
        "forecast": forecast,
        "driving_features": explanation["primary"],
        "explain_method": explanation["method"],
        "attention_features": explanation["attention"],
        "gradient_features": explanation["gradient"],
        "shap_features": explanation["shap"],
        "feature_names": FEATURE_COLS,
    }


def analyze_file(
    path: str | Path,
    model,
    scaler,
    device: torch.device,
    k: int = K_DEFAULT,
) -> dict[str, Any]:
    features = load_traffic_file(path)
    result = analyze_features(model, scaler, features, device, k)
    result["source"] = str(path)
    result["features"] = features
    return result
