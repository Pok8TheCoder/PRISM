"""
Feature Attribution and Interpretability Reports (V2 Architecture - 286 Dimensions):
Computes gradient-based saliency attributions to generate structured incident response explanations.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from src.utils.constants import UNIFIED_FLOW_FEATURES, MEDIAN_FEATURE_INDICES, MITRE_STAGES_INV
from src.prediction.attack_mapper import get_stage_guidance


def get_feature_names_292() -> List[str]:
    """Generates descriptive names for all 292 dimensions of the state vector."""
    names = []
    # 64 means
    for f in UNIFIED_FLOW_FEATURES:
        names.append(f"{f}_mean")
    # 64 stds
    for f in UNIFIED_FLOW_FEATURES:
        names.append(f"{f}_std")
    # 64 maxs
    for f in UNIFIED_FLOW_FEATURES:
        names.append(f"{f}_max")
    # 64 mins
    for f in UNIFIED_FLOW_FEATURES:
        names.append(f"{f}_min")
    # 20 medians
    for idx in MEDIAN_FEATURE_INDICES:
        f = UNIFIED_FLOW_FEATURES[idx]
        names.append(f"{f}_median")
    # 16 macro & graph descriptors
    names.extend([
        "log_flow_volume",
        "tcp_fraction",
        "udp_fraction",
        "icmp_fraction",
        "other_proto_fraction",
        "dst_port_entropy",
        "syn_ack_ratio",
        "down_up_ratio_mean",
        "burst_factor",
        "log_syn_volume",
        "max_src_out_degree_log",
        "max_dst_in_degree_log",
        "src_ip_entropy",
        "dst_ip_entropy",
        "graph_density_ratio",
        "one_way_edge_ratio"
    ])
    return names


FEATURE_NAMES_292 = get_feature_names_292()
FEATURE_NAMES_286 = FEATURE_NAMES_292  # Compatibility alias
FEATURE_NAMES_110 = FEATURE_NAMES_292  # Compatibility alias



def compute_gradient_saliency(
    model: torch.nn.Module,
    state_seq: torch.Tensor,
    target_class: int = 1
) -> np.ndarray:
    """
    Computes gradient-based feature attribution: d(P(attack)) / d(Input).
    Returns (286,) importance vector for the most recent state vector (t).
    """
    model.eval()
    if state_seq.ndim == 2:
        state_seq = state_seq.unsqueeze(0)

    x = state_seq.clone().detach().requires_grad_(True)
    (
        pred_mean,
        pred_logvar,
        pred_attack_logits,
        pred_mitre_logits,
        pred_frac,
        _
    ) = model(x)

    attack_score = pred_attack_logits[0, target_class]
    attack_score.backward()

    # Gradients with respect to the latest time step (t)
    grad_at_t = x.grad[0, -1, :].abs().cpu().numpy()
    
    # Avoid zero sum
    norm_val = np.sum(grad_at_t) + 1e-12
    importance = grad_at_t / norm_val
    return importance


def explain_state_prediction(
    model: torch.nn.Module,
    state_seq: torch.Tensor,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Generates a structured human-readable explainability report for the current prediction.
    """
    if state_seq.ndim == 2:
        state_seq = state_seq.unsqueeze(0)

    with torch.no_grad():
        _, _, p_atk, p_mit, p_frac, attn = model(state_seq)
        
        prob_attack = float(torch.softmax(p_atk, dim=-1)[0, 1].item())
        pred_stage_id = int(torch.argmax(p_mit, dim=-1)[0].item())
        stage_name = MITRE_STAGES_INV.get(pred_stage_id, "Unknown")
        predicted_fraction = float(p_frac[0].item())

    # Saliency
    saliency = compute_gradient_saliency(model, state_seq, target_class=1)
    feat_names = FEATURE_NAMES_286 if len(saliency) == 286 else FEATURE_NAMES_286[:len(saliency)]
    top_indices = np.argsort(saliency)[::-1][:top_k]

    top_features = [
        {
            "rank": rank + 1,
            "feature_index": int(idx),
            "feature_name": feat_names[idx] if idx < len(feat_names) else f"feature_{idx}",
            "importance": float(saliency[idx]),
            "importance_pct": f"{saliency[idx] * 100:.1f}%"
        }
        for rank, idx in enumerate(top_indices)
    ]

    guidance = get_stage_guidance(stage_name)

    return {
        "infiltration_probability": prob_attack,
        "predicted_mitre_stage": stage_name,
        "mitre_stage_code": pred_stage_id,
        "predicted_attack_fraction": predicted_fraction,
        "is_alarm_triggered": prob_attack >= 0.70,
        "top_features": top_features,
        "recommended_action": guidance["action"],
        "severity": guidance["severity"],
        "description": guidance["description"]
    }
