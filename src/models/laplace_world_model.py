"""
PRISM — Laplace Spatio-Temporal World Model (Gen 10 Architecture)
Tier-1 Deep Neural Engine combining:
  1. Spatio-Temporal Graph-Topology & Directional Invariant Encoder
  2. 4-Layer Causal Transformer with Learned Temporal Lookback
  3. Continuous Next-State Gaussian NLL Dynamics Forecasting Head
  4. Binary Threat Infiltration Head with Asymmetric Focal Calibration
  5. Hierarchical Gated Mixture-of-Experts (MoE) MITRE ATT&CK Stage Router
  6. Topological Supervised Contrastive (SupCon) Representation Learning
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from src.models.gen10_world_model import Gen10SpatioTemporalWorldModel
from src.models.streaming_gen10 import load_gen10_checkpoint

class LaplaceWorldModel(Gen10SpatioTemporalWorldModel):
    """
    Laplace World Model — Tier-1 Spatio-Temporal Neural Foundation.
    Learns continuous network telemetry dynamics, multi-step future forecasting,
    and routing across MITRE ATT&CK kill-chain stages.
    """
    def __init__(
        self,
        d_state: int = 249,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = 20,
        mlp_hidden: int = 512,
        dropout: float = 0.1,
        num_mitre_classes: int = 7,
        residual_dynamics: bool = True,
        num_experts: int = 4,
    ):
        super().__init__(
            d_state=d_state,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            lookback=lookback,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
            num_mitre_classes=num_mitre_classes,
            residual_dynamics=residual_dynamics,
            num_experts=num_experts,
        )


def load_laplace_checkpoint(
    ckpt_path: Optional[Path | str] = None,
    device: torch.device = torch.device("cpu"),
    d_state: int = 249,
) -> LaplaceWorldModel:
    """Load pretrained weights into LaplaceWorldModel."""
    if ckpt_path is None:
        ckpt_path = Path("weights/universal_gen10/world_model_best.pt")
    
    ckpt_path = Path(ckpt_path)
    model = LaplaceWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=20)
    
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ck.get("model_state_dict", ck)
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    return model
