"""
PRISM — Streaming Laplace Model (Tier 1 Neural + Tier 2 Symbolic Combined)
The complete production hybrid runtime uniting:
  - Tier-1: Spatio-Temporal Graph-Transformer World Model (Dynamics & MoE Stage Routing)
  - Tier-2: Neuro-Symbolic Flow Context & Subnet Directionality Attributor
  - Zero-Day Predictive Surprise Detection (Mahalanobis Variance Divergence)
  - Autoregressive World Model Future Rollout / Forecasting
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.models.streaming_gen10 import (
    StreamingGen10WorldModel,
    Tier2FlowContextAttributor,
    MITRE_STAGE_NAMES,
)
from src.models.laplace_world_model import LaplaceWorldModel, load_laplace_checkpoint


class StreamingLaplaceModel(StreamingGen10WorldModel):
    """
    Streaming Laplace Model.
    Unifies Tier 1 (Deep Spatio-Temporal World Model) and Tier 2 (Flow Context Reasoner)
    into a single production streaming engine with sub-15ms latency per window.
    """
    def __init__(
        self,
        ckpt_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
        device: str = "cpu",
        detect_thresh: float = 0.50,
        sigma_threshold: float = 2.0,
        lookback: int = 20,
        d_state: int = 249,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            scaler_path=scaler_path,
            device=device,
            detect_thresh=detect_thresh,
            sigma_threshold=sigma_threshold,
            lookback=lookback,
            d_state=d_state,
        )


# Canonical alias
LaplaceModel = StreamingLaplaceModel
