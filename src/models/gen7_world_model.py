"""
PRISM Gen 7 — Decoupled Contrastive Domain-Adaptive World Model (Gen7WorldModel)

Key Architectural Inventions:
  1. Learnable Gated Domain Feature Adaptor: Dynamically gates and routes heterogeneous 242-dim telemetry
     (NetFlow, Enterprise Flows, IoT Protocols, Cyber Range) avoiding zero-padding interference.
  2. Triple-Path Decoupled Backbone:
     - Path A (Dynamics Trunk): High-fidelity continuous time-series next-state forecaster (Ŝ_{t+1}, log σ²).
     - Path B (Binary Threat Trunk): Shared Transformer encoder → binary infiltration head (attack vs benign).
     - Path C (MITRE Stage Trunk): Dedicated 2-layer Transformer encoder branching from Path B, specialized
       for fine-grained 7-class MITRE ATT&CK stage classification. Isolates stage gradients from binary head.
  3. Supervised Contrastive Metric Head (SupCon): Projects temporal states to a unit hypersphere (z ∈ S¹²⁷),
     clustering benign traffic tightly while repelling attack vectors radially outward to maximize ROC-AUC.
  4. Temporal Attention Residual Highway: Captures stealth low-and-slow C2 beacons and 5s micro-burst floods.
"""

from __future__ import annotations

import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    MultiScaleTemporalConvBlock,
    TemporalAttentionPooling,
    LearnablePositionalEncoding,
    AsymmetricFocalLoss,
    RobustDynamicsLoss,
)
from src.utils.constants import NUM_MITRE_STAGES


class GatedDomainFeatureAdaptor(nn.Module):
    """
    Learned Gated Feature Adaptor for heterogeneous multi-domain telemetry.
    Dynamically identifies active sensor channels vs zero-padded domains
    (e.g., CTU-13 NetFlow 14 fields vs CIC-IDS 80 fields vs IoT 46 fields).
    """

    def __init__(self, d_state: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_state, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_state) -> (batch, seq_len, d_model)"""
        projected = self.proj(x)
        gated = self.gate(x)
        out = projected * gated
        return self.dropout(self.norm(out))


class SupConProjectionHead(nn.Module):
    """
    Supervised Contrastive Metric Projection Head.
    Projects representation to unit hypersphere z ∈ S¹²⁷ for contrastive margin separation.
    """

    def __init__(self, d_model: int, d_proj: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_proj),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects and L2-normalizes to unit sphere."""
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)


class Gen7DecoupledWorldModel(nn.Module):
    """
    PRISM Gen 7 Decoupled Multi-Task World Model.
    Separates continuous physics forecasting from threat classification to eliminate gradient conflict.
    """

    def __init__(
        self,
        d_state: int = 242,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = 20,
        dropout: float = 0.1,
        num_mitre_classes: int = NUM_MITRE_STAGES,
        d_contrastive: int = 128,
        residual_dynamics: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback
        self.num_mitre_classes = num_mitre_classes
        self.residual_dynamics = residual_dynamics

        # ------------------------------------------------------------------
        # 1. Shared Input Ingestion: Gated Adaptor + Multi-Scale Inception
        # ------------------------------------------------------------------
        self.feature_adaptor = GatedDomainFeatureAdaptor(d_state, d_model, dropout=dropout)
        self.inception = MultiScaleTemporalConvBlock(d_model=d_model, dropout=dropout)
        self.pos_enc = LearnablePositionalEncoding(max_len=lookback + 64, d_model=d_model)

        # ------------------------------------------------------------------
        # 2. Path A: Continuous Dynamics Transformer Trunk (Physics Forecasting)
        # ------------------------------------------------------------------
        dyn_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dyn_transformer = nn.TransformerEncoder(
            dyn_encoder_layer,
            num_layers=max(2, n_layers // 2),
            enable_nested_tensor=False,
        )
        
        self.head_state_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_state),
        )
        self.head_state_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_state),
        )

        # ------------------------------------------------------------------
        # 3. Path B: Binary Threat Discriminator Trunk (Infiltration: attack vs benign)
        # ------------------------------------------------------------------
        threat_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.threat_transformer = nn.TransformerEncoder(
            threat_encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.threat_pooler = TemporalAttentionPooling(d_model=d_model, dropout=dropout)

        # Contrastive Metric Space
        self.contrastive_head = SupConProjectionHead(d_model=d_model, d_proj=d_contrastive)

        # Binary classification head
        self.head_binary = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # ------------------------------------------------------------------
        # 4. Path C: Dedicated MITRE Stage Encoder (branches from Path B output)
        #    2 additional Transformer layers give MITRE its own representation
        #    space, decoupled from the binary head's gradients.
        # ------------------------------------------------------------------
        mitre_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mitre_transformer = nn.TransformerEncoder(
            mitre_encoder_layer,
            num_layers=2,          # 2 dedicated MITRE-specialization layers
            enable_nested_tensor=False,
        )
        self.mitre_pooler = TemporalAttentionPooling(d_model=d_model, dropout=dropout)

        # MITRE stage classification head
        self.head_mitre = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_mitre_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Initialize delta dynamics mean head near zero for smooth residual learning
        if self.residual_dynamics:
            nn.init.normal_(self.head_state_mean[-1].weight, std=1e-3)
            nn.init.zeros_(self.head_state_mean[-1].bias)
        # Initialize variance head to small negative values (initial var ≈ 0.1)
        nn.init.constant_(self.head_state_logvar[-1].weight, 0.0)
        nn.init.constant_(self.head_state_logvar[-1].bias, -2.0)

    def forward(
        self,
        state_seq: torch.Tensor,
        return_attention: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        state_seq: (batch, lookback, d_state)
        Returns dictionary of all predictions + contrastive embeddings + temporal attention.
        """
        B, T, D = state_seq.shape

        # 1. Gated Ingestion & Temporal Convolutions
        h_adapted = self.feature_adaptor(state_seq)        # (B, T, d_model)
        h_conv = self.inception(h_adapted)                 # (B, T, d_model)
        h_pos = self.pos_enc(h_conv)                       # (B, T, d_model)

        # 2. Path A: Continuous Dynamics Forecasting (Residual: S_{t+1} = S_t + Delta S_t)
        h_dyn = self.dyn_transformer(h_pos)                # (B, T, d_model)
        h_dyn_last = h_dyn[:, -1, :]                       # (B, d_model)
        delta_mean = self.head_state_mean(h_dyn_last)      # (B, d_state)
        pred_state_logvar = self.head_state_logvar(h_dyn_last).clamp(-6.0, 3.0)

        if self.residual_dynamics:
            s_t = state_seq[:, -1, :]                      # (B, d_state) instantaneous state
            pred_state_mean = s_t + delta_mean
        else:
            pred_state_mean = delta_mean

        # 3. Path B: Binary Threat Discriminator (shared threat backbone)
        h_threat = self.threat_transformer(h_pos)          # (B, T, d_model)
        h_fused, attn_weights = self.threat_pooler(h_threat) # (B, d_model)

        # Binary infiltration logits
        pred_binary = self.head_binary(h_fused)            # (B, 2)

        # 4. Path C: Dedicated MITRE Stage Encoder
        #    Input: h_threat sequence (shared threat features) → 2 MITRE-only layers
        #    Soft stop-gradient: allow 15% of MITRE gradients to flow back into the
        #    shared backbone so it learns features useful for MITRE stage classification,
        #    while preventing full gradient interference with binary head.
        if self.training:
            h_mitre_input = h_threat * 0.15 + h_threat.detach() * 0.85
        else:
            h_mitre_input = h_threat
        h_mitre_seq = self.mitre_transformer(h_mitre_input)
        h_mitre, mitre_attn_weights = self.mitre_pooler(h_mitre_seq)  # (B, d_model)
        pred_mitre = self.head_mitre(h_mitre)              # (B, num_mitre_classes)

        # Contrastive Latent Embedding (derived directly from Path C MITRE representation)
        z_contrastive = self.contrastive_head(h_mitre)     # (B, d_proj)

        return {
            "pred_state_mean": pred_state_mean,
            "pred_state_logvar": pred_state_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "contrastive_z": z_contrastive,
            "temporal_attn": attn_weights,
            "temporal_pool_weights": attn_weights,
            "mitre_attn": mitre_attn_weights,
            "latent_h": h_fused,
            "hidden": h_fused,
            "hidden_seq": h_threat,
            "attention_weights": attn_weights if return_attention else None,
        }

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Convenience: return P(attack) scalar per batch item."""
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def sample_next_state(
        self, state_seq: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Sample (or take mean of) the predicted next state distribution for K-step rollout.
        """
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-8.0, 2.0)
        std = (0.5 * logvar).exp()
        return mean + torch.randn_like(std) * std

    def compute_zero_day_anomaly(
        self,
        state_seq: torch.Tensor,
        next_state: torch.Tensor,
        sigma_threshold: float = 3.0,
    ) -> dict:
        """
        Compute Zero-Day Anomaly Surprise Score (NLL / Mahalanobis Deviation).
        """
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(state_seq)
            mean = out["pred_state_mean"]
            logvar = out["pred_state_logvar"].clamp(-6.0, 3.0)
            var = logvar.exp()

            sq_err = (next_state - mean) ** 2
            per_feature_surprise = sq_err / (var + 1e-6)
            mean_surprise = per_feature_surprise.mean(dim=-1)

            top_vals, top_indices = torch.topk(
                per_feature_surprise, k=min(10, self.d_state), dim=-1
            )

        return {
            "surprise_score": mean_surprise.cpu().numpy(),
            "per_feature_surprise": per_feature_surprise.cpu().numpy(),
            "is_zero_day_alert": (mean_surprise > (sigma_threshold ** 2)).cpu().numpy(),
            "top_anomalous_indices": top_indices.cpu().numpy(),
            "top_anomalous_values": top_vals.cpu().numpy(),
        }
