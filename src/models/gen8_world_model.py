"""
PRISM Gen 8 — Decoupled Temporal Transformer + Deep Residual MITRE MLP Expert (Gen8DecoupledMLPWorldModel)
With Discriminative Feature Highway & Contrastive Repulsion for Infiltration Disambiguation.

Key Architectural Inventions:
  1. Discriminative Telemetry Highway (Skip Connection):
     Bypasses temporal compression and feeds 13 uncompressed instantaneous telemetry
     channels (fwd_bwd_ratio, flag_syn_frac, Init Fwd/Bwd Win Bytes, Bwd IAT stats,
     port categories) directly into the Deep Residual MITRE MLP expert.
  2. Balanced Gradient Flow (50% Backprop):
     Allows 50% gradient flow from the MITRE classifier back through temporal attention,
     forcing Transformer attention heads to differentiate unidirectional SYN probes
     from bidirectional session negotiations.
  3. Hierarchical Kill-Chain Pair Disambiguation Heads:
     - Infiltration Pair Head: Reconnaissance (1) vs. Initial Access (2)
     - Egress Pair Head: Lateral Movement (3) vs. Exfiltration (5)
     - Recon/Lateral Pair Head: Reconnaissance (1) vs. Lateral Movement (3)
  4. Multi-Scale Dilated Temporal Convolutions (Horizon = 270s):
     Retains the 6-branch dilated temporal pyramid (k=3, d=1, 2, 4) to capture
     micro-burst floods (5s) and slow APT recon (270s) before self-attention.
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
)
from src.utils.constants import NUM_MITRE_STAGES


class GatedDomainFeatureAdaptor(nn.Module):
    """
    Learned Gated Feature Adaptor for heterogeneous multi-domain telemetry.
    Dynamically weights active sensor channels vs zero-padded dimensions.
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


class DeepResidualMitreMLP(nn.Module):
    """
    Dedicated Deep Residual MLP Expert for fine-grained MITRE ATT&CK attribution.
    Processes fused temporal embeddings along with direct discriminative telemetry.
    """

    def __init__(
        self,
        d_in: int = 576,
        d_hidden: int = 512,
        num_classes: int = NUM_MITRE_STAGES,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden

        # Stage 1: Wide Input Projection
        self.in_proj = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stage 2: Residual Block 1
        self.res1_fc1 = nn.Linear(d_hidden, d_hidden)
        self.res1_norm1 = nn.LayerNorm(d_hidden)
        self.res1_fc2 = nn.Linear(d_hidden, d_hidden)
        self.res1_norm2 = nn.LayerNorm(d_hidden)
        self.res1_drop = nn.Dropout(dropout)

        # Stage 3: Residual Block 2 (with Gated Feature Transformation)
        self.res2_fc1 = nn.Linear(d_hidden, d_hidden)
        self.res2_norm1 = nn.LayerNorm(d_hidden)
        self.res2_fc2 = nn.Linear(d_hidden, d_hidden)
        self.res2_norm2 = nn.LayerNorm(d_hidden)
        self.res2_drop = nn.Dropout(dropout)

        # Stage 4: Output Classifier Heads
        # Primary 7-class MITRE logit head
        self.head_mitre = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2),
            nn.LayerNorm(d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_hidden // 2, num_classes),
        )

        # Specialized Pair Disambiguation Sub-Head 1: Recon (1) vs. Initial Access (2)
        self.pair_infil_head = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Specialized Pair Disambiguation Sub-Head 2: Lateral Movement (3) vs. Exfiltration (5)
        self.pair_egress_head = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Specialized Pair Disambiguation Sub-Head 3: Reconnaissance (1) vs. Lateral Movement (3)
        self.pair_recon_lateral_head = nn.Sequential(
            nn.Linear(d_hidden, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

        # Supervised Contrastive Projection Head (projects to unit sphere S^127)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h: (B, d_in) fused temporal and discriminative representation vector.
        Returns dict with primary mitre logits, pair logits, and contrastive z.
        """
        x = self.in_proj(h)

        # ResBlock 1
        residual = x
        x = F.gelu(self.res1_norm1(self.res1_fc1(x)))
        x = self.res1_drop(x)
        x = self.res1_norm2(self.res1_fc2(x))
        x = F.gelu(x + residual)

        # ResBlock 2
        residual = x
        x = F.gelu(self.res2_norm1(self.res2_fc1(x)))
        x = self.res2_drop(x)
        x = self.res2_norm2(self.res2_fc2(x))
        x = F.gelu(x + residual)

        # Predictions
        logits_mitre = self.head_mitre(x)
        logits_infil_pair = self.pair_infil_head(x)
        logits_egress_pair = self.pair_egress_head(x)
        logits_recon_lateral_pair = self.pair_recon_lateral_head(x)
        z_contrastive = F.normalize(self.contrastive_proj(x), p=2, dim=-1)

        return {
            "pred_mitre": logits_mitre,
            "logits_infil_pair": logits_infil_pair,
            "logits_egress_pair": logits_egress_pair,
            "logits_recon_lateral_pair": logits_recon_lateral_pair,
            "contrastive_z": z_contrastive,
            "mlp_features": x,
        }


class Gen8DecoupledMLPWorldModel(nn.Module):
    """
    PRISM Gen 8 Multi-Task World Model with Temporal Transformer,
    Discriminative Telemetry Highway, and Deep Residual MITRE MLP Expert.
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
        mlp_hidden: int = 512,
        residual_dynamics: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.lookback = lookback
        self.residual_dynamics = residual_dynamics

        # 1. Learnable Gated Domain Feature Adaptor
        self.feature_adaptor = GatedDomainFeatureAdaptor(d_state, d_model, dropout=dropout)

        # 2. Multi-Scale Dilated Temporal Convolution Block (up to 270s horizon)
        self.temporal_conv = MultiScaleTemporalConvBlock(d_model, dropout=dropout)

        # 3. Learnable Positional Encoding
        self.pos_enc = LearnablePositionalEncoding(max_len=lookback + 64, d_model=d_model)

        # 4. Stream 1: Temporal Transformer Encoder (Pristine World Model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # Temporal Attention Pooling for sequence-to-vector aggregation
        self.pooler = TemporalAttentionPooling(d_model, dropout=dropout)

        # Continuous State Dynamics Forecaster (predicts next-step state mean & log-variance)
        self.head_state_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_state),
        )
        self.head_state_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_state),
        )

        # Binary Threat Infiltration Head (attack vs. benign anomaly gating)
        self.head_binary = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # Discriminative Telemetry Highway:
        # Channels: Bwd IAT (53-56), fwd_bwd_ratio (79), bytes/pkt (80), flag_syn_frac (81), flag_ack_frac (82),
        # port categories (87-89), Init Fwd/Bwd Win Bytes (91, 92) -> 13 key micro-signatures
        self.disc_indices = [53, 54, 55, 56, 79, 80, 81, 82, 87, 88, 89, 91, 92]
        self.d_disc = len(self.disc_indices)
        self.d_disc_proj = 64
        self.disc_proj = nn.Sequential(
            nn.Linear(self.d_disc, self.d_disc_proj),
            nn.LayerNorm(self.d_disc_proj),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 5. Stream 2: Dedicated Deep Residual MITRE MLP Expert (with Direct Feature Highway)
        self.mitre_mlp = DeepResidualMitreMLP(
            d_in=d_model * 2 + self.d_disc_proj,
            d_hidden=mlp_hidden,
            num_classes=num_mitre_classes,
            dropout=dropout * 1.5,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        state_seq: torch.Tensor,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        state_seq: (B, T, d_state)
        """
        B, T, D = state_seq.shape

        # 1. Gated domain adaptation + dilated temporal pyramid
        h_proj = self.feature_adaptor(state_seq)            # (B, T, d_model)
        h_conv = self.temporal_conv(h_proj)                 # (B, T, d_model)
        h_pos = self.pos_enc(h_conv)                        # (B, T, d_model)

        # 2. Stream 1: Temporal Transformer Encoder
        h_seq = self.transformer(h_pos)                     # (B, T, d_model)
        h_fused, attn_weights = self.pooler(h_seq)          # (B, d_model)

        # 3. Dynamics Forecasting (Ŝ_{t+1})
        delta_mean = self.head_state_mean(h_fused)
        pred_state_logvar = self.head_state_logvar(h_fused).clamp(-8.0, 2.0)
        if self.residual_dynamics:
            s_t = state_seq[:, -1, :]
            pred_state_mean = s_t + delta_mean
        else:
            pred_state_mean = delta_mean

        # 4. Binary Threat Infiltration Logits
        pred_binary = self.head_binary(h_fused)             # (B, 2)

        # 5. Stream 2: Deep Residual MITRE MLP Expert with Discriminative Telemetry Highway
        #    Fuses instantaneous packet state h_seq[:, -1, :] with full temporal context h_fused
        #    and direct uncompressed discriminative telemetry highway (fwd_bwd_ratio, SYN fraction, TCP windows)
        h_last = h_seq[:, -1, :]
        h_fused_comb = torch.cat([h_fused, h_last], dim=-1)  # (B, d_model * 2 = 512)

        # Telemetry highway skip connection
        if D > max(self.disc_indices):
            x_disc = state_seq[:, -1, self.disc_indices]
        else:
            x_disc = torch.zeros(B, self.d_disc, device=state_seq.device, dtype=state_seq.dtype)
        h_disc = self.disc_proj(x_disc)                        # (B, 64)
        h_mlp_comb = torch.cat([h_fused_comb, h_disc], dim=-1) # (B, 576)

        if self.training:
            h_mlp_input = h_mlp_comb * 0.50 + h_mlp_comb.detach() * 0.50
        else:
            h_mlp_input = h_mlp_comb

        mlp_out = self.mitre_mlp(h_mlp_input)

        return {
            "pred_state_mean": pred_state_mean,
            "pred_state_logvar": pred_state_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": mlp_out["pred_mitre"],
            "logits_infil_pair": mlp_out["logits_infil_pair"],
            "logits_egress_pair": mlp_out["logits_egress_pair"],
            "logits_recon_lateral_pair": mlp_out["logits_recon_lateral_pair"],
            "contrastive_z": mlp_out["contrastive_z"],
            "temporal_attn": attn_weights,
            "latent_h": h_fused,
            "mlp_features": mlp_out["mlp_features"],
        }

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Convenience: return P(attack) scalar per batch item."""
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def sample_next_state(
        self, state_seq: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Sample next-state distribution for K-step rollout."""
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
        """Compute Zero-Day Anomaly Surprise Score (NLL / Mahalanobis Deviation)."""
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
