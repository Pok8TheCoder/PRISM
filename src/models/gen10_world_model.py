"""
PRISM Gen 10 — Spatio-Temporal Graph World Model (Gen10SpatioTemporalWorldModel)

Key Architectural Breakthroughs:
  1. Graph Topology & Directional Manifold (Stage 1 & Stage 2):
     Integrates explicit graph invariants (max out-degree, max in-degree, port entropy,
     WAN->LAN ratio, LAN->LAN ratio, reciprocity, bipartite density) through a
     dedicated GraphTopologyExtractor (d_graph = 64) and NativeBipartiteGAT.
  2. Directional & Entropy Feature Highway:
     Extracts 19 instantaneous micro-features (SYN/ACK flag ratios, TCP window sizes,
     forward/backward packet size ratios, backward IAT distributions, port categories).
  3. Fused Spatio-Temporal Representation Vector:
     Combines temporal sequence context (256-d) + instantaneous state (256-d) +
     directional highway (96-d) + topological graph manifold (64-d) -> Total: 672-d.
  4. Deep Residual MITRE Expert with Supervised Contrastive Unit Hypersphere:
     2 wide residual blocks (512-d hidden) with pre-LN, GELU, and SupCon projection.
  5. Hierarchical Residual Routing:
     Ensemble-level log-odds correction that breaks the S1/S2/S3 ambiguity.
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
from src.models.native_gat import GraphTopologyExtractor, DynamicGraphEncoder
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


class DeepResidualMitreExpert(nn.Module):
    """
    Dedicated Deep Residual Expert for fine-grained MITRE ATT&CK attribution.
    Fuses pooled temporal embeddings, instantaneous state, and topological graph manifold.
    """

    def __init__(
        self,
        d_in: int = 672,  # d_model * 2 (512) + d_disc_proj (96) + d_graph_topo (64)
        d_hidden: int = 512,
        num_classes: int = NUM_MITRE_STAGES,
        dropout: float = 0.15,
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

        # Stage 3: Residual Block 2
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
            nn.Linear(d_hidden, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 2),
        )

        # Specialized Pair Disambiguation Sub-Head 2: Recon (1) vs. Lateral Movement (3)
        self.pair_recon_lateral_head = nn.Sequential(
            nn.Linear(d_hidden, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 2),
        )

        # Specialized Pair Disambiguation Sub-Head 3: Lateral Movement (3) vs. Exfiltration (5)
        self.pair_egress_head = nn.Sequential(
            nn.Linear(d_hidden, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 2),
        )

        # Supervised Contrastive Projection Head (projects to unit sphere S^127)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_hidden, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        h: (B, d_in) fused temporal, graph, and discriminative representation vector.
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
        logits_recon_lateral_pair = self.pair_recon_lateral_head(x)
        logits_egress_pair = self.pair_egress_head(x)
        z_contrastive = F.normalize(self.contrastive_proj(x), p=2, dim=-1)

        return {
            "pred_mitre": logits_mitre,
            "logits_infil_pair": logits_infil_pair,
            "logits_recon_lateral_pair": logits_recon_lateral_pair,
            "logits_egress_pair": logits_egress_pair,
            "contrastive_z": z_contrastive,
            "mlp_features": x,
        }


class Gen10SpatioTemporalWorldModel(nn.Module):
    """
    PRISM Gen 10 Spatio-Temporal Graph World Model.
    """

    def __init__(
        self,
        d_state: int = 242,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = 20,
        dropout: float = 0.1,
        mlp_hidden: int = 512,
        num_mitre_classes: int = NUM_MITRE_STAGES,
        residual_dynamics: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback
        self.num_mitre_classes = num_mitre_classes
        self.residual_dynamics = residual_dynamics

        # 1. Gated domain feature adaptor
        self.feature_adaptor = GatedDomainFeatureAdaptor(d_state, d_model, dropout=dropout)

        # 2. Multi-Scale Dilated Temporal Pyramid
        self.temporal_conv = MultiScaleTemporalConvBlock(d_model, dropout=dropout)

        # 3. Learnable Positional Encoding
        self.pos_enc = LearnablePositionalEncoding(max_len=lookback + 64, d_model=d_model)

        # 4. Stream 1: Causal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pooler = TemporalAttentionPooling(d_model, dropout=dropout)

        # Dynamics Heads: Residual Mean & LogVar
        self.head_state_mean = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_state),
        )
        self.head_state_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_state),
        )

        # Binary Threat Infiltration Head
        self.head_binary = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        # 5. Directional Highway (19 key telemetry channels)
        self.disc_indices = [10, 11, 12, 53, 54, 55, 56, 79, 80, 81, 82, 83, 84, 87, 88, 89, 91, 92, 95]
        self.d_disc = len(self.disc_indices)
        self.d_disc_proj = 96
        self.disc_proj = nn.Sequential(
            nn.Linear(self.d_disc, self.d_disc_proj),
            nn.LayerNorm(self.d_disc_proj),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # 6. Graph Topology Manifold Extractor (Stage 1 & Stage 2)
        # Dedicated topological projection for graph invariants
        self.d_graph_topo = 64
        self.graph_extractor = GraphTopologyExtractor(d_in=8, d_out=self.d_graph_topo, dropout=dropout)

        # Exact indices of graph invariants in Gen 10 canonical state vector:
        # [4: port_entropy, 5: max_out_degree, 6: max_in_degree, 7: wan_to_lan,
        #  8: lan_to_lan, 9: lan_to_wan, 10: reciprocity, 11: bipartite_density]
        self.graph_indices = [4, 5, 6, 7, 8, 9, 10, 11]

        # 7. Stream 2: Deep Residual MITRE Expert with Spatio-Temporal Graph Fusion
        self.mitre_mlp = DeepResidualMitreExpert(
            d_in=d_model * 2 + self.d_disc_proj + self.d_graph_topo,  # 512 + 96 + 64 = 672
            d_hidden=mlp_hidden,
            num_classes=num_mitre_classes,
            dropout=dropout,
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
                if m.bias is not None:
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

        # 5. Stream 2: Spatio-Temporal Graph Fusion
        h_last = h_seq[:, -1, :]
        h_fused_comb = torch.cat([h_fused, h_last], dim=-1)  # (B, d_model * 2 = 512)

        # Directional Highway Skip Connection
        if D > max(self.disc_indices):
            x_disc = state_seq[:, -1, self.disc_indices]
        else:
            x_disc = torch.zeros(B, self.d_disc, device=state_seq.device, dtype=state_seq.dtype)
            for i, idx in enumerate(self.disc_indices):
                if idx < D:
                    x_disc[:, i] = state_seq[:, -1, idx]
        h_disc = self.disc_proj(x_disc)  # (B, 96)

        # Graph Topology Manifold Extraction
        x_graph = torch.zeros(B, 8, device=state_seq.device, dtype=state_seq.dtype)
        for i, idx in enumerate(self.graph_indices):
            if idx < D:
                x_graph[:, i] = state_seq[:, -1, idx]
        h_topo = self.graph_extractor(x_graph)  # (B, 64)

        # Fuse Temporal Context (512) + Directional Highway (96) + Graph Topology (64) = 672
        h_mlp_comb = torch.cat([h_fused_comb, h_disc, h_topo], dim=-1)

        # Balanced backprop: 50% gradient flow to temporal backbone
        if self.training:
            h_mlp_input = h_mlp_comb * 0.50 + h_mlp_comb.detach() * 0.50
        else:
            h_mlp_input = h_mlp_comb

        mlp_out = self.mitre_mlp(h_mlp_input)

        out = {
            "pred_state_mean": pred_state_mean,
            "pred_state_logvar": pred_state_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": mlp_out["pred_mitre"],
            "logits_infil_pair": mlp_out["logits_infil_pair"],
            "logits_recon_lateral_pair": mlp_out["logits_recon_lateral_pair"],
            "logits_egress_pair": mlp_out["logits_egress_pair"],
            "contrastive_z": mlp_out["contrastive_z"],
            "temporal_attn": attn_weights,
            "latent_h": h_fused,
            "graph_topo_h": h_topo,
            "mlp_features": mlp_out["mlp_features"],
        }

        # Apply Hierarchical Residual Routing if in eval mode
        if not self.training:
            routed_mitre = self.hierarchical_route(
                pred_mitre=mlp_out["pred_mitre"],
                logits_infil_pair=mlp_out["logits_infil_pair"],
                logits_recon_lateral_pair=mlp_out["logits_recon_lateral_pair"],
                logits_egress_pair=mlp_out["logits_egress_pair"],
            )
            out["pred_mitre_routed"] = routed_mitre

        return out

    def hierarchical_route(
        self,
        pred_mitre: torch.Tensor,
        logits_infil_pair: torch.Tensor,
        logits_recon_lateral_pair: torch.Tensor,
        logits_egress_pair: torch.Tensor,
        alpha: float = 0.30,
    ) -> torch.Tensor:
        """
        Refines 7-stage primary MITRE logits using specialized pairwise expert heads.
        Resolves the S1 (Recon) / S2 (Initial Access) / S3 (Lateral) / S5 (Exfil) confusion.
        """
        routed = pred_mitre.clone()

        # Pair 1: S1 (Recon) vs S2 (Initial Access) -> logits_infil_pair: [0=S1, 1=S2]
        delta_12 = F.softmax(logits_infil_pair, dim=-1)  # (B, 2)
        diff_12 = delta_12[:, 1] - delta_12[:, 0]        # >0 favors S2, <0 favors S1
        routed[:, 1] = routed[:, 1] - alpha * diff_12
        routed[:, 2] = routed[:, 2] + alpha * diff_12

        # Pair 2: S1 (Recon) vs S3 (Lateral) -> logits_recon_lateral_pair: [0=S1, 1=S3]
        delta_13 = F.softmax(logits_recon_lateral_pair, dim=-1)
        diff_13 = delta_13[:, 1] - delta_13[:, 0]        # >0 favors S3, <0 favors S1
        routed[:, 1] = routed[:, 1] - alpha * diff_13
        routed[:, 3] = routed[:, 3] + alpha * diff_13

        # Pair 3: S3 (Lateral) vs S5 (Exfiltration) -> logits_egress_pair: [0=S3, 1=S5]
        delta_35 = F.softmax(logits_egress_pair, dim=-1)
        diff_35 = delta_35[:, 1] - delta_35[:, 0]        # >0 favors S5, <0 favors S3
        routed[:, 3] = routed[:, 3] - alpha * diff_35
        routed[:, 5] = routed[:, 5] + alpha * diff_35

        return routed

    def compute_zero_day_anomaly(
        self,
        state_seq: torch.Tensor,
        next_state: torch.Tensor,
        sigma_threshold: float = 3.0,
    ) -> dict:
        """
        Compute Zero-Day Anomaly Surprise Score (NLL / Mahalanobis Deviation).
        Leverages the stochastic dynamics forecasting heads (head_state_mean, head_state_logvar)
        to identify unseen out-of-distribution attacks without requiring known signatures.
        """
        if state_seq.dim() == 2:
            state_seq = state_seq.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)

        with torch.no_grad():
            out = self.forward(state_seq)
            mean = out["pred_state_mean"]
            logvar = out["pred_state_logvar"].clamp(-8.0, 2.0)
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
