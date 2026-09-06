"""
PRISM Generation 6: Google TimesFM Cyber World Model Architecture (src/models/timesfm_model.py)
Adapts Google Research's TimesFM (Time Series Foundation Model) patch-level causal attention
for multivariate cyber threat state sequence dynamics and multi-stage MITRE kill-chain classification.

Key Inventions:
1. Patch-Level Temporal Tokenizer (slices lookback sequence into multi-step temporal patches).
2. TimesFM Pre-LN Causal Transformer Backbone (4 layers, 8 heads, d_model=256).
3. Patch-Attention Threat Pooling (condenses multi-patch tactical timeline into h_fused).
4. Residual Next-State Dynamics Head: S_{t+1} = S_t + ΔS_t.
5. Dual Multi-Task Heads: Asymmetric Threat Infiltration + 7-Stage MITRE Kill-Chain.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    StatePredictionHead,
    ClassificationHead,
    TemporalAttentionPooling,
    generate_causal_mask,
)
from src.utils.constants import NUM_MITRE_STAGES

logger = logging.getLogger("prism.models.timesfm")


class PatchTokenizer(nn.Module):
    """
    TimesFM Patch Tokenizer.
    Converts a continuous multivariate sequence of shape (B, L, D_state) into
    discrete temporal patches of shape (B, N_patches, patch_len * D_state).
    """

    def __init__(self, d_state: int, patch_len: int = 4, patch_stride: int = 2):
        super().__init__()
        self.d_state = d_state
        self.patch_len = patch_len
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D_state)
        Returns: (B, N_patches, patch_len * D_state)
        """
        B, L, D = x.shape
        # Unfold along time dimension L
        # x_unfold: (B, D, N_patches, patch_len)
        x_trans = x.transpose(1, 2)  # (B, D, L)
        x_unfold = x_trans.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
        # Permute to (B, N_patches, patch_len, D) -> flatten to (B, N_patches, patch_len * D)
        x_patches = x_unfold.permute(0, 2, 3, 1).contiguous()
        N_patches = x_patches.shape[1]
        x_flat_patches = x_patches.view(B, N_patches, self.patch_len * D)
        return x_flat_patches


class PatchEmbedding(nn.Module):
    """
    MLP Patch Projection from flattened patch dimensions (P * D_state) to d_model.
    """

    def __init__(self, patch_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches)


class TimesFMPositionalEncoding(nn.Module):
    """Learnable position encodings for patch tokens."""

    def __init__(self, max_patches: int = 64, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N_patches, d_model)
        seq_len = x.size(1)
        return self.dropout(x + self.pos_embed[:, :seq_len, :])


class Gen6_TimesFMWorldModel(nn.Module):
    """
    PRISM Generation 6: Google TimesFM Cyber World Model.
    Processes multivariate network telemetry via patch-level causal attention.
    """

    def __init__(
        self,
        d_state: int = 242,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = 20,
        patch_len: int = 4,
        patch_stride: int = 2,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
        residual_dynamics: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.num_mitre_stages = num_mitre_stages
        self.residual_dynamics = residual_dynamics

        # 1. TimesFM Patch Tokenizer
        self.patch_tokenizer = PatchTokenizer(
            d_state=d_state, patch_len=patch_len, patch_stride=patch_stride
        )
        patch_dim = patch_len * d_state

        # 2. MLP Patch Embedding
        self.patch_embedding = PatchEmbedding(patch_dim, d_model, dropout=dropout)

        # 3. Patch Positional Encoding
        self.pos_enc = TimesFMPositionalEncoding(max_patches=64, d_model=d_model, dropout=dropout)

        # 4. TimesFM Pre-LN Causal Transformer Backbone
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

        # 5. Patch-Level Temporal Attention Pooling (fuses N_patches -> h_fused)
        self.patch_pool = TemporalAttentionPooling(d_model, dropout=dropout)

        # 6. Residual State Dynamics Head
        self.state_head = StatePredictionHead(d_model, d_state, head_dropout)

        # 7. Multi-Task Threat Classification Heads
        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(d_model, num_mitre_stages, head_dropout)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "Gen6_TimesFMWorldModel: d_state=%d, d_model=%d, patch_len=%d, "
            "stride=%d, layers=%d, heads=%d, params=%.2fM",
            d_state, d_model, patch_len, patch_stride, n_layers, n_heads, n_params / 1e6,
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Near-zero init for residual delta prediction head
        if hasattr(self.state_head, "mean_head"):
            last_linear = self.state_head.mean_head[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, std=1e-3)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        state_seq: torch.Tensor,
        return_attention: bool = False,
    ) -> dict:
        """
        Forward pass.
        state_seq: (B, L, D_state) — continuous sequence of past network states
        """
        B, L, D = state_seq.shape
        device = state_seq.device

        # Step 1: Tokenize sequence into temporal patches
        patches = self.patch_tokenizer(state_seq)  # (B, N_patches, patch_dim)
        N_patches = patches.size(1)

        # Step 2: Project patches to latent embedding space
        h_patches = self.patch_embedding(patches)  # (B, N_patches, d_model)
        h_patches = self.pos_enc(h_patches)

        # Step 3: Apply Causal Mask over patch sequence
        causal_mask = generate_causal_mask(N_patches, device)

        # Step 4: TimesFM Causal Transformer Encoder
        h_encoded = self.transformer(h_patches, mask=causal_mask)  # (B, N_patches, d_model)

        # Step 5: Patch-Attention Pooling (fusing multi-patch tactical context)
        h_fused, patch_attn = self.patch_pool(h_encoded)  # (B, d_model), (B, N_patches)

        # Step 6: Next-State Dynamics Prediction (Residual: S_{t+1} = S_t + ΔS_t)
        mean_delta, logvar = self.state_head(h_fused)
        if self.residual_dynamics:
            s_t = state_seq[:, -1, :]  # instantaneous latest state
            pred_state_mean = s_t + mean_delta
        else:
            pred_state_mean = mean_delta

        # Step 7: Multi-Task Threat Classification
        pred_binary = self.infiltration_head(h_fused)
        pred_mitre = self.mitre_head(h_fused)

        result = {
            "pred_state_mean": pred_state_mean,
            "pred_state_logvar": logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_fused,
            "patch_attention": patch_attn,
        }
        return result

    def predict_next_state(
        self,
        state_seq: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Convenience method for autoregressive rollout."""
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-10.0, 2.0)
        std = (0.5 * logvar).exp()
        return mean + torch.randn_like(std) * std


# Aliases
TimesFMWorldModel = Gen6_TimesFMWorldModel
Gen6_TimesFM = Gen6_TimesFMWorldModel
