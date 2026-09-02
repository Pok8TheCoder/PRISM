"""
Neural Network Components for PRISM: Positional Encodings, Transformer Encoder Blocks,
and Attention Extractor.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal or learnable positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 500, learnable: bool = True):
        super().__init__()
        self.learnable = learnable
        
        if learnable:
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.trunc_normal_(self.pe, std=0.02)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of shape (batch, seq_len, d_model) with positional encoding added.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class TransformerEncoderLayerWithAttn(nn.Module):
    """
    Standard Transformer Encoder Layer that preserves and returns self-attention weights
    for temporal interpretability.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: (batch, seq_len, d_model)
            mask: optional attention mask
        Returns:
            out: (batch, seq_len, d_model)
            attn_weights: (batch, seq_len, seq_len)
        """
        # Self-attention block
        attn_out, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=True
        )
        src = src + self.dropout1(attn_out)
        src = self.norm1(src)

        # Feed-forward block
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_out)
        src = self.norm2(src)

        return src, attn_weights
