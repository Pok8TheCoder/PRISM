"""
PRISM World Model: StateTransformerWorldModel
Predicts forward network dynamics P(S_{t+1} | S_{<=t}) along with multi-task
Infiltration and MITRE ATT&CK stage detection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from src.models.components import PositionalEncoding, TransformerEncoderLayerWithAttn
from src.utils.constants import NUM_MITRE_STAGES


class StateTransformerWorldModel(nn.Module):
    """
    Temporal Transformer World Model for Predictive Network Intrusion Detection.
    """

    def __init__(
        self,
        d_state: int = 292,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        num_mitre_stages: int = NUM_MITRE_STAGES
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model

        # 1. Embedding & Positional Encoding
        self.input_embed = nn.Sequential(
            nn.Linear(d_state, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, learnable=True)

        # 2. Transformer Encoder Layers with Attention Weight Capture
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

        # 3. Head 1: Next-State Dynamics (Mean & Log-Variance for Gaussian rollout)
        self.dynamics_mean_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_state)
        )
        self.dynamics_logvar_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Linear(dim_feedforward // 2, d_state)
        )

        # 4. Head 2: Infiltration Classification (Binary: Benign vs Attack)
        self.attack_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        # 5. Head 3: MITRE ATT&CK Stage Classification (7 classes)
        self.mitre_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_mitre_stages)
        )

        # 6. Head 4: Attack Flow Fraction (Continuous regression [0, 1])
        self.fraction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            state_seq: (batch_size, seq_len, d_state)
            mask: optional causal or padding mask
        Returns:
            pred_state_mean: (batch_size, d_state)
            pred_state_logvar: (batch_size, d_state)
            pred_attack_logits: (batch_size, 2)
            pred_mitre_logits: (batch_size, num_mitre_stages)
            pred_fraction: (batch_size, 1)
            attention_weights: (batch_size, seq_len, seq_len) from the last layer
        """
        # Embed state sequence
        x = self.input_embed(state_seq)
        x = self.pos_encoder(x)

        # Pass through Transformer encoder layers
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x, mask=mask)
            
        x = self.layer_norm(x)

        # Extract context vector from the most recent time step (t)
        h_t = x[:, -1, :]  # (batch_size, d_model)

        # Forward multi-task heads
        pred_state_mean = self.dynamics_mean_head(h_t)
        pred_state_logvar = self.dynamics_logvar_head(h_t)
        # Clamp logvar for numerical stability
        pred_state_logvar = torch.clamp(pred_state_logvar, min=-6.0, max=3.0)

        pred_attack_logits = self.attack_head(h_t)
        pred_mitre_logits = self.mitre_head(h_t)
        pred_fraction = self.fraction_head(h_t)

        return (
            pred_state_mean,
            pred_state_logvar,
            pred_attack_logits,
            pred_mitre_logits,
            pred_fraction,
            last_attn
        )

    def compute_loss(
        self,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        pred_attack: torch.Tensor,
        pred_mitre: torch.Tensor,
        pred_frac: torch.Tensor,
        target_state: torch.Tensor,
        target_attack: torch.Tensor,
        target_mitre: torch.Tensor,
        target_frac: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        mitre_weights: Optional[torch.Tensor] = None,
        lambdas: Tuple[float, float, float, float] = (0.5, 1.0, 1.2, 0.5)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite multi-task World Model loss:
        L = lambda1 * L_dynamics + lambda2 * L_attack + lambda3 * L_mitre + lambda4 * L_frac
        """
        l1, l2, l3, l4 = lambdas

        # 1. State Dynamics Loss: Stable MSE on standardized states
        dynamics_loss = F.mse_loss(pred_mean, target_state)

        # 2. Infiltration Attack Binary Cross-Entropy
        attack_loss = F.cross_entropy(pred_attack, target_attack, weight=class_weights)

        # 3. MITRE ATT&CK Multiclass Cross-Entropy with balanced class weights
        mitre_loss = F.cross_entropy(pred_mitre, target_mitre, weight=mitre_weights)

        # 4. Attack Fraction Regression (Huber / Smooth L1)
        frac_loss = F.smooth_l1_loss(pred_frac.squeeze(-1), target_frac)

        total_loss = (
            l1 * dynamics_loss +
            l2 * attack_loss +
            l3 * mitre_loss +
            l4 * frac_loss
        )

        metrics = {
            "total_loss": total_loss.item(),
            "dynamics_loss": dynamics_loss.item(),
            "attack_loss": attack_loss.item(),
            "mitre_loss": mitre_loss.item(),
            "fraction_loss": frac_loss.item()
        }

        return total_loss, metrics
