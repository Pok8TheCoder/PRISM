"""
PRISM Shared Model Components
Reusable layers: positional encoding, attention, prediction heads.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for sequence models."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, : x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class StateEmbedding(nn.Module):
    """Project raw state vector to model dimension with normalisation."""

    def __init__(self, d_state: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_state) -> (batch, seq_len, d_model)"""
        return self.dropout(self.norm(self.proj(x)))


class StatePredictionHead(nn.Module):
    """
    Predict next state as a Gaussian distribution: N(mu, sigma^2).
    Outputs mean and log-variance of the predicted next state.
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.3):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_state),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_state),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (batch, d_model) — last hidden state
        Returns: (mean, logvar), each (batch, d_state)
        """
        return self.mean_head(h), self.logvar_head(h)


class ClassificationHead(nn.Module):
    """Generic classification head with dropout."""

    def __init__(
        self, d_model: int, num_classes: int, dropout: float = 0.3
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, d_model) -> (batch, num_classes)"""
        return self.head(h)


class GaussianNLL(nn.Module):
    """Gaussian negative log-likelihood loss for state prediction."""

    def __init__(self, min_logvar: float = -10.0, max_logvar: float = 2.0):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Gaussian NLL: 0.5 * (logvar + (target - mean)^2 / var).
        """
        logvar = logvar.clamp(self.min_logvar, self.max_logvar)
        var = logvar.exp()
        nll = 0.5 * (logvar + (target - mean) ** 2 / (var + 1e-8))
        return nll.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for world model training:
      L = lambda_d * L_dynamics + lambda_i * L_infiltration + lambda_m * L_mitre
    """

    def __init__(
        self,
        lambda_dynamics: float = 1.0,
        lambda_infiltration: float = 0.5,
        lambda_mitre: float = 0.3,
        binary_class_weights: torch.Tensor | None = None,
        mitre_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.lambda_d = lambda_dynamics
        self.lambda_i = lambda_infiltration
        self.lambda_m = lambda_mitre

        self.dynamics_loss = GaussianNLL()
        self.infiltration_loss = nn.CrossEntropyLoss(
            weight=binary_class_weights
        )
        self.mitre_loss = nn.CrossEntropyLoss(
            weight=mitre_class_weights
        )

    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        target_state: torch.Tensor,
        pred_binary: torch.Tensor,
        target_binary: torch.Tensor,
        pred_mitre: torch.Tensor,
        target_mitre: torch.Tensor,
    ) -> dict:
        """
        Compute multi-task loss.

        Returns dict with total loss and individual components.
        """
        l_dyn = self.dynamics_loss(pred_mean, pred_logvar, target_state)
        l_inf = self.infiltration_loss(pred_binary, target_binary)
        l_mit = self.mitre_loss(pred_mitre, target_mitre)

        total = self.lambda_d * l_dyn + self.lambda_i * l_inf + self.lambda_m * l_mit

        return {
            "total": total,
            "dynamics": l_dyn,
            "infiltration": l_inf,
            "mitre": l_mit,
        }


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate upper-triangular causal mask for Transformer."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask
