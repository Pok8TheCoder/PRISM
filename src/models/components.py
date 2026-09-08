"""
PRISM Shared Model Components
Reusable layers: positional encoding, attention, prediction heads.
"""

import math
from typing import Optional

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


class MaskedStateEmbedding(nn.Module):
    """
    Sparsity-aware State Embedding with Missingness Indicator.
    Prevents zero-padded features (e.g. from NetFlow CTU-13 vs deep packet CIC-IDS2018)
    from skewing LayerNorm statistics across heterogeneous telemetry datasets.
    """

    def __init__(self, d_state: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.proj = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # Learnable embedding for missing / zero-padded feature indicator
        self.missing_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.missing_bias, std=0.02)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_state)
        mask: optional (batch, seq_len, d_state) or (batch, d_state) binary mask (1=active, 0=padded)
        """
        if mask is None:
            # Auto-detect zero-padded feature columns across the sequence
            with torch.no_grad():
                # 1 if feature has non-zero values in sequence, 0 if completely absent
                mask = (x.abs().sum(dim=1, keepdim=True) > 1e-6).float()  # (batch, 1, d_state)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(1)  # (batch, 1, d_state)

        # Sparsity fraction: how much of the canonical feature space is padded
        sparsity = 1.0 - mask.mean(dim=-1, keepdim=True)  # (batch, 1, 1)

        # Project and normalize
        h = self.proj(x)
        h = self.norm(h + sparsity * self.missing_bias)
        return self.dropout(h)


class HeadAdapter(nn.Module):
    """
    Dedicated MLP Projection Adapter for Task Heads.
    Decouples task-specific representations from the shared Transformer backbone,
    preventing 242-dim dynamics gradients from interfering with classification.
    """

    def __init__(self, d_model: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Residual projection: h + MLP(h)"""
        return h + self.net(h)


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


class TemporalConv1DBlock(nn.Module):
    """
    Causal 1D Temporal Convolution Block.
    Extracts local temporal dynamics (e.g. packet bursts, inter-arrival variations,
    sequential port probing) across time windows with causal padding and residual connection.
    """

    def __init__(self, d_model: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1  # Causal left padding
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        residual = x
        # (batch, d_model, seq_len)
        x_trans = x.transpose(1, 2)
        # Causal left padding along time dimension
        x_pad = F.pad(x_trans, (self.padding, 0))
        out = self.conv(x_pad)
        out = out.transpose(1, 2)  # (batch, seq_len, d_model)
        out = self.act(out)
        out = self.dropout(out)
        return self.norm(residual + out)


class RelativeTemporalBias(nn.Module):
    """
    Learnable Relative Temporal Attention Bias.
    Provides temporal inductive bias: how long ago an event happened (recency vs distant history).
    """

    def __init__(self, max_len: int = 128, n_heads: int = 8):
        super().__init__()
        self.max_len = max_len
        self.n_heads = n_heads
        # Relative distance bias: d = i - j for i >= j in causal attention
        self.bias_table = nn.Parameter(torch.zeros(n_heads, max_len))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns relative temporal bias matrix of shape (1, n_heads, seq_len, seq_len)
        or (n_heads * batch, seq_len, seq_len).
        """
        seq_len = min(seq_len, self.max_len)
        positions = torch.arange(seq_len, device=device)
        rel_dist = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq_len, seq_len), i - j
        rel_dist = rel_dist.clamp(min=0, max=self.max_len - 1)  # only non-negative for causal
        bias = self.bias_table[:, rel_dist]  # (n_heads, seq_len, seq_len)
        return bias


class TemporalAttentionPooling(nn.Module):
    """
    Temporal Sequence Attention Pooling with Gated Residual Highway.
    Aggregates full temporal context over lookback sequence L via learned attention queries,
    fusing historical threat precursors (e.g. stealth reconnaissance from 10 steps ago)
    with the instantaneous latest state h_L.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, hidden_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_seq: (batch, seq_len, d_model)
        Returns:
            fused_h: (batch, d_model) — fused instantaneous + temporal context
            attn_weights: (batch, seq_len) — temporal attention distribution
        """
        scores = self.query_proj(hidden_seq)  # (batch, seq_len, 1)
        attn_weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        context = (hidden_seq * attn_weights).sum(dim=1)  # (batch, d_model)

        h_last = hidden_seq[:, -1, :]  # instantaneous latest state
        gate = torch.sigmoid(self.gate(h_last))  # adaptive temporal gate
        fused = self.norm(h_last + gate * self.fuse(context))
        return fused, attn_weights.squeeze(-1)


class FocalLoss(nn.Module):
    """
    Focal Cross-Entropy Loss for addressing extreme class imbalance in cyber attack data.
    Down-weights easy well-classified negative (benign) examples and focuses learning on rare attacks.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch, num_classes)
        targets: (batch,)
        """
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class RobustDynamicsLoss(nn.Module):
    """
    Robust Dynamics Loss combining Smooth L1 (Huber) next-state prediction
    with bounded variance calibration and per-feature error capping.
    Prevents outlier/spiky telemetry features from dominating the gradient.
    """

    def __init__(self, beta: float = 1.0, max_sq_err: float = 5.0, var_weight: float = 0.25):
        super().__init__()
        self.beta = beta
        self.max_sq_err = max_sq_err
        self.var_weight = var_weight

    def forward(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        # Huber loss per-element
        state_loss = F.smooth_l1_loss(mean, target, beta=self.beta)
        
        # Log-variance calibration: expanded clamp range [-8.0, 2.0] for high certainty
        logvar_clamped = logvar.clamp(-8.0, 2.0)
        var = logvar_clamped.exp()
        
        # Per-feature squared error with outlier capping
        sq_err = ((target - mean).detach()) ** 2
        sq_err_capped = sq_err.clamp(max=self.max_sq_err)
        
        var_loss = self.var_weight * ((sq_err_capped / (var + 1e-4) + logvar_clamped).clamp(min=-5.0, max=10.0).mean())
        return state_loss + var_loss


# Alias for backward compatibility
RobustGaussianNLL = RobustDynamicsLoss


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


class MultiScaleTemporalConvBlock(nn.Module):
    """
    Generation 7+ Dilated Multi-Scale Causal 1D Inception Temporal Block.
    Extracts temporal dynamics across 6 parallel causal receptive fields:
      - Branch 1 (k=1, d=1): Instantaneous pointwise state anomaly.
      - Branch 2 (k=3, d=1): Fast packet burst & port scan pacing (90s window).
      - Branch 3 (k=5, d=1): Multi-window stealth drift (150s window).
      - Branch 4 (k=3, d=2): Slow-and-low reconnaissance patterns (150s effective).
      - Branch 5 (k=3, d=4): Ultra-slow APT recon over 4.5+ min horizons (270s effective).
      - Branch 6: Causal MaxPool1d + 1x1 conv (Peak flow envelope tracking).
    Fuses all branches with linear projection, LayerNorm, GELU, and residual skip.
    The dilated branches extend the receptive field to 9+ time steps without
    increasing parameter count, enabling the model to distinguish slow reconnaissance
    probes from rapid initial access exploits.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # 6 branches: allocate d_model across them
        # Each gets d_model // 6, remainder goes to the pool branch
        b_dim = d_model // 6
        remainder = d_model - 5 * b_dim  # pool branch gets the rest

        # Branch 1: k=1, dilation=1 (instantaneous)
        self.b1 = nn.Conv1d(d_model, b_dim, kernel_size=1)

        # Branch 2: k=3, dilation=1 (receptive field = 3 steps = 90s)
        self.pad3_d1 = 2  # causal padding = (k-1) * dilation
        self.b2 = nn.Conv1d(d_model, b_dim, kernel_size=3, padding=0, dilation=1)

        # Branch 3: k=5, dilation=1 (receptive field = 5 steps = 150s)
        self.pad5_d1 = 4
        self.b3 = nn.Conv1d(d_model, b_dim, kernel_size=5, padding=0, dilation=1)

        # Branch 4: k=3, dilation=2 (receptive field = 5 steps = 150s, sparser)
        self.pad3_d2 = (3 - 1) * 2  # = 4
        self.b4_dilated = nn.Conv1d(d_model, b_dim, kernel_size=3, padding=0, dilation=2)

        # Branch 5: k=3, dilation=4 (receptive field = 9 steps = 270s — catches ultra-slow APT recon)
        self.pad3_d4 = (3 - 1) * 4  # = 8
        self.b5_dilated = nn.Conv1d(d_model, b_dim, kernel_size=3, padding=0, dilation=4)

        # Branch 6: Causal MaxPool + 1x1 (peak flow envelope)
        self.pad_pool = 2
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=0)
        self.b6_pool = nn.Conv1d(d_model, remainder, kernel_size=1)

        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # Initialize projection to scale residual smoothly
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xt = x.transpose(1, 2)  # (batch, d_model, seq_len)

        out1 = self.b1(xt)                                           # k=1
        out2 = self.b2(F.pad(xt, (self.pad3_d1, 0)))                # k=3, d=1
        out3 = self.b3(F.pad(xt, (self.pad5_d1, 0)))                # k=5, d=1
        out4 = self.b4_dilated(F.pad(xt, (self.pad3_d2, 0)))        # k=3, d=2
        out5 = self.b5_dilated(F.pad(xt, (self.pad3_d4, 0)))        # k=3, d=4
        out6 = self.b6_pool(self.pool(F.pad(xt, (self.pad_pool, 0))))  # maxpool

        fused = torch.cat([out1, out2, out3, out4, out5, out6], dim=1).transpose(1, 2)
        out = self.proj(fused)
        out = self.dropout(self.act(out))
        return self.norm(residual + out)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Cost-Sensitive Focal Loss for Threat Detection.
    Applies focal loss with modulating factor (1 - p_t)^gamma and label smoothing.
    When stage_escalation or weight is provided, scales loss accordingly.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        fn_weight: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        stage_escalation: Optional[list[float]] = None,
        label_smoothing: float = 0.01,
    ):
        super().__init__()
        self.gamma = gamma
        self.fn_weight = fn_weight
        self.weight = weight
        self.label_smoothing = label_smoothing
        if stage_escalation is not None:
            self.register_buffer("stage_escalation", torch.tensor(stage_escalation, dtype=torch.float32))
        else:
            self.stage_escalation = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        probs = torch.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-7, max=1.0)
        focal_mod = (1.0 - p_t) ** self.gamma
        
        ce_loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing, reduction="none")

        loss = focal_mod * ce_loss

        if self.stage_escalation is not None and self.stage_escalation.size(0) == num_classes:
            esc = self.stage_escalation.to(targets.device)
            loss = loss * esc[targets]
        elif self.fn_weight > 1.0:
            is_positive = (targets > 0).float()
            cost_weights = 1.0 + (self.fn_weight - 1.0) * is_positive
            loss = loss * cost_weights
        
        if self.weight is not None:
            w_class = self.weight.to(targets.device)[targets]
            loss = loss * w_class

        return loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Khosla et al.).
    Pulls samples of the same class together on the unit hypersphere
    while pushing apart samples of different classes.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (batch_size, d_proj) L2-normalized embeddings
        labels: (batch_size,) integer class labels
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute dot products between all pairs: (B, B)
        similarity = torch.div(torch.matmul(features, features.T), self.temperature)

        # For numerical stability
        sim_max, _ = torch.max(similarity, dim=1, keepdim=True)
        sim = similarity - sim_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask

        # Compute log-probs
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos
        # Only compute loss for samples that have at least one positive pair
        valid_samples = (mask.sum(1) > 0)
        if valid_samples.sum() > 0:
            return loss[valid_samples].mean()
        return torch.tensor(0.0, device=device, requires_grad=True)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for world model training:
      L = lambda_d * L_dynamics + lambda_i * L_infiltration + lambda_m * L_mitre + lambda_c * L_contrastive
    Uses RobustDynamicsLoss for dynamics and AsymmetricFocalLoss for classification.
    """

    def __init__(
        self,
        lambda_dynamics: float = 0.1,
        lambda_infiltration: float = 2.0,
        lambda_mitre: float = 5.0,
        lambda_contrastive: float = 0.0,
        binary_class_weights: Optional[torch.Tensor] = None,
        mitre_class_weights: Optional[torch.Tensor] = None,
        mitre_stage_escalation: Optional[list[float]] = None,
        use_focal: bool = True,
        focal_gamma: float = 1.5,
        asymmetric_fn_weight: float = 1.0,
        label_smoothing: float = 0.01,
        max_sq_err: float = 5.0,
        contrastive_temp: float = 0.07,
    ):
        super().__init__()
        self.lambda_d = lambda_dynamics
        self.lambda_i = lambda_infiltration
        self.lambda_m = lambda_mitre
        self.lambda_c = lambda_contrastive

        self.dynamics_loss = RobustDynamicsLoss(max_sq_err=max_sq_err)
        if use_focal:
            self.infiltration_loss = AsymmetricFocalLoss(
                gamma=focal_gamma,
                fn_weight=asymmetric_fn_weight,
                weight=binary_class_weights,
                label_smoothing=label_smoothing,
            )
            self.mitre_loss = AsymmetricFocalLoss(
                gamma=focal_gamma,
                fn_weight=1.0,
                weight=mitre_class_weights,
                stage_escalation=mitre_stage_escalation,
                label_smoothing=label_smoothing,
            )
        else:
            self.infiltration_loss = nn.CrossEntropyLoss(
                weight=binary_class_weights, label_smoothing=label_smoothing
            )
            self.mitre_loss = nn.CrossEntropyLoss(
                weight=mitre_class_weights, label_smoothing=label_smoothing
            )

        if lambda_contrastive > 0.0:
            self.contrastive_loss = SupConLoss(temperature=contrastive_temp)
        else:
            self.contrastive_loss = None

    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_logvar: torch.Tensor,
        target_state: torch.Tensor,
        pred_binary: torch.Tensor,
        target_binary: torch.Tensor,
        pred_mitre: torch.Tensor,
        target_mitre: torch.Tensor,
        contrastive_z: Optional[torch.Tensor] = None,
    ) -> dict:
        l_dyn = self.dynamics_loss(pred_mean, pred_logvar, target_state)
        l_inf = self.infiltration_loss(pred_binary, target_binary)
        l_mit = self.mitre_loss(pred_mitre, target_mitre)

        total = self.lambda_d * l_dyn + self.lambda_i * l_inf + self.lambda_m * l_mit

        l_con = torch.tensor(0.0, device=target_state.device)
        l_pair = torch.tensor(0.0, device=target_state.device)
        if self.lambda_c > 0.0 and self.contrastive_loss is not None and contrastive_z is not None:
            l_con = self.contrastive_loss(contrastive_z, target_mitre)

            # Hard-Pair Disambiguation: push Recon(1) far from Initial Access(2),
            # and Lateral Movement(3) far from Exfiltration(5)
            m1 = (target_mitre == 1)
            m2 = (target_mitre == 2)
            if m1.any() and m2.any():
                sim_12 = torch.matmul(contrastive_z[m1], contrastive_z[m2].T)
                l_pair = l_pair + torch.relu(sim_12 - (-0.1)).mean()

            m3 = (target_mitre == 3)
            m5 = (target_mitre == 5)
            if m3.any() and m5.any():
                sim_35 = torch.matmul(contrastive_z[m3], contrastive_z[m5].T)
                l_pair = l_pair + torch.relu(sim_35 - (-0.1)).mean()

            total = total + self.lambda_c * (l_con + 3.0 * l_pair)

        return {
            "total": total,
            "dynamics": l_dyn,
            "infiltration": l_inf,
            "mitre": l_mit,
            "contrastive": l_con,
            "pair_disambiguation": l_pair,
        }


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate upper-triangular causal mask for Transformer."""
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )
    return mask

