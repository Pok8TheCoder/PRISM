"""
PRISM - Core Transformer World Model
Learns P(S_{t+1} | S_{t-L+1}, ..., S_t) via causal self-attention.
Three output heads: state dynamics, infiltration binary, MITRE stage.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import (
    StateEmbedding,
    MaskedStateEmbedding,
    HeadAdapter,
    LearnablePositionalEncoding,
    SinusoidalPositionalEncoding,
    TemporalConv1DBlock,
    MultiScaleTemporalConvBlock,
    TemporalAttentionPooling,
    StatePredictionHead,
    ClassificationHead,
    MultiTaskLoss,
    generate_causal_mask,
)
from src.utils.constants import NUM_MITRE_STAGES

logger = logging.getLogger("prism.models.world_model")


class TemporalTransformerWorldModel(nn.Module):
    """
    Generation 7 Multi-Scale Temporal Transformer World Model for Network Dynamics.
    Architecture:
        1. Sparsity-Aware Masked State Embedding (D_state -> D_model)
        2. Multi-Scale 1D Inception Temporal Block (k=1, 3, 5, MaxPool)
        3. Positional Encoding (learnable or sinusoidal)
        4. Causal Transformer Encoder (N layers, H heads, Pre-LN)
        5. Temporal Attention Pooling (fuses sequence context with instantaneous state)
        6. Residual State Dynamics Head: S_{t+1} = S_t + Delta S_t
        7. Decoupled Classification Adapters with Stop-Gradient protection:
           - Infiltration Binary Head: h_inf -> P(attack | trajectory)
           - MITRE Attack Stage Head: h_mitre -> P(stage | trajectory)
        8. Zero-Day Anomaly Surprise Engine
    """

    def __init__(
        self,
        d_state: int = 110,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = 20,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
        pos_encoding: str = "learnable",  # "learnable" | "sinusoidal"
        residual_dynamics: bool = True,
        conv_type: str = "multiscale",  # "multiscale" (Gen 5/7) | "single" (Gen 4/3)
        use_stop_gradient: bool = True,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback
        self.num_mitre_stages = num_mitre_stages
        self.residual_dynamics = residual_dynamics
        self.conv_type = conv_type
        self.use_stop_gradient = use_stop_gradient

        # 1. Input embedding with domain sparsity masking
        self.embedding = MaskedStateEmbedding(d_state, d_model, dropout)

        # 2. Causal 1D Temporal Convolution (Multi-scale for Gen 7/5, Single for Gen 4)
        if conv_type == "single":
            self.temporal_conv = TemporalConv1DBlock(d_model, kernel_size=3, dropout=dropout)
        else:
            self.temporal_conv = MultiScaleTemporalConvBlock(d_model, dropout=dropout)

        # 3. Positional encoding
        if pos_encoding == "learnable":
            self.pos_enc = LearnablePositionalEncoding(
                max_len=lookback + 64, d_model=d_model
            )
        else:
            self.pos_enc = SinusoidalPositionalEncoding(
                max_len=lookback + 64, d_model=d_model, dropout=dropout
            )

        # 4. Causal Transformer Encoder (Pre-LN for stability)
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

        # 5. Temporal Attention Pooling
        self.temporal_pool = TemporalAttentionPooling(d_model, dropout=dropout)

        # 6. Residual State Dynamics Head
        self.state_head = StatePredictionHead(d_model, d_state, head_dropout)

        # 7. Dedicated Task Adapters & Multi-task classification heads
        self.adapter_inf = HeadAdapter(d_model, dropout=head_dropout)
        self.adapter_mitre = HeadAdapter(d_model, dropout=head_dropout)

        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(
            d_model, num_mitre_stages, head_dropout
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "TemporalTransformerWorldModel (Gen 7): d_state=%d, d_model=%d, "
            "layers=%d, heads=%d, params=%.2fM, residual_dynamics=%s, stop_grad=%s",
            d_state, d_model, n_layers, n_heads, n_params / 1e6, residual_dynamics, use_stop_gradient,
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        return_attention: bool = False,
        use_stop_gradient: Optional[bool] = None,
    ) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        state_seq : torch.Tensor, shape (B, L, D_state)
            Sequence of past network states.
        return_attention : bool
            If True, extract and return attention weights from last layer.
        use_stop_gradient : bool, optional
            Override model-level stop_gradient behavior for classification heads.

        Returns
        -------
        dict with keys:
            pred_state_mean       : (B, D_state) — predicted next state mean
            pred_state_logvar     : (B, D_state) — predicted next state log-variance
            pred_binary           : (B, 2)       — infiltration logits
            pred_mitre            : (B, N_stage) — MITRE stage logits
            hidden                : (B, D_model) — fused representation (for SHAP/viz)
            hidden_seq            : (B, L, D_model) — full sequence representations
            attention_weights     : (B, L, L) or None
            temporal_pool_weights : (B, L)
        """
        B, L, _ = state_seq.shape
        device = state_seq.device
        stop_grad = self.use_stop_gradient if use_stop_gradient is None else use_stop_gradient

        # 1. Embed & local temporal conv
        x = self.embedding(state_seq)            # (B, L, D_model)
        x = self.pos_enc(x)                      # (B, L, D_model)
        x = self.temporal_conv(x)                # (B, L, D_model)

        # 2. Causal temporal mask: position t only attends to <= t
        causal_mask = generate_causal_mask(L, device)  # (L, L)

        # 3. Causal Transformer Encoder
        hidden_seq = self.transformer(
            x, mask=causal_mask, is_causal=True
        )                                        # (B, L, D_model)

        # 4. Temporal Attention Pooling (fuses trajectory history + latest state)
        h_fused, pool_weights = self.temporal_pool(hidden_seq)

        # 5. Prediction heads
        delta_mean, pred_logvar = self.state_head(h_fused)

        # Residual dynamics: S_{t+1} = S_t + Delta S_t
        if self.residual_dynamics:
            s_t = state_seq[:, -1, :]
            pred_mean = s_t + delta_mean
        else:
            pred_mean = delta_mean

        # 6. Decoupled Classification Path (Decouples 242-dim dynamics from 2/7-dim classification)
        if stop_grad and self.training:
            # Scaled gradient blend: 90% detached, 10% soft gradient feedback
            h_class_base = 0.1 * h_fused + 0.9 * h_fused.detach()
        else:
            h_class_base = h_fused

        h_inf = self.adapter_inf(h_class_base)
        h_mit = self.adapter_mitre(h_class_base)

        pred_binary = self.infiltration_head(h_inf)
        pred_mitre = self.mitre_head(h_mit)

        out = {
            "pred_state_mean": pred_mean,
            "pred_state_logvar": pred_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_fused,
            "hidden_seq": hidden_seq,
            "attention_weights": None,
            "temporal_pool_weights": pool_weights,
        }

        if return_attention:
            out["attention_weights"] = self._extract_attention(
                x, causal_mask, device
            )

        return out

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Convenience: return P(attack) scalar per batch item."""
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def compute_zero_day_anomaly(
        self,
        state_seq: torch.Tensor,
        next_state: torch.Tensor,
        sigma_threshold: float = 3.0,
    ) -> dict:
        """
        Compute Zero-Day Anomaly Surprise Score (Negative Log-Likelihood / Mahalanobis Deviation).
        Detects novel exploits that break learned network transition dynamics.

        Parameters
        ----------
        state_seq : (B, L, D_state) or (L, D_state)
        next_state: (B, D_state) or (D_state)
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

            # Normalized quadratic error per feature
            sq_err = (next_state - mean) ** 2
            per_feature_surprise = sq_err / (var + 1e-6)  # (B, D_state)
            mean_surprise = per_feature_surprise.mean(dim=-1)  # (B,)

            # Top contributing anomalous features
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

    def sample_next_state(
        self,
        state_seq: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Sample (or take mean of) the predicted next state distribution.
        Used for K-step rollout.
        """
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-8.0, 2.0)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mean + eps * std

    # ------------------------------------------------------------------
    # Attention extraction (for explainability)
    # ------------------------------------------------------------------
    def _extract_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Re-run the last Transformer layer and extract attention weights.
        Returns (B, L, L) averaged over heads.
        """
        last_layer = self.transformer.layers[-1]
        attn_output, attn_weights = last_layer.self_attn(
            x, x, x,
            attn_mask=mask.float().masked_fill(mask, float("-inf")),
            need_weights=True,
            average_attn_weights=True,
        )
        return attn_weights  # (B, L, L)

    def _init_weights(self):
        """Initialize weights with Xavier and small residual head weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Initialize delta prediction head to near-zero for smooth residual start
        if hasattr(self.state_head, "mean_head"):
            last_linear = self.state_head.mean_head[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.normal_(last_linear.weight, std=1e-3)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)


# Backward-compatible alias for existing code, imports, and checkpoints
StateTransformerWorldModel = TemporalTransformerWorldModel


class LSTMWorldModel(nn.Module):
    """
    LSTM-based World Model variant.
    Same input/output contract as StateTransformerWorldModel.
    """

    def __init__(
        self,
        d_state: int = 110,
        d_model: int = 256,
        lstm_layers: int = 2,
        lookback: int = 20,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        num_mitre_stages: int = NUM_MITRE_STAGES,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback

        self.embedding = StateEmbedding(d_state, d_model, dropout)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,  # causal: no look-ahead
        )

        self.state_head = StatePredictionHead(d_model, d_state, head_dropout)
        self.infiltration_head = ClassificationHead(d_model, 2, head_dropout)
        self.mitre_head = ClassificationHead(d_model, num_mitre_stages, head_dropout)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "LSTMWorldModel: d_state=%d, d_model=%d, "
            "lstm_layers=%d, params=%.2fM",
            d_state, d_model, lstm_layers, n_params / 1e6,
        )

    def forward(
        self,
        state_seq: torch.Tensor,
        hidden_state: Optional[tuple] = None,
        return_attention: bool = False,
    ) -> dict:
        """
        Parameters
        ----------
        state_seq    : (B, L, D_state)
        hidden_state : optional LSTM (h, c) for stateful inference
        """
        x = self.embedding(state_seq)                # (B, L, D_model)
        lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)  # (B, L, D_model)

        h_t = lstm_out[:, -1, :]                     # (B, D_model)

        pred_mean, pred_logvar = self.state_head(h_t)
        pred_binary = self.infiltration_head(h_t)
        pred_mitre = self.mitre_head(h_t)

        return {
            "pred_state_mean": pred_mean,
            "pred_state_logvar": pred_logvar,
            "pred_binary": pred_binary,
            "pred_mitre": pred_mitre,
            "hidden": h_t,
            "hidden_seq": lstm_out,
            "lstm_state": (h_n, c_n),
            "attention_weights": None,
        }

    def predict_infiltration_prob(self, state_seq: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(state_seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    def sample_next_state(
        self, state_seq: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        with torch.no_grad():
            out = self.forward(state_seq)
        mean = out["pred_state_mean"]
        if deterministic:
            return mean
        logvar = out["pred_state_logvar"].clamp(-10.0, 2.0)
        std = (0.5 * logvar).exp()
        return mean + torch.randn_like(std) * std


# ===========================================================================
# Generation-Specific Model Aliases
# ===========================================================================
from src.models.timesfm_model import Gen6_TimesFMWorldModel, TimesFMWorldModel
from src.models.gen7_world_model import Gen7DecoupledWorldModel
from src.models.gen8_world_model import Gen8DecoupledMLPWorldModel

Gen8_DecoupledMLPWorldModel = Gen8DecoupledMLPWorldModel
Gen7_DecoupledTemporalTransformer = Gen7DecoupledWorldModel
Gen6_TimesFMCyberWorldModel = Gen6_TimesFMWorldModel
Gen5_MultiScaleTemporalTransformer = TemporalTransformerWorldModel
Gen4_BalancedTemporalTransformer = TemporalTransformerWorldModel
Gen3_TemporalTransformer = TemporalTransformerWorldModel
Gen2_DeepTransformer = StateTransformerWorldModel
Gen1_BaselineTransformer = StateTransformerWorldModel


def build_world_model(cfg) -> nn.Module:
    """
    Factory: build the world model specified in config.

    Parameters
    ----------
    cfg : ModelConfig dataclass or dict-like
    """
    arch = getattr(cfg, "architecture", "transformer").lower()

    if arch in ("gen8", "gen8_world_model", "gen8_mlp", "gen8_decoupled"):
        return Gen8DecoupledMLPWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            lookback=getattr(cfg, "lookback", 20),
            dropout=cfg.dropout,
            residual_dynamics=getattr(cfg, "residual_dynamics", True),
        )
    elif arch in ("gen7", "gen7_world_model", "gen7_decoupled", "gen7_contrastive"):
        return Gen7DecoupledWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            lookback=getattr(cfg, "lookback", 60),
            dropout=cfg.dropout,
            residual_dynamics=getattr(cfg, "residual_dynamics", True),
        )
    elif arch in ("gen6", "gen6_timesfm", "timesfm", "timesfm_world_model"):
        return Gen6_TimesFMWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            lookback=getattr(cfg, "lookback", 20),
            patch_len=getattr(cfg, "patch_len", 4),
            patch_stride=getattr(cfg, "patch_stride", 2),
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
            residual_dynamics=getattr(cfg, "residual_dynamics", True),
        )
    elif arch in (
        "gen7", "gen7_transformer", "gen7_decoupled",
        "gen5", "gen5_transformer", "gen5_multiscale_transformer",
        "gen4", "gen4_transformer", "gen4_balanced_transformer",
        "gen3", "gen3_transformer",
        "transformer", "temporal_transformer", "temporal",
    ):
        conv_type = "single" if arch in ("gen4", "gen4_transformer", "gen4_balanced_transformer", "gen3", "gen3_transformer") else "multiscale"
        use_stop_gradient = getattr(cfg, "use_stop_gradient", True)
        return TemporalTransformerWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            lookback=getattr(cfg, "lookback", 20),
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
            residual_dynamics=getattr(cfg, "residual_dynamics", True),
            conv_type=conv_type,
            use_stop_gradient=use_stop_gradient,
        )
    elif arch in ("gen1", "gen1_transformer", "gen2", "gen2_transformer", "state_transformer"):
        return StateTransformerWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            lookback=getattr(cfg, "lookback", 20),
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    elif arch == "lstm":
        return LSTMWorldModel(
            d_state=cfg.d_state,
            d_model=cfg.d_model,
            lstm_layers=cfg.lstm_layers,
            lookback=getattr(cfg, "lookback", 20),
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    elif arch == "gnn":
        from src.models.gnn_model import GraphWorldModel
        return GraphWorldModel(
            d_node=cfg.d_state,
            d_graph=cfg.d_graph,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            gnn_type=cfg.gnn_type,
            gnn_layers=cfg.gnn_layers,
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    elif arch == "latent":
        from src.models.latent_dynamics import LatentDynamicsWorldModel
        return LatentDynamicsWorldModel(
            d_state=cfg.d_state,
            d_latent=cfg.d_latent,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            head_dropout=cfg.head_dropout,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_class": model.__class__.__name__,
        },
        path,
    )
    logger.info("Saved checkpoint -> %s (epoch %d)", path, epoch)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer=None,
    device: str = "cpu",
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    logger.info(
        "Loaded checkpoint from %s (epoch %d)", path, ckpt.get("epoch", -1)
    )
    return ckpt
