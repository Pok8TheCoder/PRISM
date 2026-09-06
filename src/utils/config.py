"""
PRISM Configuration Management
Loads YAML configs and merges with CLI overrides.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    dataset: str = "cicids2018"  # cicids2018 | ctu13
    window_size_seconds: int = 30
    lookback: int = 20
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    use_packet_features: bool = True
    max_flows_per_window: int = 10000
    num_workers: int = 0


@dataclass
class ModelConfig:
    architecture: str = "transformer"  # transformer | lstm | gnn | latent
    d_state: int = 110
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    head_dropout: float = 0.3
    # GNN-specific
    gnn_type: str = "graphsage"  # graphsage | gat
    gnn_layers: int = 2
    d_graph: int = 128
    # Latent dynamics
    d_latent: int = 64
    # LSTM-specific
    lstm_hidden: int = 256
    lstm_layers: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    lambda_dynamics: float = 1.0
    lambda_infiltration: float = 0.5
    lambda_mitre: float = 0.3
    lambda_contrastive: float = 0.0
    contrastive_temp: float = 0.07
    asymmetric_fn_weight: float = 3.5
    use_focal: bool = True
    use_stop_gradient: bool = True
    max_sq_err: float = 5.0
    label_smoothing: float = 0.01
    balanced_sampling: bool = True
    scheduler: str = "cosine"  # cosine | step | plateau
    warmup_steps: int = 500
    device: str = "auto"  # auto | cuda | cpu
    seed: int = 42
    checkpoint_dir: str = "weights"
    log_dir: str = "results/logs"
    save_every: int = 5


@dataclass
class InferenceConfig:
    model_path: str = "weights/world_model_best.pt"
    k_steps: int = 10
    num_rollouts: int = 10  # for ensemble uncertainty
    deterministic: bool = False
    threshold_critical: float = 0.85
    threshold_high: float = 0.65
    threshold_medium: float = 0.40


@dataclass
class PRISMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    project_root: str = "."


def load_config(config_path: Optional[str] = None, overrides: Optional[dict] = None) -> PRISMConfig:
    """Load config from YAML file and apply overrides."""
    cfg = PRISMConfig()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}

        if "data" in raw:
            for k, v in raw["data"].items():
                if hasattr(cfg.data, k):
                    setattr(cfg.data, k, v)
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(cfg.model, k):
                    setattr(cfg.model, k, v)
        if "train" in raw:
            for k, v in raw["train"].items():
                if hasattr(cfg.train, k):
                    setattr(cfg.train, k, v)
        if "inference" in raw:
            for k, v in raw["inference"].items():
                if hasattr(cfg.inference, k):
                    setattr(cfg.inference, k, v)
        if "project_root" in raw:
            cfg.project_root = raw["project_root"]

    if overrides:
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) == 2:
                section, attr = parts
                sub = getattr(cfg, section, None)
                if sub and hasattr(sub, attr):
                    setattr(sub, attr, value)

    return cfg


def get_device(cfg: TrainConfig) -> str:
    """Resolve device string."""
    import torch
    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device
