"""HXCore: Shaun backbone + 33-class technique head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TechniqueHead(nn.Module):
    def __init__(self, d_model: int = 256, n_classes: int = 33, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class HXCore(nn.Module):
    """Wraps a Shaun StateTransformerWorldModel and an optional technique head."""

    def __init__(self, shaun_model: nn.Module, n_tech: int = 33):
        super().__init__()
        self.shaun = shaun_model
        d_model = int(shaun_model.d_model)
        self.technique_head = TechniqueHead(d_model=d_model, n_classes=n_tech)
        self.technique_trained = False

    def encode(self, state_seq: torch.Tensor) -> torch.Tensor:
        x = self.shaun.input_embed(state_seq)
        x = self.shaun.pos_encoder(x)
        for layer in self.shaun.layers:
            x, _ = layer(x)
        x = self.shaun.layer_norm(x)
        return x[:, -1, :]

    def forward(self, state_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(state_seq)
        return {
            "hidden": h,
            "attack_logits": self.shaun.attack_head(h),
            "mitre_logits": self.shaun.mitre_head(h),
            "tech_logits": self.technique_head(h),
        }

    def freeze_backbone(self, *, last_layer: bool = True, heads: bool = False) -> None:
        for p in self.shaun.parameters():
            p.requires_grad = False
        if last_layer:
            for p in self.shaun.layers[-1].parameters():
                p.requires_grad = True
            for p in self.shaun.layer_norm.parameters():
                p.requires_grad = True
        if heads:
            for p in self.shaun.attack_head.parameters():
                p.requires_grad = True
            for p in self.shaun.mitre_head.parameters():
                p.requires_grad = True
        for p in self.technique_head.parameters():
            p.requires_grad = True


def focal_binary_fp_penalty(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    fp_weight: float = 8.0,
    fn_weight: float = 1.5,
) -> torch.Tensor:
    """CE on attack class with extra weight on false positives (pred attack, true benign)."""
    logp = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(logp, target, reduction="none")
    pred = logits.argmax(dim=-1)
    fp = (pred == 1) & (target == 0)
    fn = (pred == 0) & (target == 1)
    w = torch.ones_like(nll)
    w = torch.where(fp, torch.full_like(w, fp_weight), w)
    w = torch.where(fn, torch.full_like(w, fn_weight), w)
    return (w * nll).mean()
