"""HX-C: new causal transformer on 292-d Shaun states with a 33-way catalog head.

Attempt 2 only. Not a Shaun/ARY weight reuse. Binary attack = 1 - p(Benign);
MITRE stage is derived from the predicted catalog class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.aryan.components import (
    ClassificationHead,
    LearnablePositionalEncoding,
    StateEmbedding,
    StatePredictionHead,
    TemporalAttentionPooling,
    TemporalConv1DBlock,
    generate_causal_mask,
)
from src.aryan.constants import MITRE_STAGES, MITRE_STAGES_INV, TACTIC_TO_STAGE
from src.hx.ramx_hx import RAMXHXPredictor
from src.model.attack_catalog import get_class_names, get_mitre_map

LOOKBACK = 30
SHAUN_DIM = 292
TECH_ABSTAIN = 0.7
DETECT_THRESHOLD = 0.5
HX_C_CKPT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "Automode"
    / "train"
    / "checkpoints"
    / "hx_c_w5s.pt"
)
HX_C_V2_CKPT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "Automode"
    / "train"
    / "checkpoints"
    / "hx_c_v2_w5s.pt"
)


class CausalHXClassifier(nn.Module):
    """ARY-style causal Pre-LN transformer; primary head is catalog softmax."""

    def __init__(
        self,
        *,
        d_state: int = SHAUN_DIM,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        lookback: int = LOOKBACK,
        n_classes: int = 33,
        dropout: float = 0.1,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model
        self.lookback = lookback
        self.n_classes = n_classes
        self.embedding = StateEmbedding(d_state, d_model, dropout)
        self.temporal_conv = TemporalConv1DBlock(d_model, kernel_size=3, dropout=dropout)
        self.pos_enc = LearnablePositionalEncoding(max_len=lookback + 64, d_model=d_model)
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
        self.temporal_pool = TemporalAttentionPooling(d_model, dropout=dropout)
        self.state_head = StatePredictionHead(d_model, d_state, head_dropout)
        self.class_head = ClassificationHead(d_model, n_classes, head_dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        last_linear = self.state_head.mean_head[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, std=1e-3)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(self, state_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        seq_len = state_seq.shape[1]
        device = state_seq.device
        x = self.embedding(state_seq)
        x = self.pos_enc(x)
        x = self.temporal_conv(x)
        causal_mask = generate_causal_mask(seq_len, device)
        hidden_seq = self.transformer(x, mask=causal_mask, is_causal=True)
        h_fused, pool_weights = self.temporal_pool(hidden_seq)
        delta_mean, pred_logvar = self.state_head(h_fused)
        pred_mean = state_seq[:, -1, :] + delta_mean
        class_logits = self.class_head(h_fused)
        return {
            "class_logits": class_logits,
            "binary_logits": class_logits_to_binary(class_logits),
            "pred_state_mean": pred_mean,
            "pred_state_logvar": pred_logvar,
            "hidden": h_fused,
            "temporal_pool_weights": pool_weights,
        }


def class_logits_to_binary(class_logits: torch.Tensor) -> torch.Tensor:
    """Stack [logit(Benign), logsumexp(techniques)] as a 2-way attack head."""
    benign = class_logits[:, 0:1]
    attack = torch.logsumexp(class_logits[:, 1:], dim=-1, keepdim=True)
    return torch.cat([benign, attack], dim=-1)


def _padded_raw(buf: list[np.ndarray], lookback: int = LOOKBACK) -> np.ndarray:
    window = buf[-lookback:]
    if len(window) < lookback:
        window = [window[0]] * (lookback - len(window)) + window
    return np.stack(window)


def load_hxc_bundle(ckpt_path: Path | None = None) -> dict[str, Any]:
    path = Path(ckpt_path or HX_C_V2_CKPT if HX_C_V2_CKPT.exists() else HX_C_CKPT)
    if not path.exists():
        raise FileNotFoundError(f"Missing HX-C checkpoint: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    names = list(ckpt.get("class_names") or get_class_names())
    model = CausalHXClassifier(
        d_state=int(ckpt.get("d_state", SHAUN_DIM)),
        d_model=int(ckpt.get("d_model", 256)),
        n_layers=int(ckpt.get("n_layers", 4)),
        n_heads=int(ckpt.get("n_heads", 8)),
        lookback=int(ckpt.get("lookback", LOOKBACK)),
        n_classes=len(names),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    mean = np.asarray(ckpt.get("scaler_mean", np.zeros((1, SHAUN_DIM))), dtype=np.float32)
    std = np.asarray(ckpt.get("scaler_std", np.ones((1, SHAUN_DIM))), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "device": device,
        "class_names": names,
        "ckpt": path,
        "hx_version": str(ckpt.get("hx_version", "hx-c-2.0" if "v2" in path.name else "hx-c-1.0")),
    }


class StreamingHXC:
    """Attempt-2 live/PCAP scorer. RAMX-HX gate/cooldown/memory, no 15s confirm."""

    system_id = "hx_c"

    def __init__(self, bundle: dict[str, Any] | None = None, context_skip_steps: int = 0):
        bundle = bundle or load_hxc_bundle()
        self.model: CausalHXClassifier = bundle["model"]
        self.mean = bundle["mean"]
        self.std = bundle["std"]
        self.device = bundle["device"]
        self.class_names: list[str] = list(bundle["class_names"])
        self.hx_version: str = str(bundle.get("hx_version", "hx-c-1.0"))
        self.mitre_map = get_mitre_map()
        self.ramx = RAMXHXPredictor(context_skip_steps=context_skip_steps)
        self.buf: list[np.ndarray] = []

    def set_context_skip(self, steps: int) -> None:
        self.ramx.set_context_skip(steps)

    def begin_live_phase(self) -> None:
        self.ramx.begin_live_phase()

    def _scaled_window(self, buf: list[np.ndarray]) -> np.ndarray:
        m, s = self.mean.reshape(-1), self.std.reshape(-1)
        return ((_padded_raw(buf, self.model.lookback) - m) / s).astype(np.float32)

    def _infer_buf(self, buf: list[np.ndarray]) -> dict[str, Any]:
        seq = self._scaled_window(buf)
        x = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        m, s = self.mean.reshape(-1), self.std.reshape(-1)
        with torch.no_grad():
            out = self.model(x)
            class_p = torch.softmax(out["class_logits"], dim=-1)[0].cpu().numpy()
            p_att = float(1.0 - class_p[0])
            hidden = out["hidden"][0].detach().cpu().numpy()
            pred_scaled = out["pred_state_mean"][0].detach().cpu().numpy().reshape(-1)
        pred_raw = (pred_scaled * s + m).astype(np.float32)
        return {
            "p_att": p_att,
            "class_probs": class_p,
            "hidden": hidden,
            "traj": seq,
            "pred_raw": pred_raw,
        }

    def _forward(self) -> dict[str, Any]:
        return self._infer_buf(self.buf)

    def _fuse_scores(
        self,
        *,
        raw_p: float,
        class_p: np.ndarray,
        state: np.ndarray,
        gated: bool,
        imagined: bool = False,
    ) -> dict[str, float]:
        rel = 0.0 if imagined else self.ramx.calibrator.get_relative_anomaly_score(state)
        calibrated_raw = self.ramx.raw_calibrator.adjust(raw_p)
        in_cd = self.ramx.cooldown_left > 0
        shift = self.ramx.prior.shift_adjusted(class_p)
        fused, suppressed = self.ramx.fuse(
            raw_p=calibrated_raw,
            relative_anomaly=rel,
            in_context_gate=gated,
            in_cooldown=in_cd,
            class_shift=shift,
        )
        return {
            "fused": float(fused),
            "rel": float(rel),
            "calibrated_raw": float(calibrated_raw),
            "class_shift": float(shift),
            "suppressed": float(suppressed),
        }

    def rollout(self, horizon: int = 6) -> dict[str, Any]:
        """Open-loop K-step forecast on the state head; fuse with frozen RAMX stats."""
        if not self.buf or horizon <= 0:
            return {"forecast": [], "forecast_max": 0.0}
        buf = list(self.buf)
        future: list[float] = []
        stages: list[str] = []
        gated = self.ramx.step_count <= self.ramx.context_skip_steps
        for _ in range(horizon):
            inf = self._infer_buf(buf)
            scores = self._fuse_scores(
                raw_p=inf["p_att"], class_p=inf["class_probs"], state=inf["pred_raw"], gated=gated, imagined=True,
            )
            future.append(scores["fused"])
            idx = int(np.argmax(inf["class_probs"]))
            name = self.class_names[idx] if idx < len(self.class_names) else "Benign"
            stages.append(name)
            buf.append(inf["pred_raw"])
        return {
            "forecast": future,
            "forecast_max": float(max(future) if future else 0.0),
            "forecast_mean": float(np.mean(future) if future else 0.0),
            "forecast_tech": stages,
        }

    def _emit_technique(self, class_p: np.ndarray, confirmed: bool) -> tuple[str | None, float, int]:
        idx = int(np.argmax(class_p))
        p = float(class_p[idx])
        name = self.class_names[idx] if idx < len(self.class_names) else "Benign"
        tactic = self.mitre_map.get(name, ("Normal", ""))[0]
        stage_name = TACTIC_TO_STAGE.get(tactic, "Benign")
        stage = MITRE_STAGES.get(stage_name, 0)
        if not confirmed or name == "Benign" or p < TECH_ABSTAIN:
            return None, p, stage
        return name, p, stage

    def step(self, raw_state: np.ndarray, true_bin: int | None = None, true_mit: int | None = None) -> dict:
        del true_bin, true_mit
        raw = np.asarray(raw_state, dtype=np.float32)
        self.buf.append(raw)
        fwd = self._forward()
        raw_p = fwd["p_att"]
        class_p = fwd["class_probs"]

        self.ramx.step_count += 1
        if self.ramx.in_live_phase:
            self.ramx.live_step_count += 1
        self.ramx.calibrator.update(raw)
        in_gate = self.ramx.step_count <= self.ramx.context_skip_steps
        if in_gate:
            self.ramx.prior.observe_warmup(class_p)
        rel = self.ramx.calibrator.get_relative_anomaly_score(raw)
        calibrated_raw = self.ramx.raw_calibrator.adjust(raw_p)
        self.ramx.raw_calibrator.update(raw_p)

        in_cd = self.ramx.cooldown_left > 0
        if in_cd:
            self.ramx.cooldown_left -= 1

        shift = 0.0 if in_gate else self.ramx.prior.shift_adjusted(class_p)
        fused, suppressed = self.ramx.fuse(
            raw_p=calibrated_raw,
            relative_anomaly=rel,
            in_context_gate=in_gate,
            in_cooldown=in_cd,
            class_shift=shift,
        )
        confirmed = (not in_gate) and (not in_cd) and fused >= DETECT_THRESHOLD
        tech, tech_p, stage = self._emit_technique(fwd["class_probs"], confirmed)
        if confirmed and stage == 0:
            stage = 1
        fused, memory_written = self.ramx.maybe_memory(
            fwd["traj"], fused, confirmed, stage, in_gate, in_cd,
        )
        self.ramx.after_alert(confirmed, fused)
        stage_name = MITRE_STAGES_INV.get(stage, "Benign")
        if tech:
            tactic = self.mitre_map.get(tech, ("", ""))[0]
            mapped = TACTIC_TO_STAGE.get(tactic)
            if mapped:
                stage_name = mapped
                stage = MITRE_STAGES.get(stage_name, stage)

        return {
            "p_att": float(fused),
            "raw_p_att": float(raw_p),
            "calibrated_raw_p_att": float(calibrated_raw),
            "relative_anomaly": float(rel),
            "class_shift": float(shift),
            "confirm_p": None,
            "confirmed": bool(confirmed and fused >= DETECT_THRESHOLD),
            "suspect": False,
            "context_gated": in_gate,
            "alert_suppressed": suppressed or in_cd,
            "p_mit": None,
            "mitre_stage": stage,
            "mitre_stage_name": stage_name,
            "technique": tech,
            "technique_p": tech_p,
            "ramx_version": self.hx_version,
            "memory_written": bool(memory_written),
        }
