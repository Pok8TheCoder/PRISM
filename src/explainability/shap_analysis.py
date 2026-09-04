"""SHAP / gradient attribution for ARY and Shaun attack classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from src.aryan.feature_schema242 import FEATURE_COLS_242
from src.aryan.world_model import TemporalTransformerWorldModel
from src.explainability.feature_names import ary_block_indices, shaun_block_indices, shaun_feature_names_292


@dataclass
class ModelSpec:
    key: str
    label: str
    architecture: str  # "ary" | "shaun"
    ckpt: Path
    lookback: int
    splits_dir: Path | None = None
    shaun_root: Path | None = None
    feature_names: list[str] = field(default_factory=list)
    block_indices: dict[str, list[int]] = field(default_factory=dict)
    notes: str = ""


@dataclass
class SequenceSample:
    seq: np.ndarray  # (L, D) raw (unnormalized for shaun; ary uses split vectors as-is)
    label: int
    index: int
    p_attack: float = 0.0
    bucket: str = ""


def load_ary_model(ckpt: Path, lookback: int = 20) -> TemporalTransformerWorldModel:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(
        d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=lookback,
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def load_shaun_bundle(shaun_root: Path) -> dict[str, Any]:
    ck_path = shaun_root / "weights" / "world_model.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["input_embed.0.weight"].shape[1]
    pe_len = sd.get("pos_encoder.pe", torch.zeros(1, 50, 256)).shape[1]

    import os
    import sys

    prev_cwd = os.getcwd()
    prev_path = sys.path.copy()
    os.chdir(shaun_root)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path = [str(shaun_root.resolve())] + [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent.parent.parent]
    try:
        from src.models.world_model import StateTransformerWorldModel

        model = StateTransformerWorldModel(
            d_state=d_state, d_model=256, nhead=8, num_layers=4, max_seq_len=pe_len,
        )
        model.load_state_dict(sd)
        model.eval()
    finally:
        os.chdir(prev_cwd)
        sys.path = prev_path

    mean = np.asarray(ck.get("scaler_mean", np.zeros((1, d_state))), dtype=np.float32)
    std = np.asarray(ck.get("scaler_std", np.ones((1, d_state))), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return {"model": model, "mean": mean, "std": std}


def ary_attack_prob(model: nn.Module, seq: np.ndarray) -> float:
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        return float(torch.softmax(out["pred_binary"], dim=-1)[0, 1].item())


def shaun_attack_prob(bundle: dict[str, Any], seq: np.ndarray) -> float:
    m = bundle["mean"].reshape(-1)
    s = bundle["std"].reshape(-1)
    norm = ((seq - m) / s).astype(np.float32)
    x = torch.from_numpy(norm).unsqueeze(0)
    with torch.no_grad():
        _, _, logits, _, _, _ = bundle["model"](x)
        return float(torch.softmax(logits, dim=-1)[0, 1].item())


class LastStepAttackWrapper(nn.Module):
    """Explain current-window features with fixed historical prefix."""

    def __init__(
        self,
        architecture: str,
        model: nn.Module,
        prefix: torch.Tensor,
        shaun_mean: np.ndarray | None = None,
        shaun_std: np.ndarray | None = None,
    ):
        super().__init__()
        self.architecture = architecture
        self.model = model
        self.register_buffer("prefix", prefix.clone())
        if shaun_mean is not None and shaun_std is not None:
            self.register_buffer("shaun_mean", torch.from_numpy(shaun_mean.reshape(1, -1).astype(np.float32)))
            self.register_buffer("shaun_std", torch.from_numpy(shaun_std.reshape(1, -1).astype(np.float32)))
        else:
            self.shaun_mean = None
            self.shaun_std = None

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        # x_last: (batch, D)
        b, d = x_last.shape
        seq = self.prefix.expand(b, -1, -1).clone()
        seq[:, -1, :] = x_last
        if self.architecture == "shaun":
            assert self.shaun_mean is not None and self.shaun_std is not None
            seq = (seq - self.shaun_mean) / self.shaun_std
            _, _, logits, _, _, _ = self.model(seq)
            return torch.softmax(logits, dim=-1)[:, 1]
        out = self.model(seq)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]


def build_labeled_sequences(
    states: np.ndarray,
    labels: np.ndarray,
    lookback: int,
    *,
    label_offset: int = 1,
) -> list[SequenceSample]:
    """Build causal windows; label at t+label_offset matches ARY eval convention."""
    out: list[SequenceSample] = []
    for t in range(lookback - 1, len(states) - label_offset):
        seq = np.asarray(states[t - lookback + 1 : t + 1], dtype=np.float32)
        y = int(labels[t + label_offset])
        out.append(SequenceSample(seq=seq, label=y, index=t + label_offset))
    return out


def assign_buckets(samples: list[SequenceSample], scorer: Callable[[np.ndarray], float], threshold: float = 0.5) -> None:
    for s in samples:
        s.p_attack = scorer(s.seq)
        pred = 1 if s.p_attack >= threshold else 0
        if s.label == 1 and pred == 1:
            s.bucket = "tp"
        elif s.label == 0 and pred == 0:
            s.bucket = "tn"
        elif s.label == 0 and pred == 1:
            s.bucket = "fp"
        else:
            s.bucket = "fn"


def sample_by_bucket(samples: list[SequenceSample], max_per_bucket: int, seed: int = 0) -> dict[str, list[SequenceSample]]:
    rng = np.random.default_rng(seed)
    by: dict[str, list[SequenceSample]] = {k: [] for k in ("tp", "tn", "fp", "fn")}
    for s in samples:
        if s.bucket in by:
            by[s.bucket].append(s)
    out: dict[str, list[SequenceSample]] = {}
    for k, rows in by.items():
        if not rows:
            out[k] = []
            continue
        idx = rng.choice(len(rows), size=min(max_per_bucket, len(rows)), replace=False)
        out[k] = [rows[i] for i in idx]
    return out


def gradient_input_attribution(wrapper: LastStepAttackWrapper, x_last: np.ndarray) -> np.ndarray:
    wrapper.eval()
    x = torch.from_numpy(x_last.astype(np.float32)).unsqueeze(0).clone().detach().requires_grad_(True)
    prob = wrapper(x)
    prob.backward()
    grad = x.grad.detach().cpu().numpy()[0]
    return grad * x_last


def shap_attribution(
    wrapper: LastStepAttackWrapper,
    background: np.ndarray,
    x_last: np.ndarray,
) -> np.ndarray:
    import shap

    wrapper.eval()
    bg = torch.from_numpy(background.astype(np.float32))
    explainer = shap.GradientExplainer(wrapper, bg)
    xv = torch.from_numpy(x_last.astype(np.float32)).unsqueeze(0)
    values = explainer.shap_values(xv)
    if isinstance(values, list):
        values = values[0]
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[0]
    return arr.reshape(-1)


def aggregate_blocks(mean_abs: np.ndarray, block_indices: dict[str, list[int]]) -> dict[str, float]:
    return {
        block: float(np.mean(mean_abs[idx])) if idx else 0.0
        for block, idx in block_indices.items()
    }


def top_features(mean_abs: np.ndarray, feature_names: list[str], k: int = 15) -> list[dict[str, Any]]:
    order = np.argsort(mean_abs)[::-1][:k]
    return [
        {
            "rank": i + 1,
            "feature": feature_names[j] if j < len(feature_names) else f"dim_{j}",
            "mean_abs_attr": float(mean_abs[j]),
            "index": int(j),
        }
        for i, j in enumerate(order)
    ]


def explain_samples(
    samples: list[SequenceSample],
    wrapper_factory: Callable[[SequenceSample], LastStepAttackWrapper],
    background_last_steps: np.ndarray,
    feature_names: list[str],
    *,
    method: str = "both",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not samples:
        return np.zeros(0), []

    per_sample: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for s in samples:
        wrapper = wrapper_factory(s)
        attrs: list[np.ndarray] = []
        if method in ("gradient", "both"):
            attrs.append(np.abs(gradient_input_attribution(wrapper, s.seq[-1])))
        if method in ("shap", "both"):
            try:
                attrs.append(np.abs(shap_attribution(wrapper, background_last_steps, s.seq[-1])))
            except Exception as exc:
                if method == "shap":
                    raise
                rows.append({"index": s.index, "warning": f"shap_failed: {exc}"})
        if not attrs:
            continue
        combined = np.mean(attrs, axis=0)
        per_sample.append(combined)
        rows.append({
            "index": s.index,
            "label": s.label,
            "p_attack": round(s.p_attack, 4),
            "bucket": s.bucket,
            "top_features": top_features(combined, feature_names, k=8),
        })

    if not per_sample:
        d = samples[0].seq.shape[1]
        return np.zeros(d), rows

    mean_abs = np.mean(np.stack(per_sample), axis=0)
    return mean_abs, rows


def failure_delta(top_fn: np.ndarray, top_tp: np.ndarray, feature_names: list[str], k: int = 20) -> list[dict[str, Any]]:
    delta = top_fn - top_tp
    order = np.argsort(np.abs(delta))[::-1][:k]
    return [
        {
            "feature": feature_names[i] if i < len(feature_names) else f"dim_{i}",
            "index": int(i),
            "fn_minus_tp": float(delta[i]),
            "fn_importance": float(top_fn[i]),
            "tp_importance": float(top_tp[i]),
        }
        for i in order
    ]


def write_report_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# SHAP / Attribution Report: {payload['model']['label']}",
        "",
        f"Architecture: `{payload['model']['architecture']}` | lookback={payload['model']['lookback']}",
        f"Method: `{payload['method']}` | threshold={payload['threshold']}",
        "",
        "## Bucket counts (sampled eval set)",
        "",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for k, v in payload["bucket_counts"].items():
        lines.append(f"| {k} | {v} |")

    lines.extend(["", "## Mean |attribution| by feature block", ""])
    for bucket, blocks in payload.get("block_means", {}).items():
        lines.append(f"### {bucket}")
        lines.append("")
        for block, val in blocks.items():
            lines.append(f"- **{block}**: {val:.6f}")
        lines.append("")

    if payload.get("failure_analysis"):
        fa = payload["failure_analysis"]
        lines.extend([
            "## Failure analysis (FN vs TP attribution delta)",
            "",
            "Features with largest |FN − TP| mean |attribution| — candidates for root cause.",
            "",
            "| Feature | FN−TP delta | FN imp | TP imp |",
            "|---|---:|---:|---:|",
        ])
        for row in fa.get("top_deltas", [])[:15]:
            lines.append(
                f"| {row['feature']} | {row['fn_minus_tp']:+.6f} | "
                f"{row['fn_importance']:.6f} | {row['tp_importance']:.6f} |"
            )

    if payload["model"].get("notes"):
        lines.extend(["", "## Notes", "", payload["model"]["notes"]])

    path.write_text("\n".join(lines), encoding="utf-8")
