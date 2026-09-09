"""Full SHAP / IG attribution for forecast-demo IPS blocks (Shaun v3 + HX-C)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.explainability.feature_names import shaun_block_indices, shaun_feature_names_292
from src.hx.causal import LOOKBACK, StreamingHXC, _padded_raw
from src.shaun.streaming import StreamingShaunRamxV3, _padded_raw as shaun_padded_raw

FEATURE_NAMES = shaun_feature_names_292()
BLOCKS = shaun_block_indices()
BLOCKER_LABELS = {"sn2rx3": "Shaun v3", "hx_c": "HX-C"}
ROOT = Path(__file__).resolve().parent.parent.parent
SHAP_REPORT_DIR = ROOT / "reports" / "lab" / "hx" / "shap"


class _ShaunAttackWrapper(nn.Module):
    def __init__(self, model, prefix_norm: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("prefix", prefix_norm.clone())

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        b, _d = x_last.shape
        seq = self.prefix.expand(b, -1, -1).clone()
        seq[:, -1, :] = x_last
        _pred_mean, _, atk_logits, _, _, _ = self.model(seq)
        return torch.softmax(atk_logits, dim=-1)[:, 1]


class _ShaunSeqAttackWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _pred_mean, _, atk_logits, _, _, _ = self.model(seq)
        return torch.softmax(atk_logits, dim=-1)[:, 1]


class _HXAttackWrapper(nn.Module):
    def __init__(self, model, prefix_norm: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("prefix", prefix_norm.clone())

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        b, _d = x_last.shape
        seq = self.prefix.expand(b, -1, -1).clone()
        seq[:, -1, :] = x_last
        out = self.model(seq)
        class_p = torch.softmax(out["class_logits"], dim=-1)
        return 1.0 - class_p[:, 0]


class _HXClassWrapper(nn.Module):
    def __init__(self, model, prefix_norm: torch.Tensor, class_idx: int):
        super().__init__()
        self.model = model
        self.class_idx = class_idx
        self.register_buffer("prefix", prefix_norm.clone())

    def forward(self, x_last: torch.Tensor) -> torch.Tensor:
        b, _d = x_last.shape
        seq = self.prefix.expand(b, -1, -1).clone()
        seq[:, -1, :] = x_last
        out = self.model(seq)
        return torch.softmax(out["class_logits"], dim=-1)[:, self.class_idx]


class _HXSeqAttackWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out = self.model(seq)
        class_p = torch.softmax(out["class_logits"], dim=-1)
        return 1.0 - class_p[:, 0]


def _integrated_gradients_signed(
    wrapper: nn.Module,
    x_last: np.ndarray,
    *,
    n_steps: int = 24,
    baseline: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    wrapper.eval()
    device = next(wrapper.parameters()).device
    x = torch.from_numpy(x_last.astype(np.float32)).unsqueeze(0).to(device)
    base = torch.from_numpy((baseline if baseline is not None else np.zeros_like(x_last)).astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        base_p = float(wrapper(base).item())
    acc = torch.zeros_like(x)
    delta = x - base
    for k in range(1, n_steps + 1):
        x_k = (base + (k / n_steps) * delta).detach().requires_grad_(True)
        p = wrapper(x_k).sum()
        grads = torch.autograd.grad(p, x_k)[0]
        acc = acc + grads.detach()
    signed = (delta * (acc / n_steps))[0].cpu().numpy()
    return signed, base_p


def _integrated_gradients_sequence_signed(
    wrapper: nn.Module,
    seq: np.ndarray,
    *,
    n_steps: int = 20,
) -> np.ndarray:
    """Signed IG on full (L, D) sequence; returns per-timestep mass (L,)."""
    wrapper.eval()
    device = next(wrapper.parameters()).device
    x = torch.from_numpy(seq.astype(np.float32)).unsqueeze(0).to(device)
    base = torch.zeros_like(x)
    acc = torch.zeros_like(x)
    delta = x - base
    for k in range(1, n_steps + 1):
        x_k = (base + (k / n_steps) * delta).detach().requires_grad_(True)
        p = wrapper(x_k).sum()
        grads = torch.autograd.grad(p, x_k)[0]
        acc = acc + grads.detach()
    signed = (delta * (acc / n_steps))[0].cpu().numpy()
    per_step = np.abs(signed).sum(axis=1)
    return per_step / (per_step.sum() + 1e-9)


def _shap_gradient_signed(
    wrapper: nn.Module,
    x_last: np.ndarray,
    bg: np.ndarray,
) -> tuple[np.ndarray, float, str] | None:
    try:
        import shap
    except ImportError:
        return None
    wrapper.eval()
    device = next(wrapper.parameters()).device
    try:
        bg_t = torch.from_numpy(bg.astype(np.float32)).to(device)
        explainer = shap.GradientExplainer(wrapper, bg_t)
        xv = torch.from_numpy(x_last.astype(np.float32)).unsqueeze(0).to(device)
        vals = explainer.shap_values(xv)
        if isinstance(vals, list):
            vals = vals[0]
        arr = np.asarray(vals).reshape(-1).astype(np.float64)
        ev = explainer.expected_value
        if isinstance(ev, (list, tuple)):
            ev = ev[0]
        base_p = float(np.asarray(ev).reshape(-1)[0])
        return arr, base_p, "shap_gradient"
    except Exception:
        return None


def _background_last(hx: StreamingHXC, max_n: int = 8) -> np.ndarray:
    m, s = hx.mean.reshape(-1), hx.std.reshape(-1)
    if len(hx.buf) < 2:
        return np.zeros((4, len(m)), dtype=np.float32)
    vecs = []
    for raw in list(hx.buf)[:-1][-max_n:]:
        vecs.append(((raw - m) / s).astype(np.float32))
    while len(vecs) < 4:
        vecs.append(vecs[-1].copy())
    return np.stack(vecs)


def _background_last_shaun(sn: StreamingShaunRamxV3, max_n: int = 8) -> np.ndarray:
    m, s = sn.mean.reshape(-1), sn.std.reshape(-1)
    if len(sn.buf) < 2:
        return np.zeros((4, len(m)), dtype=np.float32)
    vecs = []
    for raw in list(sn.buf)[:-1][-max_n:]:
        vecs.append(((raw - m) / s).astype(np.float32))
    while len(vecs) < 4:
        vecs.append(vecs[-1].copy())
    return np.stack(vecs)


def _attribute_signed(
    wrapper: nn.Module,
    x_last_norm: np.ndarray,
    bg: np.ndarray,
) -> tuple[np.ndarray, float, str]:
    shap_out = _shap_gradient_signed(wrapper, x_last_norm, bg)
    if shap_out is not None:
        signed, base_p, method = shap_out
        return signed, base_p, method
    signed, base_p = _integrated_gradients_signed(wrapper, x_last_norm, baseline=bg.mean(axis=0))
    return signed, base_p, "integrated_gradients"


def _top_rows_abs(scores: np.ndarray, raw_last: np.ndarray, top_n: int = 12) -> list[dict[str, Any]]:
    abs_scores = np.abs(scores)
    order = np.argsort(abs_scores)[::-1][:top_n]
    rows = []
    for i in order:
        idx = int(i)
        rows.append({
            "feature": FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"dim_{idx}",
            "importance": round(float(abs_scores[idx] / (abs_scores.sum() + 1e-9)), 5),
            "shap_value": round(float(scores[idx]), 6),
            "value": round(float(raw_last[idx]), 4),
            "index": idx,
        })
    return rows


def _block_rows(scores: np.ndarray) -> list[dict[str, Any]]:
    abs_scores = np.abs(scores)
    rows = []
    for block, idxs in BLOCKS.items():
        if not idxs:
            continue
        val = float(np.mean(abs_scores[idxs]))
        rows.append({"block": block, "importance": round(val / (abs_scores.sum() + 1e-9), 5)})
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def _waterfall(signed: np.ndarray, raw_last: np.ndarray, base_p: float, output_p: float, top_n: int = 12) -> dict[str, Any]:
    order = np.argsort(np.abs(signed))[::-1][:top_n]
    steps: list[dict[str, Any]] = []
    cum = base_p
    for i in order:
        idx = int(i)
        v = float(signed[idx])
        cum += v
        steps.append({
            "feature": FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"dim_{idx}",
            "shap_value": round(v, 6),
            "feature_value": round(float(raw_last[idx]), 4),
            "cumulative": round(cum, 6),
            "index": idx,
        })
    return {
        "base_value": round(base_p, 6),
        "output_value": round(output_p, 6),
        "steps": steps,
    }


def _temporal_rows(per_step: np.ndarray, lookback: int) -> list[dict[str, Any]]:
    rows = []
    n = len(per_step)
    for i, imp in enumerate(per_step):
        offset = i - (n - 1)
        rows.append({
            "step": i,
            "t_rel_sec": offset,
            "label": "NOW" if offset == 0 else f"{offset}s",
            "importance": round(float(imp), 5),
        })
    return rows


def _class_prob_rows(hx: StreamingHXC, norm: np.ndarray, top_n: int = 6) -> list[dict[str, Any]]:
    x = torch.from_numpy(norm).unsqueeze(0).to(hx.device)
    with torch.no_grad():
        probs = torch.softmax(hx.model(x)["class_logits"], dim=-1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1][:top_n]
    rows = []
    for i in order:
        idx = int(i)
        rows.append({
            "class": hx.class_names[idx] if idx < len(hx.class_names) else f"class_{idx}",
            "p": round(float(probs[idx]), 4),
            "index": int(idx),
        })
    return rows


def _class_shap_top(
    hx: StreamingHXC,
    prefix: torch.Tensor,
    x_last_norm: np.ndarray,
    raw_last: np.ndarray,
    class_idx: int,
    bg: np.ndarray,
    top_n: int = 8,
) -> list[dict[str, Any]]:
    wrapper = _HXClassWrapper(hx.model, prefix, class_idx).to(hx.device)
    signed, _base, _method = _attribute_signed(wrapper, x_last_norm, bg)
    return _top_rows_abs(signed, raw_last, top_n=top_n)


def _contrast_classes(
    pred_rows: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
    top_n: int = 8,
) -> list[dict[str, Any]]:
    pred_map = {r["feature"]: r for r in pred_rows}
    gt_map = {r["feature"]: r for r in gt_rows}
    feats = set(pred_map) | set(gt_map)
    deltas = []
    for f in feats:
        p = pred_map.get(f, {}).get("shap_value", 0.0)
        g = gt_map.get(f, {}).get("shap_value", 0.0)
        deltas.append({
            "feature": f,
            "predicted_shap": p,
            "ground_truth_shap": g,
            "delta": round(p - g, 6),
        })
    deltas.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return deltas[:top_n]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def write_shap_artifacts(report: dict[str, Any], *, window_idx: int | None = None) -> dict[str, str]:
    SHAP_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"w{window_idx:05d}_" if window_idx is not None else ""
    json_path = SHAP_REPORT_DIR / f"forecast_shap_{tag}{ts}.json"
    md_path = SHAP_REPORT_DIR / f"forecast_shap_{tag}{ts}.md"
    json_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")

    lines = [
        f"# Forecast IPS SHAP report — {report.get('blocker_label', '?')}",
        "",
        report.get("narrative", ""),
        "",
        f"- Method: `{report.get('method')}`",
        f"- P(attack): {report.get('p_attack')}",
        f"- Predicted class: {report.get('predicted_class')}",
        f"- Ground truth: {report.get('ground_truth_class')}",
        "",
        "## Waterfall (P attack)",
        "",
        f"Base E[f(x)] = {report.get('waterfall', {}).get('base_value')} → "
        f"f(x) = {report.get('waterfall', {}).get('output_value')}",
        "",
        "| Feature | SHAP | Value | Cumulative |",
        "|---|---:|---:|---:|",
    ]
    for row in report.get("waterfall", {}).get("steps", []):
        lines.append(
            f"| {row['feature']} | {row['shap_value']:+.6f} | {row['feature_value']} | {row['cumulative']:.4f} |"
        )

    classes = report.get("class_analysis") or {}
    if classes.get("top_probs"):
        lines.extend(["", "## Class probabilities", ""])
        for row in classes["top_probs"]:
            lines.append(f"- **{row['class']}**: {row['p']:.4f}")

    if classes.get("contrast"):
        lines.extend([
            "",
            "## Class contrast (predicted vs ground-truth SHAP)",
            "",
            "| Feature | Predicted | Ground truth | Δ |",
            "|---|---:|---:|---:|",
        ])
        for row in classes["contrast"]:
            lines.append(
                f"| {row['feature']} | {row['predicted_shap']:+.5f} | "
                f"{row['ground_truth_shap']:+.5f} | {row['delta']:+.5f} |"
            )

    temporal = report.get("temporal") or {}
    if temporal.get("steps"):
        lines.extend(["", "## Temporal lookback", ""])
        for row in temporal["steps"]:
            lines.append(f"- t={row['label']}: {row['importance']:.4f}")

    lines.extend(["", "## Top feature blocks", ""])
    for row in report.get("top_blocks") or []:
        lines.append(f"- **{row['block']}**: {row['importance']:.4f}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _build_report(
    *,
    blocker: str,
    p_attack: float,
    predicted_class: str | None,
    signed: np.ndarray,
    raw_last: np.ndarray,
    method: str,
    base_p: float,
    ground_truth_class: str | None = None,
    class_analysis: dict[str, Any] | None = None,
    temporal: dict[str, Any] | None = None,
    waterfall: dict[str, Any] | None = None,
) -> dict[str, Any]:
    top_features = _top_rows_abs(signed, raw_last)
    top_blocks = _block_rows(signed)
    label = BLOCKER_LABELS.get(blocker, blocker)
    feat_txt = ", ".join(
        f"{r['feature']} ({r['importance'] * 100:.1f}%)"
        for r in top_features[:5]
    )
    block_txt = ", ".join(f"{r['block']} ({r['importance'] * 100:.1f}%)" for r in top_blocks[:3])
    narrative = (
        f"{label} triggered IPS at P(attack)={p_attack:.3f} "
        f"(2 consecutive windows >= 0.5). "
        f"Top traffic features: {feat_txt}. "
        f"Strongest blocks: {block_txt}."
    )
    if predicted_class and predicted_class not in ("Benign", None):
        narrative += f" Predicted class: {predicted_class}."
    if ground_truth_class and predicted_class and ground_truth_class != predicted_class:
        narrative += (
            f" Ground truth for this kill-chain phase: {ground_truth_class}"
            f" — class-specific SHAP below contrasts the two labels."
        )
    elif ground_truth_class:
        narrative += f" Ground truth: {ground_truth_class}."
    return {
        "blocker": blocker,
        "blocker_label": label,
        "p_attack": round(float(p_attack), 4),
        "predicted_class": predicted_class,
        "ground_truth_class": ground_truth_class,
        "class_match": (
            ground_truth_class is None
            or predicted_class is None
            or ground_truth_class == predicted_class
        ),
        "method": method,
        "full_shap": True,
        "top_features": top_features,
        "top_blocks": top_blocks,
        "waterfall": waterfall or _waterfall(signed, raw_last, base_p, p_attack),
        "class_analysis": class_analysis or {},
        "temporal": temporal or {},
        "narrative": narrative,
    }


def explain_shaun_block(
    sn: StreamingShaunRamxV3,
    *,
    p_attack: float,
    predicted_class: str | None,
) -> dict[str, Any]:
    if not sn.buf:
        return {}
    raw = shaun_padded_raw(sn.buf)
    m = sn.mean.reshape(-1)
    s = sn.std.reshape(-1)
    norm = ((raw - m) / s).astype(np.float32)
    prefix = torch.from_numpy(norm[:-1]).unsqueeze(0).to(sn.device)
    wrapper = _ShaunAttackWrapper(sn.model, prefix).to(sn.device)
    x_last_norm = norm[-1]
    raw_last = raw[-1]
    bg = _background_last_shaun(sn)
    signed, base_p, method = _attribute_signed(wrapper, x_last_norm, bg)
    seq_wrapper = _ShaunSeqAttackWrapper(sn.model).to(sn.device)
    per_step = _integrated_gradients_sequence_signed(seq_wrapper, norm)
    return _build_report(
        blocker="sn2rx3",
        p_attack=p_attack,
        predicted_class=predicted_class,
        signed=signed,
        raw_last=raw_last,
        method=method,
        base_p=base_p,
        temporal={"lookback": len(per_step), "steps": _temporal_rows(per_step, len(per_step))},
    )


def explain_hx_block(
    hx: StreamingHXC,
    *,
    p_attack: float,
    predicted_class: str | None,
    ground_truth_class: str | None = None,
) -> dict[str, Any]:
    if not hx.buf:
        return {}
    raw = _padded_raw(hx.buf, LOOKBACK)
    m = hx.mean.reshape(-1)
    s = hx.std.reshape(-1)
    norm = ((raw - m) / s).astype(np.float32)
    prefix = torch.from_numpy(norm[:-1]).unsqueeze(0).to(hx.device)
    x_last_norm = norm[-1]
    raw_last = raw[-1]
    bg = _background_last(hx)

    attack_wrapper = _HXAttackWrapper(hx.model, prefix).to(hx.device)
    signed, base_p, method = _attribute_signed(attack_wrapper, x_last_norm, bg)

    top_probs = _class_prob_rows(hx, norm)
    pred_idx = top_probs[0]["index"] if top_probs else 0
    pred_name = top_probs[0]["class"] if top_probs else None
    if predicted_class:
        try:
            pred_idx = hx.class_names.index(predicted_class)
            pred_name = predicted_class
        except ValueError:
            pred_name = predicted_class

    pred_class_shap = _class_shap_top(hx, prefix, x_last_norm, raw_last, pred_idx, bg)
    gt_class_shap: list[dict[str, Any]] = []
    contrast: list[dict[str, Any]] = []
    gt_idx: int | None = None
    if ground_truth_class:
        try:
            gt_idx = hx.class_names.index(ground_truth_class)
            gt_class_shap = _class_shap_top(hx, prefix, x_last_norm, raw_last, gt_idx, bg)
            if ground_truth_class != pred_name:
                contrast = _contrast_classes(pred_class_shap, gt_class_shap)
        except ValueError:
            pass

    seq_wrapper = _HXSeqAttackWrapper(hx.model).to(hx.device)
    per_step = _integrated_gradients_sequence_signed(seq_wrapper, norm)

    class_analysis = {
        "top_probs": top_probs,
        "predicted_class_shap": pred_class_shap,
        "ground_truth_class_shap": gt_class_shap,
        "contrast": contrast,
    }

    return _build_report(
        blocker="hx_c",
        p_attack=p_attack,
        predicted_class=pred_name or predicted_class,
        signed=signed,
        raw_last=raw_last,
        method=method,
        base_p=base_p,
        ground_truth_class=ground_truth_class,
        class_analysis=class_analysis,
        temporal={"lookback": LOOKBACK, "steps": _temporal_rows(per_step, LOOKBACK)},
    )


def explain_ips_block(
    *,
    first_blocker: str,
    sn: StreamingShaunRamxV3,
    hx: StreamingHXC,
    scores: dict[str, float],
    hx_meta: dict[str, Any] | None = None,
    sn_meta: dict[str, Any] | None = None,
    ground_truth_class: str | None = None,
    window_idx: int | None = None,
) -> dict[str, Any]:
    """Build full SHAP report for the model that won the IPS race."""
    hx_meta = hx_meta or {}
    sn_meta = sn_meta or {}
    if first_blocker == "hx_c":
        report = explain_hx_block(
            hx,
            p_attack=float(scores.get("hx_c", 0.0)),
            predicted_class=hx_meta.get("technique") or hx_meta.get("mitre_stage_name"),
            ground_truth_class=ground_truth_class,
        )
    else:
        report = explain_shaun_block(
            sn,
            p_attack=float(scores.get("sn2rx3", 0.0)),
            predicted_class=sn_meta.get("mitre_stage_name"),
        )
    if report:
        other = "sn2rx3" if first_blocker == "hx_c" else "hx_c"
        report["scores_at_block"] = {k: round(float(v), 4) for k, v in scores.items()}
        report["runner_up"] = BLOCKER_LABELS.get(other, other)
        report["runner_up_p"] = round(float(scores.get(other, 0.0)), 4)
        report["report_paths"] = write_shap_artifacts(report, window_idx=window_idx)
        report = _json_safe(report)
    return report
