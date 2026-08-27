"""SHAP, gradient, and attention feature attribution."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.pipeline.features import FEATURE_COLS, NUM_FEATURES


def _tile_to_sequence(features: np.ndarray, seq_len: int) -> np.ndarray:
    if len(features) < seq_len:
        pad = np.zeros((seq_len - len(features), NUM_FEATURES), dtype=np.float32)
        window = np.vstack([pad, features])
    else:
        window = features[-seq_len:]
    return window.astype(np.float32)


def gradient_attribution(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    seq_len: int,
    top_n: int = 8,
) -> list[tuple[str, float]]:
    """Input-gradient attribution on the last sequence window."""
    window = _tile_to_sequence(features, seq_len)
    norm = ((window - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)
    x = torch.from_numpy(norm).unsqueeze(0).to(device)
    x.requires_grad_(True)
    model.eval()
    _, logits = model(x)
    pred = int(logits.argmax(1).item())
    model.zero_grad(set_to_none=True)
    logits[0, pred].backward()
    if x.grad is None:
        return []
    scores = x.grad.abs().mean(dim=1)[0].detach().cpu().numpy()
    scores = scores / (scores.sum() + 1e-9)
    order = np.argsort(scores)[::-1][:top_n]
    return [(FEATURE_COLS[i], float(scores[i])) for i in order]


def attention_attribution(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    seq_len: int,
    top_n: int = 8,
) -> list[tuple[str, float]]:
    """Last-layer self-attention pooled across time, mapped onto feature energy."""
    window = _tile_to_sequence(features, seq_len)
    norm = ((window - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)
    x = torch.from_numpy(norm).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        h = model.input_proj(x) + model.pos_emb
        layers = list(model.transformer.layers)
        for layer in layers[:-1]:
            h = layer(h)
        last = layers[-1]
        _, weights = last.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        # weights: (batch, seq, seq) — importance of each timestep for the last token
        time_w = weights[0, -1].cpu().numpy()
        feat_energy = np.abs(norm) * time_w[:, None]
        scores = feat_energy.mean(axis=0)
    scores = scores / (scores.sum() + 1e-9)
    order = np.argsort(scores)[::-1][:top_n]
    return [(FEATURE_COLS[i], float(scores[i])) for i in order]


def shap_attribution(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    seq_len: int,
    top_n: int = 8,
    nsamples: int = 64,
) -> list[tuple[str, float]] | None:
    """Kernel SHAP on the last flow vector (tiled across the sequence)."""
    try:
        import shap
    except ImportError:
        return None

    window = _tile_to_sequence(features, seq_len)
    last = window[-1].astype(np.float32)

    def _predict(batch: np.ndarray) -> np.ndarray:
        seqs = []
        for row in batch:
            tiled = np.repeat(row.reshape(1, -1), seq_len, axis=0).astype(np.float32)
            tiled = ((tiled - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)
            seqs.append(tiled)
        t = torch.from_numpy(np.stack(seqs)).to(device)
        model.eval()
        with torch.no_grad():
            _, logits = model(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    try:
        background = np.zeros((8, NUM_FEATURES), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict, background)
        shap_values = explainer.shap_values(last.reshape(1, -1), nsamples=nsamples)
        if isinstance(shap_values, list):
            pred = int(np.argmax(_predict(last.reshape(1, -1))[0]))
            vals = np.abs(np.array(shap_values[pred]).reshape(-1))
        else:
            arr = np.array(shap_values)
            vals = np.abs(arr.reshape(-1, NUM_FEATURES)[0])
        vals = vals / (vals.sum() + 1e-9)
        order = np.argsort(vals)[::-1][:top_n]
        return [(FEATURE_COLS[i], float(vals[i])) for i in order]
    except Exception:
        return None


def explain_prediction(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    seq_len: int,
    top_n: int = 8,
) -> dict[str, Any]:
    """Return SHAP (if available), gradients, and attention driving features."""
    grad = gradient_attribution(model, scaler, features, device, seq_len, top_n)
    attn = attention_attribution(model, scaler, features, device, seq_len, top_n)
    shap_vals = shap_attribution(model, scaler, features, device, seq_len, top_n)
    primary = shap_vals or grad
    return {
        "primary": primary,
        "method": "shap" if shap_vals else "gradient",
        "shap": shap_vals,
        "gradient": grad,
        "attention": attn,
    }
