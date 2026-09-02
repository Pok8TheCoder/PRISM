"""Integrated Gradients (default) and optional Kernel SHAP for Aryan WM heads."""

from __future__ import annotations

import numpy as np
import torch


def feature_names(d_state: int) -> list[str]:
    return [f"feature[{i}]" for i in range(d_state)]


def integrated_gradients_infiltration(
    adapter,
    window: np.ndarray,
    n_steps: int = 16,
    top_n: int = 12,
) -> list[tuple[str, float]]:
    """IG on P(attack) for one lookback window, shape (L, D)."""
    model = adapter.inner
    model.eval()
    x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
    baseline = torch.zeros_like(x)
    acc = torch.zeros_like(x)
    delta = x - baseline
    for k in range(1, n_steps + 1):
        x_k = (baseline + (k / n_steps) * delta).detach().requires_grad_(True)
        out = model(x_k)
        p = torch.softmax(out["pred_binary"], dim=-1)[:, 1].sum()
        grads = torch.autograd.grad(p, x_k)[0]
        acc = acc + grads.detach()
    ig = delta * (acc / n_steps)
    scores = ig.abs().mean(dim=1)[0].cpu().numpy()
    scores = scores / (scores.sum() + 1e-9)
    names = feature_names(window.shape[1])
    order = np.argsort(scores)[::-1][:top_n]
    return [(names[i], float(scores[i])) for i in order]


def shap_infiltration(
    adapter,
    window: np.ndarray,
    background: np.ndarray,
    top_n: int = 12,
    nsamples: int = 32,
) -> list[tuple[str, float]] | None:
    """Kernel SHAP on mean-pooled window vector. Slow — call on demand."""
    try:
        import shap
    except ImportError:
        return None
    L, D = window.shape
    last = window.mean(axis=0).astype(np.float32)

    def _predict(batch: np.ndarray) -> np.ndarray:
        seqs = []
        for row in batch:
            tiled = np.repeat(row.reshape(1, -1), L, axis=0).astype(np.float32)
            seqs.append(tiled)
        t = torch.from_numpy(np.stack(seqs))
        adapter.eval()
        with torch.no_grad():
            _, p_att, _ = adapter.forward_full(t)
        return p_att.detach().cpu().numpy().reshape(-1, 1)

    try:
        bg = background.reshape(len(background), -1)[:, :D]
        if bg.shape[1] != D:
            bg = np.zeros((min(8, len(background)), D), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict, bg[:8])
        vals = explainer.shap_values(last.reshape(1, -1), nsamples=nsamples)
        arr = np.abs(np.array(vals).reshape(-1, D)[0])
        arr = arr / (arr.sum() + 1e-9)
        names = feature_names(D)
        order = np.argsort(arr)[::-1][:top_n]
        return [(names[i], float(arr[i])) for i in order]
    except Exception:
        return None
