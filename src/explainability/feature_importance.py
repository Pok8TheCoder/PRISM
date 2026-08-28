"""
PRISM - Feature Importance
Gradient-based and permutation-based feature importance for
model-level (global) and prediction-level (local) attribution.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("prism.explainability.feature_importance")


class FeatureImportanceAnalyser:
    """
    Compute global and local feature importance for the PRISM world model.

    Methods:
      - Integrated Gradients (local, precise)
      - Gradient × Input (local, fast)
      - Permutation Importance (global, model-agnostic)
    """

    def __init__(
        self,
        model,
        feature_names: Optional[list[str]] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.device = device

    # ------------------------------------------------------------------
    # Integrated Gradients
    # ------------------------------------------------------------------
    def integrated_gradients(
        self,
        state_seq: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        n_steps: int = 50,
    ) -> dict:
        """
        Compute Integrated Gradients attribution.

        IG(x) = (x - x') * integral_{alpha=0}^{1} [grad F(x' + alpha*(x-x'))] dalpha

        Parameters
        ----------
        state_seq : (L, D_state) — input to explain
        baseline  : (L, D_state) — reference baseline (default: zeros)
        n_steps   : number of interpolation steps (50 is usually sufficient)

        Returns
        -------
        dict with attributions (L, D_state) and mean_abs (D_state,)
        """
        import torch

        arr = np.array(state_seq, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]   # (1, L, D)

        if baseline is None:
            baseline_arr = np.zeros_like(arr)
        else:
            baseline_arr = np.array(baseline, dtype=np.float32)
            if baseline_arr.ndim == 2:
                baseline_arr = baseline_arr[np.newaxis]

        x = torch.tensor(arr, dtype=torch.float32, device=self.device)
        x_base = torch.tensor(baseline_arr, dtype=torch.float32, device=self.device)

        # Interpolated inputs
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        grads = []

        for alpha in alphas:
            interp = x_base + alpha * (x - x_base)
            interp.requires_grad_(True)

            out = self.model(interp)
            score = torch.softmax(out["pred_binary"], dim=-1)[:, 1].sum()
            score.backward()

            grads.append(interp.grad.cpu().numpy().copy())

        # Riemann sum approximation
        avg_grads = np.mean(grads, axis=0)           # (1, L, D)
        delta = arr - baseline_arr                    # (1, L, D)
        attributions = avg_grads * delta              # (1, L, D)

        mean_abs = np.abs(attributions[0]).mean(axis=0)  # (D,)
        top_features = self._format_top_features(
            mean_abs, attributions[0].mean(axis=0), arr[0, -1]
        )

        return {
            "attributions": attributions,
            "mean_abs": mean_abs,
            "top_features": top_features,
            "method": "integrated_gradients",
        }

    # ------------------------------------------------------------------
    # Gradient × Input
    # ------------------------------------------------------------------
    def gradient_x_input(self, state_seq: np.ndarray) -> dict:
        """
        Fast attribution: gradient * input at the observed point.
        Less precise than IG but much faster.
        """
        import torch

        arr = np.array(state_seq, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        x = torch.tensor(arr, dtype=torch.float32, device=self.device)
        x.requires_grad_(True)

        out = self.model(x)
        score = torch.softmax(out["pred_binary"], dim=-1)[:, 1].sum()
        score.backward()

        grads = x.grad.cpu().numpy()        # (1, L, D)
        attrs = grads * arr                 # (1, L, D) — grad × input
        mean_abs = np.abs(attrs[0]).mean(axis=0)  # (D,)

        top_features = self._format_top_features(
            mean_abs, attrs[0].mean(axis=0), arr[0, -1]
        )

        return {
            "attributions": attrs,
            "mean_abs": mean_abs,
            "top_features": top_features,
            "method": "gradient_x_input",
        }

    # ------------------------------------------------------------------
    # Global Permutation Importance
    # ------------------------------------------------------------------
    def permutation_importance(
        self,
        state_sequences: np.ndarray,
        labels: np.ndarray,
        n_repeats: int = 5,
    ) -> dict:
        """
        Global permutation feature importance over a dataset.

        Randomly shuffles each feature across samples and measures
        the drop in infiltration prediction accuracy.

        Parameters
        ----------
        state_sequences : (N, L, D_state)
        labels          : (N,) binary attack labels
        n_repeats       : number of permutation repeats per feature

        Returns
        -------
        dict with importances (D_state,), std (D_state,), and top_features list
        """
        import torch

        N, L, D = state_sequences.shape
        base_acc = self._compute_accuracy(state_sequences, labels)
        logger.info(
            "Permutation importance: N=%d, D=%d, base_acc=%.4f",
            N, D, base_acc,
        )

        importances = np.zeros(D)
        stds = np.zeros(D)

        for feat_idx in range(D):
            drops = []
            for _ in range(n_repeats):
                shuffled = state_sequences.copy()
                # Shuffle this feature across samples
                perm = np.random.permutation(N)
                shuffled[:, :, feat_idx] = shuffled[perm, :, feat_idx]
                acc = self._compute_accuracy(shuffled, labels)
                drops.append(base_acc - acc)

            importances[feat_idx] = np.mean(drops)
            stds[feat_idx] = np.std(drops)

        top_features = self._format_top_features(
            importances, importances, np.zeros(D), stds=stds
        )

        return {
            "importances": importances,
            "stds": stds,
            "top_features": top_features,
            "baseline_accuracy": base_acc,
            "method": "permutation_importance",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _compute_accuracy(
        self, state_sequences: np.ndarray, labels: np.ndarray
    ) -> float:
        import torch
        preds = []
        bs = 64
        for i in range(0, len(state_sequences), bs):
            batch = torch.tensor(
                state_sequences[i : i + bs], dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                out = self.model(batch)
                pred = torch.softmax(out["pred_binary"], dim=-1)[:, 1] > 0.5
                preds.extend(pred.cpu().numpy().tolist())
        preds = np.array(preds, dtype=int)
        return float((preds == labels[: len(preds)]).mean())

    def _format_top_features(
        self,
        mean_abs: np.ndarray,
        raw_attr: np.ndarray,
        last_state: np.ndarray,
        top_k: int = 15,
        stds: Optional[np.ndarray] = None,
    ) -> list[dict]:
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        result = []
        for rank, idx in enumerate(top_idx):
            name = (
                self.feature_names[idx]
                if idx < len(self.feature_names)
                else f"feature_{idx}"
            )
            entry = {
                "rank": rank + 1,
                "feature": name,
                "feature_index": int(idx),
                "importance": float(mean_abs[idx]),
                "attribution": float(raw_attr[idx]),
                "direction": "positive" if raw_attr[idx] > 0 else "negative",
                "feature_value": float(last_state[idx]) if idx < len(last_state) else 0.0,
            }
            if stds is not None:
                entry["std"] = float(stds[idx])
            result.append(entry)
        return result

    def plot_global_importance(
        self,
        importances: np.ndarray,
        stds: Optional[np.ndarray] = None,
        top_k: int = 20,
        title: str = "Global Feature Importance",
        save_path: Optional[str] = None,
    ):
        """Bar chart of global permutation importance."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        top_idx = np.argsort(importances)[::-1][:top_k]
        names = [
            self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
            for i in top_idx
        ]
        vals = importances[top_idx]
        errs = stds[top_idx] if stds is not None else None

        fig, ax = plt.subplots(figsize=(10, max(4, top_k * 0.35)))
        ax.barh(
            names[::-1], vals[::-1],
            xerr=errs[::-1] if errs is not None else None,
            color="#3498db", alpha=0.85,
        )
        ax.set_xlabel("Importance (accuracy drop when permuted)")
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved feature importance plot -> %s", save_path)

        return fig
