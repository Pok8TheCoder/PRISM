"""
PRISM - SHAP Explainer
SHAP-based feature attribution for world model predictions.
Identifies which specific traffic features drive the infiltration score,
satisfying the interpretability requirement.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("prism.explainability.shap_explainer")


class PRISMShapExplainer:
    """
    SHAP-based feature attribution for the PRISM world model.

    Uses shap.DeepExplainer (fast, PyTorch-native) when possible,
    falls back to shap.KernelExplainer (model-agnostic) otherwise.
    """

    def __init__(
        self,
        model,
        background_data: np.ndarray,
        feature_names: Optional[list[str]] = None,
        device: str = "cpu",
        use_deep_explainer: bool = True,
    ):
        """
        Parameters
        ----------
        model           : trained world model (PyTorch nn.Module)
        background_data : np.ndarray, shape (N_bg, L, D_state)
                          Sample of background (training set) state sequences.
                          N_bg typically 50-200 samples.
        feature_names   : list of feature names (length D_state)
        device          : "cpu" | "cuda"
        use_deep_explainer: if True, use DeepExplainer; else KernelExplainer
        """
        import shap
        import torch

        self.model = model
        self.feature_names = feature_names or []
        self.device = device
        self._explainer = None

        # Wrap model for SHAP: takes (B, L, D) -> infiltration prob scalar
        self._torch_model = _SHAPModelWrapper(model, device)

        bg_tensor = torch.tensor(background_data, dtype=torch.float32)

        if use_deep_explainer:
            try:
                self._explainer = shap.DeepExplainer(
                    self._torch_model, bg_tensor
                )
                self._mode = "deep"
                logger.info(
                    "SHAP DeepExplainer initialised with %d background samples",
                    len(background_data),
                )
            except Exception as e:
                logger.warning(
                    "DeepExplainer failed (%s); falling back to KernelExplainer.", e
                )
                use_deep_explainer = False

        if not use_deep_explainer:
            # Flatten for KernelExplainer
            bg_flat = background_data.reshape(len(background_data), -1)

            def predict_fn(x_flat):
                import torch
                x_3d = x_flat.reshape(
                    len(x_flat), background_data.shape[1], background_data.shape[2]
                )
                t = torch.tensor(x_3d, dtype=torch.float32)
                with torch.no_grad():
                    prob = self._torch_model(t)
                return prob.numpy()

            self._explainer = shap.KernelExplainer(
                predict_fn,
                shap.kmeans(bg_flat, k=min(50, len(bg_flat))),
            )
            self._mode = "kernel"
            self._bg_shape = background_data.shape
            logger.info(
                "SHAP KernelExplainer initialised (mode=kernel, bg=%d)",
                len(background_data),
            )

    def explain(
        self,
        state_sequence: np.ndarray,
        top_k: int = 10,
    ) -> dict:
        """
        Compute SHAP values for a single state sequence prediction.

        Parameters
        ----------
        state_sequence : np.ndarray, shape (L, D_state) or (1, L, D_state)
        top_k          : number of top features to return

        Returns
        -------
        dict:
            shap_values       : np.ndarray, same shape as input
            mean_abs_shap     : np.ndarray, (D_state,) — averaged over L steps
            top_features      : list of dicts with feature, value, shap_value, direction
            infiltration_prob : float — model prediction for this input
            explanation_text  : str — human-readable summary
        """
        import torch

        arr = np.array(state_sequence, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]   # (1, L, D)

        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32)
            inf_prob = float(self._torch_model(t)[0])

        if self._mode == "deep":
            shap_vals = self._explainer.shap_values(
                torch.tensor(arr, dtype=torch.float32)
            )
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_arr = np.array(shap_vals)
        else:
            arr_flat = arr.reshape(1, -1)
            shap_vals = self._explainer.shap_values(arr_flat, nsamples=100)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            shap_arr = np.array(shap_vals).reshape(arr.shape)

        # Average absolute SHAP over L time steps -> (D_state,)
        mean_abs = np.abs(shap_arr[0]).mean(axis=0)

        top_features = self._format_top_features(
            mean_abs,
            raw_shap=shap_arr[0].mean(axis=0),
            state_last=arr[0, -1],
            top_k=top_k,
        )

        explanation_text = self._build_explanation_text(
            top_features, inf_prob
        )

        return {
            "shap_values": shap_arr,
            "mean_abs_shap": mean_abs,
            "top_features": top_features,
            "infiltration_prob": inf_prob,
            "explanation_text": explanation_text,
        }

    def explain_batch(
        self, state_sequences: np.ndarray, top_k: int = 10
    ) -> list[dict]:
        """Explain a batch of state sequences."""
        results = []
        for i in range(state_sequences.shape[0]):
            results.append(self.explain(state_sequences[i], top_k))
        return results

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    def _format_top_features(
        self,
        mean_abs: np.ndarray,
        raw_shap: np.ndarray,
        state_last: np.ndarray,
        top_k: int,
    ) -> list[dict]:
        """Build sorted top-K feature list with SHAP values."""
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        result = []

        for rank, idx in enumerate(top_idx):
            name = (
                self.feature_names[idx]
                if idx < len(self.feature_names)
                else f"feature_{idx}"
            )
            shap_val = float(raw_shap[idx])
            feature_val = float(state_last[idx]) if idx < len(state_last) else 0.0
            direction = "increases" if shap_val > 0 else "decreases"

            result.append({
                "rank": rank + 1,
                "feature": name,
                "feature_value": feature_val,
                "shap_value": shap_val,
                "abs_shap": float(mean_abs[idx]),
                "direction": direction,
                "explanation": self._feature_explanation(name, shap_val, feature_val),
            })

        return result

    @staticmethod
    def _feature_explanation(
        name: str, shap_val: float, feature_val: float
    ) -> str:
        """Generate a one-line explanation for a feature's SHAP contribution."""
        direction = "increases" if shap_val > 0 else "reduces"
        abs_s = abs(shap_val)

        if "syn" in name.lower() and shap_val > 0:
            return f"High SYN flag rate ({feature_val:.3f}) {direction} attack probability — suggests active scanning."
        if "port_entropy" in name.lower() and shap_val > 0:
            return f"High port diversity ({feature_val:.3f}) {direction} attack probability — consistent with port scanning."
        if "iat_std" in name.lower() and shap_val > 0:
            return f"Low/high IAT variance ({feature_val:.3f}) {direction} attack probability — irregular timing detected."
        if "ttl_std" in name.lower() and shap_val > 0:
            return f"TTL variance ({feature_val:.3f}) {direction} attack probability — possible OS spoofing."
        if "rst" in name.lower() and shap_val > 0:
            return f"RST rate ({feature_val:.3f}) {direction} attack probability — repeated connection resets observed."
        if "bytes" in name.lower() and shap_val > 0:
            return f"Byte transfer ({feature_val:.3f}) {direction} attack probability — abnormal data volume."
        return (
            f"Feature '{name}' = {feature_val:.3f} {direction} "
            f"attack probability (SHAP={shap_val:+.4f})."
        )

    @staticmethod
    def _build_explanation_text(
        top_features: list[dict], inf_prob: float
    ) -> str:
        """Build concise human-readable explanation paragraph."""
        if not top_features:
            return f"Infiltration probability: {inf_prob:.1%}. No feature attribution available."

        top3 = top_features[:3]
        feature_desc = "; ".join(
            f"{f['feature']} ({'+' if f['shap_value'] > 0 else ''}{f['shap_value']:.3f})"
            for f in top3
        )
        verdict = (
            "High confidence attack trajectory" if inf_prob > 0.65
            else "Moderate threat indicators" if inf_prob > 0.40
            else "Low-level anomaly"
        )

        return (
            f"{verdict} detected (P={inf_prob:.1%}). "
            f"Primary drivers: {feature_desc}. "
            f"Top feature: {top3[0]['explanation']}"
        )

    def plot_waterfall(
        self,
        explain_result: dict,
        max_display: int = 10,
        save_path: Optional[str] = None,
    ):
        """
        Plot SHAP waterfall chart for a single prediction.
        Returns matplotlib Figure.
        """
        import shap
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        mean_abs = explain_result["mean_abs_shap"]
        raw_shap = explain_result["shap_values"][0].mean(axis=0)
        names = (
            self.feature_names
            if self.feature_names
            else [f"f{i}" for i in range(len(mean_abs))]
        )

        # Build shap Explanation object
        exp = shap.Explanation(
            values=raw_shap,
            base_values=0.5,
            feature_names=names[:len(raw_shap)],
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=max_display, show=False)

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved SHAP waterfall -> %s", save_path)

        return plt.gcf()

    def plot_bar(
        self,
        explain_result: dict,
        top_k: int = 15,
        save_path: Optional[str] = None,
    ):
        """Plot horizontal bar chart of mean |SHAP| values."""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")

        top_feats = explain_result["top_features"][:top_k]
        names = [f["feature"] for f in top_feats]
        shap_vals = [f["shap_value"] for f in top_feats]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
        bars = ax.barh(names[::-1], shap_vals[::-1], color=colors[::-1])
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on infiltration probability)")
        ax.set_title("Feature Attribution — SHAP Values")
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


class _SHAPModelWrapper(object):
    """Thin PyTorch wrapper that returns infiltration prob as scalar."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def __call__(self, x):
        import torch
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x)
        return torch.softmax(out["pred_binary"], dim=-1)[:, 1]

    # Required by DeepExplainer for module detection
    def parameters(self):
        return self.model.parameters()
