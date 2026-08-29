"""
PRISM - Attention Visualisation
Extracts and formats Transformer attention weights for temporal explainability.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("prism.explainability.attention_viz")


class AttentionVisualiser:
    """
    Extract and interpret Transformer attention weights.

    Attention weights reveal WHICH past time windows the model focuses on
    when making an infiltration prediction — enabling temporal explanations
    like "the model is most influenced by traffic patterns from 5 minutes ago."
    """

    def __init__(self, feature_names: Optional[list[str]] = None):
        self.feature_names = feature_names or []

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------
    def extract_weights(
        self,
        model,
        state_seq_tensor,
        return_all_layers: bool = False,
    ) -> dict:
        """
        Run model forward pass and extract attention weights.

        Parameters
        ----------
        model           : StateTransformerWorldModel
        state_seq_tensor: torch.Tensor, shape (1, L, D_state)
        return_all_layers: if True, hook all layers (not just last)

        Returns
        -------
        dict:
            last_layer_attn : (L, L) numpy array — averaged over heads
            all_layers_attn : list of (L, L) arrays (if return_all_layers)
            top_attended_steps: list of (step_idx, weight) sorted by weight
        """
        import torch

        hooks = []
        attn_matrices = []

        if return_all_layers:
            def make_hook(layer_idx):
                def hook(module, input, output):
                    # output[1] is the attention weight tensor if need_weights=True
                    if isinstance(output, tuple) and output[1] is not None:
                        attn_matrices.append(
                            output[1].detach().cpu().numpy()
                        )
                return hook

            for i, layer in enumerate(model.transformer.layers):
                h = layer.self_attn.register_forward_hook(make_hook(i))
                hooks.append(h)

        try:
            out = model(state_seq_tensor, return_attention=True)
            last_attn = out.get("attention_weights")
        finally:
            for h in hooks:
                h.remove()

        result = {}

        if last_attn is not None:
            if hasattr(last_attn, "detach"):
                last_np = last_attn[0].detach().cpu().numpy()
            elif hasattr(last_attn, "cpu"):
                last_np = last_attn[0].cpu().numpy()
            else:
                last_np = np.array(last_attn)
            result["last_layer_attn"] = last_np  # (L, L)
            result["top_attended_steps"] = self._top_attended(last_np[-1])
        else:
            result["last_layer_attn"] = None
            result["top_attended_steps"] = []

        result["all_layers_attn"] = attn_matrices if return_all_layers else []
        return result

    # ------------------------------------------------------------------
    # Temporal attention interpretation
    # ------------------------------------------------------------------
    def interpret_temporal(
        self,
        attn_weights: np.ndarray,
        window_size_seconds: int = 30,
    ) -> list[dict]:
        """
        Convert attention weights into human-readable temporal explanations.

        Parameters
        ----------
        attn_weights     : (L,) — attention weight of last query over all keys
        window_size_seconds: duration of each time window

        Returns
        -------
        list of dicts: [{"step": i, "seconds_ago": s, "weight": w, "interpretation": str}]
        """
        L = len(attn_weights)
        explanations = []

        for i, w in enumerate(attn_weights):
            steps_ago = L - 1 - i
            seconds_ago = steps_ago * window_size_seconds

            if seconds_ago == 0:
                time_desc = "current window"
            elif seconds_ago < 60:
                time_desc = f"{seconds_ago}s ago"
            else:
                time_desc = f"{seconds_ago // 60}m {seconds_ago % 60}s ago"

            if w > 0.20:
                interp = "HIGH INFLUENCE"
            elif w > 0.10:
                interp = "Moderate influence"
            elif w > 0.05:
                interp = "Low influence"
            else:
                interp = "Minimal influence"

            explanations.append({
                "step": i,
                "steps_ago": steps_ago,
                "seconds_ago": seconds_ago,
                "time_desc": time_desc,
                "weight": float(w),
                "interpretation": interp,
            })

        # Sort by weight descending
        explanations.sort(key=lambda x: x["weight"], reverse=True)
        return explanations

    def format_attention_summary(
        self,
        attn_weights: np.ndarray,
        window_size_seconds: int = 30,
        top_k: int = 5,
    ) -> str:
        """Return a human-readable attention summary string."""
        interps = self.interpret_temporal(attn_weights, window_size_seconds)
        lines = ["Temporal Attention Summary (most influential windows):"]
        for item in interps[:top_k]:
            lines.append(
                f"  {item['time_desc']:>15} | weight={item['weight']:.3f} | {item['interpretation']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _top_attended(attn_row: np.ndarray, top_k: int = 5) -> list[tuple]:
        """Return top-K (step_idx, weight) sorted by weight."""
        top_idx = np.argsort(attn_row)[::-1][:top_k]
        return [(int(i), float(attn_row[i])) for i in top_idx]

    def plot_attention_heatmap(
        self,
        attn_weights: np.ndarray,
        title: str = "Attention Heatmap",
        save_path: Optional[str] = None,
    ):
        """
        Plot (L, L) attention weight matrix as a heatmap.
        Returns matplotlib Figure.
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")  # non-interactive backend (safe for server use)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn_weights, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Attention Weight")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Key (Source Window)")
        ax.set_ylabel("Query (Prediction Step)")

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved attention heatmap -> %s", save_path)

        return fig

    def plot_temporal_bar(
        self,
        attn_weights: np.ndarray,
        window_size_seconds: int = 30,
        title: str = "Temporal Influence",
        save_path: Optional[str] = None,
    ):
        """
        Bar chart of attention weights over time.
        """
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        L = len(attn_weights)
        seconds = [(L - 1 - i) * window_size_seconds for i in range(L)]
        labels = [f"-{s}s" if s > 0 else "now" for s in seconds]

        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["#e74c3c" if w > 0.15 else "#3498db" for w in attn_weights]
        ax.bar(labels, attn_weights, color=colors)
        ax.set_xlabel("Time Step (relative to prediction)")
        ax.set_ylabel("Attention Weight")
        ax.set_title(title, fontsize=13)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig
