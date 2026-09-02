"""
Attention Visualization: Extracts and formats Transformer self-attention weights
for temporal importance explanations.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def format_attention_heatmap(
    attn_matrix: np.ndarray,
    window_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Formats an (L, L) attention matrix from the Transformer into temporal heatmap data.

    Args:
        attn_matrix: (L, L) 2D array of attention weights.
        window_labels: Optional human-readable labels for each time window (e.g. ['t-9', 't-8', ... 't']).

    Returns:
        Dictionary containing formatted heatmap grid, peak attention steps, and narrative.
    """
    if attn_matrix.ndim == 3:
        # Take first sample if batch dimension is present
        attn_matrix = attn_matrix[0]

    seq_len = attn_matrix.shape[0]
    labels = window_labels or [f"t-{seq_len - 1 - i}" if i < seq_len - 1 else "t (Now)" for i in range(seq_len)]

    # Temporal attention received by past steps from the current active step (last row)
    current_step_attn = attn_matrix[-1, :]
    peak_attended_idx = int(np.argmax(current_step_attn))
    peak_attended_label = labels[peak_attended_idx]
    peak_weight = float(current_step_attn[peak_attended_idx])

    narrative = (
        f"The model focuses {peak_weight * 100:.1f}% of its temporal attention on window "
        f"'{peak_attended_label}' to evaluate current infiltration risk."
    )

    return {
        "matrix": attn_matrix.tolist(),
        "window_labels": labels,
        "current_step_attn": current_step_attn.tolist(),
        "peak_window": peak_attended_label,
        "peak_weight": peak_weight,
        "narrative": narrative
    }
