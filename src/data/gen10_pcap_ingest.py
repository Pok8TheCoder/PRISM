"""PCAP → canonical 249-d Gen10/Laplace states + per-window flow context.

Uses the same fixed feature layout as universal Gen10 training (109 numeric cols
at indices 12–229). Missing CIC columns are zero-filled at their canonical slots
instead of shifting flag/proto/port tail features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.gen10_feature_schema import (
    GEN10_STATE_DIM,
    aggregate_window_gen10_canonical,
    enrich_flow_priv_flags,
    top_flow_context_for_window,
)
from src.pipeline.extract import pcap_to_rows

DEFAULT_WINDOW_SEC = 5.0


def _assign_windows(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    df = df.copy()
    ts = None
    for col in ("time", "Timestamp", "timestamp"):
        if col in df.columns:
            ts = pd.to_numeric(df[col], errors="coerce")
            break
    if ts is not None and ts.notna().any():
        t0 = float(ts.dropna().iloc[0])
        df["window_id"] = ((ts.fillna(t0) - t0) // window_sec).astype(int)
    else:
        df["window_id"] = np.arange(len(df)) // 32
    return df


def pcap_to_gen10_windows(
    pcap_path: str | Path,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> dict[str, Any]:
    """
    Build canonical raw 249-d states and Tier-2 flow contexts from a PCAP.

    Returns dict with keys: states (T, 249), flow_contexts, feature_dim, window_ids.
    """
    rows = pcap_to_rows(pcap_path)
    if not rows:
        return {
            "states": np.zeros((0, GEN10_STATE_DIM), dtype=np.float32),
            "flow_contexts": [],
            "feature_dim": GEN10_STATE_DIM,
            "window_ids": np.array([], dtype=np.int64),
        }

    df = enrich_flow_priv_flags(pd.DataFrame(rows))
    df = _assign_windows(df, window_sec)

    states: list[np.ndarray] = []
    contexts: list[dict[str, Any]] = []
    window_ids: list[int] = []

    for wid, window in df.groupby("window_id", sort=True):
        states.append(aggregate_window_gen10_canonical(window))
        contexts.append(top_flow_context_for_window(window))
        window_ids.append(int(wid))

    return {
        "states": np.stack(states).astype(np.float32) if states else np.zeros((0, GEN10_STATE_DIM), dtype=np.float32),
        "flow_contexts": contexts,
        "feature_dim": GEN10_STATE_DIM,
        "window_ids": np.array(window_ids, dtype=np.int64),
    }
