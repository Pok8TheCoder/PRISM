"""Canonical 249-d Gen10 / Laplace state layout for training-aligned inference.

Layout (matches universal Gen10 compiler):
  [0:5]   meta counts
  [5:12]  graph topology invariants (Gen10)
  [12:230] mean/std pairs for 109 fixed CIC numeric columns
  [230:236] TCP flag fractions
  [236:239] protocol fractions
  [239:249] top attacked port counts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.state_builder import StateBuilder, _is_private_ip_fast

GEN10_STATE_DIM = 249
_COLS_PATH = Path(__file__).with_name("gen10_canonical_numeric_cols.json")


def canonical_numeric_cols() -> list[str]:
    if _COLS_PATH.is_file():
        return list(json.loads(_COLS_PATH.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"Missing canonical column list: {_COLS_PATH}")


CANONICAL_NUMERIC_COLS: list[str] = canonical_numeric_cols()


def aggregate_window_gen10_canonical(window: pd.DataFrame) -> np.ndarray:
    """Build a fixed 249-d raw state vector (always full width)."""
    builder = StateBuilder(mode="vector")
    vec = builder._aggregate_window_vector(window, CANONICAL_NUMERIC_COLS)
    if len(vec) < GEN10_STATE_DIM:
        out = np.zeros(GEN10_STATE_DIM, dtype=np.float32)
        out[: len(vec)] = vec
        return out
    return vec[:GEN10_STATE_DIM].astype(np.float32)


def top_flow_context_for_window(window: pd.DataFrame) -> dict[str, Any]:
    """Pick dominant flow row for Tier-2 attribution (matches streaming_gen10.process_pcap)."""
    from src.models.streaming_gen10 import Tier2FlowContextAttributor

    if window.empty:
        return {}

    scored_rows: list[tuple[int, dict[str, Any]]] = []
    for _, r in window.iterrows():
        score = 0
        dp = r.get("dst_port") or r.get("Dst Port")
        try:
            dp_int = int(dp) if dp is not None else None
        except Exception:
            dp_int = None
        if dp_int in Tier2FlowContextAttributor.LATERAL_PORTS:
            score += 5
        if dp_int in Tier2FlowContextAttributor.INITIAL_ACCESS_PORTS:
            score += 4
        if dp_int in Tier2FlowContextAttributor.C2_PORTS:
            score += 3
        fwd_b = float(
            r.get("fwd_bytes")
            or r.get("_fwd_bytes")
            or r.get("TotLen Fwd Pkts")
            or 0.0
        )
        if fwd_b > 10000:
            score += 5
        scored_rows.append((score, r.to_dict()))

    scored_rows.sort(key=lambda x: x[0], reverse=True)
    row = scored_rows[0][1]
    src_ip = str(row.get("src_ip") or row.get("Src IP") or "")
    dst_ip = str(row.get("dst_ip") or row.get("Dst IP") or "")
    return {
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "dst_port": int(row.get("dst_port") or row.get("Dst Port") or 0),
        "fwd_bytes": float(row.get("fwd_bytes") or row.get("_fwd_bytes") or row.get("TotLen Fwd Pkts") or 0.0),
        "bwd_bytes": float(row.get("bwd_bytes") or row.get("_bwd_bytes") or row.get("TotLen Bwd Pkts") or 0.0),
        "flow_count": int(len(window)),
        "unique_dst_ports": int(window["Dst Port"].nunique()) if "Dst Port" in window.columns else 1,
    }


def enrich_flow_priv_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add _src_priv / _dst_priv for graph directionality ratios."""
    df = df.copy()
    src_col = next((c for c in ("Src IP", "src_ip") if c in df.columns), None)
    dst_col = next((c for c in ("Dst IP", "dst_ip") if c in df.columns), None)
    if src_col:
        df["_src_priv"] = df[src_col].astype(str).map(_is_private_ip_fast)
    else:
        df["_src_priv"] = False
    if dst_col:
        df["_dst_priv"] = df[dst_col].astype(str).map(_is_private_ip_fast)
    else:
        df["_dst_priv"] = False
    return df
