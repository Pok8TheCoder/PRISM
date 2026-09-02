"""Lab PCAP -> Shaun V2 292D states via SchemaAligner + StateBuilder (fair ingest)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
sys.path.insert(0, str(ROOT))

from src.pipeline.extract import pcap_to_rows  # noqa: E402


def prism_rows_to_cic_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Map PRISM flow rows to CIC-IDS-style CSV column names for SchemaAligner."""
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        fwd = float(r.get("Tot Fwd Pkts", 0) or 0)
        bwd = float(r.get("Tot Bwd Pkts", 0) or 0)
        total_pkts = fwd + bwd
        flow_pkts_s = float(r.get("Flow Pkts/s", 0) or 0)
        fwd_mean = float(r.get("Fwd Pkt Len Mean", 0) or 0)
        bwd_mean = float(r.get("Bwd Pkt Len Mean", 0) or 0)
        fwd_min = float(r.get("Fwd Pkt Len Min", 0) or 0)
        bwd_min = float(r.get("Bwd Pkt Len Min", 0) or 0)
        fwd_max = float(r.get("Fwd Pkt Len Max", 0) or 0)
        bwd_max = float(r.get("Bwd Pkt Len Max", 0) or 0)
        pkt_min = min(x for x in (fwd_min, bwd_min) if x > 0) if (fwd_min or bwd_min) else 0.0
        pkt_max = max(fwd_max, bwd_max)
        pkt_mean = (fwd_mean * fwd + bwd_mean * bwd) / max(total_pkts, 1)

        records.append({
            "Timestamp": r.get("Timestamp"),
            "Src IP": str(r.get("Src IP", r.get("src", "0.0.0.0"))),
            "Dst IP": str(r.get("Dst IP", "0.0.0.0")),
            "Destination Port": float(r.get("Dst Port", 0) or 0),
            "Protocol": int(r.get("Protocol", 0) or 0),
            "Flow Duration": float(r.get("Flow Duration", 0) or 0),
            "Total Fwd Packets": fwd,
            "Total Backward Packets": bwd,
            "Total Length of Fwd Packets": fwd_mean * fwd,
            "Total Length of Bwd Packets": bwd_mean * bwd,
            "Fwd Packet Length Max": fwd_max,
            "Fwd Packet Length Min": fwd_min,
            "Fwd Packet Length Mean": fwd_mean,
            "Bwd Packet Length Max": bwd_max,
            "Bwd Packet Length Min": bwd_min,
            "Bwd Packet Length Mean": bwd_mean,
            "Bwd Packet Length Std": float(r.get("Bwd Pkt Len Std", 0) or 0),
            "Flow Bytes/s": float(r.get("Flow Byts/s", 0) or 0),
            "Flow Packets/s": flow_pkts_s,
            "Fwd Packets/s": float(r.get("Fwd Pkts/s", flow_pkts_s * (fwd / max(total_pkts, 1)))),
            "Bwd Packets/s": float(r.get("Bwd Pkts/s", flow_pkts_s * (bwd / max(total_pkts, 1)))),
            "Flow IAT Mean": float(r.get("Flow IAT Mean", 0) or 0),
            "Flow IAT Std": float(r.get("Flow IAT Std", 0) or 0),
            "Flow IAT Min": float(r.get("Flow IAT Min", 0) or 0),
            "Flow IAT Max": float(r.get("Flow IAT Max", 0) or 0),
            "Min Packet Length": pkt_min,
            "Max Packet Length": pkt_max,
            "Packet Length Mean": pkt_mean,
            "Packet Length Std": 0.0,
            "Packet Length Variance": 0.0,
            "Average Packet Size": pkt_mean,
            "SYN Flag Count": float(r.get("SYN Flag Cnt", 0) or 0),
            "FIN Flag Count": float(r.get("FIN Flag Cnt", 0) or 0),
            "RST Flag Count": float(r.get("RST Flag Cnt", 0) or 0),
            "PSH Flag Count": float(r.get("PSH Flag Cnt", 0) or 0),
            "ACK Flag Count": float(r.get("ACK Flag Cnt", 0) or 0),
            "URG Flag Count": 0.0,
            "Down/Up Ratio": float(r.get("Down/Up Ratio", bwd / max(fwd, 1))),
        })
    return pd.DataFrame(records)


def _load_shaun_modules(shaun_root: Path):
    """Import Shaun SchemaAligner + StateBuilder with isolated sys.path."""
    shaun_root = shaun_root.resolve()
    prev_cwd = os.getcwd()
    os.chdir(shaun_root)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path.insert(0, str(shaun_root))
    try:
        from src.data.schema_aligner import SchemaAligner
        from src.data.state_builder import StateBuilder
    finally:
        os.chdir(prev_cwd)
    return SchemaAligner, StateBuilder


def rows_to_shaun_states(
    rows: list[dict],
    shaun_root: Path | None = None,
    window_sec: float = 15.0,
) -> np.ndarray:
    """PRISM flow rows -> 292D states through Shaun's full training pipeline."""
    shaun_root = (shaun_root or SHAUN_ROOT).resolve()
    if not shaun_root.exists():
        raise FileNotFoundError(f"Shaun worktree not found: {shaun_root}")

    raw_df = prism_rows_to_cic_dataframe(rows)
    if raw_df.empty:
        return np.zeros((0, 292), dtype=np.float32)

    SchemaAligner, StateBuilder = _load_shaun_modules(shaun_root)
    prev_cwd = os.getcwd()
    os.chdir(shaun_root)
    try:
        aligned = SchemaAligner().align_dataframe(raw_df)
        builder = StateBuilder(window_size_seconds=int(window_sec))
        states, _, _, _, _ = builder.build_states_from_dataframe(aligned)
    finally:
        os.chdir(prev_cwd)
    return states.astype(np.float32)


def pcap_to_shaun_states(pcap: Path, window_sec: float = 15.0, shaun_root: Path | None = None) -> np.ndarray:
    return rows_to_shaun_states(pcap_to_rows(pcap), shaun_root=shaun_root, window_sec=window_sec)
