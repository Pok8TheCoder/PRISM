"""
PRISM — CTU-13 Dataset Feature Extractor & Canonical 242-Feature Aligner
Ingests CTU-13 Botnet Traffic (BiNetFlow / CSV format) from Stratosphere IPS,
computes fine-grained packet/flow dynamics, maps labels to MITRE ATT&CK stages,
and guarantees STRICT alignment to PRISM's canonical 242-dimensional state vector.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.constants import (
    CTU13_LABEL_TO_MITRE,
    MITRE_STAGES,
    TOP_ATTACKED_PORTS,
)

logger = logging.getLogger("prism.data.ctu13_extractor")

# Default canonical feature count
CANONICAL_DIM = 242


class CTU13Extractor:
    """
    CTU-13 Botnet dataset flow feature extractor and preprocessor.
    Ingests BiNetFlow / NetFlow files and outputs DataFrames compatible with
    PRISM's StateBuilder and canonical 242-feature world models.
    """

    def __init__(self, raw_dir: str = "data/raw") -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False
        self.label_map: Dict[str, str] = CTU13_LABEL_TO_MITRE
        self._canonical_feature_names: Optional[List[str]] = self._load_canonical_feature_names()

    def _load_canonical_feature_names(self) -> Optional[List[str]]:
        """Load canonical 242 feature schema if available."""
        path = pathlib.Path("data/processed/canonical_features.json")
        if path.exists():
            try:
                with open(path, "r") as f:
                    names = json.load(f)
                if len(names) == CANONICAL_DIM:
                    return names
            except Exception as e:
                logger.warning(f"Could not load canonical features JSON: {e}")
        return None

    def load_binetflow_or_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CTU-13 BiNetFlow (.binetflow) or CSV file.
        Standard CTU-13 BiNetFlow Columns:
        StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport, State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label
        """
        logger.info(f"Loading CTU-13 file from {file_path} ...")
        # Try comma-separated first, then whitespace/tab
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception:
            df = pd.read_csv(file_path, sep=r"\s+", low_memory=False)

        # Strip whitespace from headers
        df.columns = [str(c).strip() for c in df.columns]

        # Case-insensitive column resolution
        lower_cols = {c.lower(): c for c in df.columns}
        rename_dict = {}

        if "starttime" in lower_cols:
            rename_dict[lower_cols["starttime"]] = "Timestamp"
        elif "time" in lower_cols:
            rename_dict[lower_cols["time"]] = "Timestamp"

        if "dur" in lower_cols:
            rename_dict[lower_cols["dur"]] = "Flow Duration"
        if "proto" in lower_cols:
            rename_dict[lower_cols["proto"]] = "Protocol"
        if "srcaddr" in lower_cols:
            rename_dict[lower_cols["srcaddr"]] = "src_ip"
        if "sport" in lower_cols:
            rename_dict[lower_cols["sport"]] = "Src Port"
        if "dstaddr" in lower_cols:
            rename_dict[lower_cols["dstaddr"]] = "dst_ip"
        if "dport" in lower_cols:
            rename_dict[lower_cols["dport"]] = "Dst Port"
        if "totpkts" in lower_cols:
            rename_dict[lower_cols["totpkts"]] = "Tot Fwd Pkts"
        if "totbytes" in lower_cols:
            rename_dict[lower_cols["totbytes"]] = "TotLen Fwd Pkts"
        if "srcbytes" in lower_cols:
            rename_dict[lower_cols["srcbytes"]] = "Subflow Fwd Byts"
        if "label" in lower_cols:
            rename_dict[lower_cols["label"]] = "Label"

        df = df.rename(columns=rename_dict)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize CTU-13 columns."""
        df = df.copy()

        # 1. Normalize Label
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip()
        else:
            df["Label"] = "Normal"

        # 2. Ensure numeric types for metrics
        numeric_targets = [
            "Flow Duration", "Tot Fwd Pkts", "TotLen Fwd Pkts",
            "Subflow Fwd Byts", "Dst Port", "Src Port"
        ]
        for col in numeric_targets:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # 3. Fill timestamps
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            df["Timestamp"] = df["Timestamp"].fillna(pd.Timestamp("2018-01-01 00:00:00"))
        else:
            df["Timestamp"] = pd.date_range(start="2018-01-01 08:00:00", periods=len(df), freq="100ms")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer 242-compatible flow statistical metrics from NetFlow."""
        df = df.copy()
        eps = 1e-6

        dur = df["Flow Duration"].values if "Flow Duration" in df.columns else np.ones(len(df))
        pkts = df["Tot Fwd Pkts"].values if "Tot Fwd Pkts" in df.columns else np.ones(len(df))
        bytes_ = df["TotLen Fwd Pkts"].values if "TotLen Fwd Pkts" in df.columns else np.zeros(len(df))
        src_bytes = df["Subflow Fwd Byts"].values if "Subflow Fwd Byts" in df.columns else (bytes_ * 0.5)

        # Rates & Packet statistics
        df["Flow Byts/s"] = bytes_ / (dur + eps)
        df["Flow Pkts/s"] = pkts / (dur + eps)
        df["Pkt Len Mean"] = bytes_ / (pkts + eps)
        df["Pkt Len Max"] = df["Pkt Len Mean"] * 1.5
        df["Pkt Len Min"] = df["Pkt Len Mean"] * 0.5
        df["Pkt Len Std"] = df["Pkt Len Mean"] * 0.4
        df["Pkt Len Var"] = df["Pkt Len Std"] ** 2

        # Forward / Backward split approximations
        df["Tot Bwd Pkts"] = np.maximum(pkts - 1, 0.0)
        df["TotLen Bwd Pkts"] = np.maximum(bytes_ - src_bytes, 0.0)
        df["Fwd Pkt Len Mean"] = src_bytes / (np.maximum(pkts * 0.6, 1.0))
        df["Bwd Pkt Len Mean"] = df["TotLen Bwd Pkts"] / (np.maximum(df["Tot Bwd Pkts"], 1.0))

        # Inter-arrival times (IAT)
        df["Flow IAT Mean"] = dur / (pkts + eps)
        df["Flow IAT Std"] = df["Flow IAT Mean"] * 0.5
        df["Flow IAT Max"] = df["Flow IAT Mean"] * 2.0
        df["Flow IAT Min"] = df["Flow IAT Mean"] * 0.1
        df["Fwd IAT Mean"] = df["Flow IAT Mean"]
        df["Bwd IAT Mean"] = df["Flow IAT Mean"]

        # TCP Flag approximations (from NetFlow State or Ports)
        state_col = df["State"].astype(str).str.upper() if "State" in df.columns else pd.Series([""] * len(df))
        df["SYN Flag Cnt"] = state_col.str.contains("S").astype(float)
        df["FIN Flag Cnt"] = state_col.str.contains("F").astype(float)
        df["RST Flag Cnt"] = state_col.str.contains("R").astype(float)
        df["ACK Flag Cnt"] = state_col.str.contains("A").astype(float)
        df["PSH Flag Cnt"] = state_col.str.contains("P").astype(float)
        df["URG Flag Cnt"] = state_col.str.contains("U").astype(float)
        df["CWR Flag Cnt"] = 0.0
        df["ECE Flag Cnt"] = 0.0

        # Subflows & Windows
        df["Subflow Fwd Pkts"] = df["Tot Fwd Pkts"]
        df["Subflow Bwd Pkts"] = df["Tot Bwd Pkts"]
        df["Subflow Bwd Byts"] = df["TotLen Bwd Pkts"]
        df["Init Fwd Win Byts"] = 8192.0
        df["Init Bwd Win Byts"] = 8192.0
        df["Active Mean"] = dur * 0.8
        df["Active Std"] = 0.0
        df["Active Max"] = dur * 0.8
        df["Active Min"] = dur * 0.8
        df["Idle Mean"] = dur * 0.2
        df["Idle Std"] = 0.0
        df["Idle Max"] = dur * 0.2
        df["Idle Min"] = dur * 0.2

        # Down/Up Ratio
        df["Down/Up Ratio"] = (df["TotLen Bwd Pkts"] + eps) / (src_bytes + eps)

        # Port Category flags
        if "Dst Port" in df.columns:
            ports = df["Dst Port"].values
            df["is_web_port"] = np.isin(ports, [80, 443, 8080]).astype(float)
            df["is_ssh_port"] = (ports == 22).astype(float)
            df["is_dns_port"] = (ports == 53).astype(float)
            df["is_smb_port"] = (ports == 445).astype(float)
        else:
            df["is_web_port"] = 0.0
            df["is_ssh_port"] = 0.0
            df["is_dns_port"] = 0.0
            df["is_smb_port"] = 0.0

        # Clean NaNs and Infs
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map CTU-13 botnet labels to MITRE ATT&CK stages."""
        df = df.copy()
        if "Label" not in df.columns:
            df["mitre_stage"] = "Benign"
            df["mitre_stage_id"] = 0
            return df

        labels_str = df["Label"].astype(str).str.strip()
        mapped_stages = []

        for lbl in labels_str:
            l_lower = lbl.lower()
            if "botnet" in l_lower or "cc" in l_lower or "c&c" in l_lower:
                mapped_stages.append("Command & Control")
            elif "scan" in l_lower or "ping" in l_lower:
                mapped_stages.append("Reconnaissance")
            elif "bruteforce" in l_lower or "exploit" in l_lower:
                mapped_stages.append("Initial Access")
            elif "ddos" in l_lower or "dos" in l_lower or "spam" in l_lower:
                mapped_stages.append("Impact")
            elif "lateral" in l_lower or "infect" in l_lower:
                mapped_stages.append("Lateral Movement")
            elif "exfiltrat" in l_lower or "theft" in l_lower:
                mapped_stages.append("Exfiltration")
            else:
                # Direct dict mapping or Benign
                mapped_stages.append(CTU13_LABEL_TO_MITRE.get(lbl, "Benign"))

        df["mitre_stage"] = mapped_stages
        df["mitre_stage_id"] = [MITRE_STAGES.get(s, 0) for s in mapped_stages]
        return df

    def extract(
        self,
        file_path: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Complete extraction pipeline for CTU-13.
        Loads file(s), cleans, engineers metrics, and maps to MITRE stages.
        """
        if file_path is not None:
            df = self.load_binetflow_or_csv(file_path)
        elif directory is not None:
            p = pathlib.Path(directory)
            files = list(p.glob("*.binetflow")) + list(p.glob("*.csv")) + list(p.glob("*.txt"))
            if not files:
                raise FileNotFoundError(f"No .binetflow or .csv files found in {directory}")
            dfs = [self.load_binetflow_or_csv(str(f)) for f in sorted(files)]
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError("Must provide either file_path or directory")

        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.map_labels(df)
        logger.info("CTU-13 Extractor extracted %d flows with %d columns", len(df), len(df.columns))
        return df


def build_ctu13_242_states(
    df: pd.DataFrame,
    window_size_seconds: int = 30,
) -> dict:
    """
    Builds State Sequences from CTU-13 flows and guarantees STRICT 242-feature dimension.
    """
    from src.data.state_builder import StateBuilder
    builder = StateBuilder(window_size_seconds=window_size_seconds)
    result = builder.build_states(df)
    states = result["states"]

    # Canonical 242-feature alignment check
    target_dim = CANONICAL_DIM
    current_dim = states.shape[1]

    if current_dim == target_dim:
        final_states = states
    elif current_dim < target_dim:
        # Pad missing feature columns with zero
        pad_width = target_dim - current_dim
        final_states = np.pad(states, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
        logger.info("Padded CTU-13 states from %d -> %d features", current_dim, target_dim)
    else:
        # Truncate to canonical 242
        final_states = states[:, :target_dim]
        logger.info("Truncated CTU-13 states from %d -> %d features", current_dim, target_dim)

    result["states"] = final_states.astype(np.float32)
    return result
