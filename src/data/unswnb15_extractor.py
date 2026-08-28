"""
PRISM — UNSW-NB15 Feature Extraction
Ingests UNSW-NB15 CSV records and outputs cleaned, normalised DataFrames
with MITRE ATT&CK stage mappings compatible with the PRISM world model.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.constants import (
    UNSWNB15_LABEL_TO_MITRE,
    MITRE_STAGES,
    TOP_ATTACKED_PORTS,
)

logger = logging.getLogger("prism.data.unswnb15_extractor")


class UNSWNB15Extractor:
    """UNSW-NB15 dataset flow feature extractor and preprocessor."""

    def __init__(self, raw_dir: str = "data/raw") -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False
        self.label_map = UNSWNB15_LABEL_TO_MITRE

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single UNSW-NB15 CSV file."""
        logger.info(f"Loading UNSW-NB15 CSV from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)

        # Handle headerless UNSW-NB15 4-part files if present
        if "dur" not in [str(c).lower() for c in df.columns] and len(df.columns) in (47, 49):
            headers = [
                "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
                "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts",
                "swin", "dwin", "stcpb", "dtcpb", "smean", "dmean", "trans_depth", "response_body_len",
                "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
                "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
                "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
            ]
            if len(df.columns) == len(headers):
                df.columns = headers

        df.columns = [str(c).strip() for c in df.columns]
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean UNSW-NB15 dataset."""
        df = df.copy()
        
        # Determine label column (attack_cat or label)
        if "attack_cat" in df.columns:
            df["Label"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
        elif "label" in df.columns:
            df["Label"] = df["label"].apply(lambda x: "Normal" if x == 0 else "Attack")
        else:
            df["Label"] = "Normal"

        # Standardise column names
        rename_map = {
            "dur": "Flow Duration",
            "spkts": "Tot Fwd Pkts",
            "dpkts": "Tot Bwd Pkts",
            "sbytes": "TotLen Fwd Pkts",
            "dbytes": "TotLen Bwd Pkts",
            "sload": "Flow Byts/s",
            "spkts": "Flow Pkts/s",
            "dsport": "Dst Port",
            "sport": "Src Port",
            "srcip": "src_ip",
            "dstip": "dst_ip",
            "proto": "Protocol",
            "stime": "Timestamp",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Ensure Dst Port is numeric
        if "Dst Port" in df.columns:
            df["Dst Port"] = pd.to_numeric(df["Dst Port"], errors="coerce").fillna(0).astype(int)

        # Replace infinite values and fill NaNs
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for downstream model compatibility."""
        df = df.copy()

        # Fwd/Bwd Packet ratio
        tot_fwd = df.get("Tot Fwd Pkts", 1.0)
        tot_bwd = df.get("Tot Bwd Pkts", 1.0)
        df["fwd_bwd_ratio"] = tot_fwd / (tot_bwd + 1e-6)

        # Port categories and top port indicators
        dst_port = df.get("Dst Port", pd.Series(0, index=df.index))
        df["port_cat_well_known"] = ((dst_port >= 0) & (dst_port <= 1023)).astype(float)
        df["port_cat_registered"] = ((dst_port >= 1024) & (dst_port <= 49151)).astype(float)
        df["port_cat_ephemeral"] = ((dst_port >= 49152) & (dst_port <= 65535)).astype(float)

        for port in TOP_ATTACKED_PORTS:
            df[f"port_{port}"] = (dst_port == port).astype(float)

        # Dummy flag counts for compatibility with flow_extractor features
        for flag in ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]:
            df[f"flag_{flag.lower()}_frac"] = 0.0

        return df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map dataset attack categories to MITRE ATT&CK stages."""
        df = df.copy()
        raw_labels = df["Label"].astype(str)

        def map_fn(label_str: str) -> str:
            cleaned = label_str.strip()
            if cleaned in self.label_map:
                return self.label_map[cleaned]
            # Partial substring matching
            for k, v in self.label_map.items():
                if k.lower() in cleaned.lower():
                    return v
            return "Benign"

        df["mitre_stage"] = raw_labels.apply(map_fn)
        df["mitre_stage_id"] = df["mitre_stage"].map(MITRE_STAGES).fillna(0).astype(int)
        return df

    def extract(self, file_path: Optional[str] = None, directory: Optional[str] = None, fit_scaler: bool = True) -> pd.DataFrame:
        """Full extraction pipeline."""
        if file_path:
            df = self.load_csv(file_path)
        elif directory:
            files = list(pathlib.Path(directory).glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {directory}")
            dfs = [self.load_csv(str(f)) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            default_path = self.raw_dir / "unsw_nb15.csv"
            if default_path.exists():
                df = self.load_csv(str(default_path))
            else:
                logger.warning("No input file found. Generating synthetic UNSW-NB15 frame.")
                df = self._generate_synthetic_frame()

        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.map_labels(df)
        logger.info(f"UNSW-NB15 extraction complete: {len(df)} records.")
        return df

    def _generate_synthetic_frame(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic UNSW-NB15 records for pipeline verification."""
        rng = np.random.default_rng(42)
        categories = ["Normal", "Exploits", "Fuzzers", "DoS", "Reconnaissance", "Generic"]
        data = {
            "srcip": ["192.168.1.10" for _ in range(num_records)],
            "sport": rng.integers(1024, 65535, num_records),
            "dstip": ["10.0.0.5" for _ in range(num_records)],
            "dsport": rng.choice([80, 443, 22, 21, 3389, 8080], num_records),
            "proto": rng.choice(["tcp", "udp", "icmp"], num_records),
            "state": ["FIN" for _ in range(num_records)],
            "dur": rng.uniform(0.001, 10.0, num_records),
            "sbytes": rng.integers(64, 10000, num_records),
            "dbytes": rng.integers(64, 20000, num_records),
            "sttl": rng.choice([64, 128, 255], num_records),
            "dttl": rng.choice([64, 128, 255], num_records),
            "sloss": rng.integers(0, 5, num_records),
            "dloss": rng.integers(0, 5, num_records),
            "service": rng.choice(["http", "dns", "ftp", "smtp", "-"], num_records),
            "sload": rng.uniform(100, 100000, num_records),
            "dload": rng.uniform(100, 100000, num_records),
            "spkts": rng.integers(1, 100, num_records),
            "dpkts": rng.integers(1, 100, num_records),
            "swin": [255] * num_records,
            "dwin": [255] * num_records,
            "stcpb": rng.integers(0, 1e9, num_records),
            "dtcpb": rng.integers(0, 1e9, num_records),
            "smean": rng.uniform(40, 1000, num_records),
            "dmean": rng.uniform(40, 1000, num_records),
            "trans_depth": [0] * num_records,
            "response_body_len": [0] * num_records,
            "sjit": rng.uniform(0, 100, num_records),
            "djit": rng.uniform(0, 100, num_records),
            "stime": rng.integers(1500000000, 1500050000, num_records),
            "ltime": rng.integers(1500050000, 1500100000, num_records),
            "sintpkt": rng.uniform(0, 10, num_records),
            "dintpkt": rng.uniform(0, 10, num_records),
            "tcprtt": rng.uniform(0.001, 0.5, num_records),
            "synack": rng.uniform(0.001, 0.2, num_records),
            "ackdat": rng.uniform(0.001, 0.2, num_records),
            "is_sm_ips_ports": [0] * num_records,
            "ct_state_ttl": rng.integers(1, 6, num_records),
            "ct_flw_http_mthd": [0] * num_records,
            "is_ftp_login": [0] * num_records,
            "ct_ftp_cmd": [0] * num_records,
            "ct_srv_src": rng.integers(1, 10, num_records),
            "ct_srv_dst": rng.integers(1, 10, num_records),
            "ct_dst_ltm": rng.integers(1, 10, num_records),
            "ct_src_ltm": rng.integers(1, 10, num_records),
            "ct_src_dport_ltm": rng.integers(1, 10, num_records),
            "ct_dst_sport_ltm": rng.integers(1, 10, num_records),
            "ct_dst_src_ltm": rng.integers(1, 10, num_records),
            "attack_cat": rng.choice(categories, num_records, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.05]),
            "label": [0] * num_records,
        }
        return pd.DataFrame(data)
