"""
PRISM — CICIoT2023 Feature Extraction
Ingests CICIoT2023 CSV records (IoT network security telemetry) and processes
them into normalised feature DataFrames mapped to MITRE ATT&CK stages.
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
    CICIOT2023_LABEL_TO_MITRE,
    MITRE_STAGES,
    TOP_ATTACKED_PORTS,
)

logger = logging.getLogger("prism.data.ciciot2023_extractor")


class CICIoT2023Extractor:
    """CICIoT2023 dataset feature extractor and preprocessor."""

    def __init__(self, raw_dir: str = "data/raw") -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False
        self.label_map = CICIOT2023_LABEL_TO_MITRE

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CICIoT2023 CSV file."""
        logger.info(f"Loading CICIoT2023 CSV from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CICIoT2023 DataFrame."""
        df = df.copy()

        # Determine label column
        label_col = None
        for c in ["label", "Label", "attack_type"]:
            if c in df.columns:
                label_col = c
                break
        
        if label_col:
            df["Label"] = df[label_col].astype(str).str.strip()
        else:
            df["Label"] = "Benign"

        # Map column aliases to PRISM canonical standards
        rename_map = {
            "flow_duration": "Flow Duration",
            "Header_Length": "Fwd Header Len",
            "Duration": "Flow Duration",
            "Rate": "Flow Byts/s",
            "fin_flag_number": "FIN Flag Cnt",
            "syn_flag_number": "SYN Flag Cnt",
            "rst_flag_number": "RST Flag Cnt",
            "psh_flag_number": "PSH Flag Cnt",
            "ack_flag_number": "ACK Flag Cnt",
            "ece_flag_number": "ECE Flag Cnt",
            "AVG": "Pkt Size Avg",
            "Min": "Pkt Len Min",
            "Max": "Pkt Len Max",
            "Std": "Pkt Len Std",
            "Protocol Type": "Protocol",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Add synthetic Dst Port if missing (based on application flags like HTTP/DNS/SSH)
        if "Dst Port" not in df.columns:
            ports = pd.Series(0, index=df.index)
            if "HTTP" in df.columns:
                ports = np.where(df["HTTP"] == 1, 80, ports)
            if "HTTPS" in df.columns:
                ports = np.where(df["HTTPS"] == 1, 443, ports)
            if "DNS" in df.columns:
                ports = np.where(df["DNS"] == 1, 53, ports)
            if "SSH" in df.columns:
                ports = np.where(df["SSH"] == 1, 22, ports)
            if "Telnet" in df.columns:
                ports = np.where(df["Telnet"] == 1, 23, ports)
            df["Dst Port"] = ports

        # Fill NaNs and replace infs
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features for world model state aggregation."""
        df = df.copy()

        # Flag fractions
        syn_cnt = df.get("SYN Flag Cnt", 0.0)
        ack_cnt = df.get("ACK Flag Cnt", 0.0)
        fin_cnt = df.get("FIN Flag Cnt", 0.0)
        rst_cnt = df.get("RST Flag Cnt", 0.0)
        psh_cnt = df.get("PSH Flag Cnt", 0.0)
        tot_pkts = df.get("Number", 1.0)

        df["flag_syn_frac"] = syn_cnt / (tot_pkts + 1e-6)
        df["flag_ack_frac"] = ack_cnt / (tot_pkts + 1e-6)
        df["flag_fin_frac"] = fin_cnt / (tot_pkts + 1e-6)
        df["flag_rst_frac"] = rst_cnt / (tot_pkts + 1e-6)
        df["flag_psh_frac"] = psh_cnt / (tot_pkts + 1e-6)
        df["flag_urg_frac"] = 0.0

        # Port one-hots
        dst_port = df["Dst Port"]
        df["port_cat_well_known"] = ((dst_port >= 0) & (dst_port <= 1023)).astype(float)
        df["port_cat_registered"] = ((dst_port >= 1024) & (dst_port <= 49151)).astype(float)
        df["port_cat_ephemeral"] = ((dst_port >= 49152) & (dst_port <= 65535)).astype(float)

        for port in TOP_ATTACKED_PORTS:
            df[f"port_{port}"] = (dst_port == port).astype(float)

        return df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map IoT attack categories to MITRE ATT&CK stages."""
        df = df.copy()
        raw_labels = df["Label"].astype(str)

        def map_fn(label_str: str) -> str:
            cleaned = label_str.strip()
            if cleaned in self.label_map:
                return self.label_map[cleaned]
            for k, v in self.label_map.items():
                if k.lower() in cleaned.lower():
                    return v
            return "Benign"

        df["mitre_stage"] = raw_labels.apply(map_fn)
        df["mitre_stage_id"] = df["mitre_stage"].map(MITRE_STAGES).fillna(0).astype(int)
        return df

    def extract(self, file_path: Optional[str] = None, directory: Optional[str] = None, fit_scaler: bool = True) -> pd.DataFrame:
        """Full extraction pipeline execution."""
        if file_path:
            df = self.load_csv(file_path)
        elif directory:
            files = list(pathlib.Path(directory).glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"No CSV files found in {directory}")
            dfs = [self.load_csv(str(f)) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            default_path = self.raw_dir / "ciciot2023.csv"
            if default_path.exists():
                df = self.load_csv(str(default_path))
            else:
                logger.warning("No input file found. Generating synthetic CICIoT2023 frame.")
                df = self._generate_synthetic_frame()

        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.map_labels(df)
        logger.info(f"CICIoT2023 extraction complete: {len(df)} records.")
        return df

    def _generate_synthetic_frame(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic CICIoT2023 records for pipeline verification."""
        rng = np.random.default_rng(43)
        attack_types = [
            "Benign", "DDoS-ICMP_Flood", "DDoS-UDP_Flood", "Mirai-greeth_flood",
            "Recon-PortScan", "VulnerabilityScan", "DNS_Spoofing", "CommandInjection"
        ]
        data = {
            "flow_duration": rng.uniform(0.01, 30.0, num_records),
            "Header_Length": rng.integers(20, 60, num_records),
            "Protocol Type": rng.choice([6.0, 17.0, 1.0], num_records),
            "Duration": rng.uniform(0.01, 30.0, num_records),
            "Rate": rng.uniform(10.0, 50000.0, num_records),
            "Srate": rng.uniform(10.0, 25000.0, num_records),
            "Drate": rng.uniform(0.0, 25000.0, num_records),
            "fin_flag_number": rng.choice([0, 1], num_records, p=[0.9, 0.1]),
            "syn_flag_number": rng.choice([0, 1], num_records, p=[0.7, 0.3]),
            "rst_flag_number": rng.choice([0, 1], num_records, p=[0.95, 0.05]),
            "psh_flag_number": rng.choice([0, 1], num_records, p=[0.6, 0.4]),
            "ack_flag_number": rng.choice([0, 1], num_records, p=[0.3, 0.7]),
            "ece_flag_number": [0] * num_records,
            "cwr_flag_number": [0] * num_records,
            "HTTP": rng.choice([0, 1], num_records, p=[0.7, 0.3]),
            "HTTPS": rng.choice([0, 1], num_records, p=[0.6, 0.4]),
            "DNS": rng.choice([0, 1], num_records, p=[0.9, 0.1]),
            "Telnet": rng.choice([0, 1], num_records, p=[0.95, 0.05]),
            "SSH": rng.choice([0, 1], num_records, p=[0.9, 0.1]),
            "AVG": rng.uniform(40.0, 1400.0, num_records),
            "Min": rng.uniform(40.0, 100.0, num_records),
            "Max": rng.uniform(100.0, 1500.0, num_records),
            "Std": rng.uniform(0.0, 300.0, num_records),
            "Number": rng.integers(1, 200, num_records),
            "label": rng.choice(attack_types, num_records, p=[0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        }
        return pd.DataFrame(data)
