"""
PRISM — DARPA Intrusion Detection Dataset Feature Extractor
Ingests DARPA 1998/1999/2000 evaluation flow records and converts them to
the PRISM temporal state representation.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import DARPA_LABEL_TO_MITRE, MITRE_STAGES, TOP_ATTACKED_PORTS

logger = logging.getLogger("prism.data.darpa_extractor")


class DARPAExtractor:
    """DARPA Intrusion Detection Dataset flow extractor."""

    def __init__(self, raw_dir: str = "data/raw") -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.label_map = DARPA_LABEL_TO_MITRE

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load DARPA flow dump or kddcup-like CSV file."""
        logger.info(f"Loading DARPA dataset file from {file_path}")
        df = pd.read_csv(file_path, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardise DARPA records."""
        df = df.copy()

        # Find label column
        label_col = "label" if "label" in df.columns else ("target" if "target" in df.columns else None)
        if label_col:
            df["Label"] = df[label_col].astype(str).str.rstrip(".").str.strip()
        else:
            df["Label"] = "normal"

        rename_map = {
            "duration": "Flow Duration",
            "src_bytes": "TotLen Fwd Pkts",
            "dst_bytes": "TotLen Bwd Pkts",
            "wrong_fragment": "ip_frag_flag",
            "protocol_type": "Protocol",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "Dst Port" not in df.columns:
            df["Dst Port"] = 80

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features matching PRISM flow structure."""
        df = df.copy()
        dst_port = df["Dst Port"]

        df["Tot Fwd Pkts"] = np.ceil(df.get("TotLen Fwd Pkts", 100) / 500.0)
        df["Tot Bwd Pkts"] = np.ceil(df.get("TotLen Bwd Pkts", 100) / 500.0)
        df["Flow Byts/s"] = (df.get("TotLen Fwd Pkts", 0) + df.get("TotLen Bwd Pkts", 0)) / (df.get("Flow Duration", 1) + 1e-6)
        df["Flow Pkts/s"] = (df["Tot Fwd Pkts"] + df["Tot Bwd Pkts"]) / (df.get("Flow Duration", 1) + 1e-6)
        df["fwd_bwd_ratio"] = df["Tot Fwd Pkts"] / (df["Tot Bwd Pkts"] + 1e-6)

        df["port_cat_well_known"] = ((dst_port >= 0) & (dst_port <= 1023)).astype(float)
        df["port_cat_registered"] = ((dst_port >= 1024) & (dst_port <= 49151)).astype(float)
        df["port_cat_ephemeral"] = ((dst_port >= 49152) & (dst_port <= 65535)).astype(float)

        for port in TOP_ATTACKED_PORTS:
            df[f"port_{port}"] = (dst_port == port).astype(float)

        for flag in ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]:
            df[f"flag_{flag.lower()}_frac"] = 0.0

        return df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map DARPA labels to MITRE ATT&CK stages."""
        df = df.copy()
        raw_labels = df["Label"].astype(str)

        def map_fn(lbl: str) -> str:
            cleaned = lbl.strip().lower()
            if cleaned in self.label_map:
                return self.label_map[cleaned]
            for k, v in self.label_map.items():
                if k in cleaned:
                    return v
            return "Benign"

        df["mitre_stage"] = raw_labels.apply(map_fn)
        df["mitre_stage_id"] = df["mitre_stage"].map(MITRE_STAGES).fillna(0).astype(int)
        return df

    def extract(self, file_path: Optional[str] = None, directory: Optional[str] = None, fit_scaler: bool = True) -> pd.DataFrame:
        """Full DARPA extraction pipeline."""
        if file_path:
            df = self.load_csv(file_path)
        else:
            default_path = self.raw_dir / "darpa.csv"
            if default_path.exists():
                df = self.load_csv(str(default_path))
            else:
                logger.warning("No DARPA input file found. Generating synthetic DARPA frame.")
                df = self._generate_synthetic_frame()

        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.map_labels(df)
        logger.info(f"DARPA extraction complete: {len(df)} records.")
        return df

    def _generate_synthetic_frame(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic DARPA dataset records."""
        rng = np.random.default_rng(45)
        categories = ["normal", "probe", "dos", "u2r", "r2l"]
        data = {
            "duration": rng.uniform(0, 100, num_records),
            "protocol_type": rng.choice(["tcp", "udp", "icmp"], num_records),
            "service": rng.choice(["http", "smtp", "domain_u", "ftp", "ecr_i"], num_records),
            "flag": rng.choice(["SF", "S0", "REJ", "RSTR"], num_records),
            "src_bytes": rng.integers(0, 50000, num_records),
            "dst_bytes": rng.integers(0, 50000, num_records),
            "land": [0] * num_records,
            "wrong_fragment": [0] * num_records,
            "urgent": [0] * num_records,
            "label": rng.choice(categories, num_records, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
        }
        return pd.DataFrame(data)
