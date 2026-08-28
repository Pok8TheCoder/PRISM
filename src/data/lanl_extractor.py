"""
PRISM — LANL Authentication Dataset Feature Extractor
Ingests Los Alamos National Laboratory (LANL) authentication event logs
and processes them into time-windowed feature representations for PRISM.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.constants import LANL_LABEL_TO_MITRE, MITRE_STAGES, TOP_ATTACKED_PORTS

logger = logging.getLogger("prism.data.lanl_extractor")


class LANLExtractor:
    """LANL Unified Host and Network Authentication log extractor."""

    def __init__(self, raw_dir: str = "data/raw") -> None:
        self.raw_dir = pathlib.Path(raw_dir)
        self.label_map = LANL_LABEL_TO_MITRE

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load LANL auth.txt or logon.csv file."""
        logger.info(f"Loading LANL authentication log from {file_path}")
        # LANL auth records: time, src_user@domain, dst_user@domain, src_comp, dst_comp, auth_type, logon_type, orientation, result
        df = pd.read_csv(file_path, low_memory=False, header=None)
        if len(df.columns) == 9:
            df.columns = [
                "time", "src_user", "dst_user", "src_comp", "dst_comp",
                "auth_type", "logon_type", "orientation", "result"
            ]
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format LANL logs to flow-compatible schema."""
        df = df.copy()

        # Fill missing values
        df = df.fillna("?")

        # Standardise names
        df["Timestamp"] = pd.to_numeric(df.get("time", 0), errors="coerce").fillna(0)
        df["Flow Duration"] = 1.0  # Discrete point authentication events
        df["Tot Fwd Pkts"] = 1
        df["Tot Bwd Pkts"] = 1
        df["TotLen Fwd Pkts"] = 64
        df["TotLen Bwd Pkts"] = 64
        df["Dst Port"] = 88  # Kerberos default or 445 SMB
        df["Protocol"] = 6  # TCP

        # Failures count as potential unauthorized access attempt
        result_col = df.get("result", pd.Series("Success", index=df.index))
        df["Label"] = np.where(result_col.astype(str).str.upper().str.startswith("FAIL"), "AuthenticationFailure", "Normal")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer feature columns matching PRISM flow standard."""
        df = df.copy()
        dst_port = df["Dst Port"]

        df["Flow Byts/s"] = 128.0
        df["Flow Pkts/s"] = 2.0
        df["fwd_bwd_ratio"] = 1.0
        df["port_cat_well_known"] = 1.0
        df["port_cat_registered"] = 0.0
        df["port_cat_ephemeral"] = 0.0

        for port in TOP_ATTACKED_PORTS:
            df[f"port_{port}"] = (dst_port == port).astype(float)

        for flag in ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]:
            df[f"flag_{flag.lower()}_frac"] = 0.0

        return df

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map authentication events to MITRE stages."""
        df = df.copy()
        raw_labels = df["Label"].astype(str)

        def map_fn(lbl: str) -> str:
            return self.label_map.get(lbl.strip(), "Benign")

        df["mitre_stage"] = raw_labels.apply(map_fn)
        df["mitre_stage_id"] = df["mitre_stage"].map(MITRE_STAGES).fillna(0).astype(int)
        return df

    def extract(self, file_path: Optional[str] = None, directory: Optional[str] = None, fit_scaler: bool = True) -> pd.DataFrame:
        """Extract pipeline for LANL auth dataset."""
        if file_path:
            df = self.load_csv(file_path)
        else:
            default_path = self.raw_dir / "lanl_auth.csv"
            if default_path.exists():
                df = self.load_csv(str(default_path))
            else:
                logger.warning("No LANL input file found. Generating synthetic LANL auth frame.")
                df = self._generate_synthetic_frame()

        df = self.clean(df)
        df = self.engineer_features(df)
        df = self.map_labels(df)
        logger.info(f"LANL auth log extraction complete: {len(df)} records.")
        return df

    def _generate_synthetic_frame(self, num_records: int = 500) -> pd.DataFrame:
        """Generate synthetic LANL authentication records."""
        rng = np.random.default_rng(44)
        data = {
            "time": np.sort(rng.integers(1, 86400, num_records)),
            "src_user": [f"U{rng.integers(100, 999)}@DOM1" for _ in range(num_records)],
            "dst_user": [f"U{rng.integers(100, 999)}@DOM1" for _ in range(num_records)],
            "src_comp": [f"C{rng.integers(1, 500)}" for _ in range(num_records)],
            "dst_comp": [f"C{rng.integers(1, 500)}" for _ in range(num_records)],
            "auth_type": rng.choice(["Kerberos", "NTLM", "Negotiate", "?"], num_records),
            "logon_type": rng.choice(["Network", "Interactive", "Service", "Batch"], num_records),
            "orientation": rng.choice(["LogOn", "LogOff", "TGS", "TGT"], num_records),
            "result": rng.choice(["Success", "Fail"], num_records, p=[0.9, 0.1]),
        }
        return pd.DataFrame(data)
