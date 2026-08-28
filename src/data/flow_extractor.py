"""
PRISM — Flow-Level Feature Extraction

Ingests CIC-IDS-2018 and CTU-13 CSV flow records and outputs cleaned,
normalised feature DataFrames ready for the downstream Mamba/Transformer
state-space model.

Usage:
    extractor = FlowExtractor(dataset_type="cicids2018", raw_dir="data/raw")
    df = extractor.extract(directory="data/raw/cicids2018")
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
    CICIDS_LABEL_TO_MITRE,
    CTU13_LABEL_TO_MITRE,
    LOG_TRANSFORM_FEATURES,
    MITRE_STAGES,
    TCP_FLAGS,
    TOP_ATTACKED_PORTS,
)

logger = logging.getLogger("prism.data.flow_extractor")

# Column name aliases — CIC-IDS-2018 CSVs are inconsistent across releases.
# We normalise to the canonical names used in constants.py.
_COLUMN_ALIASES = {
    # destination port
    "dst port": "Dst Port",
    "destination port": "Dst Port",
    "dst_port": "Dst Port",
    # protocol
    "protocol": "Protocol",
    # label
    "label": "Label",
    # forward / backward packets
    "tot fwd pkts": "Tot Fwd Pkts",
    "total fwd packets": "Tot Fwd Pkts",
    "tot bwd pkts": "Tot Bwd Pkts",
    "total backward packets": "Tot Bwd Pkts",
    # lengths
    "totlen fwd pkts": "TotLen Fwd Pkts",
    "total length of fwd packets": "TotLen Fwd Pkts",
    "totlen bwd pkts": "TotLen Bwd Pkts",
    "total length of bwd packets": "TotLen Bwd Pkts",
    # flow duration
    "flow duration": "Flow Duration",
    # bytes / packets per second
    "flow byts/s": "Flow Byts/s",
    "flow bytes/s": "Flow Byts/s",
    "flow pkts/s": "Flow Pkts/s",
    "flow packets/s": "Flow Pkts/s",
    # header lengths
    "fwd header len": "Fwd Header Len",
    "fwd header length": "Fwd Header Len",
    "fwd header length.1": "Fwd Header Len",
    "bwd header len": "Bwd Header Len",
    "bwd header length": "Bwd Header Len",
    # flag counts
    "fin flag cnt": "FIN Flag Cnt",
    "fin flag count": "FIN Flag Cnt",
    "syn flag cnt": "SYN Flag Cnt",
    "syn flag count": "SYN Flag Cnt",
    "rst flag cnt": "RST Flag Cnt",
    "rst flag count": "RST Flag Cnt",
    "psh flag cnt": "PSH Flag Cnt",
    "psh flag count": "PSH Flag Cnt",
    "ack flag cnt": "ACK Flag Cnt",
    "ack flag count": "ACK Flag Cnt",
    "urg flag cnt": "URG Flag Cnt",
    "urg flag count": "URG Flag Cnt",
    # timestamp
    "timestamp": "Timestamp",
}

# Flag count column names (canonical) used for distribution calculation
_FLAG_CNT_COLS = {
    "SYN": "SYN Flag Cnt",
    "ACK": "ACK Flag Cnt",
    "FIN": "FIN Flag Cnt",
    "RST": "RST Flag Cnt",
    "PSH": "PSH Flag Cnt",
    "URG": "URG Flag Cnt",
}

# IAT columns that commonly contain infinity in CIC-IDS-2018
_IAT_COLUMNS = [
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
]


class FlowExtractor:
    """End-to-end feature extraction pipeline for network flow CSVs."""

    def __init__(
        self,
        dataset_type: str = "cicids2018",
        raw_dir: str = "data/raw",
    ) -> None:
        """
        Parameters
        ----------
        dataset_type : str
            One of ``"cicids2018"`` or ``"ctu13"``.
        raw_dir : str
            Root directory that contains raw CSV files.
        """
        if dataset_type not in ("cicids2018", "ctu13"):
            raise ValueError(
                f"Unknown dataset_type '{dataset_type}'. "
                "Expected 'cicids2018' or 'ctu13'."
            )
        self.dataset_type: str = dataset_type
        self.raw_dir: str = raw_dir
        self.scaler: StandardScaler = StandardScaler()
        self._scaler_fitted: bool = False
        self._numeric_columns: Optional[List[str]] = None
        logger.info(
            "FlowExtractor initialised  ·  dataset=%s  ·  raw_dir=%s",
            dataset_type,
            raw_dir,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CSV flow file with robust type handling.

        Parameters
        ----------
        file_path : str
            Path to a ``.csv`` file.

        Returns
        -------
        pd.DataFrame
        """
        logger.info("Loading CSV: %s", file_path)
        df = pd.read_csv(
            file_path,
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
            on_bad_lines="warn",
        )

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Normalise column names via alias table (case-insensitive lookup)
        rename_map = {}
        for col in df.columns:
            key = col.lower().strip()
            if key in _COLUMN_ALIASES and col != _COLUMN_ALIASES[key]:
                rename_map[col] = _COLUMN_ALIASES[key]
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.debug("Renamed columns: %s", rename_map)

        # Coerce numeric columns — some CSVs contain non-numeric artefacts
        for col in df.columns:
            if col in ("Label", "Timestamp", "Src IP", "Dst IP"):
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(
            "Loaded %d rows × %d columns from %s",
            len(df),
            len(df.columns),
            file_path,
        )
        return df

    def load_all_csvs(self, directory: Optional[str] = None) -> pd.DataFrame:
        """Load and concatenate every CSV in *directory*.

        Parameters
        ----------
        directory : str, optional
            Directory to scan.  Falls back to ``self.raw_dir``.

        Returns
        -------
        pd.DataFrame
        """
        directory = directory or self.raw_dir
        csv_files = sorted(
            str(p)
            for p in pathlib.Path(directory).rglob("*.csv")
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in '{directory}'."
            )
        logger.info("Found %d CSV file(s) in %s", len(csv_files), directory)

        frames: list[pd.DataFrame] = []
        for fpath in csv_files:
            try:
                frames.append(self.load_csv(fpath))
            except Exception:
                logger.exception("Failed to load %s — skipping.", fpath)

        if not frames:
            raise RuntimeError(
                "All CSV files failed to load.  Check data integrity."
            )

        df = pd.concat(frames, ignore_index=True)
        logger.info(
            "Concatenated into %d rows × %d columns", len(df), len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw DataFrame.

        * Replace infinite values in IAT columns with column-wise max of
          finite values.
        * Drop rows that still contain NaN or Inf after IAT replacement.
        * Drop duplicate rows.

        Returns a copy — the input is not mutated.
        """
        logger.info("Cleaning: input shape %s", df.shape)
        df = df.copy()

        # Replace +/-Inf in IAT columns with column max of finite values
        iat_cols_present = [c for c in _IAT_COLUMNS if c in df.columns]
        for col in iat_cols_present:
            finite_mask = np.isfinite(df[col])
            if finite_mask.any():
                col_max = df.loc[finite_mask, col].max()
            else:
                col_max = 0.0
            df[col] = df[col].replace([np.inf, -np.inf], col_max)

        # Replace remaining infinities across all numeric columns with NaN,
        # then drop rows with NaN.
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        before = len(df)
        df.dropna(inplace=True)
        dropped_nan = before - len(df)

        before = len(df)
        df.drop_duplicates(inplace=True)
        dropped_dup = before - len(df)

        df.reset_index(drop=True, inplace=True)
        logger.info(
            "Cleaning done: dropped %d NaN/Inf rows, %d duplicates  ·  "
            "output shape %s",
            dropped_nan,
            dropped_dup,
            df.shape,
        )
        return df

    # ------------------------------------------------------------------
    # Feature Engineering
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from the raw flow columns.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame  (new columns appended in-place on a copy)
        """
        logger.info("Engineering features …")
        df = df.copy()

        # --- Forward / Backward ratio ------------------------------------
        fwd_col = "Tot Fwd Pkts" if "Tot Fwd Pkts" in df.columns else None
        bwd_col = "Tot Bwd Pkts" if "Tot Bwd Pkts" in df.columns else None
        if fwd_col and bwd_col:
            df["fwd_bwd_ratio"] = df[fwd_col] / (df[bwd_col] + 1)
        else:
            logger.warning(
                "Cannot compute fwd_bwd_ratio — missing columns "
                "(Tot Fwd Pkts / Tot Bwd Pkts)."
            )

        # --- Bytes per packet ---------------------------------------------
        total_len_cols = [
            c for c in ("TotLen Fwd Pkts", "TotLen Bwd Pkts") if c in df.columns
        ]
        total_pkt_cols = [
            c for c in ("Tot Fwd Pkts", "Tot Bwd Pkts") if c in df.columns
        ]
        if total_len_cols and total_pkt_cols:
            total_bytes = df[total_len_cols].sum(axis=1)
            total_packets = df[total_pkt_cols].sum(axis=1)
            df["bytes_per_packet"] = total_bytes / (total_packets + 1)
        else:
            logger.warning(
                "Cannot compute bytes_per_packet — missing length/packet columns."
            )

        # --- TCP flag distribution ----------------------------------------
        flag_cols_present = {
            flag: col
            for flag, col in _FLAG_CNT_COLS.items()
            if col in df.columns
        }
        if flag_cols_present:
            flag_total = sum(df[c] for c in flag_cols_present.values()) + 1e-9
            for flag in TCP_FLAGS:
                if flag in flag_cols_present:
                    df[f"flag_{flag.lower()}_frac"] = (
                        df[flag_cols_present[flag]] / flag_total
                    )
                else:
                    df[f"flag_{flag.lower()}_frac"] = 0.0
        else:
            logger.warning(
                "No TCP flag count columns found — skipping flag distribution."
            )

        # --- Port binning -------------------------------------------------
        if "Dst Port" in df.columns:
            conditions = [
                df["Dst Port"].between(0, 1023),
                df["Dst Port"].between(1024, 49151),
                df["Dst Port"].between(49152, 65535),
            ]
            choices = ["well_known", "registered", "ephemeral"]
            df["port_category"] = np.select(
                conditions, choices, default="unknown"
            )
            # One-hot encode port categories
            for cat in choices:
                df[f"port_cat_{cat}"] = (df["port_category"] == cat).astype(
                    np.int8
                )
        else:
            logger.warning(
                "Dst Port column missing — skipping port binning."
            )

        # --- One-hot for top-20 attacked ports ----------------------------
        if "Dst Port" in df.columns:
            for port in TOP_ATTACKED_PORTS:
                df[f"port_{port}"] = (df["Dst Port"] == port).astype(np.int8)
        else:
            logger.warning(
                "Dst Port column missing — skipping top-port one-hot."
            )

        logger.info(
            "Feature engineering complete  ·  shape %s", df.shape
        )
        return df

    # ------------------------------------------------------------------
    # Log Transform
    # ------------------------------------------------------------------

    def log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply ``log1p`` to heavily skewed numeric features.

        Only transforms columns that exist in the DataFrame to stay robust
        when a dataset has a different schema.
        """
        logger.info("Applying log1p transform …")
        df = df.copy()
        transformed: list[str] = []
        for col in LOG_TRANSFORM_FEATURES:
            if col in df.columns:
                # Clip at zero before log1p to avoid NaN from small negatives
                # introduced by floating-point arithmetic.
                df[col] = np.log1p(df[col].clip(lower=0))
                transformed.append(col)
        logger.info("Log-transformed %d columns: %s", len(transformed), transformed)
        return df

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> pd.DataFrame:
        """Z-score normalise all numeric features.

        Parameters
        ----------
        df : pd.DataFrame
        fit : bool
            If ``True`` (default), fit the scaler and store statistics.
            If ``False``, re-use previously fitted statistics (for inference).

        Returns
        -------
        pd.DataFrame  — same index and columns, numeric columns normalised.
        """
        logger.info("Normalising  ·  fit=%s", fit)
        df = df.copy()

        # Identify numeric columns, excluding engineered categoricals and labels
        _exclude = {
            "Label",
            "mitre_stage",
            "mitre_stage_id",
            "port_category",
            "Timestamp",
        }
        numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in _exclude
        ]

        if fit:
            self.scaler.fit(df[numeric_cols])
            self._scaler_fitted = True
            self._numeric_columns = numeric_cols
            logger.info("Scaler fitted on %d columns.", len(numeric_cols))
        else:
            if not self._scaler_fitted:
                raise RuntimeError(
                    "Scaler has not been fitted yet.  "
                    "Call normalize(fit=True) or load_scaler() first."
                )
            # Use the columns the scaler was fitted on; if any are missing in
            # the incoming frame we fill with 0 (already centred).
            numeric_cols = self._numeric_columns or numeric_cols
            missing = [c for c in numeric_cols if c not in df.columns]
            if missing:
                logger.warning(
                    "Columns present at fit time but missing now — "
                    "filling with 0: %s",
                    missing,
                )
                for c in missing:
                    df[c] = 0.0

        df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        logger.info("Normalisation complete.")
        return df

    # ------------------------------------------------------------------
    # Label Mapping
    # ------------------------------------------------------------------

    def map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map dataset-specific attack labels to MITRE ATT&CK stages.

        Adds two columns:
        * ``mitre_stage``    — human-readable stage name (str)
        * ``mitre_stage_id`` — integer ID from ``MITRE_STAGES``

        Unknown labels are mapped to ``"Benign"`` / ``0`` with a warning.
        """
        logger.info("Mapping labels to MITRE ATT&CK stages …")
        df = df.copy()

        if "Label" not in df.columns:
            logger.warning("No 'Label' column found — skipping label mapping.")
            return df

        # Strip whitespace from labels
        df["Label"] = df["Label"].astype(str).str.strip()

        if self.dataset_type == "cicids2018":
            label_map = CICIDS_LABEL_TO_MITRE
        else:
            label_map = CTU13_LABEL_TO_MITRE

        # Map to MITRE stage string
        df["mitre_stage"] = df["Label"].map(label_map)

        # Warn about unmapped labels and default them to Benign
        unmapped_mask = df["mitre_stage"].isna()
        if unmapped_mask.any():
            unmapped_labels = df.loc[unmapped_mask, "Label"].unique().tolist()
            logger.warning(
                "Unmapped labels (defaulting to 'Benign'): %s", unmapped_labels
            )
            df["mitre_stage"] = df["mitre_stage"].fillna("Benign")

        # Map to integer ID
        df["mitre_stage_id"] = df["mitre_stage"].map(MITRE_STAGES)

        stage_counts = df["mitre_stage"].value_counts().to_dict()
        logger.info("MITRE stage distribution: %s", stage_counts)
        return df

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def extract(
        self,
        file_path: Optional[str] = None,
        directory: Optional[str] = None,
        fit_scaler: bool = True,
    ) -> pd.DataFrame:
        """Run the complete extraction pipeline.

        Exactly one of *file_path* or *directory* must be provided.

        Pipeline: load → clean → engineer → log_transform → normalize →
        map_labels.

        Parameters
        ----------
        file_path : str, optional
            Path to a single CSV.
        directory : str, optional
            Directory of CSVs to concatenate.
        fit_scaler : bool
            Passed through to :meth:`normalize`.

        Returns
        -------
        pd.DataFrame — fully processed, ready for modelling.
        """
        if file_path is None and directory is None:
            raise ValueError(
                "Provide either 'file_path' or 'directory'."
            )

        # 1. Load
        if file_path is not None:
            df = self.load_csv(file_path)
        else:
            df = self.load_all_csvs(directory)

        # 2. Clean
        df = self.clean(df)

        # 3. Engineer features
        df = self.engineer_features(df)

        # 4. Log transform skewed features
        df = self.log_transform(df)

        # 5. Normalise
        df = self.normalize(df, fit=fit_scaler)

        # 6. Map labels
        df = self.map_labels(df)

        logger.info(
            "Extraction pipeline complete  ·  final shape %s", df.shape
        )
        return df

    # ------------------------------------------------------------------
    # Scaler Persistence
    # ------------------------------------------------------------------

    def save_scaler(self, path: str) -> None:
        """Persist the fitted :class:`StandardScaler` to disk.

        Also saves the list of numeric column names so that inference-time
        DataFrames are aligned correctly.
        """
        if not self._scaler_fitted:
            raise RuntimeError("Scaler has not been fitted — nothing to save.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "numeric_columns": self._numeric_columns,
        }
        joblib.dump(payload, path)
        logger.info("Scaler saved to %s", path)

    def load_scaler(self, path: str) -> None:
        """Load a previously saved scaler from *path*."""
        payload = joblib.load(path)
        self.scaler = payload["scaler"]
        self._numeric_columns = payload["numeric_columns"]
        self._scaler_fitted = True
        logger.info(
            "Scaler loaded from %s  ·  %d columns",
            path,
            len(self._numeric_columns) if self._numeric_columns else 0,
        )


# ======================================================================
# Standalone demo
# ======================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="PRISM Flow-Level Feature Extractor",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single CSV flow file.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing CSV flow files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cicids2018", "ctu13"],
        default="cicids2018",
        help="Dataset type (default: cicids2018).",
    )
    parser.add_argument(
        "--save-scaler",
        type=str,
        default=None,
        help="Path to save the fitted scaler (e.g. weights/scaler.joblib).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the processed DataFrame as parquet.",
    )
    args = parser.parse_args()

    if args.file is None and args.dir is None:
        parser.error("Provide at least one of --file or --dir.")

    extractor = FlowExtractor(dataset_type=args.dataset)
    result = extractor.extract(file_path=args.file, directory=args.dir)

    print("\n--- Processed DataFrame ---")
    print(f"Shape : {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(result.head())

    if "mitre_stage" in result.columns:
        print("\n--- MITRE Stage Distribution ---")
        print(result["mitre_stage"].value_counts())

    if args.save_scaler:
        extractor.save_scaler(args.save_scaler)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.to_parquet(args.output, index=False)
        logger.info("Output written to %s", args.output)
        print(f"\nSaved to {args.output}")
