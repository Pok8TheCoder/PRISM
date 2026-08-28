"""
PRISM Feature Merger
Joins flow-level and packet-level features into a unified feature matrix.
"""

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger("prism.data.feature_merger")


class FeatureMerger:
    """Merge flow-level and packet-level features on 5-tuple + time window."""

    def __init__(self):
        self._merge_stats = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def merge(
        self,
        flow_df: pd.DataFrame,
        packet_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Merge flow and packet DataFrames.

        Parameters
        ----------
        flow_df : pd.DataFrame
            Output of FlowExtractor.extract(). Must contain ``window_id``.
        packet_df : pd.DataFrame | None
            Output of PacketExtractor (per-flow, per-window).
            If None, packet features are filled with zeros.

        Returns
        -------
        pd.DataFrame
            Unified feature matrix with a ``has_packet_features`` flag.
        """
        flow_df = flow_df.copy()

        # Ensure window_id exists in flow_df
        if "window_id" not in flow_df.columns:
            flow_df = self._assign_window_ids(flow_df)

        if packet_df is None or packet_df.empty:
            logger.info("No packet features provided; filling with zeros.")
            merged = self._fill_packet_zeros(flow_df)
            self._merge_stats = {
                "flow_rows": len(flow_df),
                "packet_rows": 0,
                "merged_rows": len(merged),
                "match_rate": 0.0,
            }
            return merged

        # Normalise join keys
        join_keys = self._resolve_join_keys(flow_df, packet_df)
        logger.info("Joining on keys: %s", join_keys)

        merged = pd.merge(
            flow_df,
            packet_df,
            on=join_keys,
            how="left",
            suffixes=("", "_pkt"),
        )

        # Fill unmatched packet features with 0
        pkt_feat_cols = [
            c for c in packet_df.columns if c not in join_keys
        ]
        merged["has_packet_features"] = merged[pkt_feat_cols[0]].notna().astype(int)
        for col in pkt_feat_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)

        matched = int(merged["has_packet_features"].sum())
        self._merge_stats = {
            "flow_rows": len(flow_df),
            "packet_rows": len(packet_df),
            "merged_rows": len(merged),
            "match_rate": matched / max(len(merged), 1),
        }
        logger.info(
            "Merged %d flow rows with %d packet rows -> %d rows (%.1f%% matched)",
            len(flow_df),
            len(packet_df),
            len(merged),
            self._merge_stats["match_rate"] * 100,
        )
        return merged

    @property
    def stats(self) -> dict:
        return dict(self._merge_stats)

    def save(self, df: pd.DataFrame, path: str) -> None:
        """Save merged features to CSV or Parquet."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Saved merged features to %s (%d rows)", path, len(df))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _assign_window_ids(
        self, df: pd.DataFrame, window_seconds: int = 30
    ) -> pd.DataFrame:
        """Assign window_id based on Timestamp column."""
        ts_col = None
        for candidate in ["Timestamp", "timestamp", "flow_start"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            logger.warning(
                "No timestamp column found; assigning sequential window_id."
            )
            df["window_id"] = np.arange(len(df)) // 100
            return df

        ts = pd.to_datetime(df[ts_col], errors="coerce")
        epoch = (ts - ts.min()).dt.total_seconds()
        df["window_id"] = (epoch // window_seconds).astype(int)
        return df

    def _resolve_join_keys(
        self, flow_df: pd.DataFrame, packet_df: pd.DataFrame
    ) -> list[str]:
        """Find common join columns present in both DataFrames."""
        candidates = [
            "window_id",
            "src_ip", "dst_ip",
            "src_port", "dst_port",
            "protocol",
            # hashed variants
            "src_ip_hash", "dst_ip_hash",
        ]
        keys = [c for c in candidates if c in flow_df.columns and c in packet_df.columns]
        if not keys:
            logger.warning("No common join keys found; falling back to window_id only.")
            return ["window_id"]
        return keys

    def _fill_packet_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add zero-valued packet-feature columns."""
        from src.utils.constants import PACKET_FEATURES

        for feat in PACKET_FEATURES:
            if feat not in df.columns:
                df[feat] = 0.0
        df["has_packet_features"] = 0
        return df


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Merge flow + packet features")
    parser.add_argument("--flow-csv", required=True, help="Path to flow CSV")
    parser.add_argument("--packet-csv", default=None, help="Path to packet CSV")
    parser.add_argument("--output", default="data/processed/merged.csv")
    args = parser.parse_args()

    merger = FeatureMerger()
    flow = pd.read_csv(args.flow_csv)
    pkt = pd.read_csv(args.packet_csv) if args.packet_csv else None
    result = merger.merge(flow, pkt)
    merger.save(result, args.output)
    print(f"Merge stats: {merger.stats}")
