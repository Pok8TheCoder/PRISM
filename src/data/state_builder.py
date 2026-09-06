"""
PRISM State Builder
Converts merged feature DataFrames into time-windowed state vectors or
graph representations for consumption by the world model.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("prism.data.state_builder")


class StateBuilder:
    """
    Build state representations from merged flow+packet features.

    Each time window ``t`` produces one state vector ``S_t`` (feature-vector
    mode) or one graph ``G_t`` (graph mode).
    """

    def __init__(
        self,
        window_size_seconds: int = 30,
        mode: str = "vector",  # "vector" | "graph"
    ):
        self.window_size_seconds = window_size_seconds
        self.mode = mode
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_states(
        self,
        df: pd.DataFrame,
        label_col: str = "mitre_stage_id",
    ) -> dict:
        """
        Build the full sequence of states from the merged DataFrame.

        Returns
        -------
        dict with keys:
            states : np.ndarray (T, D_state)  — state vectors
            labels_binary : np.ndarray (T,)    — 0=benign, 1=attack per window
            labels_mitre  : np.ndarray (T,)    — MITRE stage id per window
            window_ids    : np.ndarray (T,)    — window identifiers
            feature_names : list[str]          — ordered feature names in state
        """
        df = df.copy()

        # Ensure window_id
        if "window_id" not in df.columns:
            df = self._assign_window_ids(df)

        if self.mode == "vector":
            return self._build_vector_states(df, label_col)
        elif self.mode == "graph":
            return self._build_graph_states(df, label_col)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def state_dim(self) -> int:
        return len(self._feature_names)

    # ------------------------------------------------------------------
    # Vector state builder
    # ------------------------------------------------------------------
    def _build_vector_states(self, df: pd.DataFrame, label_col: str) -> dict:
        """Aggregate per-window into a fixed-size feature vector."""
        from src.utils.constants import (
            TCP_FLAGS,
            PROTOCOLS,
            PACKET_FEATURES,
        )

        # Identify numeric columns for aggregation
        exclude_cols = {
            "window_id", "Label", "label", "mitre_stage",
            "mitre_stage_id", "Timestamp", "timestamp",
            "Src IP", "src_ip", "Dst IP", "dst_ip",
            "src_ip_hash", "dst_ip_hash",
            "Flow ID", "flow_id",
        }
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

        grouped = df.groupby("window_id")
        window_ids = sorted(df["window_id"].unique())
        states = []
        labels_binary = []
        labels_mitre = []

        for wid in window_ids:
            window = grouped.get_group(wid)
            state = self._aggregate_window_vector(window, numeric_cols)
            states.append(state)

            # Labels — attack if any attack flow present in window
            if label_col in window.columns:
                has_attack = int((window[label_col] > 0).any())
                if has_attack:
                    attack_flows = window[window[label_col] > 0]
                    stage_counts = attack_flows[label_col].value_counts()
                    if 5 in stage_counts.index:
                        dominant_stage = 5
                    elif 3 in stage_counts.index:
                        dominant_stage = 3
                    else:
                        dominant_stage = int(stage_counts.idxmax())
                else:
                    dominant_stage = 0
            else:
                has_attack = 0
                dominant_stage = 0

            labels_binary.append(has_attack)
            labels_mitre.append(dominant_stage)

        states = np.array(states, dtype=np.float32)
        # Replace any remaining NaN/Inf
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            "Built %d vector states, dim=%d (attack windows: %d / %d)",
            len(states),
            states.shape[1] if len(states) > 0 else 0,
            sum(labels_binary),
            len(labels_binary),
        )

        return {
            "states": states,
            "labels_binary": np.array(labels_binary, dtype=np.int64),
            "labels_mitre": np.array(labels_mitre, dtype=np.int64),
            "window_ids": np.array(window_ids, dtype=np.int64),
            "feature_names": list(self._feature_names),
        }

    def _aggregate_window_vector(
        self, window: pd.DataFrame, numeric_cols: list[str]
    ) -> np.ndarray:
        """Produce a single state vector for one time window."""
        features = []
        names = []

        # --- Counts ---
        features.append(len(window))
        names.append("num_flows")

        # Unique IPs
        for col in ["Src IP", "src_ip", "src_ip_hash"]:
            if col in window.columns:
                features.append(window[col].nunique())
                names.append("num_unique_src_ips")
                break
        else:
            features.append(0)
            names.append("num_unique_src_ips")

        for col in ["Dst IP", "dst_ip", "dst_ip_hash"]:
            if col in window.columns:
                features.append(window[col].nunique())
                names.append("num_unique_dst_ips")
                break
        else:
            features.append(0)
            names.append("num_unique_dst_ips")

        # Unique dst ports
        for col in ["Dst Port", "dst_port"]:
            if col in window.columns:
                features.append(window[col].nunique())
                names.append("num_unique_dst_ports")
                break
        else:
            features.append(0)
            names.append("num_unique_dst_ports")

        # Port entropy
        for col in ["Dst Port", "dst_port"]:
            if col in window.columns:
                features.append(self._shannon_entropy(window[col].values))
                names.append("port_entropy")
                break
        else:
            features.append(0.0)
            names.append("port_entropy")

        # --- Numeric feature statistics (mean + std) ---
        for col in numeric_cols:
            if col in window.columns:
                vals = window[col].values.astype(np.float64)
                features.append(np.nanmean(vals))
                names.append(f"mean_{col}")
                features.append(np.nanstd(vals))
                names.append(f"std_{col}")

        # --- TCP flag distribution ---
        flag_cols_map = {
            "SYN": "SYN Flag Cnt",
            "ACK": "ACK Flag Cnt",
            "FIN": "FIN Flag Cnt",
            "RST": "RST Flag Cnt",
            "PSH": "PSH Flag Cnt",
            "URG": "URG Flag Cnt",
        }
        total_flag_packets = max(len(window), 1)
        for flag_name, col_name in flag_cols_map.items():
            if col_name in window.columns:
                features.append(
                    window[col_name].sum() / total_flag_packets
                )
            else:
                features.append(0.0)
            names.append(f"flag_frac_{flag_name}")

        # --- Protocol distribution ---
        for col in ["Protocol", "protocol"]:
            if col in window.columns:
                proto_counts = window[col].value_counts(normalize=True)
                for proto_id in [6, 17, 1]:  # TCP, UDP, ICMP
                    features.append(proto_counts.get(proto_id, 0.0))
                    names.append(f"proto_frac_{proto_id}")
                break
        else:
            for proto_id in [6, 17, 1]:
                features.append(0.0)
                names.append(f"proto_frac_{proto_id}")

        # --- Top port counts ---
        from src.utils.constants import TOP_ATTACKED_PORTS

        for col in ["Dst Port", "dst_port"]:
            if col in window.columns:
                port_vals = window[col].values
                for port in TOP_ATTACKED_PORTS[:10]:
                    features.append(int(np.sum(port_vals == port)))
                    names.append(f"top_port_{port}_count")
                break
        else:
            for port in TOP_ATTACKED_PORTS[:10]:
                features.append(0)
                names.append(f"top_port_{port}_count")

        self._feature_names = names
        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # Graph state builder
    # ------------------------------------------------------------------
    def _build_graph_states(self, df: pd.DataFrame, label_col: str) -> dict:
        """
        Build per-window graphs.

        Returns dict with 'graphs' key containing a list of dicts
        (node_features, edge_index, edge_features, label).
        This can be converted to PyG Data objects downstream.
        """
        grouped = df.groupby("window_id")
        window_ids = sorted(df["window_id"].unique())
        graphs = []
        labels_binary = []
        labels_mitre = []

        for wid in window_ids:
            window = grouped.get_group(wid)
            graph = self._build_single_graph(window)
            graphs.append(graph)

            if label_col in window.columns:
                has_attack = int((window[label_col] > 0).any())
                if has_attack:
                    attack_flows = window[window[label_col] > 0]
                    dominant_stage = int(attack_flows[label_col].value_counts().idxmax()) if len(attack_flows) > 0 else 0
                else:
                    dominant_stage = 0
            else:
                has_attack = 0
                dominant_stage = 0

            labels_binary.append(has_attack)
            labels_mitre.append(dominant_stage)

        logger.info(
            "Built %d graph states (attack windows: %d / %d)",
            len(graphs), sum(labels_binary), len(labels_binary),
        )

        return {
            "graphs": graphs,
            "labels_binary": np.array(labels_binary, dtype=np.int64),
            "labels_mitre": np.array(labels_mitre, dtype=np.int64),
            "window_ids": np.array(window_ids, dtype=np.int64),
        }

    def _build_single_graph(self, window: pd.DataFrame) -> dict:
        """Build a single graph from one time window."""
        # Resolve IP columns
        src_col = dst_col = None
        for c in ["src_ip_hash", "Src IP", "src_ip"]:
            if c in window.columns:
                src_col = c
                break
        for c in ["dst_ip_hash", "Dst IP", "dst_ip"]:
            if c in window.columns:
                dst_col = c
                break

        if src_col is None or dst_col is None:
            return {
                "node_features": np.zeros((1, 8), dtype=np.float32),
                "edge_index": np.zeros((2, 0), dtype=np.int64),
                "edge_features": np.zeros((0, 8), dtype=np.float32),
            }

        # Map IPs to node indices
        all_ips = pd.concat([window[src_col], window[dst_col]]).unique()
        ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
        n_nodes = len(all_ips)

        # Node features: per-IP aggregates
        node_features = np.zeros((n_nodes, 8), dtype=np.float32)
        for ip, idx in ip_to_idx.items():
            sent = window[window[src_col] == ip]
            recv = window[window[dst_col] == ip]
            node_features[idx, 0] = len(sent)  # flows initiated
            node_features[idx, 1] = len(recv)  # flows received
            for feat_col, feat_idx in [
                ("TotLen Fwd Pkts", 2), ("TotLen Bwd Pkts", 3),
            ]:
                if feat_col in sent.columns:
                    node_features[idx, feat_idx] = sent[feat_col].mean() if len(sent) > 0 else 0
            # Unique ports contacted
            for port_col in ["Dst Port", "dst_port"]:
                if port_col in sent.columns:
                    node_features[idx, 4] = sent[port_col].nunique()
                    break
            # Flag profile (mean of SYN, ACK, RST counts)
            for fi, fc in enumerate(["SYN Flag Cnt", "ACK Flag Cnt", "RST Flag Cnt"]):
                if fc in sent.columns:
                    node_features[idx, 5 + fi] = sent[fc].mean() if len(sent) > 0 else 0

        # Edge index and features
        edges = window.groupby([src_col, dst_col])
        edge_src = []
        edge_dst = []
        edge_feats = []

        for (s, d), grp in edges:
            edge_src.append(ip_to_idx[s])
            edge_dst.append(ip_to_idx[d])
            ef = [
                len(grp),  # num flows
                grp.get("TotLen Fwd Pkts", pd.Series([0])).sum(),
                grp.get("Flow Duration", pd.Series([0])).mean(),
                grp.get("Protocol", pd.Series([0])).mode().iloc[0] if "Protocol" in grp.columns else 0,
                grp.get("SYN Flag Cnt", pd.Series([0])).sum(),
                grp.get("ACK Flag Cnt", pd.Series([0])).sum(),
                grp.get("Flow IAT Mean", pd.Series([0])).mean(),
                self._shannon_entropy(
                    grp["Dst Port"].values if "Dst Port" in grp.columns
                    else np.array([0])
                ),
            ]
            edge_feats.append(ef)

        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_features = np.array(edge_feats, dtype=np.float32) if edge_feats else np.zeros((0, 8), dtype=np.float32)

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "edge_features": edge_features,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _assign_window_ids(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        ts_col = None
        for candidate in ["Timestamp", "timestamp", "flow_start"]:
            if candidate in df.columns:
                ts_col = candidate
                break

        if ts_col is None:
            df["window_id"] = np.arange(len(df)) // 100
            return df

        ts = pd.to_datetime(df[ts_col], errors="coerce")
        epoch = (ts - ts.min()).dt.total_seconds()
        df["window_id"] = (epoch // self.window_size_seconds).astype(int)
        return df

    @staticmethod
    def _shannon_entropy(values: np.ndarray) -> float:
        """Compute Shannon entropy of a discrete distribution."""
        if len(values) == 0:
            return 0.0
        _, counts = np.unique(values, return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build state representations")
    parser.add_argument("--input", required=True, help="Merged CSV/Parquet")
    parser.add_argument("--mode", default="vector", choices=["vector", "graph"])
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--output", default="data/processed/states.npz")
    args = parser.parse_args()

    import os

    df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_parquet(args.input)

    builder = StateBuilder(window_size_seconds=args.window, mode=args.mode)
    result = builder.build_states(df)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.mode == "vector":
        np.savez(
            args.output,
            states=result["states"],
            labels_binary=result["labels_binary"],
            labels_mitre=result["labels_mitre"],
            window_ids=result["window_ids"],
        )
    print(f"Saved {len(result.get('states', result.get('graphs', [])))} states to {args.output}")
