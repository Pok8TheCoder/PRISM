"""
State Builder (V2 Architecture): Aggregates aligned flow streams into sequential 15-second
286-dimensional state vectors (S_t) for high-resolution temporal sequence modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from src.utils.constants import (
    UNIFIED_FLOW_FEATURES,
    MEDIAN_FEATURE_INDICES,
    STATE_DIM,
    MITRE_STAGES,
    MITRE_STAGES_INV,
    NUM_MITRE_STAGES
)
from src.utils.logger import setup_logger

logger = setup_logger("StateBuilder")


def calculate_entropy(series: pd.Series) -> float:
    """Computes Shannon entropy of a categorical or discrete feature series."""
    if len(series) == 0:
        return 0.0
    val_counts = series.value_counts(normalize=True).to_numpy()
    entropy = -np.sum(val_counts * np.log2(val_counts + 1e-12))
    return float(entropy)


class StateBuilder:
    """
    Constructs compact fixed 15-second temporal state vectors (286 dims) from aligned flows.
    """

    def __init__(self, window_size_seconds: int = 15, features: Optional[List[str]] = None):
        self.window_size = window_size_seconds
        self.features = features or UNIFIED_FLOW_FEATURES
        self.median_indices = MEDIAN_FEATURE_INDICES

    def build_states_from_dataframe(
        self,
        df: pd.DataFrame,
        freq: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        Groups flows by 15-second intervals into 286-dimensional sequence state vectors.

        Returns:
            states: (T, 286) array of state vectors S_t
            attack_labels: (T,) binary attack labels (1=Attack, 0=Benign)
            mitre_labels: (T,) multiclass MITRE stage labels (0-6)
            attack_fractions: (T,) fraction of attack flows in each window
            timestamps: list of window start timestamps
        """
        if df.empty:
            return (
                np.empty((0, STATE_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                []
            )

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        resample_freq = freq or f"{self.window_size}s"
        grouped = df.groupby(pd.Grouper(key="timestamp", freq=resample_freq))

        states_list = []
        attack_labels_list = []
        mitre_labels_list = []
        attack_fractions_list = []
        timestamps_list = []

        for window_time, group in grouped:
            if len(group) == 0:
                continue

            num_flows = len(group)
            num_flows_log = np.log1p(num_flows)

            # Ensure all 64 features are present, defaulting missing to 0.0
            feat_matrix = group.reindex(columns=self.features, fill_value=0.0).to_numpy(dtype=np.float32)
            
            # 1. 64 Means, 64 Stds, 64 Maxs, 64 Mins (Total = 256)
            means = np.mean(feat_matrix, axis=0)
            stds = np.std(feat_matrix, axis=0)
            maxs = np.max(feat_matrix, axis=0)
            mins = np.min(feat_matrix, axis=0)

            # 2. 20 Medians on key rate/packet metrics (Total = 20)
            medians = np.median(feat_matrix[:, self.median_indices], axis=0)

            # 3. 16 Macro & Graph Topological Descriptors
            proto_col = group["protocol_type"].to_numpy() if "protocol_type" in group.columns else np.ones(num_flows)
            tcp_frac = float(np.mean(proto_col == 1.0))
            udp_frac = float(np.mean(proto_col == 2.0))
            icmp_frac = float(np.mean(proto_col == 3.0))
            other_proto_frac = float(np.mean(~np.isin(proto_col, [1.0, 2.0, 3.0])))

            dst_port_entropy = calculate_entropy(group["dst_port_binned"]) if "dst_port_binned" in group.columns else 0.0
            
            # Flag entropy & SYN/ACK dynamics
            syn_sum = np.sum(group["syn_flag_cnt"].to_numpy()) if "syn_flag_cnt" in group.columns else 0.0
            ack_sum = np.sum(group["ack_flag_cnt"].to_numpy()) if "ack_flag_cnt" in group.columns else 0.0
            syn_ack_ratio = float((syn_sum + 1.0) / (ack_sum + 1.0))
            
            down_up_mean = float(np.mean(group["down_up_ratio"].to_numpy())) if "down_up_ratio" in group.columns else 1.0
            
            # Burst factor: max rate / mean rate
            pkts_s_mean = means[2]  # flow_pkts_s
            pkts_s_max = maxs[2]
            burst_factor = float((pkts_s_max + 1.0) / (pkts_s_mean + 1.0))

            # --- Graph Topological Feature Extraction (Spatial Intelligence without GNN) ---
            # 1. Max Out-Degree (Fan-Out: Host scanning many targets)
            if "src_ip" in group.columns and "dst_ip" in group.columns and group["src_ip"].notna().any():
                max_src_out_degree = float(group.groupby("src_ip")["dst_ip"].nunique().max())
            elif "src_ip" in group.columns and "dst_port_binned" in group.columns and group["src_ip"].notna().any():
                max_src_out_degree = float(group.groupby("src_ip")["dst_port_binned"].nunique().max())
            else:
                max_src_out_degree = float(np.log1p(maxs[8]))

            # 2. Max In-Degree (Fan-In: Many bots hitting 1 target)
            if "dst_ip" in group.columns and "src_ip" in group.columns and group["dst_ip"].notna().any():
                max_dst_in_degree = float(group.groupby("dst_ip")["src_ip"].nunique().max())
            elif "dst_port_binned" in group.columns:
                max_dst_in_degree = float(group.groupby("dst_port_binned")["flow_duration"].count().max())
            else:
                max_dst_in_degree = float(np.log1p(maxs[6]))

            # 3. Source IP Entropy (Catches IP Spoofing)
            if "src_ip" in group.columns and group["src_ip"].notna().any():
                src_ip_entropy = calculate_entropy(group["src_ip"])
            else:
                src_ip_entropy = calculate_entropy(group["srate"]) if "srate" in group.columns else 0.0

            # 4. Destination IP Entropy (Catches Subnet traversal)
            if "dst_ip" in group.columns and group["dst_ip"].notna().any():
                dst_ip_entropy = calculate_entropy(group["dst_ip"])
            else:
                dst_ip_entropy = dst_port_entropy

            # 5. Graph Density Ratio (Active Connections vs Log-scale Endpoints)
            graph_density_ratio = float(num_flows / (np.log1p(num_flows) + 1.0))

            # 6. One-Way Edge Ratio (Unidirectional Half-Open / Spoofed Scans)
            bwd_pkts = group["tot_bwd_pkts"].to_numpy() if "tot_bwd_pkts" in group.columns else np.zeros(num_flows)
            one_way_edge_ratio = float(np.mean(bwd_pkts == 0.0))

            macro_and_graph_descriptors = np.array([
                num_flows_log,
                tcp_frac,
                udp_frac,
                icmp_frac,
                other_proto_frac,
                dst_port_entropy,
                syn_ack_ratio,
                down_up_mean,
                burst_factor,
                float(np.log1p(syn_sum)),
                np.log1p(max_src_out_degree),
                np.log1p(max_dst_in_degree),
                src_ip_entropy,
                dst_ip_entropy,
                graph_density_ratio,
                one_way_edge_ratio
            ], dtype=np.float32)

            # Concatenate into full 292-dim state vector
            state_vec = np.concatenate([
                means,                          # 64
                stds,                           # 64
                maxs,                           # 64
                mins,                           # 64
                medians,                        # 20
                macro_and_graph_descriptors     # 16
            ], dtype=np.float32)

            state_vec = np.nan_to_num(state_vec, nan=0.0, posinf=1e5, neginf=-1e5)

            # Window-level labels
            is_attack_col = group["is_attack"].to_numpy() if "is_attack" in group.columns else np.zeros(num_flows)
            attack_count = np.sum(is_attack_col > 0)
            attack_fraction = float(attack_count / num_flows)
            is_attack_window = 1 if attack_count > 0 else 0

            # Determine dominant MITRE stage
            if attack_count > 0 and "mitre_code" in group.columns:
                attack_mitre = group.loc[group["is_attack"] > 0, "mitre_code"].to_numpy()
                dominant_mitre = int(pd.Series(attack_mitre).mode()[0])
            else:
                dominant_mitre = 0

            states_list.append(state_vec)
            attack_labels_list.append(is_attack_window)
            mitre_labels_list.append(dominant_mitre)
            attack_fractions_list.append(attack_fraction)
            timestamps_list.append(window_time)

        if not states_list:
            return (
                np.empty((0, STATE_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
                []
            )

        states_arr = np.stack(states_list, axis=0)
        attack_labels_arr = np.array(attack_labels_list, dtype=np.int64)
        mitre_labels_arr = np.array(mitre_labels_list, dtype=np.int64)
        attack_fractions_arr = np.array(attack_fractions_list, dtype=np.float32)

        return states_arr, attack_labels_arr, mitre_labels_arr, attack_fractions_arr, timestamps_list
