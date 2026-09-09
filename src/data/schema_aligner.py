"""
Schema Aligner: Standardizes raw network flow CSVs from diverse datasets
(CICIOT2023, CIC-IDS2018, CIC-IDS2017, UNSW-NB15) into the unified 64-feature representation.
"""

import re
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from src.utils.constants import (
    UNIFIED_FLOW_FEATURES,
    LABEL_TO_MITRE,
    MITRE_STAGES,
    COMMON_ATTACK_PORTS
)
from src.utils.logger import setup_logger

logger = setup_logger("SchemaAligner")

# Comprehensive column mapping dictionary for CICIOT23, CIC-IDS2018, CIC-IDS2017, and UNSW-NB15
MULTI_DATASET_COLUMN_MAP = {
    # CIC-IDS2017 / 2018
    "src ip": "src_ip",
    "src_ip": "src_ip",
    "source ip": "src_ip",
    "srcip": "src_ip",
    "dst ip": "dst_ip",
    "dst_ip": "dst_ip",
    "destination ip": "dst_ip",
    "dstip": "dst_ip",
    "dst port": "dst_port",
    "destination port": "dst_port",
    "protocol": "protocol_type",
    "flow duration": "flow_duration",
    "tot fwd pkts": "tot_fwd_pkts",

    "total fwd packets": "tot_fwd_pkts",
    "tot bwd pkts": "tot_bwd_pkts",
    "total backward packets": "tot_bwd_pkts",
    "totlen fwd pkts": "tot_fwd_bytes",
    "total length of fwd packets": "tot_fwd_bytes",
    "totlen bwd pkts": "tot_bwd_bytes",
    "total length of bwd packets": "tot_bwd_bytes",
    "fwd pkt len max": "fwd_pkt_len_max",
    "fwd packet length max": "fwd_pkt_len_max",
    "fwd pkt len min": "fwd_pkt_len_min",
    "fwd packet length min": "fwd_pkt_len_min",
    "fwd pkt len mean": "fwd_pkt_len_mean",
    "fwd packet length mean": "fwd_pkt_len_mean",
    "fwd pkt len std": "fwd_pkt_len_std",
    "fwd packet length std": "fwd_pkt_len_std",
    "bwd pkt len max": "bwd_pkt_len_max",
    "bwd packet length max": "bwd_pkt_len_max",
    "bwd pkt len min": "bwd_pkt_len_min",
    "bwd packet length min": "bwd_pkt_len_min",
    "bwd pkt len mean": "bwd_pkt_len_mean",
    "bwd packet length mean": "bwd_pkt_len_mean",
    "bwd pkt len std": "bwd_pkt_len_std",
    "bwd packet length std": "bwd_pkt_len_std",
    "flow byts/s": "flow_byts_s",
    "flow bytes/s": "flow_byts_s",
    "flow pkts/s": "flow_pkts_s",
    "flow packets/s": "flow_pkts_s",
    "fwd pkts/s": "fwd_pkts_s",
    "bwd pkts/s": "bwd_pkts_s",
    "flow iat mean": "flow_iat_mean",
    "flow iat std": "flow_iat_std",
    "flow iat max": "flow_iat_max",
    "flow iat min": "flow_iat_min",
    "fwd iat tot": "fwd_iat_tot",
    "fwd iat total": "fwd_iat_tot",
    "fwd iat mean": "fwd_iat_mean",
    "fwd iat std": "fwd_iat_std",
    "fwd iat min": "fwd_iat_min",
    "fwd iat max": "fwd_iat_max",
    "bwd iat tot": "bwd_iat_tot",
    "bwd iat total": "bwd_iat_tot",
    "bwd iat mean": "bwd_iat_mean",
    "bwd iat std": "bwd_iat_std",
    "bwd iat min": "bwd_iat_min",
    "bwd iat max": "bwd_iat_max",
    "pkt len min": "pkt_len_min",
    "min packet length": "pkt_len_min",
    "pkt len max": "pkt_len_max",
    "max packet length": "pkt_len_max",
    "pkt len mean": "pkt_len_mean",
    "packet length mean": "pkt_len_mean",
    "pkt len std": "pkt_len_std",
    "packet length std": "pkt_len_std",
    "pkt len var": "pkt_len_var",
    "packet length variance": "pkt_len_var",
    "pkt size avg": "pkt_size_avg",
    "average packet size": "pkt_size_avg",
    "fin flag cnt": "fin_flag_cnt",
    "fin flag count": "fin_flag_cnt",
    "syn flag cnt": "syn_flag_cnt",
    "syn flag count": "syn_flag_cnt",
    "rst flag cnt": "rst_flag_cnt",
    "rst flag count": "rst_flag_cnt",
    "psh flag cnt": "psh_flag_cnt",
    "psh flag count": "psh_flag_cnt",
    "ack flag cnt": "ack_flag_cnt",
    "ack flag count": "ack_flag_cnt",
    "urg flag cnt": "urg_flag_cnt",
    "urg flag count": "urg_flag_cnt",
    "ece flag cnt": "ece_flag_cnt",
    "ece flag count": "ece_flag_cnt",
    "cwe flag count": "cwr_flag_cnt",
    "cwr flag count": "cwr_flag_cnt",
    "fwd header len": "fwd_header_len",
    "fwd header length": "fwd_header_len",
    "bwd header len": "bwd_header_len",
    "bwd header length": "bwd_header_len",
    "init fwd win byts": "init_win_bytes_fwd",
    "init_win_bytes_forward": "init_win_bytes_fwd",
    "init bwd win byts": "init_win_bytes_bwd",
    "init_win_bytes_backward": "init_win_bytes_bwd",
    "down/up ratio": "down_up_ratio",
    "active mean": "active_mean",
    "active std": "active_std",
    "active max": "active_max",
    "active min": "active_min",
    "idle mean": "idle_mean",
    "idle std": "idle_std",
    "idle max": "idle_max",
    "idle min": "idle_min",
    "timestamp": "timestamp",
    "label": "label",
    
    # CICIOT2023 Specific Columns
    "header_length": "fwd_header_len",
    "protocol type": "protocol_type",
    "duration": "flow_duration",
    "rate": "rate",
    "srate": "srate",
    "drate": "drate",
    "fin_flag_number": "fin_flag_cnt",
    "syn_flag_number": "syn_flag_cnt",
    "rst_flag_number": "rst_flag_cnt",
    "psh_flag_number": "psh_flag_cnt",
    "ack_flag_number": "ack_flag_cnt",
    "ece_flag_number": "ece_flag_cnt",
    "cwr_flag_number": "cwr_flag_cnt",
    "ack_count": "ack_flag_cnt",
    "syn_count": "syn_flag_cnt",
    "fin_count": "fin_flag_cnt",
    "urg_count": "urg_flag_cnt",
    "rst_count": "rst_flag_cnt",
    "min": "pkt_len_min",
    "max": "pkt_len_max",
    "avg": "pkt_len_mean",
    "std": "pkt_len_std",
    "tot sum": "tot_fwd_bytes",
    "tot size": "tot_bwd_bytes",
    "iat": "flow_iat_mean",
    "number": "tot_fwd_pkts",
    "magnitue": "pkt_size_avg",
    "radius": "fwd_pkt_len_std",
    "covariance": "pkt_len_var",
    "variance": "pkt_len_var",
    "weight": "flow_pkts_s",
    
    # UNSW-NB15 Specific Columns
    "dur": "flow_duration",
    "spkts": "tot_fwd_pkts",
    "dpkts": "tot_bwd_pkts",
    "sbytes": "tot_fwd_bytes",
    "dbytes": "tot_bwd_bytes",
    "sintpkt": "fwd_iat_mean",
    "dintpkt": "bwd_iat_mean",
    "smeansz": "fwd_pkt_len_mean",
    "dmeansz": "bwd_pkt_len_mean",
    "swin": "init_win_bytes_fwd",
    "dwin": "init_win_bytes_bwd",
    "dsport": "dst_port",
    "proto": "protocol_type",
    "stime": "timestamp",
    "attack_cat": "label",
    "sload": "flow_byts_s",

    # CTU-13 Specific Columns (NetFlow format)
    "starttime": "timestamp",
    "srcaddr": "src_ip",
    "dstaddr": "dst_ip",
    "sport": "src_port",
    "dport": "dst_port",
    "totpkts": "tot_fwd_pkts",
    "totbytes": "tot_fwd_bytes",
    "srcbytes": "fwd_act_data_pkts"
}

CIC_COLUMN_MAP = MULTI_DATASET_COLUMN_MAP


def clean_label_string(val: str) -> str:
    """Normalizes label text by stripping whitespace, non-ascii, and lowercasing."""
    if not isinstance(val, str):
        return "benign"
    cleaned = re.sub(r"[^a-zA-Z0-9\-\s_]", " ", val).strip().lower()
    return cleaned


def map_label_to_mitre(label_str: str) -> Tuple[int, str, int]:
    """
    Maps a raw dataset label string to (mitre_code, mitre_name, is_attack).
    """
    clean_lbl = clean_label_string(label_str)
    
    # Priority check for CTU-13 Botnet and Background patterns
    if "botnet" in clean_lbl or "from-botnet" in clean_lbl or "to-botnet" in clean_lbl:
        return 4, "Command & Control", 1
    if "background" in clean_lbl or "from-normal" in clean_lbl or "to-normal" in clean_lbl:
        return 0, "Benign", 0
        
    mitre_name = LABEL_TO_MITRE.get(clean_lbl, None)
    if mitre_name is None:
        for k, v in LABEL_TO_MITRE.items():
            if k in clean_lbl or clean_lbl in k:
                mitre_name = v
                break
                
    if mitre_name is None:
        mitre_name = "Benign" if ("benign" in clean_lbl or "normal" in clean_lbl) else "Initial Access"
        
    mitre_code = MITRE_STAGES.get(mitre_name, 0)
    is_attack = 0 if mitre_code == 0 else 1
    return mitre_code, mitre_name, is_attack


class SchemaAligner:
    """Standardizes heterogeneous flow datasets into a unified 64-feature schema."""

    def __init__(self, target_features=None):
        self.target_features = target_features or UNIFIED_FLOW_FEATURES

    def align_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts any supported input dataframe chunk into the unified 64-feature representation.
        """
        df_cols_clean = {c: c.strip().lower() for c in df.columns}
        df_renamed = df.rename(columns=df_cols_clean)
        
        # Apply mapping dictionary
        rename_map = {}
        for col in df_renamed.columns:
            if col in CIC_COLUMN_MAP:
                rename_map[col] = CIC_COLUMN_MAP[col]
            else:
                for k, v in CIC_COLUMN_MAP.items():
                    if k in col or col in k:
                        rename_map[col] = v
                        break
        df_mapped = df_renamed.rename(columns=rename_map)
        
        result_df = pd.DataFrame(index=df_mapped.index)
        
        # 1. Fill unified numerical flow features
        for feat in self.target_features:
            if feat in df_mapped.columns:
                col_data = df_mapped[feat]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                series = pd.to_numeric(col_data, errors="coerce").fillna(0.0)
                series = series.replace([np.inf, -np.inf], 0.0)
                result_df[feat] = series
            else:
                # Custom feature derivations
                if feat == "dst_port_binned":
                    if "dst_port" in df_mapped.columns:
                        col_data = df_mapped["dst_port"]
                        if isinstance(col_data, pd.DataFrame):
                            col_data = col_data.iloc[:, 0]
                        ports = pd.to_numeric(col_data, errors="coerce").fillna(0).astype(int)
                        result_df["dst_port_binned"] = ports.map(COMMON_ATTACK_PORTS).fillna(0).astype(float)
                    else:
                        result_df["dst_port_binned"] = 0.0
                elif feat == "protocol_type":
                    if "protocol_type" in df_mapped.columns:
                        col_data = df_mapped["protocol_type"]
                        if isinstance(col_data, pd.DataFrame):
                            col_data = col_data.iloc[:, 0]
                        protos = pd.to_numeric(col_data, errors="coerce").fillna(6)
                        proto_map = {6: 1.0, 17: 2.0, 1: 3.0}
                        result_df["protocol_type"] = protos.map(proto_map).fillna(0.0)
                    else:
                        result_df["protocol_type"] = 1.0  # Default TCP
                elif feat == "app_proto_flag":
                    # Check CICIOT23 app flags if present
                    app_code = np.zeros(len(df_mapped), dtype=np.float32)
                    if "http" in df_renamed.columns and df_renamed["http"].sum() > 0:
                        app_code = np.where(df_renamed["http"] > 0, 1.0, app_code)
                    if "https" in df_renamed.columns and df_renamed["https"].sum() > 0:
                        app_code = np.where(df_renamed["https"] > 0, 2.0, app_code)
                    if "dns" in df_renamed.columns and df_renamed["dns"].sum() > 0:
                        app_code = np.where(df_renamed["dns"] > 0, 3.0, app_code)
                    if "ssh" in df_renamed.columns and df_renamed["ssh"].sum() > 0:
                        app_code = np.where(df_renamed["ssh"] > 0, 4.0, app_code)
                    if "telnet" in df_renamed.columns and df_renamed["telnet"].sum() > 0:
                        app_code = np.where(df_renamed["telnet"] > 0, 5.0, app_code)
                    result_df["app_proto_flag"] = app_code
                else:
                    result_df[feat] = 0.0
                    
        # 2. Timestamp extraction / normalization
        if "timestamp" in df_mapped.columns:
            ts_data = df_mapped["timestamp"]
            if isinstance(ts_data, pd.DataFrame):
                ts_data = ts_data.iloc[:, 0]
            parsed_ts = pd.to_datetime(
                ts_data.astype(str),
                errors="coerce",
                format="mixed"
            )
            # If dataset has minute-level truncation (seconds are all 00), distribute flows across the 60s
            if parsed_ts.notna().any() and len(parsed_ts) > 50:
                valid_mask = parsed_ts.notna()
                if (parsed_ts[valid_mask].dt.second == 0).all():
                    minute_order = parsed_ts.groupby(parsed_ts).cumcount()
                    minute_totals = parsed_ts.map(parsed_ts.value_counts())
                    sub_sec_offsets = (minute_order / np.maximum(minute_totals, 1)) * 59.0
                    parsed_ts = parsed_ts + pd.to_timedelta(sub_sec_offsets, unit="s")
            result_df["timestamp"] = parsed_ts
        else:
            # Synthetic 15s-compatible timestamps
            result_df["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="100ms")
            
        # 3. Label & MITRE ATT&CK mapping
        lbl_col = None
        if "attack_cat" in df_renamed.columns:
            lbl_col = df_renamed["attack_cat"]
        elif "label" in df_mapped.columns:
            lbl_col = df_mapped["label"]
        elif "activity" in df_renamed.columns:
            lbl_col = df_renamed["activity"]
            
        if lbl_col is not None:
            if isinstance(lbl_col, pd.DataFrame):
                lbl_col = lbl_col.iloc[:, 0]
            labels = lbl_col.astype(str)
            mapped_tuples = [map_label_to_mitre(lbl) for lbl in labels]
            result_df["mitre_code"] = [t[0] for t in mapped_tuples]
            result_df["mitre_stage"] = [t[1] for t in mapped_tuples]
            result_df["is_attack"] = [t[2] for t in mapped_tuples]
        else:
            result_df["mitre_code"] = 0
            result_df["mitre_stage"] = "Benign"
            result_df["is_attack"] = 0

        # Sort chronologically if timestamps are valid
        if not result_df["timestamp"].isna().all():
            result_df = result_df.sort_values(by="timestamp").reset_index(drop=True)
            
        return result_df
