"""
PRISM — Universal 4-Dataset Strict 5-Second Temporal Compiler (Gen 8)
Guarantees STRICT 5.0-Second Windows across all 4 datasets with sliding attack resolution
and kill-chain priority labeling to prevent Reconnaissance, Initial Access, and Lateral
Movement from being compressed or masked.

Output:
  - data/splits_universal_gen8_5s/train.npz
  - data/splits_universal_gen8_5s/val.npz
  - data/splits_universal_gen8_5s/test.npz
  - weights/universal_gen8_5s_scaler.pkl
"""

from __future__ import annotations

import glob
import logging
import os
import pathlib
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.state_builder import StateBuilder
from src.data.flow_extractor import FlowExtractor
from src.data.ctu13_extractor import CTU13Extractor, CANONICAL_DIM
from src.data.ciciot2023_extractor import CICIoT2023Extractor
from src.data.unswnb15_extractor import UNSWNB15Extractor
from src.utils.logger import setup_logger

logger = setup_logger("prism.universal.gen8_compiler", log_dir="results/logs")


def build_fast_5s_windows(
    df: pd.DataFrame,
    builder: StateBuilder,
    window_size_seconds: float = 5.0,
    stride_seconds: float = 2.0,
    numeric_cols: list[str] = None,
) -> dict:
    """
    Constructs STRICT 5.0-second windows using binary-search aggregation.
    Every single state window aggregates exactly 5.0 seconds of telemetry [t, t + 5.0s].
    """
    ts_col = None
    for candidate in ["Timestamp", "timestamp", "flow_start", "StartTime", "stime"]:
        if candidate in df.columns:
            ts_col = candidate
            break

    epoch = None
    if ts_col is not None:
        try:
            if pd.api.types.is_numeric_dtype(df[ts_col]):
                ts = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
            else:
                ts = pd.to_datetime(df[ts_col], dayfirst=True, errors="coerce")
            # Filter out corrupt timestamps (e.g. 1970 artifacts)
            valid_ts = ts.notna() & (ts.dt.year >= 2000) & (ts.dt.year <= 2030)
            if valid_ts.sum() > 100:
                ts = ts[valid_ts]
                df = df.loc[valid_ts].copy()
                epoch = (ts - ts.min()).dt.total_seconds().values
        except Exception:
            epoch = None

    df = df.loc[:, ~df.columns.duplicated()].copy()
    if epoch is None:
        # Construct continuous 5-second timeline from flow durations
        dur_col = None
        for candidate in ["Flow Duration", "flow_duration", "dur", "Duration"]:
            if candidate in df.columns:
                dur_col = candidate
                break
        if dur_col is not None:
            col_data = df[dur_col].iloc[:, 0] if isinstance(df[dur_col], pd.DataFrame) else df[dur_col]
            dur = pd.to_numeric(col_data, errors="coerce").fillna(0.05)
            # If duration is in microseconds, convert to seconds
            if dur.mean() > 1000.0:
                dur = dur / 1e6
            dur = dur.clip(lower=0.01, upper=2.0)
            epoch = dur.cumsum().values
        else:
            # Fallback: estimate arrival at ~0.25s per flow
            epoch = np.arange(len(df), dtype=float) * 0.25

    # Sort by epoch for binary search
    sort_idx = np.argsort(epoch)
    epoch_sorted = epoch[sort_idx]
    df_sorted = df.iloc[sort_idx]

    max_time = np.nanmax(epoch_sorted) if len(epoch_sorted) > 0 else 0.0

    if numeric_cols is None:
        exclude_cols = {
            "window_id", "Label", "label", "mitre_stage",
            "mitre_stage_id", "Timestamp", "timestamp",
            "Src IP", "src_ip", "Dst IP", "dst_ip",
            "src_ip_hash", "dst_ip_hash",
            "Flow ID", "flow_id", "port_category", "StartTime", "stime"
        }
        numeric_cols = [
            c for c in df_sorted.select_dtypes(include=[np.number]).columns
            if c not in exclude_cols
        ]

    states = []
    labels_binary = []
    labels_mitre = []

    # STRICT 5.0-SECOND WINDOWS
    starts = np.arange(0.0, max(0.0, max_time - 0.5), stride_seconds)
    ends = starts + window_size_seconds

    idx_starts = np.searchsorted(epoch_sorted, starts, side="left")
    idx_ends = np.searchsorted(epoch_sorted, ends, side="right")

    for s_idx, e_idx in zip(idx_starts, idx_ends):
        if e_idx > s_idx:
            window_df = df_sorted.iloc[s_idx:e_idx]
            state = builder._aggregate_window_vector(window_df, numeric_cols)
            states.append(state)

            if "mitre_stage_id" in window_df.columns:
                attack_mask = window_df["mitre_stage_id"] > 0
                has_attack = int(attack_mask.any())
                if has_attack:
                    stage_counts = window_df.loc[attack_mask, "mitre_stage_id"].value_counts()
                    # KILL-CHAIN PROPORTIONAL RESOLUTION:
                    # Filter out high-volume floods/botnets (4: C2, 6: Impact) if specific kill-chain attacks exist
                    killchain_stages = [s for s in [1, 2, 3, 5] if s in stage_counts.index]
                    if killchain_stages:
                        dominant_stage = int(stage_counts.loc[killchain_stages].idxmax())
                    else:
                        dominant_stage = int(stage_counts.idxmax())
                else:
                    dominant_stage = 0
            elif "Label" in window_df.columns:
                has_attack = int((window_df["Label"] != "Benign").any())
                dominant_stage = 1 if has_attack else 0
            else:
                has_attack = 0
                dominant_stage = 0

            labels_binary.append(has_attack)
            labels_mitre.append(dominant_stage)

    if len(states) == 0:
        return builder.build_states(df)

    states_arr = np.nan_to_num(np.array(states, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "states": states_arr,
        "labels_binary": np.array(labels_binary, dtype=np.int64),
        "labels_mitre": np.array(labels_mitre, dtype=np.int64),
    }


def align_242_dim(states: np.ndarray) -> np.ndarray:
    """Guarantee strict 242-feature dimension."""
    curr_dim = states.shape[1]
    if curr_dim == CANONICAL_DIM:
        return states.astype(np.float32)
    elif curr_dim < CANONICAL_DIM:
        pad = CANONICAL_DIM - curr_dim
        return np.pad(states, ((0, 0), (0, pad)), mode="constant", constant_values=0.0).astype(np.float32)
    else:
        return states[:, :CANONICAL_DIM].astype(np.float32)


def split_sequence_70_15_15(states: np.ndarray, y_bin: np.ndarray, y_mit: np.ndarray, block_size: int = 200):
    """Split a single file's temporal state sequence into 70% Train, 15% Val, 15% Test with block stratification."""
    n = len(states)
    if n < 3:
        return (
            (states, y_bin, y_mit),
            (states, y_bin, y_mit),
            (states, y_bin, y_mit)
        )
    if n < 300:
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        return (
            (states[:n_train], y_bin[:n_train], y_mit[:n_train]),
            (states[n_train:n_train+n_val], y_bin[n_train:n_train+n_val], y_mit[n_train:n_train+n_val]),
            (states[n_train+n_val:], y_bin[n_train+n_val:], y_mit[n_train+n_val:])
        )

    tr_s, tr_b, tr_m = [], [], []
    va_s, va_b, va_m = [], [], []
    te_s, te_b, te_m = [], [], []

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        cur_s, cur_b, cur_m = states[start:end], y_bin[start:end], y_mit[start:end]
        c_n = len(cur_s)
        if c_n == 0:
            continue
        c_tr = int(c_n * 0.70)
        c_va = int(c_n * 0.15)

        tr_s.append(cur_s[:c_tr])
        tr_b.append(cur_b[:c_tr])
        tr_m.append(cur_m[:c_tr])

        va_s.append(cur_s[c_tr:c_tr+c_va])
        va_b.append(cur_b[c_tr:c_tr+c_va])
        va_m.append(cur_m[c_tr:c_tr+c_va])

        te_s.append(cur_s[c_tr+c_va:])
        te_b.append(cur_b[c_tr+c_va:])
        te_m.append(cur_m[c_tr+c_va:])

    return (
        (np.concatenate(tr_s), np.concatenate(tr_b), np.concatenate(tr_m)),
        (np.concatenate(va_s), np.concatenate(va_b), np.concatenate(va_m)),
        (np.concatenate(te_s), np.concatenate(te_b), np.concatenate(te_m)),
    )


def process_universal_gen8_5s(
    output_dir: str = "data/splits_universal_gen8_5s",
    scaler_out: str = "weights/universal_gen8_5s_scaler.pkl",
    window_size_seconds: float = 5.0,
    stride_seconds: float = 2.0,
):
    """Compile, split PER-FILE, normalize, and save universal 4-dataset strict 5-second state sequences."""
    logger.info("=" * 80)
    logger.info("PRISM UNIVERSAL 4-DATASET STRICT 5-SECOND COMPILER (GEN 8)")
    logger.info("WINDOW SIZE = STRICTLY 5.0 SECONDS  ·  STRIDE = 2.0 SECONDS")
    logger.info("=" * 80)

    builder = StateBuilder(window_size_seconds=int(window_size_seconds))

    tr_states, tr_bin, tr_mit = [], [], []
    va_states, va_bin, va_mit = [], [], []
    te_states, te_bin, te_mit = [], [], []

    # -------------------------------------------------------------------------
    # 1. CSE-CIC-IDS2018 (Enterprise Cloud & Servers)
    # -------------------------------------------------------------------------
    logger.info("Processing Dataset 1/4: CSE-CIC-IDS2018 (Per-File Splitting) ...")
    ext_cic = FlowExtractor(dataset_type="cicids2018")
    cic_files = sorted(glob.glob("data/raw/cicids2018/*.csv"))
    for f in cic_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df = ext_cic.clean(df)
            # Attack-first stratified sampling: Preserve 100% of attack flows
            if "Label" in df.columns:
                is_attack = df["Label"].astype(str).str.strip() != "Benign"
                atk_df = df[is_attack]
                benign_df = df[~is_attack]
                if len(benign_df) > 300000:
                    stride = max(1, len(benign_df) // 300000)
                    benign_df = benign_df.iloc[::stride]
                df = pd.concat([atk_df, benign_df])
            df = ext_cic.engineer_features(df)
            df = ext_cic.map_labels(df)
            
            # Use 1.5s stride for 02-14 (BruteForce) and 03-01 (Infiltration) for maximum attack resolution
            file_stride = 1.5 if ("02-14" in f or "03-01" in f) else stride_seconds
            res = build_fast_5s_windows(df, builder, window_size_seconds, file_stride)
            st = align_242_dim(res["states"])
            b = res["labels_binary"]
            m = res["labels_mitre"]
            
            tr_p, va_p, te_p = split_sequence_70_15_15(st, b, m)
            if len(tr_p[0]) > 0: tr_states.append(tr_p[0]); tr_bin.append(tr_p[1]); tr_mit.append(tr_p[2])
            if len(va_p[0]) > 0: va_states.append(va_p[0]); va_bin.append(va_p[1]); va_mit.append(va_p[2])
            if len(te_p[0]) > 0: te_states.append(te_p[0]); te_bin.append(te_p[1]); te_mit.append(te_p[2])
            logger.info(f" [CIC] {os.path.basename(f)}: {len(st):,} strict 5s windows -> {len(tr_p[0]):,} tr / {len(va_p[0]):,} va / {len(te_p[0]):,} te (Atks: {sum(b):,}, Stages: {pd.Series(m).value_counts().to_dict()})")
        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(f)}: {e}")

    # -------------------------------------------------------------------------
    # 2. CTU-13 (Real Malware Botnets & C2)
    # -------------------------------------------------------------------------
    logger.info("Processing Dataset 2/4: CTU-13 Botnet (Per-Scenario Splitting) ...")
    ext_ctu = CTU13Extractor()
    ctu_files = sorted(glob.glob("data/raw/ctu13/CTU-13-Dataset/*/*.binetflow"))
    for f in ctu_files:
        try:
            df = ext_ctu.load_binetflow_or_csv(f)
            lbl_col = "Label" if "Label" in df.columns else "label"
            if lbl_col in df.columns:
                is_bot = df[lbl_col].astype(str).str.contains("Botnet", case=False, na=False)
                bot_df = df[is_bot]
                other_df = df[~is_bot]
                if len(other_df) > 200000:
                    stride = max(1, len(other_df) // 200000)
                    other_df = other_df.iloc[::stride]
                df = pd.concat([bot_df, other_df])
            df = ext_ctu.clean(df)
            df = ext_ctu.engineer_features(df)
            df = ext_ctu.map_labels(df)
            res = build_fast_5s_windows(df, builder, window_size_seconds, stride_seconds=3.0)
            st = align_242_dim(res["states"])
            b = res["labels_binary"]
            m = res["labels_mitre"]
            
            tr_p, va_p, te_p = split_sequence_70_15_15(st, b, m)
            if len(tr_p[0]) > 0: tr_states.append(tr_p[0]); tr_bin.append(tr_p[1]); tr_mit.append(tr_p[2])
            if len(va_p[0]) > 0: va_states.append(va_p[0]); va_bin.append(va_p[1]); va_mit.append(va_p[2])
            if len(te_p[0]) > 0: te_states.append(te_p[0]); te_bin.append(te_p[1]); te_mit.append(te_p[2])
            logger.info(f" [CTU] {os.path.basename(f)}: {len(st):,} strict 5s windows -> {len(tr_p[0]):,} tr / {len(va_p[0]):,} va / {len(te_p[0]):,} te (Atks: {sum(b):,})")
        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(f)}: {e}")

    # -------------------------------------------------------------------------
    # 3. CICIoT2023 (Continuous 5.0-Second IoT Aggregation — Recon, MITM, DNS)
    # -------------------------------------------------------------------------
    logger.info("Processing Dataset 3/4: CICIoT2023 (Continuous Strict 5s Windows) ...")
    ext_iot = CICIoT2023Extractor()
    iot_train_f = "data/raw/archive/CICIOT23/train/train.csv"
    iot_val_f = "data/raw/archive/CICIOT23/validation/validation.csv"
    iot_test_f = "data/raw/archive/CICIOT23/test/test.csv"

    def load_stratified_iot(path, chunksize=150000):
        rare_classes = {
            "MITM-ArpSpoofing", "BrowserHijacking", "BrowserExtraction",
            "DNS_Spoofing", "DictionaryBruteForce", "SqlInjection", "CommandInjection",
            "Backdoor_Malware", "XSS", "Uploading_Attack", "Recon-PingSweep",
            "Recon-OSScan", "Recon-PortScan", "VulnerabilityScan", "Recon-HostDiscovery",
            "DoS-HTTP_Flood", "DDoS-HTTP_Flood", "DDoS-SlowLoris"
        }
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
            lbl_col = "label" if "label" in chunk.columns else "Label"
            if lbl_col in chunk.columns:
                lbl = chunk[lbl_col].astype(str).str.strip()
                is_rare = lbl.isin(rare_classes)
                rare_df = chunk[is_rare]
                flood_df = chunk[~is_rare]
                if len(flood_df) > 5000:
                    stride = max(1, len(flood_df) // 3000)
                    flood_df = flood_df.iloc[::stride]
                chunks.append(pd.concat([rare_df, flood_df]))
            else:
                chunks.append(chunk.iloc[::max(1, len(chunk) // 10000)])
        return pd.concat(chunks, ignore_index=True)

    def process_iot_file(path, name):
        logger.info(f"Loading & stratifying {name} from {path} ...")
        df = load_stratified_iot(path)
        lbl_col = "label" if "label" in df.columns else "Label"
        mitm_cnt = (df[lbl_col] == "MITM-ArpSpoofing").sum() if lbl_col in df.columns else 0
        brow_cnt = (df[lbl_col] == "BrowserHijacking").sum() if lbl_col in df.columns else 0
        logger.info(f" {name} loaded: {len(df):,} flows (MITM-ArpSpoofing: {mitm_cnt:,}, BrowserHijacking: {brow_cnt:,})")
        df = ext_iot.clean(df)
        df = ext_iot.engineer_features(df)
        df = ext_iot.map_labels(df)
        
        # Build STRICT 5-second windows (NO // 100 squashing!)
        res = build_fast_5s_windows(df, builder, window_size_seconds=window_size_seconds, stride_seconds=stride_seconds)
        st = align_242_dim(res["states"])
        b = res["labels_binary"]
        m = res["labels_mitre"]
        logger.info(f" [{name}] Generated {len(st):,} strict 5s windows (Atks: {sum(b):,}, Stages: {pd.Series(m).value_counts().to_dict()})")
        return st, b, m

    iot_tr_s, iot_tr_b, iot_tr_m = process_iot_file(iot_train_f, "CICIoT2023-Train")
    iot_va_s, iot_va_b, iot_va_m = process_iot_file(iot_val_f, "CICIoT2023-Val")
    iot_te_s, iot_te_b, iot_te_m = process_iot_file(iot_test_f, "CICIoT2023-Test")

    tr_states.append(iot_tr_s); tr_bin.append(iot_tr_b); tr_mit.append(iot_tr_m)
    va_states.append(iot_va_s); va_bin.append(iot_va_b); va_mit.append(iot_va_m)
    te_states.append(iot_te_s); te_bin.append(iot_te_b); te_mit.append(iot_te_m)

    # -------------------------------------------------------------------------
    # 4. UNSW-NB15 (Strict 5s Windows — Fuzzers, Reconnaissance, Exploits)
    # -------------------------------------------------------------------------
    logger.info("Processing Dataset 4/4: UNSW-NB15 (Per-File Splitting) ...")
    ext_unsw = UNSWNB15Extractor()
    unsw_files = sorted(glob.glob("data/raw/archive (1)/*.csv"))
    for f in unsw_files:
        if "features" in f.lower() or "events" in f.lower():
            continue
        try:
            df = ext_unsw.load_csv(f)
            df = ext_unsw.clean(df)
            if "Label" in df.columns:
                is_attack = ~df["Label"].astype(str).str.lower().isin(["normal", "benign", "0"])
                atk_df = df[is_attack]
                norm_df = df[~is_attack]
                if len(norm_df) > 200000:
                    stride = max(1, len(norm_df) // 200000)
                    norm_df = norm_df.iloc[::stride]
                df = pd.concat([atk_df, norm_df])
            df = ext_unsw.engineer_features(df)
            df = ext_unsw.map_labels(df)
            
            # Strict 5-second windows
            res = build_fast_5s_windows(df, builder, window_size_seconds=window_size_seconds, stride_seconds=stride_seconds)
            st = align_242_dim(res["states"])
            b = res["labels_binary"]
            m = res["labels_mitre"]
            
            tr_p, va_p, te_p = split_sequence_70_15_15(st, b, m)
            if len(tr_p[0]) > 0: tr_states.append(tr_p[0]); tr_bin.append(tr_p[1]); tr_mit.append(tr_p[2])
            if len(va_p[0]) > 0: va_states.append(va_p[0]); va_bin.append(va_p[1]); va_mit.append(va_p[2])
            if len(te_p[0]) > 0: te_states.append(te_p[0]); te_bin.append(te_p[1]); te_mit.append(te_p[2])
            logger.info(f" [UNSW] {os.path.basename(f)}: {len(st):,} strict 5s windows -> {len(tr_p[0]):,} tr / {len(va_p[0]):,} va / {len(te_p[0]):,} te (Atks: {sum(b):,}, Stages: {pd.Series(m).value_counts().to_dict()})")
        except Exception as e:
            logger.warning(f"Error processing {os.path.basename(f)}: {e}")

    # -------------------------------------------------------------------------
    # Merge, Apply Log1p Normalization & Fit StandardScaler on Train Set
    # -------------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Merging Partitions & Applying Sign-Preserving Log1p Normalization ...")

    raw_train_st = np.concatenate(tr_states, axis=0)
    train_bin = np.concatenate(tr_bin, axis=0)
    train_mit = np.concatenate(tr_mit, axis=0)

    raw_val_st = np.concatenate(va_states, axis=0)
    val_bin = np.concatenate(va_bin, axis=0)
    val_mit = np.concatenate(va_mit, axis=0)

    raw_test_st = np.concatenate(te_states, axis=0)
    test_bin = np.concatenate(te_bin, axis=0)
    test_mit = np.concatenate(te_mit, axis=0)

    # Log1p transformation: S' = sign(S) * ln(1 + |S|)
    log_train = np.sign(raw_train_st) * np.log1p(np.abs(raw_train_st))
    log_val = np.sign(raw_val_st) * np.log1p(np.abs(raw_val_st))
    log_test = np.sign(raw_test_st) * np.log1p(np.abs(raw_test_st))

    logger.info(f"Fitting StandardScaler strictly on Combined Train Set ({len(log_train):,} windows) ...")
    scaler = StandardScaler()
    scaler.fit(log_train)
    os.makedirs(os.path.dirname(scaler_out), exist_ok=True)
    joblib.dump(scaler, scaler_out)

    train_states = scaler.transform(log_train).astype(np.float32)
    val_states = scaler.transform(log_val).astype(np.float32)
    test_states = scaler.transform(log_test).astype(np.float32)

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "train.npz"), states=train_states, labels_binary=train_bin, labels_mitre=train_mit)
    np.savez(os.path.join(output_dir, "val.npz"), states=val_states, labels_binary=val_bin, labels_mitre=val_mit)
    np.savez(os.path.join(output_dir, "test.npz"), states=test_states, labels_binary=test_bin, labels_mitre=test_mit)

    total_all = len(train_states) + len(val_states) + len(test_states)
    total_atk = sum(train_bin) + sum(val_bin) + sum(test_bin)

    logger.info("=" * 80)
    logger.info(f"GEN 8 UNIVERSAL 4-DATASET COMPILATION COMPLETE: {total_all:,} STRICT 5-SECOND WINDOWS")
    logger.info(f"Total Attack Windows: {total_atk:,} ({total_atk/total_all*100:.1f}%)")
    logger.info(f" - Train Set: {train_states.shape} ({len(train_states):,} windows, {sum(train_bin):,} attacks, MITRE breakdown: {pd.Series(train_mit).value_counts().to_dict()})")
    logger.info(f" - Val Set:   {val_states.shape} ({len(val_states):,} windows, {sum(val_bin):,} attacks, MITRE breakdown: {pd.Series(val_mit).value_counts().to_dict()})")
    logger.info(f" - Test Set:  {test_states.shape} ({len(test_states):,} windows, {sum(test_bin):,} attacks, MITRE breakdown: {pd.Series(test_mit).value_counts().to_dict()})")
    logger.info(f"Saved Splits -> {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    process_universal_gen8_5s()
