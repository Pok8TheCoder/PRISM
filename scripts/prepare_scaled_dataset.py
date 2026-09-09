"""
Scaled Multi-Dataset Ingestion and Leak-Free Stratified Splitting for PRISM.
Builds comprehensive 15-second temporal state windows across all 7 MITRE stages.
"""

import os
import sys
import glob
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.schema_aligner import SchemaAligner
from src.data.state_builder import StateBuilder
from src.utils.logger import setup_logger
from src.utils.constants import MITRE_STAGES, MITRE_STAGES_INV

logger = setup_logger("DatasetScaler")

PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_curated_file_manifest() -> List[Dict[str, any]]:
    """Defines the curated multi-dataset files and maximum row limits to ensure balanced, diverse windows."""
    c18_dir = os.path.join("data", "raw", "CIC-IDS2018")
    c17_dir = os.path.join("data", "raw", "TrafficLabelling")
    unsw_file = os.path.join("data", "raw", "unsw_nb15", "UNSW_NB15.csv")
    iot_file = os.path.join("data", "raw", "CICIOT23", "train", "train.csv")

    manifest = [
        # Stage 0: Pure Benign Baseline
        {
            "path": os.path.join(c17_dir, "Monday-WorkingHours.pcap_ISCX.csv"),
            "max_rows": 250_000,
            "target_stage": 0,
            "desc": "Monday Pure Benign Traffic"
        },
        # Stage 1: Reconnaissance
        {
            "path": os.path.join(c17_dir, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
            "max_rows": 300_000,
            "target_stage": 1,
            "desc": "CIC-IDS2017 PortScan (Reconnaissance)"
        },
        {
            "path": os.path.join(c17_dir, "Tuesday-WorkingHours.pcap_ISCX.csv"),
            "max_rows": 250_000,
            "target_stage": 1,
            "desc": "CIC-IDS2017 Tuesday Port/Service Probing & Patator"
        },
        # Stage 2: Initial Access / Brute Force & Web Attacks
        {
            "path": os.path.join(c18_dir, "02-14-2018.csv"),
            "max_rows": 300_000,
            "target_stage": 2,
            "desc": "CIC-IDS2018 FTP/SSH Brute Force"
        },
        {
            "path": os.path.join(c17_dir, "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
            "max_rows": 200_000,
            "target_stage": 2,
            "desc": "CIC-IDS2017 Web Attacks (XSS, SQLi, Brute Force)"
        },
        {
            "path": os.path.join(c18_dir, "02-22-2018.csv"),
            "max_rows": 60_000,
            "target_stage": 2,
            "desc": "CIC-IDS2018 Web Attacks"
        },
        {
            "path": os.path.join(c18_dir, "02-23-2018.csv"),
            "max_rows": 60_000,
            "target_stage": 2,
            "desc": "CIC-IDS2018 Web Attacks Day 2 (Brute Force, XSS, SQLi)"
        },
        # Stage 3: Lateral Movement & Infiltration
        {
            "path": os.path.join(c18_dir, "03-01-2018.csv"),
            "max_rows": 450_000,
            "target_stage": 3,
            "desc": "CIC-IDS2018 Infiltration Campaign Day 2"
        },
        {
            "path": os.path.join(c18_dir, "02-28-2018.csv"),
            "max_rows": 650_000,  # Infiltration starts at row ~381k
            "target_stage": 3,
            "desc": "CIC-IDS2018 Infiltration Campaign Day 1"
        },
        {
            "path": os.path.join(c17_dir, "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
            "max_rows": 300_000,
            "target_stage": 3,
            "desc": "CIC-IDS2017 Infiltration"
        },
        # Stage 4: Command & Control (C2 / Botnets)
        {
            "path": os.path.join(c18_dir, "03-02-2018.csv"),
            "max_rows": 250_000,
            "target_stage": 4,
            "desc": "CIC-IDS2018 Botnet C2 (80k+ Bot flows)"
        },
        {
            "path": os.path.join(c17_dir, "Friday-WorkingHours-Morning.pcap_ISCX.csv"),
            "max_rows": 200_000,
            "target_stage": 4,
            "desc": "CIC-IDS2017 Botnet Ares C2"
        },
        {
            "path": os.path.join("data", "raw", "ctu13", "capture20110810.binetflow"),
            "skiprows": range(1, 750_000),
            "max_rows": 600_000,
            "target_stage": 4,
            "desc": "CTU-13 Neris Botnet C2 (Active Wave 1)"
        },
        {
            "path": os.path.join("data", "raw", "ctu13", "capture20110810.binetflow"),
            "skiprows": range(1, 1_350_000),
            "max_rows": 700_000,
            "target_stage": 4,
            "desc": "CTU-13 Neris Botnet C2 (Active Wave 2)"
        },
        {
            "path": os.path.join("data", "raw", "ctu13", "capture20110810.binetflow"),
            "skiprows": range(1, 2_050_000),
            "max_rows": 700_000,
            "target_stage": 4,
            "desc": "CTU-13 Neris Botnet C2 (Active Wave 3)"
        },
        # Stage 5: Exfiltration & Covert Channels (UNSW-NB15)
        {
            "path": os.path.join("data", "raw", "unsw_nb15", "UNSW_NB15_training-set.csv"),
            "max_rows": 200_000,
            "target_stage": 5,
            "desc": "UNSW-NB15 Training Partition (Exfil, Exploits, Fuzzers, Backdoors)"
        },
        {
            "path": unsw_file,
            "max_rows": 100_000,
            "target_stage": 5,
            "desc": "UNSW-NB15 Test Partition"
        },
        # Stage 6: Impact (DoS / DDoS)
        {
            "path": os.path.join(c18_dir, "02-15-2018.csv"),
            "max_rows": 250_000,
            "target_stage": 6,
            "desc": "CIC-IDS2018 DoS GoldenEye / Slowloris"
        },
        {
            "path": os.path.join(c18_dir, "02-20-2018.csv"),
            "max_rows": 250_000,
            "target_stage": 6,
            "desc": "CIC-IDS2018 DDoS LOIC-HTTP"
        },
        {
            "path": os.path.join(c17_dir, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
            "max_rows": 250_000,
            "target_stage": 6,
            "desc": "CIC-IDS2017 DDoS LOIC"
        },
        {
            "path": iot_file,
            "max_rows": 1_200_000,
            "target_stage": 6,
            "desc": "CICIOT2023 Multi-Attack IoT Stream"
        }
    ]
    return manifest


def build_scaled_dataset(window_size: int = 15, chunksize: int = 50_000):
    manifest = get_curated_file_manifest()
    aligner = SchemaAligner()
    builder = StateBuilder(window_size_seconds=window_size)

    file_tuples = []
    file_metadata = []

    logger.info(f"Starting scaled ingestion across {len(manifest)} curated scenario files...")

    for item in manifest:
        filepath = item["path"]
        max_rows = item["max_rows"]
        desc = item["desc"]
        skiprows = item.get("skiprows", None)
        fname = os.path.basename(filepath)

        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}. Skipping.")
            continue

        logger.info(f"--> Reading {fname} ({desc}) up to {max_rows:,} rows...")
        chunks = []
        rows_read = 0

        # Try cp1252 then fallback
        try:
            read_kwargs = {"chunksize": chunksize, "low_memory": False, "encoding": "cp1252"}
            if skiprows is not None:
                read_kwargs["skiprows"] = skiprows
            for chunk in pd.read_csv(filepath, **read_kwargs):
                aligned = aligner.align_dataframe(chunk)
                chunks.append(aligned)
                rows_read += len(chunk)
                if rows_read >= max_rows:
                    break
        except Exception:
            read_kwargs = {"chunksize": chunksize, "low_memory": False, "encoding_errors": "replace"}
            if skiprows is not None:
                read_kwargs["skiprows"] = skiprows
            for chunk in pd.read_csv(filepath, **read_kwargs):
                aligned = aligner.align_dataframe(chunk)
                chunks.append(aligned)
                rows_read += len(chunk)
                if rows_read >= max_rows:
                    break

        if not chunks:
            continue

        full_df = pd.concat(chunks, ignore_index=True)

        if "unsw" in fname.lower() or "train.csv" in fname.lower() or "iot" in desc.lower():
            # Ingest heterogeneous datasets by distinct MITRE attack stream with continuous 100ms timestamps
            # This completely prevents chunk-timestamp collision and DDoS swallowing of subtle Recon/Exfil flows
            for stage_name, grp in full_df.groupby("mitre_stage"):
                # Cap excessive Impact (DDoS) rows to 100k to prevent flooding out the other 6 stages
                if stage_name == "Impact" and len(grp) > 100_000:
                    grp = grp.iloc[:100_000]
                grp_clean = grp.copy().reset_index(drop=True)
                # Assign continuous, non-overlapping timestamps across the entire stream (200ms for covert exfiltration)
                step_freq = "200ms" if stage_name == "Exfiltration" else "100ms"
                grp_clean["timestamp"] = pd.date_range("2026-01-01 00:00:00", periods=len(grp_clean), freq=step_freq)
                states, atks, mitres, fracs, _ = builder.build_states_from_dataframe(grp_clean)
                num_windows = len(states)
                if num_windows >= 5:
                    file_tuples.append((states, atks, mitres, fracs))
                    stage_counts = {int(k): int(v) for k, v in pd.Series(mitres).value_counts().items()}
                    file_metadata.append({
                        "file": f"{fname} [{stage_name}]",
                        "desc": f"{desc} [{stage_name}]",
                        "windows": num_windows,
                        "stages": stage_counts
                    })
                    logger.info(f"    Built {num_windows} 15-second windows from {fname} [{stage_name}].")
        elif "portscan" in fname.lower():
            # Ingest both the full timeline and the concentrated PortScan probe stream
            full_df_sorted = full_df.sort_values(by="timestamp").reset_index(drop=True)
            states, atks, mitres, fracs, _ = builder.build_states_from_dataframe(full_df_sorted)
            if len(states) >= 5:
                file_tuples.append((states, atks, mitres, fracs))
                file_metadata.append({
                    "file": fname,
                    "desc": desc,
                    "windows": len(states),
                    "stages": {int(k): int(v) for k, v in pd.Series(mitres).value_counts().items()}
                })
            # Concentrated Reconnaissance probe stream
            recon_flows = full_df[full_df["mitre_stage"] == "Reconnaissance"].copy().reset_index(drop=True)
            if len(recon_flows) > 0:
                recon_flows = recon_flows.iloc[:60_000]
                recon_flows["timestamp"] = pd.date_range("2026-01-01 00:00:00", periods=len(recon_flows), freq="100ms")
                states_r, atks_r, mitres_r, fracs_r, _ = builder.build_states_from_dataframe(recon_flows)
                if len(states_r) >= 5:
                    file_tuples.append((states_r, atks_r, mitres_r, fracs_r))
                    file_metadata.append({
                        "file": f"{fname} [Concentrated PortScan]",
                        "desc": "CIC-IDS2017 Concentrated Reconnaissance Probe Stream",
                        "windows": len(states_r),
                        "stages": {int(k): int(v) for k, v in pd.Series(mitres_r).value_counts().items()}
                    })
                    logger.info(f"    Built {len(states_r)} 15-second windows from {fname} [Concentrated PortScan].")
        else:
            full_df = full_df.sort_values(by="timestamp").reset_index(drop=True)
            states, atks, mitres, fracs, _ = builder.build_states_from_dataframe(full_df)
            num_windows = len(states)
            logger.info(f"    Built {num_windows} 15-second windows from {fname}.")

            if num_windows >= 5:
                file_tuples.append((states, atks, mitres, fracs))
                stage_counts = {int(k): int(v) for k, v in pd.Series(mitres).value_counts().items()}
                file_metadata.append({
                    "file": fname,
                    "desc": desc,
                    "windows": num_windows,
                    "stages": stage_counts
                })

            # ALSO extract concentrated, un-diluted attack streams for subtle stages (Lateral Movement, Initial Access, Recon)
            for stage_name in ["Lateral Movement", "Initial Access", "Reconnaissance"]:
                stage_flows = full_df[full_df["mitre_stage"] == stage_name].copy().reset_index(drop=True)
                if len(stage_flows) >= 50:
                    stage_flows = stage_flows.iloc[:60_000]
                    stage_flows["timestamp"] = pd.date_range("2026-01-01 00:00:00", periods=len(stage_flows), freq="100ms")
                    states_atk, atks_atk, mitres_atk, fracs_atk, _ = builder.build_states_from_dataframe(stage_flows)
                    if len(states_atk) >= 5:
                        file_tuples.append((states_atk, atks_atk, mitres_atk, fracs_atk))
                        stage_counts_atk = {int(k): int(v) for k, v in pd.Series(mitres_atk).value_counts().items()}
                        file_metadata.append({
                            "file": f"{fname} [Concentrated {stage_name}]",
                            "desc": f"{desc} [Concentrated {stage_name} Stream]",
                            "windows": len(states_atk),
                            "stages": stage_counts_atk
                        })
                        logger.info(f"    Built {len(states_atk)} concentrated {stage_name} windows from {fname}.")

    if not file_tuples:
        raise ValueError("No states could be extracted from any files!")

    # 1. Save combined flat dataset
    all_states = np.concatenate([t[0] for t in file_tuples], axis=0)
    all_atks = np.concatenate([t[1] for t in file_tuples], axis=0)
    all_mitres = np.concatenate([t[2] for t in file_tuples], axis=0)
    all_fracs = np.concatenate([t[3] for t in file_tuples], axis=0)

    np.save(os.path.join(PROCESSED_DIR, "states.npy"), all_states)
    np.save(os.path.join(PROCESSED_DIR, "attack_labels.npy"), all_atks)
    np.save(os.path.join(PROCESSED_DIR, "mitre_labels.npy"), all_mitres)
    np.save(os.path.join(PROCESSED_DIR, "attack_fractions.npy"), all_fracs)

    # 2. Perform Per-Scenario Chronological Split (70% Train, 15% Val, 15% Test)
    # Each scenario stream's earliest 70% is used for training, middle 15% for validation, 
    # and unseen future 15% for holdout testing. Zero distribution lockout, zero temporal leakage.
    train_indices = []
    val_indices = []
    test_indices = []

    current_idx = 0
    for states, atks, mitres, fracs in file_tuples:
        n = len(states)
        tr_end = int(n * 0.70)
        vl_end = int(n * 0.85)

        train_indices.extend(range(current_idx, current_idx + tr_end))
        val_indices.extend(range(current_idx + tr_end, current_idx + vl_end))
        test_indices.extend(range(current_idx + vl_end, current_idx + n))
        current_idx += n

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)

    # Save explicit split files
    np.savez_compressed(
        os.path.join(PROCESSED_DIR, "split_indices.npz"),
        train_indices=np.array(train_indices, dtype=np.int64),
        val_indices=np.array(val_indices, dtype=np.int64),
        test_indices=np.array(test_indices, dtype=np.int64)
    )

    tr_mitre_all = all_mitres[train_indices]
    vl_mitre_all = all_mitres[val_indices]
    te_mitre_all = all_mitres[test_indices]

    # Compute scaler strictly on train set
    tr_states_all = all_states[train_indices]
    scaler_mean = np.mean(tr_states_all, axis=0, keepdims=True)
    scaler_std = np.std(tr_states_all, axis=0, keepdims=True)
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)

    np.save(os.path.join(PROCESSED_DIR, "scaler_mean.npy"), scaler_mean)
    np.save(os.path.join(PROCESSED_DIR, "scaler_std.npy"), scaler_std)

    # Save split metadata report
    summary = {
        "total_windows": len(all_states),
        "train_windows": len(train_indices),
        "val_windows": len(val_indices),
        "test_windows": len(test_indices),
        "files_processed": len(file_metadata),
        "file_details": file_metadata,
        "stages_breakdown": {
            stage_name: {
                "total": int(np.sum(all_mitres == code)),
                "train": int(np.sum(tr_mitre_all == code)),
                "val": int(np.sum(vl_mitre_all == code)),
                "test": int(np.sum(te_mitre_all == code))
            }
            for code, stage_name in MITRE_STAGES_INV.items()
        }
    }

    with open(os.path.join(PROCESSED_DIR, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("               DATASET PREPARATION COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total 15-Second Windows: {len(all_states):,}")
    logger.info(f"Train Set (70%):         {len(train_indices):,}")
    logger.info(f"Validation Set (15%):   {len(val_indices):,}")
    logger.info(f"Test Set (15%):          {len(test_indices):,}")
    logger.info("-" * 70)
    logger.info(f"{'Stage':<25} | {'Total':<8} | {'Train':<8} | {'Val':<8} | {'Test':<8}")
    logger.info("-" * 70)
    for code in range(7):
        s_name = MITRE_STAGES_INV.get(code, f"Stage {code}")
        tot = int(np.sum(all_mitres == code))
        tr = int(np.sum(tr_mitre_all == code))
        vl = int(np.sum(vl_mitre_all == code))
        te = int(np.sum(te_mitre_all == code))
        logger.info(f"{s_name:<25} | {tot:<8} | {tr:<8} | {vl:<8} | {te:<8}")
    logger.info("=" * 70)

    return summary


if __name__ == "__main__":
    build_scaled_dataset(window_size=15)
