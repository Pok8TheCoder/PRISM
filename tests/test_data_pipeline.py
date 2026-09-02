"""Unit tests for SchemaAligner, StateBuilder, and Dataset."""
import numpy as np
import pandas as pd
import torch
import pytest
from src.data.schema_aligner import SchemaAligner, map_label_to_mitre
from src.data.state_builder import StateBuilder, STATE_DIM
from src.data.dataset import StateSequenceDataset, chronological_split


def test_schema_aligner():
    df_raw = pd.DataFrame({
        " Destination Port": [80, 22, 443],
        " Flow Duration": [1000, 2000, 3000],
        " Total Fwd Packets": [5, 10, 15],
        " Total Backward Packets": [4, 8, 12],
        " Timestamp": ["04/07/2017 08:54:00", "04/07/2017 08:54:30", "04/07/2017 08:55:00"],
        " Label": ["BENIGN", "SSH-Patator", "PortScan"]
    })
    
    aligner = SchemaAligner()
    aligned = aligner.align_dataframe(df_raw)
    
    assert len(aligned) == 3
    assert "flow_duration" in aligned.columns
    assert "mitre_code" in aligned.columns
    assert aligned["mitre_stage"].tolist() == ["Benign", "Initial Access", "Reconnaissance"]
    assert aligned["is_attack"].tolist() == [0, 1, 1]


def test_state_builder():
    dates = pd.date_range("2026-01-01 00:00:00", periods=10, freq="20s")
    df_aligned = pd.DataFrame({
        "timestamp": dates,
        "dst_port_binned": [0.0] * 10,
        "protocol_type": [1.0] * 10,
        "is_attack": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        "mitre_code": [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    })
    for col in ["flow_duration", "tot_fwd_pkts", "tot_bwd_pkts"]:
        df_aligned[col] = np.random.rand(10)
        
    builder = StateBuilder(window_size_seconds=60)
    states, atks, mitres, fracs, t_stamps = builder.build_states_from_dataframe(df_aligned)
    
    assert states.shape[1] == STATE_DIM
    assert len(states) >= 3
    assert 1 in atks  # Attack window present


def test_dataset_and_chronological_split():
    states = np.random.randn(50, STATE_DIM).astype(np.float32)
    atks = np.random.randint(0, 2, size=50)
    mitres = np.random.randint(0, 7, size=50)
    
    train_ds, val_ds, test_ds = chronological_split(states, atks, mitres, lookback=10)
    
    assert len(train_ds) > 0
    sample = train_ds[0]
    assert sample["state_seq"].shape == (10, STATE_DIM)
    assert sample["next_state"].shape == (STATE_DIM,)
