"""
PRISM Dataset and DataLoader: Constructs sliding temporal sequences for PyTorch
and handles chronological, leakage-safe dataset splitting and normalization.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict
from src.utils.logger import setup_logger

logger = setup_logger("Dataset")


class StateSequenceDataset(Dataset):
    """
    Sliding window sequence dataset for World Model training.
    
    Item tuple:
        - state_sequence: (L, D_state) historical sequence [S_{t-L+1} ... S_t]
        - next_state: (D_state,) target next state vector S_{t+1}
        - is_attack_next: (1,) binary classification target
        - mitre_next: (1,) multiclass MITRE target (0-6)
        - is_attack_curr: (1,) current window attack status
        - mitre_curr: (1,) current window MITRE stage
    """

    def __init__(
        self,
        states: np.ndarray,
        attack_labels: np.ndarray,
        mitre_labels: np.ndarray,
        attack_fractions: Optional[np.ndarray] = None,
        lookback: int = 30,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None
    ):
        self.lookback = lookback
        self.raw_states = states.astype(np.float32)
        self.attack_labels = attack_labels.astype(np.int64)
        self.mitre_labels = mitre_labels.astype(np.int64)
        self.attack_fractions = (
            attack_fractions.astype(np.float32)
            if attack_fractions is not None
            else self.attack_labels.astype(np.float32)
        )

        # Standardize state vectors
        if scaler_mean is not None and scaler_std is not None:
            self.scaler_mean = scaler_mean.astype(np.float32)
            self.scaler_std = np.where(scaler_std == 0, 1.0, scaler_std).astype(np.float32)
        else:
            self.scaler_mean = np.mean(self.raw_states, axis=0, keepdims=True)
            std = np.std(self.raw_states, axis=0, keepdims=True)
            self.scaler_std = np.where(std == 0, 1.0, std).astype(np.float32)

        self.normalized_states = (self.raw_states - self.scaler_mean) / self.scaler_std
        # Number of valid sequences where we have lookback past states + 1 future state
        self.num_samples = max(0, len(self.normalized_states) - self.lookback)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx
        end_idx = idx + self.lookback  # non-inclusive for sequence, points to next state
        
        seq = self.normalized_states[start_idx:end_idx]          # (L, D)
        next_s = self.normalized_states[end_idx]                 # (D,)
        
        curr_attack = self.attack_labels[end_idx - 1]
        curr_mitre = self.mitre_labels[end_idx - 1]
        next_attack = self.attack_labels[end_idx]
        next_mitre = self.mitre_labels[end_idx]
        next_frac = self.attack_fractions[end_idx]

        return {
            "state_seq": torch.from_numpy(seq),
            "next_state": torch.from_numpy(next_s),
            "curr_attack": torch.tensor(curr_attack, dtype=torch.long),
            "curr_mitre": torch.tensor(curr_mitre, dtype=torch.long),
            "next_attack": torch.tensor(next_attack, dtype=torch.long),
            "next_mitre": torch.tensor(next_mitre, dtype=torch.long),
            "next_fraction": torch.tensor(next_frac, dtype=torch.float32)
        }


def chronological_split(
    states: np.ndarray,
    attack_labels: np.ndarray,
    mitre_labels: np.ndarray,
    attack_fractions: Optional[np.ndarray] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    lookback: int = 10
) -> Tuple[StateSequenceDataset, StateSequenceDataset, StateSequenceDataset]:
    """
    Partitions a contiguous timeline into Train, Validation, and Test sets
    strictly by chronological order, fitting normalizer statistics only on the training slice.
    """
    total_len = len(states)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    train_states = states[:train_end]
    train_atk = attack_labels[:train_end]
    train_mitre = mitre_labels[:train_end]
    train_frac = attack_fractions[:train_end] if attack_fractions is not None else None

    # Fit scaler strictly on training data
    scaler_mean = np.mean(train_states, axis=0, keepdims=True)
    scaler_std = np.std(train_states, axis=0, keepdims=True)
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)

    train_dataset = StateSequenceDataset(
        train_states, train_atk, train_mitre, train_frac,
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    val_dataset = StateSequenceDataset(
        states[train_end:val_end],
        attack_labels[train_end:val_end],
        mitre_labels[train_end:val_end],
        attack_fractions[train_end:val_end] if attack_fractions is not None else None,
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    test_dataset = StateSequenceDataset(
        states[val_end:],
        attack_labels[val_end:],
        mitre_labels[val_end:],
        attack_fractions[val_end:] if attack_fractions is not None else None,
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    return train_dataset, val_dataset, test_dataset


def multi_file_chronological_split(
    file_state_tuples: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    lookback: int = 10
) -> Tuple[StateSequenceDataset, StateSequenceDataset, StateSequenceDataset]:
    """
    Splits each file's state sequence chronologically into Train (70%), Val (15%), Test (15%),
    fits the scaler strictly across all training portions pooled together, and returns datasets.
    """
    train_states_list, train_atk_list, train_mitre_list, train_frac_list = [], [], [], []
    val_states_list, val_atk_list, val_mitre_list, val_frac_list = [], [], [], []
    test_states_list, test_atk_list, test_mitre_list, test_frac_list = [], [], [], []

    for states, atks, mitres, fracs in file_state_tuples:
        total_len = len(states)
        if total_len <= lookback + 2:
            continue
        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        train_states_list.append(states[:train_end])
        train_atk_list.append(atks[:train_end])
        train_mitre_list.append(mitres[:train_end])
        train_frac_list.append(fracs[:train_end])

        val_states_list.append(states[train_end:val_end])
        val_atk_list.append(atks[train_end:val_end])
        val_mitre_list.append(mitres[train_end:val_end])
        val_frac_list.append(fracs[train_end:val_end])

        test_states_list.append(states[val_end:])
        test_atk_list.append(atks[val_end:])
        test_mitre_list.append(mitres[val_end:])
        test_frac_list.append(fracs[val_end:])

    # Pool training states to compute shared normalization parameters
    all_train_states = np.concatenate(train_states_list, axis=0)
    scaler_mean = np.mean(all_train_states, axis=0, keepdims=True)
    scaler_std = np.std(all_train_states, axis=0, keepdims=True)
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std)

    # Combine into single arrays
    train_ds = StateSequenceDataset(
        np.concatenate(train_states_list, axis=0),
        np.concatenate(train_atk_list, axis=0),
        np.concatenate(train_mitre_list, axis=0),
        np.concatenate(train_frac_list, axis=0),
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    val_ds = StateSequenceDataset(
        np.concatenate(val_states_list, axis=0),
        np.concatenate(val_atk_list, axis=0),
        np.concatenate(val_mitre_list, axis=0),
        np.concatenate(val_frac_list, axis=0),
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    test_ds = StateSequenceDataset(
        np.concatenate(test_states_list, axis=0),
        np.concatenate(test_atk_list, axis=0),
        np.concatenate(test_mitre_list, axis=0),
        np.concatenate(test_frac_list, axis=0),
        lookback=lookback, scaler_mean=scaler_mean, scaler_std=scaler_std
    )

    return train_ds, val_ds, test_ds

