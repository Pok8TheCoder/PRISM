"""
PRISM Dataset
PyTorch Datasets and DataLoaders for state-sequence world model training.
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

logger = logging.getLogger("prism.data.dataset")


class StateSequenceDataset(Dataset):
    """
    PyTorch Dataset that yields (state_sequence, next_state, binary_label,
    mitre_label) tuples for world-model training.

    Given a sequence of states [S_1, ..., S_T], each sample is:
        X = [S_{i}, S_{i+1}, ..., S_{i+L-1}]   (lookback window)
        y_state = S_{i+L}                        (next state to predict)
        y_binary = labels_binary[i+L]            (attack or not)
        y_mitre  = labels_mitre[i+L]             (MITRE stage)
    """

    def __init__(
        self,
        states: np.ndarray,
        labels_binary: np.ndarray,
        labels_mitre: np.ndarray,
        lookback: int = 20,
        stride: int = 1,
    ):
        """
        Parameters
        ----------
        states : np.ndarray, shape (T, D_state)
        labels_binary : np.ndarray, shape (T,)
        labels_mitre : np.ndarray, shape (T,)
        lookback : int
            Number of past windows to include as context.
        stride : int
            Step size between consecutive samples.
        """
        assert len(states) == len(labels_binary) == len(labels_mitre), (
            f"Length mismatch: states={len(states)}, "
            f"binary={len(labels_binary)}, mitre={len(labels_mitre)}"
        )
        self.states = torch.tensor(states, dtype=torch.float32)
        self.labels_binary = torch.tensor(labels_binary, dtype=torch.long)
        self.labels_mitre = torch.tensor(labels_mitre, dtype=torch.long)
        self.lookback = lookback
        self.stride = stride

        # Valid indices: we need lookback states + 1 target state
        self.valid_indices = list(
            range(lookback, len(states), stride)
        )
        logger.info(
            "StateSequenceDataset: %d total states, lookback=%d, "
            "stride=%d -> %d samples",
            len(states), lookback, stride, len(self.valid_indices),
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    @property
    def sample_labels_binary(self) -> np.ndarray:
        """Returns binary labels for all valid sequence target windows."""
        return np.array([self.labels_binary[t].item() for t in self.valid_indices])

    def __getitem__(self, idx: int) -> dict:
        t = self.valid_indices[idx]
        return {
            "state_seq": self.states[t - self.lookback : t],    # (L, D)
            "next_state": self.states[t],                        # (D,)
            "label_binary": self.labels_binary[t],               # scalar
            "label_mitre": self.labels_mitre[t],                 # scalar
        }


class GraphSequenceDataset(Dataset):
    """
    PyTorch Dataset for graph-based world model.

    Each sample is a sequence of L graphs plus labels for the next step.
    """

    def __init__(
        self,
        graphs: list[dict],
        labels_binary: np.ndarray,
        labels_mitre: np.ndarray,
        lookback: int = 20,
        stride: int = 1,
    ):
        self.graphs = graphs
        self.labels_binary = torch.tensor(labels_binary, dtype=torch.long)
        self.labels_mitre = torch.tensor(labels_mitre, dtype=torch.long)
        self.lookback = lookback
        self.stride = stride

        self.valid_indices = list(range(lookback, len(graphs), stride))
        logger.info(
            "GraphSequenceDataset: %d graphs, lookback=%d -> %d samples",
            len(graphs), lookback, len(self.valid_indices),
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        t = self.valid_indices[idx]
        return {
            "graph_seq": self.graphs[t - self.lookback : t],
            "next_graph": self.graphs[t],
            "label_binary": self.labels_binary[t],
            "label_mitre": self.labels_mitre[t],
        }


# -----------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------
def load_states_from_npz(path: str) -> dict:
    """Load pre-built states from an .npz file."""
    data = np.load(path)
    return {
        "states": data["states"],
        "labels_binary": data["labels_binary"],
        "labels_mitre": data["labels_mitre"],
        "window_ids": data.get("window_ids", None),
    }


def create_temporal_splits(
    states: np.ndarray,
    labels_binary: np.ndarray,
    labels_mitre: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """
    Split data temporally (by time order, not random) to prevent leakage.

    Returns dict with 'train', 'val', 'test' keys, each containing
    (states, labels_binary, labels_mitre) arrays.
    """
    T = len(states)
    t_train = int(T * train_ratio)
    t_val = int(T * (train_ratio + val_ratio))

    splits = {
        "train": {
            "states": states[:t_train],
            "labels_binary": labels_binary[:t_train],
            "labels_mitre": labels_mitre[:t_train],
        },
        "val": {
            "states": states[t_train:t_val],
            "labels_binary": labels_binary[t_train:t_val],
            "labels_mitre": labels_mitre[t_train:t_val],
        },
        "test": {
            "states": states[t_val:],
            "labels_binary": labels_binary[t_val:],
            "labels_mitre": labels_mitre[t_val:],
        },
    }

    for name, split in splits.items():
        n_attack = int(split["labels_binary"].sum())
        logger.info(
            "Split '%s': %d windows (%d attack, %d benign)",
            name, len(split["states"]),
            n_attack, len(split["states"]) - n_attack,
        )

    return splits


def save_splits(splits: dict, output_dir: str) -> None:
    """Save train/val/test splits to separate .npz files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, split in splits.items():
        path = os.path.join(output_dir, f"{name}.npz")
        np.savez(path, **split)
        logger.info("Saved %s split to %s", name, path)


def create_dataloaders(
    splits: dict,
    lookback: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
    stride: int = 1,
    balanced_sampling: bool = True,
) -> dict:
    """
    Create DataLoaders for train/val/test splits.

    Parameters
    ----------
    splits : dict
        Dict mapping split name ('train', 'val', 'test') to split dict.
    balanced_sampling : bool
        If True, use WeightedRandomSampler on the training split to enforce
        a 50/50 balance of attack and benign windows per batch.

    Returns dict mapping split name -> DataLoader.
    """
    loaders = {}
    for name, split in splits.items():
        ds = StateSequenceDataset(
            states=split["states"],
            labels_binary=split["labels_binary"],
            labels_mitre=split["labels_mitre"],
            lookback=lookback,
            stride=stride,
        )

        sampler = None
        shuffle = (name == "train")

        if balanced_sampling and name == "train":
            labels = ds.sample_labels_binary
            counts = np.bincount(labels, minlength=2).astype(np.float64)
            present = counts > 0
            class_weights = np.zeros(len(counts), dtype=np.float64)
            if present.all():
                class_weights[present] = 1.0 / counts[present]
                sample_weights = torch.tensor(class_weights[labels], dtype=torch.double)
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False
                logger.info(
                    "Train DataLoader: WeightedRandomSampler active (Class counts: %s -> balanced batches)",
                    counts.tolist(),
                )

        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(name == "train"),
        )
        logger.info(
            "DataLoader '%s': %d batches (batch_size=%d)",
            name, len(loaders[name]), batch_size,
        )

    return loaders


def compute_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights for present classes."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if present.any():
        # Smoothed inverse square root frequency prevents extreme skew while boosting minority
        inv = 1.0 / np.sqrt(counts[present])
        weights[present] = inv / inv.sum() * present.sum()
    weights[~present] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--states", required=True, help="Path to states.npz")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data = load_states_from_npz(args.states)
    splits = create_temporal_splits(
        data["states"], data["labels_binary"], data["labels_mitre"]
    )
    save_splits(splits, args.output_dir)

    loaders = create_dataloaders(
        splits, lookback=args.lookback, batch_size=args.batch_size, num_workers=0,
    )
    # Smoke test
    for name, loader in loaders.items():
        batch = next(iter(loader))
        print(
            f"{name}: state_seq={batch['state_seq'].shape}, "
            f"next_state={batch['next_state'].shape}, "
            f"binary={batch['label_binary'].shape}"
        )
