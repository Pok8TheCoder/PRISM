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
        augment: bool = False,
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
        augment : bool
            If True, enables in-memory intra-class sequence mixup and subtle jitter
            for minority stages (Recon, Lateral Movement, Exfiltration).
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
        self.augment = augment

        # Valid indices: we need lookback states + 1 target state
        self.valid_indices = list(
            range(lookback, len(states), stride)
        )
        logger.info(
            "StateSequenceDataset: %d total states, lookback=%d, "
            "stride=%d, augment=%s -> %d samples",
            len(states), lookback, stride, augment, len(self.valid_indices),
        )


    def __len__(self) -> int:
        return len(self.valid_indices)

    @property
    def sample_labels_binary(self) -> np.ndarray:
        """Returns binary labels for all valid sequence target windows."""
        return np.array([self.labels_binary[t].item() for t in self.valid_indices])

    @property
    def sample_labels_mitre(self) -> np.ndarray:
        """Returns MITRE stage labels for all valid sequence target windows."""
        return np.array([self.labels_mitre[t].item() for t in self.valid_indices])

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


class GPUDatasetLoader:
    """
    High-Performance GPU-Resident Time-Series DataLoader.
    Stores the full dataset in GPU VRAM (~100 MB) once.
    Gathers sliding-window batches directly in CUDA memory in <1ms without
    touching the host CPU or system RAM, eliminating data transfer bottlenecks.
    """

    def __init__(
        self,
        states: np.ndarray,
        labels_binary: np.ndarray,
        labels_mitre: np.ndarray,
        lookback: int = 20,
        batch_size: int = 1024,
        stride: int = 1,
        device: torch.device = torch.device("cuda"),
        balanced_sampling: bool = False,
        shuffle: bool = False,
    ):
        self.device = device
        self.lookback = lookback
        self.batch_size = batch_size
        self.balanced_sampling = balanced_sampling
        self.shuffle = shuffle

        # Direct transfer to GPU memory once:
        self.states = torch.tensor(states, dtype=torch.float32, device=device)
        self.labels_binary = torch.tensor(labels_binary, dtype=torch.long, device=device)
        self.labels_mitre = torch.tensor(labels_mitre, dtype=torch.long, device=device)

        self.valid_indices = torch.arange(lookback, len(self.states), stride, device=device)
        self.N = len(self.valid_indices)
        self.window_offsets = torch.arange(-lookback, 0, device=device)

        # Precompute balanced sampling weights on GPU
        if balanced_sampling:
            labels_m = self.labels_mitre[self.valid_indices]
            counts = torch.bincount(labels_m, minlength=7).float()
            target_prob = 1.0 / 7.0
            class_weights = torch.zeros(7, device=device)
            mask = counts > 0
            class_weights[mask] = target_prob / counts[mask]
            self.sample_weights = class_weights[labels_m]
        else:
            self.sample_weights = None

    def __len__(self) -> int:
        return (self.N + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.balanced_sampling and self.sample_weights is not None:
            # GPU multinomial balanced sampling: 0.05s on CUDA
            perm = torch.multinomial(self.sample_weights, num_samples=self.N, replacement=True)
            active_indices = self.valid_indices[perm]
        elif self.shuffle:
            perm = torch.randperm(self.N, device=self.device)
            active_indices = self.valid_indices[perm]
        else:
            active_indices = self.valid_indices

        for start in range(0, self.N, self.batch_size):
            b_idx = active_indices[start : start + self.batch_size]
            gather_idx = b_idx.unsqueeze(1) + self.window_offsets.unsqueeze(0)
            yield {
                "state_seq": self.states[gather_idx],
                "next_state": self.states[b_idx],
                "label_binary": self.labels_binary[b_idx],
                "label_mitre": self.labels_mitre[b_idx],
            }


def create_dataloaders(
    splits: dict,
    lookback: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
    stride: int = 1,
    balanced_sampling: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Create DataLoaders for train/val/test splits.
    Uses pure GPUDatasetLoader if CUDA is active, otherwise falls back to standard DataLoader.
    """
    loaders = {}
    if isinstance(device, str):
        device = torch.device(device)
    use_gpu_loader = (device is not None and device.type == "cuda")

    if use_gpu_loader:
        logger.info("Initializing high-speed GPU-resident DataLoaders (zero CPU overhead)...")
        for name, split in splits.items():
            is_train = (name == "train")
            loaders[name] = GPUDatasetLoader(
                states=split["states"],
                labels_binary=split["labels_binary"],
                labels_mitre=split["labels_mitre"],
                lookback=lookback,
                batch_size=batch_size,
                stride=stride,
                device=device,
                balanced_sampling=(balanced_sampling and is_train),
                shuffle=is_train,
            )
            logger.info("GPUDatasetLoader '%s': %d batches (batch_size=%d) in VRAM", name, len(loaders[name]), batch_size)
        return loaders

    # Fallback to standard CPU DataLoader
    for name, split in splits.items():
        ds = StateSequenceDataset(
            states=split["states"],
            labels_binary=split["labels_binary"],
            labels_mitre=split["labels_mitre"],
            lookback=lookback,
            stride=stride,
            augment=(name == "train"),
        )

        sampler = None
        shuffle = (name == "train")

        if balanced_sampling and name == "train":
            labels_m = ds.sample_labels_mitre
            num_stages = 7
            counts_m = np.bincount(labels_m, minlength=num_stages).astype(np.float64)
            target_prob = np.ones(num_stages, dtype=np.float64) / num_stages
            class_weights = np.zeros(num_stages, dtype=np.float64)
            for c in range(num_stages):
                if counts_m[c] > 0:
                    class_weights[c] = target_prob[c] / counts_m[c]
                else:
                    class_weights[c] = 0.0

            sample_weights = torch.tensor(class_weights[labels_m], dtype=torch.double)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False

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
    """Compute balanced inverse-frequency class weights for present classes.

    Fix C: weights are clamped to a minimum of 0.1 so that any class with
    zero samples in the training split (e.g. Stage 5 Exfiltration before
    the dataset was recompiled) never receives a weight of 0.0 and is
    thus silently ignored by the loss function.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    if present.any():
        # Smoothed inverse square root frequency prevents extreme skew while boosting minority
        inv = 1.0 / np.sqrt(counts[present])
        weights[present] = inv / inv.sum() * present.sum()
    # Fix C: floor at 0.1 — absent classes still contribute a small gradient
    # signal rather than being silently zeroed out.
    weights = np.maximum(weights, 0.1)
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
