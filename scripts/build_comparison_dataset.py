"""Build the shared ZMT.01 / YMT.01 benchmark dataset.

Both models are trained and scored on the *same flows in the same order*, split
by capture so overlapping windows can never straddle train and test. The only
difference between the two tensors written here is the feature representation.

Output: data/processed/zmt_ymt_dataset.npz
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.attack_catalog import get_class_to_idx, resolve_pcap_class  # noqa: E402
from src.pipeline.extract_v2 import (  # noqa: E402
    base_to_v1,
    blocks_to_v2,
    cic_df_to_base,
    pcap_to_base,
)
from src.pipeline.features_v2 import NUM_BASE  # noqa: E402

SEQ_LEN = 10
PCAP_STRIDE = 1
CSV_STRIDE = 4
CSV_CHUNK_FLOWS = 250        # flows per synthetic "capture" carved from the CSVs
MAX_SAMPLES_PER_CSV_CLASS = 2600
SEED = 42

ADV_DIR = ROOT / "data" / "raw" / "adversarial"
CIC_FILES = [
    ROOT / "data" / "raw" / "thursday_01_03_2018.csv",
    ROOT / "data" / "raw" / "wednesday_28_02_2018.csv",
]
OUT = ROOT / "data" / "processed" / "zmt_ymt_dataset.npz"
BLOCKS_OUT = ROOT / "data" / "processed" / "zmt_ymt_blocks.npy"

CIC_USE_COLS = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Fwd Pkt Len Mean", "Fwd Pkt Len Max",
    "Fwd Pkt Len Min", "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Std",
    "SYN Flag Cnt", "ACK Flag Cnt", "FIN Flag Cnt", "RST Flag Cnt",
    "PSH Flag Cnt", "Pkt Size Avg", "Pkt Len Std", "Init Fwd Win Byts", "Label",
]


def windows_from_capture(base: np.ndarray, stride: int) -> np.ndarray:
    """Slice one capture's flow matrix into (k, SEQ_LEN, NUM_BASE) blocks.

    Captures shorter than SEQ_LEN are cyclically tiled to one full block, which
    preserves their real port/protocol diversity instead of inventing port-0
    padding flows that would distort window entropy.
    """
    n = len(base)
    if n == 0:
        return np.zeros((0, SEQ_LEN, NUM_BASE), dtype=np.float32)
    if n < SEQ_LEN:
        reps = int(np.ceil(SEQ_LEN / n))
        return np.tile(base, (reps, 1))[:SEQ_LEN][None, ...]
    starts = range(0, n - SEQ_LEN + 1, stride)
    return np.stack([base[s:s + SEQ_LEN] for s in starts])


def collect_pcap_captures() -> list[tuple[str, str, np.ndarray]]:
    """Return (capture_id, class_name, base_matrix) for every labeled PCAP."""
    out = []
    pcaps = sorted(ADV_DIR.rglob("*.pcap"))
    for i, p in enumerate(pcaps):
        cls = resolve_pcap_class(p.name)
        if cls is None:
            continue
        base = pcap_to_base(p)
        if len(base) == 0:
            continue
        out.append((f"pcap_{i:04d}_{p.stem}", cls, base))
    print(f"  PCAP captures usable: {len(out)}")
    return out


def collect_csv_captures() -> list[tuple[str, str, np.ndarray]]:
    """Carve the CIC CSVs into fixed-size single-label pseudo-captures.

    Both CSVs contain only Benign and Infilteration, so the label mapping used
    by the shipped v1 trainer (all non-benign -> T1021_remote_services) is
    reproduced exactly rather than re-derived.
    """
    frames = []
    for f in CIC_FILES:
        if not f.exists() or f.stat().st_size == 0:
            continue
        df = pd.read_csv(f, usecols=lambda c: c in CIC_USE_COLS, low_memory=False)
        df = df[df["Label"] != "Label"]
        df["Label"] = df["Label"].astype(str).str.strip()
        frames.append(df)
    if not frames:
        print("  WARNING: no CIC CSVs found")
        return []
    df = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(SEED)
    # Samples per capture after windowing, used to cap each CSV class.
    per_chunk = len(range(0, CSV_CHUNK_FLOWS - SEQ_LEN + 1, CSV_STRIDE))
    max_chunks = max(1, MAX_SAMPLES_PER_CSV_CLASS // per_chunk)

    out = []
    for label_val, cls in (("benign", "Benign"), ("attack", "T1021_remote_services")):
        mask = (df["Label"].str.lower() == "benign")
        sub = df[mask] if label_val == "benign" else df[~mask]
        n_chunks_avail = len(sub) // CSV_CHUNK_FLOWS
        if n_chunks_avail == 0:
            continue
        pick = rng.choice(n_chunks_avail, size=min(max_chunks, n_chunks_avail),
                          replace=False)
        for j in sorted(pick):
            chunk = sub.iloc[j * CSV_CHUNK_FLOWS:(j + 1) * CSV_CHUNK_FLOWS]
            out.append((f"csv_{cls}_{j:06d}", cls, cic_df_to_base(chunk)))
    print(f"  CSV pseudo-captures: {len(out)}")
    return out


def split_captures(captures: list[tuple[str, str, np.ndarray]]) -> dict[str, str]:
    """Per-class capture-level split so no capture appears in two splits."""
    rng = np.random.default_rng(SEED)
    by_class: dict[str, list[str]] = defaultdict(list)
    for cid, cls, _ in captures:
        by_class[cls].append(cid)

    assignment: dict[str, str] = {}
    for cls, cids in by_class.items():
        cids = sorted(cids)
        rng.shuffle(cids)
        n = len(cids)
        if n == 1:
            assignment[cids[0]] = "train"
            continue
        n_test = max(1, int(round(0.2 * n)))
        n_val = max(1, int(round(0.2 * n))) if n >= 3 else 0
        for k, cid in enumerate(cids):
            if k < n_test:
                assignment[cid] = "test"
            elif k < n_test + n_val:
                assignment[cid] = "val"
            else:
                assignment[cid] = "train"
    return assignment


def main() -> None:
    print("Collecting captures...")
    captures = collect_pcap_captures() + collect_csv_captures()
    class_to_idx = get_class_to_idx()
    assignment = split_captures(captures)

    blocks_all, y_all, split_all, src_all = [], [], [], []
    for cid, cls, base in captures:
        if cls not in class_to_idx:
            continue
        stride = CSV_STRIDE if cid.startswith("csv_") else PCAP_STRIDE
        blocks = windows_from_capture(base, stride)
        if len(blocks) == 0:
            continue
        blocks_all.append(blocks.astype(np.float32))
        y_all.append(np.full(len(blocks), class_to_idx[cls], dtype=np.int64))
        split_all.append(np.full(len(blocks), assignment[cid], dtype=object))
        src_all.append(np.full(len(blocks), "csv" if cid.startswith("csv_") else "pcap",
                               dtype=object))

    blocks = np.concatenate(blocks_all)
    y = np.concatenate(y_all)
    split = np.concatenate(split_all)
    source = np.concatenate(src_all)
    print(f"\nTotal samples: {len(blocks)}  (block shape {blocks.shape})")

    print("Projecting to v1 (27-d) and v2 (64-d)...")
    X1 = base_to_v1(blocks)
    X2 = blocks_to_v2(blocks)
    print(f"  v1 {X1.shape}   v2 {X2.shape}")

    for s in ("train", "val", "test"):
        m = split == s
        print(f"  {s:<6} {m.sum():>7} samples | {len(np.unique(y[m]))} classes")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, X1=X1, X2=X2, y=y,
        split=split.astype(str), source=source.astype(str),
    )
    # Raw base blocks are kept so alternative schemas can be benchmarked on
    # byte-identical flow records without re-parsing every capture.
    np.save(BLOCKS_OUT, blocks)
    print(f"\nSaved -> {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved -> {BLOCKS_OUT}  ({BLOCKS_OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
