#!/usr/bin/env python3
"""Rebuild ARY CIC-IDS splits at configurable window size (5s / 30s).

Uses the same pipeline as ``origin/aryan`` @ ``1ec06bc``:
  FlowExtractor -> log/normalize -> StateBuilder -> temporal 70/15/15 split

Streams 200k rows per CSV from the public CIC-IDS-2018 S3 bucket (same
four days as ``scripts/preprocess_cicids.py`` on the aryan branch).

Usage:
  python scripts/build_aryan_cic_splits.py --window-sec 5 --out data/aryan_splits_5s
  python scripts/build_aryan_cic_splits.py --window-sec 30 --out data/aryan_splits_30s_verify
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORKTREE = ROOT.parent / "PRISM-aryan-build"
ARYAN_COMMIT = "1ec06bc"

S3 = "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed+Traffic+Data+for+ML+Algorithms"
CIC_FILES = [
    ("02-14-2018.csv", f"{S3}/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"),
    ("02-16-2018.csv", f"{S3}/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"),
    ("02-22-2018.csv", f"{S3}/Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"),
    ("03-01-2018.csv", f"{S3}/Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"),
]
NROWS = 200_000


def ensure_worktree() -> Path:
    if (WORKTREE / "src" / "data" / "flow_extractor.py").exists():
        return WORKTREE
    print(f"Creating git worktree at {WORKTREE} @ {ARYAN_COMMIT} ...")
    WORKTREE.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["git", "worktree", "add", str(WORKTREE), ARYAN_COMMIT],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    if r.returncode != 0 and not (WORKTREE / "src").exists():
        raise RuntimeError(f"git worktree add failed: {r.stderr}")
    return WORKTREE


def load_cic_sample(raw_dir: Path) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dfs = []
    for fname, url in CIC_FILES:
        local = raw_dir / fname
        if local.exists() and local.stat().st_size > 1000:
            print(f"  {fname} (cached)")
            chunk = pd.read_csv(local, nrows=NROWS, low_memory=False)
        else:
            print(f"  {fname} streaming {NROWS} rows ...")
            chunk = pd.read_csv(url, nrows=NROWS, low_memory=False)
            chunk.to_csv(local, index=False)
        chunk.columns = [str(c).strip() for c in chunk.columns]
        print(f"    -> {len(chunk)} rows")
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows: {len(df)}")
    return df


def temporal_split(states, bin_labels, mit_labels, train_ratio=0.70, val_ratio=0.15):
    t = len(states)
    t_train = int(t * train_ratio)
    t_val = int(t * (train_ratio + val_ratio))
    return {
        "train": (states[:t_train], bin_labels[:t_train], mit_labels[:t_train]),
        "val": (states[t_train:t_val], bin_labels[t_train:t_val], mit_labels[t_train:t_val]),
        "test": (states[t_val:], bin_labels[t_val:], mit_labels[t_val:]),
    }


def process_file(df: pd.DataFrame, extractor, builder) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = extractor.clean(df)
    df = extractor.engineer_features(df)
    df = extractor.log_transform(df)
    df = extractor.normalize(df, fit=False)
    df = extractor.map_labels(df)
    result = builder.build_states(df, label_col="mitre_stage_id")
    return (
        result["states"].astype(np.float32),
        result["labels_binary"].astype(np.int64),
        result["labels_mitre"].astype(np.int64),
    )


def merge_splits(parts: list[dict]) -> dict:
    out = {"train": [[], [], []], "val": [[], [], []], "test": [[], [], []]}
    for part in parts:
        for name in out:
            for i in range(3):
                out[name][i].append(part[name][i])
    merged = {}
    for name in out:
        merged[name] = tuple(np.concatenate(chunks) for chunks in out[name])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-sec", type=int, default=5)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "aryan_splits_5s")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "cicids2018")
    args = parser.parse_args()

    wt = ensure_worktree()
    sys.path.insert(0, str(wt))

    from src.data.flow_extractor import FlowExtractor  # noqa: E402
    from src.data.state_builder import StateBuilder  # noqa: E402

    print(f"\n=== CIC-IDS-2018 -> {args.window_sec}s windows (per-day split) ===")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    extractor = FlowExtractor(dataset_type="cicids2018", raw_dir=str(args.raw_dir))
    builder = StateBuilder(window_size_seconds=args.window_sec)

    # Fit scaler on all rows first (matches single-pass preprocess).
    all_chunks = []
    for fname, url in CIC_FILES:
        local = args.raw_dir / fname
        if local.exists() and local.stat().st_size > 1000:
            chunk = pd.read_csv(local, nrows=NROWS, low_memory=False)
        else:
            chunk = pd.read_csv(url, nrows=NROWS, low_memory=False)
            chunk.to_csv(local, index=False)
        chunk.columns = [str(c).strip() for c in chunk.columns]
        all_chunks.append(chunk)
    full = pd.concat(all_chunks, ignore_index=True)
    full = extractor.clean(full)
    full = extractor.engineer_features(full)
    full = extractor.log_transform(full)
    extractor.normalize(full, fit=True)

    split_parts = []
    total_windows = 0
    for fname, _url in CIC_FILES:
        local = args.raw_dir / fname
        chunk = pd.read_csv(local, nrows=NROWS, low_memory=False)
        chunk.columns = [str(c).strip() for c in chunk.columns]
        states, bin_labels, mit_labels = process_file(chunk, extractor, builder)
        total_windows += len(states)
        part = temporal_split(states, bin_labels, mit_labels)
        split_parts.append(part)
        print(f"  {fname}: {len(states)} windows, attack {bin_labels.mean():.1%}")

    splits = merge_splits(split_parts)
    states = np.concatenate([splits["train"][0], splits["val"][0], splits["test"][0]])
    bin_labels = np.concatenate([splits["train"][1], splits["val"][1], splits["test"][1]])
    print(f"Built {total_windows} windows total (attack {bin_labels.mean():.1%})")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"window_sec": args.window_sec, "nrows_per_csv": NROWS, "splits": {}}
    for name, (s, b, m) in splits.items():
        out = args.out / f"{name}.npz"
        np.savez_compressed(out, states=s, labels_binary=b, labels_mitre=m, window_ids=np.arange(len(s)))
        summary["splits"][name] = {
            "windows": int(len(s)),
            "attack_pct": round(float(b.mean()), 4),
            "mitre_stages": sorted(set(m.tolist())),
        }
        print(f"  {name}: {len(s)} windows, attack {b.mean():.1%}")

    (args.out / "meta.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote -> {args.out}")


if __name__ == "__main__":
    main()
