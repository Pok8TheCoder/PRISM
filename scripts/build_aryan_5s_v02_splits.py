#!/usr/bin/env python3
"""Build ARY-5sV02 splits: CIC 5s base + lab PCAP windows (5s).

Adds all lab PCAPs to **train**; oversamples failed-capture PCAPs so the model
sees quiet recon traffic more often. Val/test stay CIC-only for early stopping.

Usage:
  python scripts/build_aryan_5s_v02_splits.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_xmt_splits import _class_from_pcap, _mitre_id  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import pcap_to_states  # noqa: E402

BASE_SPLITS = ROOT / "data" / "aryan_splits_5s"
OUT_DIR = ROOT / "data" / "aryan_splits_5s_v02"

# PCAPs that failed ARY-5sV01 + RAMX strict bench — oversample in train
FAILED_PCAPS = {
    "live_port_scan_sequential_none.pcap",
    "r44_a1_T1135_share_discovery_none.pcap",
}
OVERSAMPLE = 20


def lab_windows(pcap: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    states, _, _ = pcap_to_states(pcap, window_sec=5.0)
    if len(states) == 0:
        return None
    cls = _class_from_pcap(pcap.name)
    mid = _mitre_id(cls)
    binary = 0 if cls == "Benign" else 1
    b = np.full(len(states), binary, dtype=np.int64)
    m = np.full(len(states), mid, dtype=np.int64)
    return states.astype(np.float32), b, m


def main() -> int:
    tr_s, tr_b, tr_m = load_all_splits(BASE_SPLITS)["train"]
    va_s, va_b, va_m = load_all_splits(BASE_SPLITS)["val"]
    te_s, te_b, te_m = load_all_splits(BASE_SPLITS)["test"]

    lab_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = []
    pcaps = sorted(SAVE_DIR.glob("*.pcap"))
    if not pcaps:
        print(f"No PCAPs in {SAVE_DIR}", file=sys.stderr)
        return 1

    for pcap in pcaps:
        row = lab_windows(pcap)
        if row is None:
            continue
        rep = OVERSAMPLE if pcap.name in FAILED_PCAPS else 1
        lab_rows.append((*row, pcap.name, rep))

    extra_s, extra_b, extra_m = [], [], []
    for s, b, m, name, rep in lab_rows:
        for _ in range(rep):
            extra_s.append(s)
            extra_b.append(b)
            extra_m.append(m)

    tr_s = np.concatenate([tr_s] + extra_s, axis=0)
    tr_b = np.concatenate([tr_b] + extra_b, axis=0)
    tr_m = np.concatenate([tr_m] + extra_m, axis=0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in [
        ("train", (tr_s, tr_b, tr_m)),
        ("val", (va_s, va_b, va_m)),
        ("test", (te_s, te_b, te_m)),
    ]:
        s, b, m = blob
        np.savez_compressed(
            OUT_DIR / f"{name}.npz",
            states=s, labels_binary=b, labels_mitre=m,
        )

    meta = {
        "base": str(BASE_SPLITS),
        "window_sec": 5.0,
        "lab_pcaps": len(lab_rows),
        "failed_oversample": OVERSAMPLE,
        "failed_pcaps": sorted(FAILED_PCAPS),
        "train_windows": int(len(tr_s)),
        "val_windows": int(len(va_s)),
        "test_windows": int(len(te_s)),
        "lab_pcap_windows": int(sum(len(x[0]) * x[4] for x in lab_rows)),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"Wrote -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
