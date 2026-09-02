"""Build XMT.01 NPZ splits from PRISM lab PCAPs (242-d, ARY.01 feature schema).

ARY.01/02 train on CIC-IDS-2018 CSV days. XMT uses Docker-lab captures under
``data/raw/adversarial/`` — different traffic, same 30s StateBuilder geometry.

Output: ``data/xmt_splits/{train,val,test}.npz``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.constants import MITRE_STAGES, TACTIC_TO_STAGE  # noqa: E402
from src.aryan.ingest import pcap_to_states  # noqa: E402
from src.model.attack_catalog import get_mitre_map, resolve_pcap_class  # noqa: E402

OUT_DIR = ROOT / "data" / "xmt_splits"
_TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.I)


def _class_from_pcap(name: str) -> str:
    cls = resolve_pcap_class(name)
    if cls:
        return cls
    m = _TID.search(name)
    if not m:
        return "Benign"
    tid = m.group(0).upper()
    for class_id in get_mitre_map():
        if class_id == "Benign":
            continue
        if tid in class_id or tid.replace(".", "_") in class_id.replace(".", "_"):
            return class_id
    return "Benign"


def _mitre_id(class_id: str) -> int:
    if class_id == "Benign":
        return 0
    tactic = get_mitre_map().get(class_id, ("Normal", ""))[0]
    stage = TACTIC_TO_STAGE.get(tactic, "Benign")
    return MITRE_STAGES.get(stage, 0)


def _split_files(files: list[Path], train_frac=0.70, val_frac=0.15) -> dict[str, list[Path]]:
    n = len(files)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:n_train + n_val + n_test],
    }


def _build_split(files: list[Path]) -> dict[str, np.ndarray]:
    states, bin_labels, mit_labels, window_ids, sources = [], [], [], [], []
    wid = 0
    for pcap in files:
        st, _, _ = pcap_to_states(pcap)
        if len(st) == 0:
            continue
        cls = _class_from_pcap(pcap.name)
        mid = _mitre_id(cls)
        binary = 0 if cls == "Benign" else 1
        states.append(st)
        bin_labels.append(np.full(len(st), binary, dtype=np.int64))
        mit_labels.append(np.full(len(st), mid, dtype=np.int64))
        window_ids.append(np.arange(wid, wid + len(st), dtype=np.int64))
        sources.append(np.array([pcap.name] * len(st), dtype=object))
        wid += len(st)
    if not states:
        empty = np.zeros((0, 242), dtype=np.float32)
        return {
            "states": empty,
            "labels_binary": np.zeros(0, dtype=np.int64),
            "labels_mitre": np.zeros(0, dtype=np.int64),
            "window_ids": np.zeros(0, dtype=np.int64),
            "source_pcap": np.array([], dtype=object),
        }
    return {
        "states": np.concatenate(states).astype(np.float32),
        "labels_binary": np.concatenate(bin_labels),
        "labels_mitre": np.concatenate(mit_labels),
        "window_ids": np.concatenate(window_ids),
        "source_pcap": np.concatenate(sources),
    }


def main() -> None:
    pcaps = sorted(SAVE_DIR.glob("*.pcap"), key=lambda p: p.name)
    if not pcaps:
        raise SystemExit(f"No PCAPs in {SAVE_DIR}. Run lab captures first.")

    parts = _split_files(pcaps)
    meta = {"source": "prism_lab_pcaps", "pcap_dir": str(SAVE_DIR), "schema": "ary242_30s"}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split, files in parts.items():
        blob = _build_split(files)
        out = OUT_DIR / f"{split}.npz"
        np.savez_compressed(out, **blob)
        attack_pct = float(blob["labels_binary"].mean()) if len(blob["labels_binary"]) else 0.0
        summary[split] = {
            "windows": int(len(blob["states"])),
            "pcaps": len(files),
            "attack_pct": round(attack_pct, 4),
            "mitre_stages": sorted(set(blob["labels_mitre"].tolist())),
        }
        print(f"{split}: {summary[split]['windows']} windows from {len(files)} pcaps "
              f"(attack {attack_pct:.1%})")

    (OUT_DIR / "meta.json").write_text(json.dumps({**meta, "splits": summary}, indent=2))
    print(f"Wrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
