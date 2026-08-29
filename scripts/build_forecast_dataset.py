"""Build continuous per-capture state trajectories for one-step forecasting.

Unlike the classification dataset (10-flow chopped windows), this keeps each
capture as one continuous trajectory of length T (its full flow count), so
consecutive positions are genuinely t -> t+1 in real time, not two independent
samples. blocks_to_v2 / blocks_to_aryan aggregate a trailing window ending at
each position regardless of total sequence length, so they apply directly to a
whole capture without chunking.

Output: data/processed/forecast_captures.pkl
  list of {"cid", "cls", "split", "source", "v2": (T,64), "amt": (T,82)}
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_comparison_dataset import (  # noqa: E402
    collect_csv_captures,
    collect_pcap_captures,
    split_captures,
)
from src.model.attack_catalog import get_class_to_idx  # noqa: E402
from src.pipeline.extract_aryan import blocks_to_aryan  # noqa: E402
from src.pipeline.extract_v2 import blocks_to_v2  # noqa: E402

OUT = ROOT / "data" / "processed" / "forecast_captures.pkl"


def main() -> None:
    print("Collecting captures...")
    pcap_caps = collect_pcap_captures()
    csv_caps = collect_csv_captures()
    captures = pcap_caps + csv_caps
    class_to_idx = get_class_to_idx()
    assignment = split_captures(captures)

    out = []
    skipped = 0
    for cid, cls, base in captures:
        if cls not in class_to_idx or len(base) < 3:
            skipped += 1
            continue
        v2 = blocks_to_v2(base[None, :, :])[0]
        amt = blocks_to_aryan(base[None, :, :])[0]
        out.append({
            "cid": cid, "cls": cls, "split": assignment[cid],
            "source": "csv" if cid.startswith("csv_") else "pcap",
            "v2": v2.astype(np.float32), "amt": amt.astype(np.float32),
        })

    print(f"Captures kept: {len(out)}  (skipped {skipped}, too short for a pair)")
    lengths = [len(c["v2"]) for c in out]
    print(f"Trajectory length: min={min(lengths)} median={int(np.median(lengths))} "
          f"max={max(lengths)}  total flows={sum(lengths)}")
    for s in ("train", "val", "test"):
        n = sum(1 for c in out if c["split"] == s)
        pairs = sum(len(c["v2"]) - 1 for c in out if c["split"] == s)
        print(f"  {s:<6} {n:>4} captures  {pairs:>6} forecast pairs")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
