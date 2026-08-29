"""Measure flows-per-capture so the v2 window size can be chosen from data."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.attack_catalog import resolve_pcap_class  # noqa: E402
from src.pipeline.extract import pcap_to_rows  # noqa: E402

ADV = ROOT / "data" / "raw" / "adversarial"


def main() -> None:
    pcaps = sorted(ADV.rglob("*.pcap"))
    print(f"{len(pcaps)} pcaps found")

    per_class: dict[str, list[int]] = defaultdict(list)
    unlabeled = 0
    for p in pcaps:
        cls = resolve_pcap_class(p.name)
        if cls is None:
            unlabeled += 1
            continue
        per_class[cls].append(len(pcap_to_rows(p)))

    print(f"unlabeled: {unlabeled}\n")
    print(f"{'class':<32}{'caps':>5}{'flows_tot':>10}{'min':>6}{'med':>7}{'max':>7}")
    print("-" * 67)
    totals = []
    for cls in sorted(per_class):
        counts = np.array(per_class[cls])
        totals.extend(counts.tolist())
        print(
            f"{cls:<32}{len(counts):>5}{counts.sum():>10}"
            f"{counts.min():>6}{int(np.median(counts)):>7}{counts.max():>7}"
        )

    t = np.array(totals)
    print("-" * 67)
    print(f"{'ALL':<32}{len(t):>5}{t.sum():>10}{t.min():>6}{int(np.median(t)):>7}{t.max():>7}")
    print(f"\npercentiles of flows/capture: "
          f"p10={np.percentile(t,10):.0f} p25={np.percentile(t,25):.0f} "
          f"p50={np.percentile(t,50):.0f} p75={np.percentile(t,75):.0f} "
          f"p90={np.percentile(t,90):.0f}")
    print(f"captures with >=8 flows : {(t>=8).sum()}/{len(t)}")
    print(f"captures with >=16 flows: {(t>=16).sum()}/{len(t)}")
    print(f"captures with >=32 flows: {(t>=32).sum()}/{len(t)}")


if __name__ == "__main__":
    main()
