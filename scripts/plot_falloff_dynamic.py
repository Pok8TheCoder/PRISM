#!/usr/bin/env python3
"""Plot overlap solo vs blend vs fixed baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "results" / "falloff_dynamic" / "falloff_dynamic.json"
OUT = ROOT / "results" / "falloff_dynamic" / "falloff_dynamic.png"

COLORS = {
    "overlap_solo": "#a371f7",
    "overlap_blend": "#d29922",
    "fixed_5s": "#3fb950",
    "fixed_30s": "#58a6ff",
}
LABELS = {
    "overlap_solo": "Overlap SOLO",
    "overlap_blend": "Overlap BLEND (mean of 2 finest)",
    "fixed_5s": "Fixed 5s",
    "fixed_30s": "Fixed 30s",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=JSON_PATH)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    if not args.json.exists():
        print("Run: python scripts/falloff_dynamic_tiers.py")
        return 1

    data = json.loads(args.json.read_text())
    tiers = data.get("tiers", [])

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0d1117")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 1, 1.1], hspace=0.35, wspace=0.28)

    # Schedule diagram
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_xlim(0, 20)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.add_patch(plt.Rectangle((0, 0.55), 10, 0.25, color="#238636", alpha=0.9))
    ax0.text(5, 0.675, "10 min CONTEXT", ha="center", color="white", fontsize=10, fontweight="bold")
    colors = ["#f85149", "#d29922", "#a371f7", "#58a6ff"]
    x0 = 10.0
    for i, t in enumerate(tiers):
        w = (t["solo_end"] - t["solo_start"] + 1) / 60.0 * (10 / 20)
        w = max(0.4, min(w, 10 - (x0 - 10)))
        if x0 >= 20:
            break
        ax0.add_patch(plt.Rectangle((x0, 0.35), w, 0.35, color=colors[i % 4], alpha=0.85))
        ax0.text(x0 + w / 2, 0.52, f"{t['window_sec']:.0f}s solo\n[{t['solo_start']:.0f}-{t['solo_end']:.0f}s]",
                 ha="center", va="center", color="white", fontsize=8)
        x0 += w
    ax0.text(10, 0.82, "Overlapping tiers — coarse models also predict into fine zones (blend = mean of 2 finest)",
             color="#8b949e", fontsize=9)

    # Median bar chart
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor("#161b22")
    keys = ["overlap_solo", "overlap_blend", "fixed_5s", "fixed_30s"]
    meds = [data[k].get("falloff_min_median") or 0 for k in keys]
    ax1.bar(range(4), meds, color=[COLORS[k] for k in keys])
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([LABELS[k].replace(" ", "\n") for k in keys], color="white", fontsize=8)
    ax1.set_ylabel("Median falloff (min)", color="white")
    ax1.set_title("Median falloff into 20-min forecast", color="white", loc="left")
    ax1.axhline(20, color="#f85149", ls=":", lw=1)
    ax1.tick_params(colors="white")

    # Histogram solo vs blend
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor("#161b22")
    bins = np.linspace(0, 20, 21)
    for key in ("overlap_solo", "overlap_blend"):
        mins = [f["falloff_min"] for f in data[key]["features"] if f.get("falloff_min")]
        ax2.hist(mins, bins=bins, alpha=0.55, color=COLORS[key], label=LABELS[key])
    ax2.legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    ax2.set_xlabel("Min to falloff", color="white")
    ax2.tick_params(colors="white")
    ax2.set_title("SOLO vs BLEND distribution", color="white", loc="left")

    # Example error curves
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor("#161b22")
    f0 = data["overlap_solo"]["features"][0]["feature_name"]
    for key in keys:
        err = data[key]["features"][0].get("per_step_err")
        if not err:
            continue
        if key.startswith("fixed"):
            step_sec = 5 if "5s" in key else 30
            t = np.arange(1, len(err) + 1) * step_sec / 60.0
        else:
            t = np.arange(1, len(err) + 1) / 60.0
        ax3.plot(t, err, color=COLORS[key], lw=1.5, label=LABELS[key], alpha=0.9)
    ax3.set_xlim(0, 20)
    ax3.set_xlabel("Minutes into forecast", color="white")
    ax3.set_ylabel(f"|error| ({f0})", color="white")
    ax3.legend(facecolor="#161b22", labelcolor="white", ncol=2)
    ax3.tick_params(colors="white")
    ax3.set_title("Example feature error", color="white", loc="left")

    fig.suptitle("Overlapping dynamic tiers: SOLO vs BLEND  |  10 min ctx → 20 min forecast",
                 color="white", fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
