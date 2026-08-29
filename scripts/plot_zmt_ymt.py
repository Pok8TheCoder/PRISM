"""Render the ZMT.01 / YMT.01 / AMT.01 comparison figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RES = ROOT / "results" / "zmt_ymt"
COLOR = {"ZMT.01": "#e2574c", "YMT.01": "#2e86c1", "AMT.01": "#7d3c98"}
LABEL = {
    "ZMT.01": "ZMT.01  (v1, 27f)",
    "YMT.01": "YMT.01  (v2, 64f)",
    "AMT.01": "AMT.01  (aryan StateBuilder, 82f)",
}


def main() -> None:
    m = json.load(open(RES / "metrics.json"))
    a = json.load(open(RES / "analysis.json"))
    models = [n for n in ("ZMT.01", "YMT.01", "AMT.01") if n in m["summary"]]
    w = 0.8 / len(models)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("PRISM  ·  feature-schema head-to-head on identical traffic",
                 fontsize=16, fontweight="bold")

    # ── Headline metrics ──
    ax = axes[0, 0]
    keys = ["accuracy", "f1_macro", "binary_f1", "binary_precision"]
    labels = ["Accuracy\n(33-class)", "Macro F1", "Binary F1", "Binary\nprecision"]
    x = np.arange(len(keys))
    for i, n in enumerate(models):
        off = (i - len(models) / 2 + 0.5) * w
        vals = [m["summary"][n][k]["mean"] for k in keys]
        errs = [m["summary"][n][k]["std"] for k in keys]
        ax.bar(x + off, vals, w * 0.92, yerr=errs, capsize=3,
               label=LABEL[n], color=COLOR[n])
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 0.02, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Test metrics (mean ± std, 3 seeds) — higher is better")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)

    # ── Lab captures + scan family ──
    ax = axes[0, 1]
    groups = ["PCAP-only\naccuracy", "PCAP-only\nmacro F1", "Scan/flood\nfamily F1"]
    x = np.arange(3)
    for i, n in enumerate(models):
        off = (i - len(models) / 2 + 0.5) * w
        vals = [a["pcap_only"][n]["accuracy"], a["pcap_only"][n]["f1_macro"],
                a["scan_family_mean"][n]]
        ax.bar(x + off, vals, w * 0.92, label=LABEL[n], color=COLOR[n])
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, groups, fontsize=9)
    ax.set_ylim(0.8, 1.05)
    ax.set_title("Lab captures and the scan/flood family")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # ── False positives ──
    ax = axes[1, 0]
    fp = a["false_positives"]
    x = np.arange(len(models))
    correct = [fp[n]["correct"] for n in models]
    to_inf = [fp[n]["to_infiltration"] for n in models]
    to_oth = [fp[n]["to_other_attacks"] for n in models]
    ax.bar(x, correct, 0.55, label="correctly benign", color="#27ae60")
    ax.bar(x, to_inf, 0.55, bottom=correct,
           label="→ CIC Infiltration (known-ambiguous)", color="#f39c12")
    ax.bar(x, to_oth, 0.55, bottom=np.array(correct) + np.array(to_inf),
           label="→ other attack (true false alarm)", color="#c0392b")
    for i, n in enumerate(models):
        ax.text(i, 505, f"{to_oth[i]} false alarms\nFPR* {fp[n]['fpr_excl_infiltration']:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, models)
    ax.set_ylabel("benign test windows")
    ax.set_ylim(0, 600)
    ax.set_title("Where the 488 benign windows go   (FPR* excludes Infiltration)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    # ── Per-class, largest movers ──
    ax = axes[1, 1]
    rows = sorted(a["per_class"], key=lambda r: -abs(r["delta"]))[:14]
    rows.sort(key=lambda r: r["delta"])
    y = np.arange(len(rows))
    h = 0.8 / len(models)
    for i, n in enumerate(models):
        off = (i - len(models) / 2 + 0.5) * h
        ax.barh(y + off, [r[n] for r in rows], h * 0.9,
                label=LABEL[n], color=COLOR[n])
    ax.set_yticks(y, [r["class"] for r in rows], fontsize=7.5)
    ax.set_xlabel("per-class F1")
    ax.set_xlim(0, 1.05)
    ax.set_title("Classes where the schemas disagree most")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out = RES / "comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
