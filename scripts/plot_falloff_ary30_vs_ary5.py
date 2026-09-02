#!/usr/bin/env python3
"""Plot 10min context + 20min forecast falloff comparison."""

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
DEFAULT_JSON = ROOT / "results" / "falloff_ary30_vs_ary5" / "falloff_report.json"
OUT_DIR = ROOT / "results" / "falloff_ary30_vs_ary5"

COL_30 = "#58a6ff"
COL_5 = "#3fb950"
COL_BG = "#0d1117"
COL_PANEL = "#161b22"
COL_GRID = "#30363d"
HORIZON_MIN = 20.0


def style_ax(ax):
    ax.set_facecolor(COL_PANEL)
    for spine in ax.spines.values():
        spine.set_color(COL_GRID)
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.2, color=COL_GRID)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=DEFAULT_JSON)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if not args.json.exists():
        print(f"Missing {args.json}. Run: python scripts/falloff_ary30_vs_ary5.py")
        return 1

    data = json.loads(args.json.read_text())
    r30, r5 = data["ary30"], data["ary5"]
    cmp = data["comparison"]

    min30 = [f["falloff_min_into_forecast"] for f in r30["features"] if f.get("falloff_min_into_forecast") is not None]
    min5 = [f["falloff_min_into_forecast"] for f in r5["features"] if f.get("falloff_min_into_forecast") is not None]

    paired = [
        (d["ary30_falloff_sec"] / 60.0, d["ary5_falloff_sec"] / 60.0)
        for d in cmp["features"]
        if d["ary30_falloff_sec"] is not None and d["ary5_falloff_sec"] is not None
    ]

    # --- Main figure ---
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor(COL_BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 1.2, 1.2], hspace=0.38, wspace=0.28)

    # Scenario strip
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(COL_PANEL)
    ax0.set_xlim(0, 30)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.add_patch(plt.Rectangle((0, 0.35), 10, 0.3, color="#238636", alpha=0.85))
    ax0.add_patch(plt.Rectangle((10, 0.35), 20, 0.3, color="#1f6feb", alpha=0.85))
    ax0.text(5, 0.5, "10 min CONTEXT\n(real observed windows)", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax0.text(20, 0.5, "20 min FORECAST\n(open-loop rollout)", ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    ax0.annotate("", xy=(10, 0.15), xytext=(0, 0.15), arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax0.annotate("", xy=(30, 0.15), xytext=(10, 0.15), arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax0.text(15, 0.08, "30 min total timeline  |  falloff = when forecast error exceeds 2x baseline inside the 20 min window",
             ha="center", color="#8b949e", fontsize=10)
    ax0.text(5, 0.82, f"ARY.01: {r30['context_steps']} ctx + {r30['horizon_steps']} fcst steps @ 30s", color=COL_30, fontsize=9)
    ax0.text(20, 0.82, f"ARY-5s: {r5['context_steps']} ctx + {r5['horizon_steps']} fcst steps @ 5s", color=COL_5, fontsize=9)

    # A: minutes into forecast
    ax1 = fig.add_subplot(gs[1, 0])
    style_ax(ax1)
    bins = np.linspace(0, HORIZON_MIN, 25)
    ax1.hist(min30, bins=bins, alpha=0.65, color=COL_30, label=f"ARY.01 (n={len(min30)})", edgecolor="none")
    ax1.hist(min5, bins=bins, alpha=0.65, color=COL_5, label=f"ARY-5sV01 (n={len(min5)})", edgecolor="none")
    med30 = r30["falloff_sec_median"] / 60.0 if r30["falloff_sec_median"] else None
    med5 = r5["falloff_sec_median"] / 60.0 if r5["falloff_sec_median"] else None
    if med30:
        ax1.axvline(med30, color=COL_30, ls="--", lw=2, label=f"30s median {med30:.1f} min")
    if med5:
        ax1.axvline(med5, color=COL_5, ls="--", lw=2, label=f"5s median {med5:.1f} min")
    ax1.axvline(HORIZON_MIN, color="#f85149", ls=":", lw=1.5, label="20 min horizon end")
    ax1.set_xlabel("Minutes into 20-min forecast before falloff", color="white")
    ax1.set_ylabel("# features", color="white")
    ax1.set_title("A · When trust breaks (inside forecast window)", color="white", loc="left")
    ax1.legend(fontsize=8, facecolor=COL_PANEL, edgecolor=COL_GRID, labelcolor="white")

    # B: paired scatter (minutes)
    ax2 = fig.add_subplot(gs[1, 1])
    style_ax(ax2)
    if paired:
        px30, px5 = zip(*paired)
        ax2.scatter(px30, px5, c="#e6edf3", alpha=0.45, s=24, edgecolors="none")
        ax2.plot([0, HORIZON_MIN], [0, HORIZON_MIN], color="#f85149", ls="--", lw=1.2, label="equal")
        below = sum(1 for a, b in paired if b < a)
        ax2.text(0.04, 0.96, f"5s falls off sooner: {below}/{len(paired)}", transform=ax2.transAxes,
                 color=COL_5, fontsize=10, va="top")
    ax2.set_xlim(0, HORIZON_MIN)
    ax2.set_ylim(0, HORIZON_MIN)
    ax2.set_xlabel("ARY.01 falloff (min into forecast)", color="white")
    ax2.set_ylabel("ARY-5sV01 falloff (min into forecast)", color="white")
    ax2.set_title("B · Per-feature paired (20-min window)", color="white", loc="left")
    ax2.legend(fontsize=8, facecolor=COL_PANEL, edgecolor=COL_GRID, labelcolor="white")

    # C: example error curves (top-variance feature)
    ax3 = fig.add_subplot(gs[2, 0])
    style_ax(ax3)
    f30 = r30["features"][0]
    f5 = r5["features"][0]
    if f30.get("per_step_err") and f5.get("per_step_err"):
        t30 = np.arange(1, len(f30["per_step_err"]) + 1) * r30["window_sec"] / 60.0
        t5 = np.arange(1, len(f5["per_step_err"]) + 1) * r5["window_sec"] / 60.0
        ax3.plot(t30, f30["per_step_err"], color=COL_30, lw=1.8, label=f"ARY.01 ({f30['feature_name']})")
        ax3.plot(t5, f5["per_step_err"], color=COL_5, lw=1.8, label=f"ARY-5s ({f5['feature_name']})")
        ax3.axhline(f30["threshold"], color=COL_30, ls=":", alpha=0.7)
        ax3.axhline(f5["threshold"], color=COL_5, ls=":", alpha=0.7)
        if f30["falloff_min_into_forecast"]:
            ax3.axvline(f30["falloff_min_into_forecast"], color=COL_30, ls="--", alpha=0.8)
        if f5["falloff_min_into_forecast"]:
            ax3.axvline(f5["falloff_min_into_forecast"], color=COL_5, ls="--", alpha=0.8)
    ax3.set_xlim(0, HORIZON_MIN)
    ax3.set_xlabel("Minutes into forecast", color="white")
    ax3.set_ylabel("|error| (example feature)", color="white")
    ax3.set_title("C · Example error vs time (highest-variance feature)", color="white", loc="left")
    ax3.legend(fontsize=8, facecolor=COL_PANEL, edgecolor=COL_GRID, labelcolor="white")

    # D: summary bars + never falloff
    ax4 = fig.add_subplot(gs[2, 1])
    style_ax(ax4)
    cats = ["Median\n(min)", "Mean\n(min)", "Never falloff\n(/242)"]
    v30 = [
        (r30["falloff_sec_median"] or 0) / 60.0,
        (r30["falloff_sec_mean"] or 0) / 60.0,
        r30["n_never_falloff"],
    ]
    v5 = [
        (r5["falloff_sec_median"] or 0) / 60.0,
        (r5["falloff_sec_mean"] or 0) / 60.0,
        r5["n_never_falloff"],
    ]
    x = np.arange(3)
    w = 0.35
    ax4.bar(x[:2] - w / 2, v30[:2], w, color=COL_30, label="ARY.01 30s")
    ax4.bar(x[:2] + w / 2, v5[:2], w, color=COL_5, label="ARY-5sV01")
    ax4b = ax4.twinx()
    ax4b.bar(x[2] - w / 2, v30[2], w, color=COL_30, alpha=0.5)
    ax4b.bar(x[2] + w / 2, v5[2], w, color=COL_5, alpha=0.5)
    ax4b.set_ylabel("Count (never falloff)", color="white")
    ax4b.tick_params(colors="white")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cats, color="white")
    ax4.set_ylabel("Minutes into forecast", color="white")
    ax4.set_title("D · Summary", color="white", loc="left")
    ax4.legend(fontsize=9, loc="upper left", facecolor=COL_PANEL, edgecolor=COL_GRID, labelcolor="white")

    fig.suptitle(
        "10 min context → 20 min forecast  |  ARY.01 (30s) vs ARY-5sV01 (5s)",
        color="white", fontsize=14, y=0.995,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "falloff_comparison.png"
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
