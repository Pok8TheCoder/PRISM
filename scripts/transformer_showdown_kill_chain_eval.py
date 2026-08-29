"""The GRU-vs-Transformer showdown, with RAM.01's memory+TTT layer applied to
both backbones, on the identical synthetic ~1000-step multi-attack kill-chain
timeline used in v6 (context=60, horizon=40).

Four candidates, v2 (YMT) schema only (this is a same-schema, apples-to-apples
comparison -- Temporal-A used a different feature schema in v6 and is
intentionally left out here):

  Temporal-Y   -- frozen GRU forecaster                  (backbone: TemporalForecaster)
  TFT.01       -- frozen Transformer forecaster           (backbone: TransformerForecaster)
  RAM.01       -- GRU        + test-time training + episodic memory
  RAMT.01      -- Transformer + test-time training + episodic memory

RAM.01's `EpisodicMemoryBank` / `OnlineAdaptive` (scripts/adaptive_memory_forecaster.py)
are already model-agnostic -- they just call `model(seq)` and backprop through
whatever nn.Module they're given -- so RAMT.01 needed zero new memory/TTT
code, only a Transformer backbone with the same forward(seq) -> pred contract
(see TransformerForecaster in scripts/train_transformer_forecaster.py).

This directly tests the hypothesis from the "do we have a chance to beat
Temporal Transformer-like models with RAM?" discussion: that RAM's real edge
is the memory+adaptation layer, which should be backbone-agnostic and help
a Transformer roughly as much as it helped the GRU.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adaptive_memory_forecaster import EpisodicMemoryBank, OnlineAdaptive  # noqa: E402,F401 (EpisodicMemoryBank used by seed_memory return type)
from scripts.ram01_kill_chain_eval import (  # noqa: E402
    CONTEXT,
    HORIZON,
    PICK_V2,
    build_timeline,
    calibrate_match_thresh,
    calibrate_val_error,
    run_frozen,
    run_ram01,
    seed_memory,
    segment_mse,
)
from scripts.train_temporal_forecaster import TemporalForecaster  # noqa: E402
from scripts.train_transformer_forecaster import TransformerForecaster  # noqa: E402
from src.pipeline.features_v2 import FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "results" / "forecast" / "v7_transformer_showdown"

ADAPT_LR = 3e-4
ADAPT_STEPS = 3
PULLBACK = 5e-3


def load_gru(tag: str, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_temporal_ctx60.pth", map_location=device, weights_only=False)
    m = TemporalForecaster(ck["dim"], torch.tensor(ck["max_step"], dtype=torch.float32)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["scaler_mean"], ck["scaler_scale"]


def load_transformer(tag: str, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_transformer_ctx60.pth", map_location=device, weights_only=False)
    m = TransformerForecaster(ck["dim"], torch.tensor(ck["max_step"], dtype=torch.float32),
                               context=ck.get("context", CONTEXT)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["scaler_mean"], ck["scaler_scale"]


def overall_mse(pred, true, mean, scale):
    p, t = (pred - mean) / scale, (true - mean) / scale
    return float(np.mean((p - t) ** 2))


def prep_ram_candidate(name, model, mean, scale, captures):
    print(f"\nCalibrating RAM anomaly threshold ({name})...")
    anomaly_thresh = calibrate_val_error(model, mean, scale, captures, "v2", CONTEXT)
    print(f"  anomaly threshold = {anomaly_thresh:.4f}")
    print(f"Seeding episodic memory from TRAIN captures ({name})...")
    mem_bank = seed_memory(model, mean, scale, captures, CONTEXT, HORIZON, anomaly_thresh)
    print(f"  by class: {dict(Counter(e['label'] for e in mem_bank.entries).most_common(10))}")
    match_thresh = calibrate_match_thresh(mem_bank)
    return mem_bank, anomaly_thresh, match_thresh


def report_retrieval(name, retrievals):
    matched = [r for r in retrievals if r["match"]]
    correct = [r for r in matched if r["retrieved_label"] == r["true_label"]]
    acc = 100 * len(correct) / len(matched) if matched else 0.0
    print(f"  {name}: {len(retrievals)} surprises, {len(matched)} confident matches, "
          f"{len(correct)} correct ({acc:.1f}% top-1 accuracy)")
    return {"n_surprises": len(retrievals), "n_matched": len(matched),
            "n_correct": len(correct), "accuracy": acc / 100 if matched else None}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    with open(DATA, "rb") as f:
        captures = pickle.load(f)

    print("Loading frozen GRU (Temporal-Y) and frozen Transformer (TFT.01), both v2/ctx60...")
    gru_model, gru_mean, gru_scale = load_gru("v2", device)
    tft_model, tft_mean, tft_scale = load_transformer("v2", device)
    print(f"  GRU params:         {sum(p.numel() for p in gru_model.parameters()):,}")
    print(f"  Transformer params: {sum(p.numel() for p in tft_model.parameters()):,}")

    print("\nBuilding synthetic ~1000-step multi-attack timeline (v2 schema)...")
    full_v2, labels, segments = build_timeline(captures, "v2")

    gru_bank, gru_anom_t, gru_match_t = prep_ram_candidate("RAM.01 / GRU", gru_model, gru_mean, gru_scale, captures)
    tft_bank, tft_anom_t, tft_match_t = prep_ram_candidate("RAMT.01 / Transformer", tft_model, tft_mean, tft_scale, captures)

    print("\nRunning frozen Temporal-Y (GRU) across the timeline...")
    gru_pred, gru_true, gru_steps = run_frozen(gru_model, gru_mean, gru_scale, full_v2, CONTEXT, HORIZON)
    print("Running frozen TFT.01 (Transformer) across the timeline...")
    tft_pred, tft_true, tft_steps = run_frozen(tft_model, tft_mean, tft_scale, full_v2, CONTEXT, HORIZON)
    print("Running RAM.01 (GRU + TTT + memory) across the timeline...")
    ram_pred, ram_true, ram_steps, ram_anomalies, ram_retrievals = run_ram01(
        gru_model, gru_mean, gru_scale, full_v2, labels, CONTEXT, HORIZON, gru_bank,
        gru_anom_t, gru_match_t, adapt_lr=ADAPT_LR, adapt_steps=ADAPT_STEPS, pullback=PULLBACK)
    print("Running RAMT.01 (Transformer + TTT + memory) across the timeline...")
    ramt_pred, ramt_true, ramt_steps, ramt_anomalies, ramt_retrievals = run_ram01(
        tft_model, tft_mean, tft_scale, full_v2, labels, CONTEXT, HORIZON, tft_bank,
        tft_anom_t, tft_match_t, adapt_lr=ADAPT_LR, adapt_steps=ADAPT_STEPS, pullback=PULLBACK)

    mse = {
        "Temporal-Y (GRU, frozen)": overall_mse(gru_pred, gru_true, gru_mean, gru_scale),
        "TFT.01 (Transformer, frozen)": overall_mse(tft_pred, tft_true, tft_mean, tft_scale),
        "RAM.01 (GRU+memory)": overall_mse(ram_pred, ram_true, gru_mean, gru_scale),
        "RAMT.01 (Transformer+memory)": overall_mse(ramt_pred, ramt_true, tft_mean, tft_scale),
    }
    seg = {
        "Temporal-Y (GRU, frozen)": segment_mse(gru_pred, gru_true, gru_steps, segments, HORIZON, gru_mean, gru_scale),
        "TFT.01 (Transformer, frozen)": segment_mse(tft_pred, tft_true, tft_steps, segments, HORIZON, tft_mean, tft_scale),
        "RAM.01 (GRU+memory)": segment_mse(ram_pred, ram_true, ram_steps, segments, HORIZON, gru_mean, gru_scale),
        "RAMT.01 (Transformer+memory)": segment_mse(ramt_pred, ramt_true, ramt_steps, segments, HORIZON, tft_mean, tft_scale),
    }

    print("\n" + "=" * 90)
    print("Overall forecast MSE (standardized units, v2 schema, all four directly comparable):")
    for k, v in mse.items():
        print(f"  {k:32s} mse={v:.4f}")

    gru_delta = 100 * (mse["RAM.01 (GRU+memory)"] - mse["Temporal-Y (GRU, frozen)"]) / mse["Temporal-Y (GRU, frozen)"]
    tft_delta = 100 * (mse["RAMT.01 (Transformer+memory)"] - mse["TFT.01 (Transformer, frozen)"]) / mse["TFT.01 (Transformer, frozen)"]
    print(f"\nDoes memory+TTT help each backbone by a similar margin?")
    print(f"  GRU:         frozen -> +memory  =  {gru_delta:+.1f}%")
    print(f"  Transformer: frozen -> +memory  =  {tft_delta:+.1f}%")

    print("\nPer-segment MSE (each model normalized by its OWN overall mean, i.e. relative difficulty):")
    for lbl in ["Benign", "T1595_active_scan", "T1499_http_flood", "T1498_network_dos"]:
        row = "  " + f"{lbl:22s}"
        for k in mse:
            val = seg[k].get(lbl, float("nan")) / mse[k]
            row += f"  {k.split(' ')[0]:12s}={val:5.2f}x"
        print(row)

    print("\nMemory retrieval accuracy:")
    ram_retr_summary = report_retrieval("RAM.01 (GRU)", ram_retrievals)
    ramt_retr_summary = report_retrieval("RAMT.01 (Transformer)", ramt_retrievals)

    summary = {
        "overall_mse": mse,
        "segment_mse": seg,
        "segments": segments,
        "memory_delta_pct": {"GRU": gru_delta, "Transformer": tft_delta},
        "retrieval": {"RAM.01": ram_retr_summary, "RAMT.01": ramt_retr_summary},
        "n_params": {
            "GRU": sum(p.numel() for p in gru_model.parameters()),
            "Transformer": sum(p.numel() for p in tft_model.parameters()),
        },
    }
    with open(OUT_DIR / "showdown_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    colors_seg = {"Benign": "#eef6ee", "T1595_active_scan": "#fde2e2",
                  "T1499_http_flood": "#e2ecfd", "T1498_network_dos": "#fdf0d5"}

    def idx_map(steps, arr):
        return np.array([i for st in steps for i in range(st, st + HORIZON)])[:len(arr)]

    im_gru, im_tft = idx_map(gru_steps, gru_pred), idx_map(tft_steps, tft_pred)
    im_ram, im_ramt = idx_map(ram_steps, ram_pred), idx_map(ramt_steps, ramt_pred)

    fig, axes = plt.subplots(len(PICK_V2), 1, figsize=(15, 2.8 * len(PICK_V2)), sharex=True)
    fig.suptitle("Frozen GRU vs Frozen Transformer  ·  synthetic multi-attack timeline "
                 f"(context={CONTEXT}, horizon={HORIZON})", fontsize=12, fontweight="bold")
    for ax, name in zip(axes, PICK_V2):
        j = FEATURE_COLS_V2.index(name)
        for lbl, a, b in segments:
            if lbl != "Benign":
                ax.axvspan(a, b, color=colors_seg[lbl], alpha=0.6, zorder=0)
        ax.plot(range(len(full_v2)), full_v2[:, j], color="black", lw=1.2, label="actual", zorder=3)
        ax.plot(im_gru, gru_pred[:, j], color="#e67e22", lw=1.0, ls=":", label="Temporal-Y (GRU, frozen)", zorder=2)
        ax.plot(im_tft, tft_pred[:, j], color="#8e44ad", lw=1.0, ls="--", label="TFT.01 (Transformer, frozen)", zorder=2)
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlabel("step across ~1000-step synthetic session (shaded = inserted attack)")
    plt.tight_layout()
    out1 = OUT_DIR / "showdown_frozen_gru_vs_transformer.png"
    fig.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out1}")

    fig, axes = plt.subplots(len(PICK_V2), 1, figsize=(15, 2.8 * len(PICK_V2)), sharex=True)
    fig.suptitle("RAM.01 (GRU+memory) vs RAMT.01 (Transformer+memory)  ·  same timeline "
                 f"(context={CONTEXT}, horizon={HORIZON})", fontsize=12, fontweight="bold")
    for ax, name in zip(axes, PICK_V2):
        j = FEATURE_COLS_V2.index(name)
        for lbl, a, b in segments:
            if lbl != "Benign":
                ax.axvspan(a, b, color=colors_seg[lbl], alpha=0.6, zorder=0)
        ax.plot(range(len(full_v2)), full_v2[:, j], color="black", lw=1.2, label="actual", zorder=3)
        ax.plot(im_ram, ram_pred[:, j], color="#27ae60", lw=1.2, label="RAM.01 (GRU+memory)", zorder=4)
        ax.plot(im_ramt, ramt_pred[:, j], color="#2980b9", lw=1.2, ls="--", label="RAMT.01 (Transformer+memory)", zorder=4)
        for r in ram_retrievals:
            if r["match"]:
                ax.axvline(r["step"], color="green", alpha=0.3, lw=1, zorder=1)
        for r in ramt_retrievals:
            if r["match"]:
                ax.axvline(r["step"], color="blue", alpha=0.3, lw=1, zorder=1)
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("green vline = RAM.01 memory match used | blue vline = RAMT.01 memory match used")
    plt.tight_layout()
    out2 = OUT_DIR / "showdown_ram01_vs_ramt01.png"
    fig.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out2}")
    print(f"Saved summary -> {OUT_DIR / 'showdown_summary.json'}")


if __name__ == "__main__":
    main()
