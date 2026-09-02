"""Sensitivity sweeps for RAM activation threshold and forecast horizon.

Models (5):
  ary_base, ary_V8, ary_V10, timesfm, timesfm_ram

Sweep A — attack-gate threshold (when RAM memory write/blend is allowed):
  Fixed context=100, horizon=200. Thresholds 0.0 = RAM always on (ungated),
  1.0 = RAM never on. V8/V10/timesfm_ram retested at each threshold.

Sweep B — forecast horizon:
  Fixed context=100. Horizons 20..300. All 5 models at each horizon (ungated RAM).

Outputs:
  results/ram_improve/sensitivity/gate_threshold/
  results/ram_improve/sensitivity/horizon/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.timesfm_ram_compare import (  # noqa: E402
    ARY_LOOKBACK,
    RunConfig,
    ary_state_rollout,
    build_timeline,
    calibrate_thresholds,
    labels_to_ids,
    load_all_splits,
    load_model,
    timesfm_rollout,
)
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402

OUT = ROOT / "results" / "ram_improve" / "sensitivity"
CONTEXT = 100
FEATURE_IDX = 3
GATE_THRESHOLDS = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
HORIZONS = [20, 50, 100, 200, 300]


def mse_1d(pred: np.ndarray, actual: np.ndarray) -> tuple[float, int]:
    mask = ~np.isnan(pred)
    if mask.sum() == 0:
        return float("nan"), 0
    return float(np.mean((pred[mask] - actual[mask]) ** 2)), int(mask.sum())


def run_five_models(
    full,
    bin_labels,
    mit_labels,
    series,
    feature_idx,
    context,
    horizon,
    base_model,
    forecaster,
    raw_thresh,
    hidden_thresh,
    anomaly_thresh,
    *,
    attack_gate: float = 0.5,
    gate_ram: bool = False,
) -> dict[str, dict]:
    """Run all 5 models; RAM variants use attack gate when gate_ram=True."""
    feat_name = FEATURE_COLS_242[feature_idx]
    cfg = RunConfig(
        context=context, horizon=horizon, feature_idx=feature_idx,
        feature_name=feat_name, gate_ram_on_attack=gate_ram, attack_gate=attack_gate,
    )

    out: dict[str, dict] = {}

    pred_base, gs = ary_state_rollout(
        base_model, full, bin_labels, mit_labels, cfg,
        adapt=False, loss_type="mse", use_memory=False, key_space="raw", knn_k=1,
        match_thresh=raw_thresh, anomaly_thresh=anomaly_thresh, gate_on_attack=False,
    )
    mse, n = mse_1d(pred_base[:, feature_idx], series)
    out["ary_base"] = {"mse": mse, "n_covered": n, "gate_stats": gs}

    for name, kwargs in [
        ("ary_V8", dict(adapt=False, loss_type="mse", use_memory=True, key_space="raw", knn_k=1)),
        ("ary_V10", dict(adapt=True, loss_type="multitask", use_memory=True, key_space="hidden", knn_k=3)),
    ]:
        thresh = hidden_thresh if kwargs["key_space"] == "hidden" else raw_thresh
        pred, gs = ary_state_rollout(
            base_model, full, bin_labels, mit_labels, cfg,
            match_thresh=thresh, anomaly_thresh=anomaly_thresh,
            gate_on_attack=gate_ram, **kwargs,
        )
        mse, n = mse_1d(pred[:, feature_idx], series)
        out[name] = {"mse": mse, "n_covered": n, "gate_stats": gs}

    pred_tfm, gs = timesfm_rollout(forecaster, series, cfg, use_memory=False)
    mse, n = mse_1d(pred_tfm, series)
    out["timesfm"] = {"mse": mse, "n_covered": n, "gate_stats": gs}

    pred_ram, gs = timesfm_rollout(
        forecaster, series, cfg, use_memory=True,
        gate_model=base_model, full_states=full, gate_on_attack=gate_ram,
    )
    mse, n = mse_1d(pred_ram, series)
    out["timesfm_ram"] = {"mse": mse, "n_covered": n, "gate_stats": gs}

    return out


def sweep_gate_threshold(full, bin_labels, mit_labels, series, feature_idx, base_model, forecaster,
                         raw_thresh, hidden_thresh, anomaly_thresh) -> list[dict]:
    rows = []
    # Baselines (no gate) — same at every threshold row for reference
    baseline = run_five_models(
        full, bin_labels, mit_labels, series, feature_idx, CONTEXT, 200,
        base_model, forecaster, raw_thresh, hidden_thresh, anomaly_thresh,
        gate_ram=False,
    )
    base_mse = baseline["ary_base"]["mse"]
    tfm_mse = baseline["timesfm"]["mse"]

    for thr in GATE_THRESHOLDS:
        t0 = time.time()
        gate_ram = thr > 0.0
        if thr == 0.0:
            results = baseline
        else:
            results = run_five_models(
                full, bin_labels, mit_labels, series, feature_idx, CONTEXT, 200,
                base_model, forecaster, raw_thresh, hidden_thresh, anomaly_thresh,
                attack_gate=thr, gate_ram=True,
            )
        row = {
            "attack_gate": thr,
            "gate_mode": "ungated" if thr == 0.0 else ("disabled" if thr >= 1.0 else "gated"),
            "elapsed_sec": round(time.time() - t0, 1),
            "models": {},
        }
        for model, data in results.items():
            mse = data["mse"]
            gs = data.get("gate_stats", {})
            ref = base_mse if model.startswith("ary") else tfm_mse
            row["models"][model] = {
                "mse": mse,
                "n_covered": data["n_covered"],
                "delta_vs_noram_ref": mse - ref if model in ("ary_V8", "ary_V10", "timesfm_ram") else 0.0,
                "pct_vs_noram_ref": (mse / ref - 1.0) * 100 if model in ("ary_V8", "ary_V10", "timesfm_ram") and ref else 0.0,
                "gate_stats": gs,
            }
        rows.append(row)
        v8 = row["models"]["ary_V8"]["mse"]
        ram = row["models"]["timesfm_ram"]["mse"]
        print(f"  gate={thr:.2f}  V8={v8:.0f}  timesfm_ram={ram:.0f}  "
              f"writes_v8={row['models']['ary_V8']['gate_stats'].get('memory_writes', 0)}")
    return rows


def sweep_horizon(full, bin_labels, mit_labels, series, feature_idx, base_model, forecaster,
                  raw_thresh, hidden_thresh, anomaly_thresh) -> list[dict]:
    rows = []
    for hor in HORIZONS:
        t0 = time.time()
        anomaly_h = anomaly_thresh  # reuse; could recalibrate per horizon
        results = run_five_models(
            full, bin_labels, mit_labels, series, feature_idx, CONTEXT, hor,
            base_model, forecaster, raw_thresh, hidden_thresh, anomaly_h,
            gate_ram=False,
        )
        row = {"horizon": hor, "elapsed_sec": round(time.time() - t0, 1), "models": {}}
        base_mse = results["ary_base"]["mse"]
        tfm_mse = results["timesfm"]["mse"]
        for model, data in results.items():
            mse = data["mse"]
            ref = base_mse if model.startswith("ary") else tfm_mse
            row["models"][model] = {
                "mse": mse,
                "n_covered": data["n_covered"],
                "delta_vs_noram_ref": mse - ref if model in ("ary_V8", "ary_V10", "timesfm_ram") else 0.0,
                "pct_vs_noram_ref": (mse / ref - 1.0) * 100 if model in ("ary_V8", "ary_V10", "timesfm_ram") and ref else 0.0,
                "gate_stats": data.get("gate_stats", {}),
            }
        rows.append(row)
        print(f"  hor={hor:3d}  base={row['models']['ary_base']['mse']:.0f}  "
              f"V8={row['models']['ary_V8']['mse']:.0f}  timesfm={row['models']['timesfm']['mse']:.0f}  "
              f"ram={row['models']['timesfm_ram']['mse']:.0f}")
    return rows


def plot_gate_sweep(rows: list[dict], out_dir: Path, feature_idx: int):
    gates = [r["attack_gate"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")

    ram_models = ["ary_V8", "ary_V10", "timesfm_ram"]
    colors = {"ary_V8": "#27ae60", "ary_V10": "#e67e22", "timesfm_ram": "#e74c3c"}
    refs = {"ary_V8": "ary_base", "ary_V10": "ary_base", "timesfm_ram": "timesfm"}

    for ax, model in zip(axes, ["ary_V8", "timesfm_ram"]):
        ax.set_facecolor("#161b22")
        mses = [r["models"][model]["mse"] for r in rows]
        ref_mse = rows[0]["models"][refs[model]]["mse"]
        ax.plot(gates, mses, "o-", color=colors[model], lw=2, label=model)
        ax.axhline(ref_mse, color="#7f8c8d" if "ary" in model else "#3498db",
                   ls="--", label=f"{refs[model]} (no RAM)")
        ax.set_xlabel("P(attack) gate threshold", color="white")
        ax.set_ylabel("MSE (lower = better)", color="white")
        ax.set_title(f"{model} vs gate | feat[{feature_idx}]", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#161b22", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    fig.suptitle(f"RAM gate sensitivity (ctx={CONTEXT}, hor=200)", color="white", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "mse_vs_gate.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Combined panel all RAM models + pct delta
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for model in ram_models:
        pct = [r["models"][model]["pct_vs_noram_ref"] for r in rows]
        ax.plot(gates, pct, "o-", color=colors[model], lw=2, label=model)
    ax.axhline(0, color="white", ls="--", alpha=0.5)
    ax.set_xlabel("P(attack) gate threshold", color="white")
    ax.set_ylabel("% MSE change vs no-RAM baseline", color="white")
    ax.set_title("RAM benefit/harm (% vs base/timesfm)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out_dir / "pct_delta_vs_gate.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_horizon_sweep(rows: list[dict], out_dir: Path, feature_idx: int):
    hors = [r["horizon"] for r in rows]
    models = ["ary_base", "ary_V8", "ary_V10", "timesfm", "timesfm_ram"]
    colors = {
        "ary_base": "#7f8c8d", "ary_V8": "#27ae60", "ary_V10": "#e67e22",
        "timesfm": "#3498db", "timesfm_ram": "#e74c3c",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for model in models:
        mses = [r["models"][model]["mse"] for r in rows]
        ax.plot(hors, mses, "o-", color=colors[model], lw=2, label=model)
    ax.set_xlabel("Forecast horizon (steps per chunk)", color="white")
    ax.set_ylabel("MSE (lower = better)", color="white")
    ax.set_title(f"Horizon sensitivity (ctx={CONTEXT}, ungated RAM) | feat[{feature_idx}]", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out_dir / "mse_vs_horizon.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    for model in ["ary_V8", "ary_V10", "timesfm_ram"]:
        pct = [r["models"][model]["pct_vs_noram_ref"] for r in rows]
        ax.plot(hors, pct, "o-", lw=2, label=model)
    ax.axhline(0, color="white", ls="--", alpha=0.5)
    ax.set_xlabel("Forecast horizon", color="white")
    ax.set_ylabel("% MSE vs no-RAM baseline", color="white")
    ax.set_title("RAM harm grows with longer horizons (ungated)", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out_dir / "pct_delta_vs_horizon.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def summarize_best(rows: list[dict], key: str) -> dict:
    """Find any RAM config that beats its no-RAM reference."""
    ram_models = ["ary_V8", "ary_V10", "timesfm_ram"]
    best = {}
    for model in ram_models:
        candidates = []
        for r in rows:
            d = r["models"][model]
            if d["delta_vs_noram_ref"] < 0:
                candidates.append({key: r[key], "mse": d["mse"], "delta": d["delta_vs_noram_ref"],
                                   "pct": d["pct_vs_noram_ref"]})
        best[model] = min(candidates, key=lambda x: x["delta"]) if candidates else None
    return best


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--feature-idx", type=int, default=FEATURE_IDX)
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--skip-horizon", action="store_true")
    args = p.parse_args()

    feat_idx = args.feature_idx
    feat_name = FEATURE_COLS_242[feat_idx]

    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    va_s, va_b, _ = val
    full, labels, _ = build_timeline(train, val, test, 1000)
    bin_labels, mit_labels = labels_to_ids(labels)
    series = full[:, feat_idx]

    print(f"Feature[{feat_idx}] {feat_name} | ctx={CONTEXT}\n")
    base_model = load_model()
    raw_thresh, hidden_thresh, anomaly_thresh = calibrate_thresholds(base_model, va_s, va_b, CONTEXT)

    print("Loading TimesFM-3...")
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=16, device="cuda")
    )

    meta = {"feature_idx": feat_idx, "feature_name": feat_name, "context": CONTEXT}

    if not args.skip_gate:
        print("\n--- Sweep A: attack-gate threshold (hor=200) ---")
        gate_dir = OUT / f"feat{feat_idx:03d}" / "gate_threshold"
        gate_dir.mkdir(parents=True, exist_ok=True)
        gate_rows = sweep_gate_threshold(
            full, bin_labels, mit_labels, series, feat_idx, base_model, forecaster,
            raw_thresh, hidden_thresh, anomaly_thresh,
        )
        gate_best = summarize_best(gate_rows, "attack_gate")
        gate_payload = {**meta, "horizon": 200, "thresholds": GATE_THRESHOLDS,
                        "rows": gate_rows, "best_beat_baseline": gate_best}
        (gate_dir / "sweep.json").write_text(json.dumps(gate_payload, indent=2))
        plot_gate_sweep(gate_rows, gate_dir, feat_idx)
        print(f"  best beat baseline: {gate_best}")
        print(f"  -> {gate_dir}")

    if not args.skip_horizon:
        print("\n--- Sweep B: forecast horizon (ungated RAM) ---")
        hor_dir = OUT / f"feat{feat_idx:03d}" / "horizon"
        hor_dir.mkdir(parents=True, exist_ok=True)
        hor_rows = sweep_horizon(
            full, bin_labels, mit_labels, series, feat_idx, base_model, forecaster,
            raw_thresh, hidden_thresh, anomaly_thresh,
        )
        hor_best = summarize_best(hor_rows, "horizon")
        hor_payload = {**meta, "horizons": HORIZONS, "rows": hor_rows, "best_beat_baseline": hor_best}
        (hor_dir / "sweep.json").write_text(json.dumps(hor_payload, indent=2))
        plot_horizon_sweep(hor_rows, hor_dir, feat_idx)
        print(f"  best beat baseline: {hor_best}")
        print(f"  -> {hor_dir}")


if __name__ == "__main__":
    main()
