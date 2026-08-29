"""Round 3 of the recursive-rollout stress test: adds a genuinely TEMPORAL
(GRU, recurrent) forecaster alongside everything tested in v3_recursive_5way,
to isolate "does real recurrence help vs. a flat concatenated window?"

All six candidates get the same WARMUP=6 real ground-truth states (the GRU's
window size, used as the common warmup for a fair comparison), then must
roll forward for the rest of the capture using nothing but their own
previous predictions.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_forecast_models import MLPForecaster  # noqa: E402
from scripts.train_hybrid_forecaster import HybridForecaster  # noqa: E402
from scripts.train_temporal_forecaster import TemporalForecaster  # noqa: E402
from src.pipeline.extract_aryan import FEATURE_COLS_ARYAN  # noqa: E402
from src.pipeline.features_v2 import FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "results" / "forecast" / "v4_recursive_temporal"

WARMUP = 6  # matches the GRU's window so every model gets the same head start
PHI = 0.9

TARGET_CIDS = {"attack": "pcap_0126_r34_a1_T1595_active_scan_none_20260827T184405Z"}
PICK_V2 = ["win_dst_port_entropy", "win_syn_frac", "cur_duration_log"]
PICK_AMT = ["port_entropy", "flag_frac_SYN", "mean_duration_us"]


def load_ae(tag, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_ae.pth", map_location=device, weights_only=False)
    m = MLPForecaster(ck["dim"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["scaler_mean"], ck["scaler_scale"]


def load_hybrid(tag, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_hybrid.pth", map_location=device, weights_only=False)
    m = HybridForecaster(ck["dim"], ck["window"], torch.tensor(ck["max_step"], dtype=torch.float32)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["scaler_mean"], ck["scaler_scale"], ck["window"]


def load_temporal(tag, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_temporal.pth", map_location=device, weights_only=False)
    m = TemporalForecaster(ck["dim"], torch.tensor(ck["max_step"], dtype=torch.float32)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["scaler_mean"], ck["scaler_scale"], ck["window"]


def load_joblib(tag, kind):
    d = joblib.load(CKPT_DIR / f"forecast_{tag}_{kind}.joblib")
    return d["model"], d["scaler_mean"], d["scaler_scale"]


def rollout_memoryless(predict_fn, mean, scale, warm_states, n_steps):
    cur = (warm_states[-1] - mean) / scale
    out = []
    for _ in range(n_steps):
        cur = predict_fn(cur)
        out.append(cur * scale + mean)
    return np.stack(out)


def rollout_windowed(model, mean, scale, window, warm_states, n_steps, device, recurrent: bool):
    buf = [(s - mean) / scale for s in warm_states[-window:]]
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            if recurrent:
                x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0).to(device)  # (1, window, dim)
            else:
                x = torch.from_numpy(np.concatenate(buf).astype(np.float32)).unsqueeze(0).to(device)  # (1, window*dim)
            nxt = model(x).squeeze(0).cpu().numpy()
            buf = buf[1:] + [nxt]
            out.append(nxt * scale + mean)
    return np.stack(out)


def rollout_holt_damped(warm_states, n_steps):
    level = warm_states[-1].copy()
    trend = np.mean(np.diff(warm_states, axis=0), axis=0)
    out, phi_sum = [], 0.0
    for _ in range(n_steps):
        phi_sum = PHI + PHI * phi_sum
        out.append(level + trend * phi_sum)
    return np.stack(out)


def range_coverage(rollout, actual):
    r_rng = rollout.max() - rollout.min()
    a_rng = actual.max() - actual.min()
    if a_rng < 1e-9:
        return float("nan")
    return float(r_rng / a_rng)


def run_schema(cap, schema_key, cols, pick, tag, device):
    traj = cap[schema_key]
    n = len(traj) - 1 - WARMUP
    if n < 5:
        return None
    warm = traj[:WARMUP]
    actual = traj[WARMUP:WARMUP + n]

    ae_model, ae_mean, ae_scale = load_ae(tag, device)
    rf_model, rf_mean, rf_scale = load_joblib(tag, "rf")
    xgb_model, xgb_mean, xgb_scale = load_joblib(tag, "xgb")
    hy_model, hy_mean, hy_scale, hy_window = load_hybrid(tag, device)
    tm_model, tm_mean, tm_scale, tm_window = load_temporal(tag, device)

    def ae_predict(cur):
        with torch.no_grad():
            x = torch.from_numpy(cur.astype(np.float32)).unsqueeze(0).to(device)
            return ae_model(x).squeeze(0).cpu().numpy()

    ae_roll = rollout_memoryless(ae_predict, ae_mean, ae_scale, warm, n)
    rf_roll = rollout_memoryless(lambda c: rf_model.predict(c.reshape(1, -1))[0], rf_mean, rf_scale, warm, n)
    xgb_roll = rollout_memoryless(lambda c: xgb_model.predict(c.reshape(1, -1))[0], xgb_mean, xgb_scale, warm, n)
    hy_roll = rollout_windowed(hy_model, hy_mean, hy_scale, hy_window, warm, n, device, recurrent=False)
    tm_roll = rollout_windowed(tm_model, tm_mean, tm_scale, tm_window, warm, n, device, recurrent=True)
    holt_roll = rollout_holt_damped(warm, n)

    rolls = {"AE-MLP": ae_roll, "RF": rf_roll, "XGBoost": xgb_roll,
             "Hybrid (flat window)": hy_roll, "Temporal-GRU (recurrent)": tm_roll,
             "Holt damped-trend": holt_roll}

    coverage = {}
    for name in pick:
        j = cols.index(name)
        coverage[name] = {k: range_coverage(v[:, j], actual[:, j]) for k, v in rolls.items()}

    colors = {"AE-MLP": "#2e86c1", "RF": "#e67e22", "XGBoost": "#c0392b",
              "Hybrid (flat window)": "#95a5a6", "Temporal-GRU (recurrent)": "#27ae60",
              "Holt damped-trend": "#8e44ad"}
    styles = {"AE-MLP": "--", "RF": ":", "XGBoost": "-.",
              "Hybrid (flat window)": (0, (3, 1, 1, 1)), "Temporal-GRU (recurrent)": "-",
              "Holt damped-trend": (0, (1, 1))}
    widths = {"Temporal-GRU (recurrent)": 2.0}

    fig, axes = plt.subplots(len(pick), 1, figsize=(12, 2.6 * len(pick)), sharex=True)
    fig.suptitle(f"{tag.upper()} recursive rollout (6 candidates incl. real GRU, warmup={WARMUP})  ·  "
                 f"{cap['cls']}, {n} rolled steps", fontsize=12, fontweight="bold")
    t = np.arange(1, n + 1)
    for ax, name in zip(axes, pick):
        j = cols.index(name)
        ax.plot(t, actual[:, j], color="black", lw=1.8, label="actual", zorder=5)
        for k, roll in rolls.items():
            ax.plot(t, roll[:, j], color=colors[k], lw=widths.get(k, 1.1), ls=styles[k], label=k)
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right", ncol=3)
    axes[-1].set_xlabel("step (recursive, self-fed after warmup)")
    plt.tight_layout()
    out_path = OUT_DIR / f"rollout3_{tag}_{'attack' if cap['source']=='pcap' else 'benign'}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {"n_steps": n, "coverage": coverage, "plot": str(out_path.name)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    with open(DATA, "rb") as f:
        captures = {c["cid"]: c for c in pickle.load(f)}

    attack_cap = captures[TARGET_CIDS["attack"]]
    benign_cap = next(c for c in captures.values()
                       if c["split"] == "test" and c["cls"] == "Benign" and c["source"] == "csv")

    results = {}
    for label, cap in (("attack (T1595_active_scan)", attack_cap), ("benign (CIC CSV)", benign_cap)):
        results[label] = {}
        for schema_key, cols, pick, tag in (
            ("v2", FEATURE_COLS_V2, PICK_V2, "v2"),
            ("amt", FEATURE_COLS_ARYAN, PICK_AMT, "amt"),
        ):
            r = run_schema(cap, schema_key, cols, pick, tag, device)
            results[label][tag] = r

    with open(OUT_DIR / "rollout3_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nRange coverage = (rollout's own min..max) / (actual's true min..max).")
    print("~0.0 = frozen flat line.  ~1.0 = healthy, tracks the real dynamic range.")
    print("=" * 110)
    for label, r in results.items():
        for tag, m in r.items():
            print(f"\n{label} / {tag}:")
            for feat, cov in m["coverage"].items():
                row = "  ".join(f"{k}={v:.2f}" for k, v in cov.items())
                print(f"  {feat:24s} {row}")
    print(f"\nSaved -> {OUT_DIR / 'rollout3_summary.json'}")


if __name__ == "__main__":
    main()
