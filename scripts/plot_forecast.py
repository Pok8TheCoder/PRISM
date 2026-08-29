"""Predicted-vs-actual forecast plots: AE-MLP and RF against ground truth.

For one attack capture and one benign capture (both held-out test captures),
walks the true trajectory step by step, forecasts state[t+1] from the true
state[t] (one-step-ahead, never fed its own predictions), and plots forecast
vs actual for a few interpretable dimensions per schema.
"""

from __future__ import annotations

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
from src.pipeline.extract_aryan import FEATURE_COLS_ARYAN  # noqa: E402
from src.pipeline.features_v2 import FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "results" / "forecast" / "v1_onestep_ae_rf"

TARGET_CIDS = {
    "attack": "pcap_0126_r34_a1_T1595_active_scan_none_20260827T184405Z",
    "benign": None,  # first Benign CSV test capture, resolved at runtime
}

PICK_V2 = ["win_dst_port_entropy", "win_syn_frac", "win_ttl_mean", "cur_duration_log"]
PICK_AMT = ["port_entropy", "flag_frac_SYN", "mean_ttl_mean", "mean_duration_us"]


def load_ae(tag: str, dim: int, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_ae.pth", map_location=device, weights_only=False)
    model = MLPForecaster(dim).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["scaler_mean"], ck["scaler_scale"]


def load_rf(tag: str):
    d = joblib.load(CKPT_DIR / f"forecast_{tag}_rf.joblib")
    return d["model"], d["scaler_mean"], d["scaler_scale"]


def forecast_trajectory(traj: np.ndarray, ae_model, ae_mean, ae_scale,
                         rf_model, rf_mean, rf_scale, device):
    """One-step forecasts at every position, using the *true* previous state.

    Returns both natural-unit arrays (for plotting individual features people
    can read directly) and standardized-unit arrays (for an MSE that isn't
    dominated by whichever raw column happens to have the largest magnitude,
    e.g. a byte rate vs. a 0-1 fraction).
    """
    Xs = ((traj[:-1] - ae_mean) / ae_scale).astype(np.float32)
    Ys = ((traj[1:] - ae_mean) / ae_scale).astype(np.float32)
    with torch.no_grad():
        ae_pred_s = ae_model(torch.from_numpy(Xs).to(device)).cpu().numpy()
    ae_pred = ae_pred_s * ae_scale + ae_mean

    Xs_rf = (traj[:-1] - rf_mean) / rf_scale
    rf_pred_s = rf_model.predict(Xs_rf)
    rf_pred = rf_pred_s * rf_scale + rf_mean

    naive_pred = traj[:-1]  # "nothing changes"
    naive_pred_s = Xs  # same scaler for both schemas here, but keep symmetric
    return ae_pred, rf_pred, naive_pred, ae_pred_s, rf_pred_s, naive_pred_s, Ys


def plot_one(cap: dict, schema_key: str, cols: list[str], pick: list[str],
             tag: str, device, out_path: Path) -> dict:
    dim = len(cols)
    ae_model, ae_mean, ae_scale = load_ae(tag, dim, device)
    rf_model, rf_mean, rf_scale = load_rf(tag)

    traj = cap[schema_key]
    ae_pred, rf_pred, naive_pred, ae_pred_s, rf_pred_s, naive_pred_s, Ys = forecast_trajectory(
        traj, ae_model, ae_mean, ae_scale, rf_model, rf_mean, rf_scale, device)
    actual_next = traj[1:]

    # MSE in standardized units -- comparable across dims of very different
    # natural scale (e.g. a byte rate vs. a 0-1 fraction), and directly
    # comparable to the test_mse figures in results/forecast/metrics.json.
    mse_ae = float(np.mean((ae_pred_s - Ys) ** 2))
    mse_rf = float(np.mean((rf_pred_s - Ys) ** 2))
    mse_naive = float(np.mean((naive_pred_s - Ys) ** 2))

    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 2.4 * len(pick)), sharex=True)
    fig.suptitle(f"{tag.upper()} one-step forecast  ·  {cap['cls']}  "
                 f"({'lab capture' if cap['source']=='pcap' else 'CIC CSV chunk'}, "
                 f"{len(traj)} flows)", fontsize=12, fontweight="bold")
    t = np.arange(1, len(traj))
    for ax, name in zip(axes, pick):
        j = cols.index(name)
        ax.plot(t, actual_next[:, j], color="black", lw=1.8, label="actual")
        ax.plot(t, ae_pred[:, j], color="#2e86c1", lw=1.3, ls="--", label="AE-MLP forecast")
        ax.plot(t, rf_pred[:, j], color="#e67e22", lw=1.1, ls=":", label="RF forecast")
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlabel("flow index in capture")
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {"mse_ae": mse_ae, "mse_rf": mse_rf, "mse_naive": mse_naive, "n_steps": len(t)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(DATA, "rb") as f:
        captures = {c["cid"]: c for c in pickle.load(f)}

    attack_cap = captures[TARGET_CIDS["attack"]]
    benign_cap = next(
        c for c in captures.values()
        if c["split"] == "test" and c["cls"] == "Benign" and c["source"] == "csv"
    )

    summary = {}
    for label, cap in (("attack (T1595_active_scan)", attack_cap), ("benign (CIC CSV)", benign_cap)):
        summary[label] = {}
        for schema_key, cols, pick, tag in (
            ("v2", FEATURE_COLS_V2, PICK_V2, "v2"),
            ("amt", FEATURE_COLS_ARYAN, PICK_AMT, "amt"),
        ):
            fname = f"forecast_{tag}_{'attack' if 'attack' in label else 'benign'}.png"
            m = plot_one(cap, schema_key, cols, pick, tag, device, OUT_DIR / fname)
            summary[label][tag] = m
            print(f"{label:<28}{tag:<6} steps={m['n_steps']:<5} "
                  f"AE_MSE={m['mse_ae']:.4f}  RF_MSE={m['mse_rf']:.4f}  "
                  f"naive_MSE={m['mse_naive']:.4f}   -> {fname}")

    import json
    with open(OUT_DIR / "plot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved plots + summary -> {OUT_DIR}")


if __name__ == "__main__":
    main()
