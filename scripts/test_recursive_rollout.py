"""Test the "leaf-node bound" claim: does RF collapse under recursive rollout?

One-step forecasting (plot_forecast.py) always feeds the model the TRUE
previous state. This script instead does a genuine multi-step rollout: predict
state[t+1] from state[t], then feed that *prediction* back in as the input for
state[t+2], and so on for the whole capture, never touching ground truth again
after step 0.

Claim under test: Random Forest regressors, being averages over training-set
leaf values, cannot extrapolate outside the numeric range they were trained on.
Fed their own output repeatedly, they should regress toward the training mean
and lose amplitude/variance over the rollout. A neural net is not
leaf-bounded and may or may not show the same collapse.

Reports, per model per schema: variance of the rollout vs variance of the true
trajectory (a "collapse ratio" -- near 0 means the rollout went flat).
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
from src.pipeline.extract_aryan import FEATURE_COLS_ARYAN  # noqa: E402
from src.pipeline.features_v2 import FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "forecast_captures.pkl"
CKPT_DIR = ROOT / "models" / "checkpoints"
OUT_DIR = ROOT / "results" / "forecast" / "v2_recursive_ae_rf"

TARGET_CIDS = {"attack": "pcap_0126_r34_a1_T1595_active_scan_none_20260827T184405Z"}
PICK_V2 = ["win_dst_port_entropy", "win_syn_frac", "cur_duration_log"]
PICK_AMT = ["port_entropy", "flag_frac_SYN", "mean_duration_us"]


def load_ae(tag: str, dim: int, device):
    ck = torch.load(CKPT_DIR / f"forecast_{tag}_ae.pth", map_location=device, weights_only=False)
    model = MLPForecaster(dim).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["scaler_mean"], ck["scaler_scale"]


def load_rf(tag: str):
    d = joblib.load(CKPT_DIR / f"forecast_{tag}_rf.joblib")
    return d["model"], d["scaler_mean"], d["scaler_scale"]


def rollout_ae(model, mean, scale, s0: np.ndarray, n_steps: int, device) -> np.ndarray:
    """s0 is one true starting state (natural units). Returns (n_steps, dim)."""
    cur_s = (s0 - mean) / scale
    out = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.from_numpy(cur_s.astype(np.float32)).unsqueeze(0).to(device)
            cur_s = model(x).squeeze(0).cpu().numpy()
            out.append(cur_s * scale + mean)
    return np.stack(out)


def rollout_rf(model, mean, scale, s0: np.ndarray, n_steps: int) -> np.ndarray:
    cur_s = (s0 - mean) / scale
    out = []
    for _ in range(n_steps):
        cur_s = model.predict(cur_s.reshape(1, -1))[0]
        out.append(cur_s * scale + mean)
    return np.stack(out)


def collapse_ratio(rollout_s: np.ndarray, actual_s: np.ndarray) -> float:
    """Variance of the rollout vs variance of the true trajectory, averaged
    across dims, in standardized units. Near 0 => rollout went flat."""
    v_roll = rollout_s.var(axis=0)
    v_true = actual_s.var(axis=0) + 1e-9
    return float(np.mean(v_roll / v_true))


def run_schema(cap: dict, schema_key: str, cols: list[str], pick: list[str],
               tag: str, device) -> dict:
    traj = cap[schema_key]
    n = len(traj) - 1
    ae_model, ae_mean, ae_scale = load_ae(tag, traj.shape[1], device)
    rf_model, rf_mean, rf_scale = load_rf(tag)

    ae_roll = rollout_ae(ae_model, ae_mean, ae_scale, traj[0], n, device)
    rf_roll = rollout_rf(rf_model, rf_mean, rf_scale, traj[0], n)
    actual = traj[1:]

    # Standardize everything the same way for the collapse-ratio comparison.
    to_s = lambda A: (A - ae_mean) / ae_scale  # noqa: E731
    ae_collapse = collapse_ratio(to_s(ae_roll), to_s(actual))
    rf_collapse = collapse_ratio(to_s(rf_roll), to_s(actual))

    ae_mse = float(np.mean((to_s(ae_roll) - to_s(actual)) ** 2))
    rf_mse = float(np.mean((to_s(rf_roll) - to_s(actual)) ** 2))

    fig, axes = plt.subplots(len(pick), 1, figsize=(11, 2.4 * len(pick)), sharex=True)
    fig.suptitle(f"{tag.upper()} RECURSIVE rollout (no ground truth after step 0)  ·  "
                 f"{cap['cls']}, {n+1} flows", fontsize=12, fontweight="bold")
    t = np.arange(1, n + 1)
    for ax, name in zip(axes, pick):
        j = cols.index(name)
        ax.plot(t, actual[:, j], color="black", lw=1.8, label="actual")
        ax.plot(t, ae_roll[:, j], color="#2e86c1", lw=1.2, ls="--", label="AE-MLP rollout")
        ax.plot(t, rf_roll[:, j], color="#e67e22", lw=1.2, ls=":", label="RF rollout")
        ax.set_ylabel(name, fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper right", ncol=3)
    axes[-1].set_xlabel("step (recursive -- each point built on the model's own prior guess)")
    plt.tight_layout()
    out_path = OUT_DIR / f"rollout_{tag}_{'attack' if cap['source']=='pcap' else 'benign'}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_steps": n,
        "ae_mse": ae_mse, "rf_mse": rf_mse,
        "ae_collapse_ratio": ae_collapse, "rf_collapse_ratio": rf_collapse,
        "plot": str(out_path.name),
    }


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

    results = {}
    for label, cap in (("attack (T1595_active_scan)", attack_cap), ("benign (CIC CSV)", benign_cap)):
        results[label] = {}
        for schema_key, cols, pick, tag in (
            ("v2", FEATURE_COLS_V2, PICK_V2, "v2"),
            ("amt", FEATURE_COLS_ARYAN, PICK_AMT, "amt"),
        ):
            r = run_schema(cap, schema_key, cols, pick, tag, device)
            results[label][tag] = r
            print(f"{label:<28}{tag:<5} steps={r['n_steps']:<4} "
                  f"AE: mse={r['ae_mse']:.4f} collapse={r['ae_collapse_ratio']:.4f}   "
                  f"RF: mse={r['rf_mse']:.4f} collapse={r['rf_collapse_ratio']:.4f}")

    with open(OUT_DIR / "rollout_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("Collapse ratio = variance(rollout) / variance(actual), std units.")
    print("~1.0 = keeps realistic variation.  ~0.0 = flatlined toward a mean.")
    print("=" * 78)
    for label, r in results.items():
        for tag, m in r.items():
            print(f"  {label:<28}{tag:<5} AE_collapse={m['ae_collapse_ratio']:.3f}  "
                  f"RF_collapse={m['rf_collapse_ratio']:.3f}")
    print(f"\nSaved -> {OUT_DIR / 'rollout_summary.json'}")


if __name__ == "__main__":
    main()
