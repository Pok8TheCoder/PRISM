"""Lab attack eval: TimesFM + ARY.01 classifier vs ARY + RAM (V8/V10).

TimesFM+ARY hybrid:
  - Attack / MITRE classification: ARY.01 heads on **real** causal windows (same
    protocol as ram_improve_eval.py — no label leakage).
  - Dynamics accuracy: 1-step TimesFM forecast MSE vs ARY pred_state MSE.

Compared variants (classify-blend RAM from ram_improve_eval):
  ary_frozen, ary_v8_ram, ary_v10_ram, timesfm_ary_cls

Datasets:
  aryan  — CIC-IDS test split (data/aryan_splits)
  xmt    — Docker lab PCAPs (data/xmt_splits)

Outputs JSON for the viewer (--mode lab):
  results/ram_improve/lab_attack/{dataset}/eval.json

Run:
  python scripts/timesfm_ary_lab_eval.py --dataset both
  python scripts/falloff_viewer.py --mode lab
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ram_improve_eval import (  # noqa: E402
    ADAPT_EVERY,
    ADAPT_LR,
    ADAPT_STEPS,
    BLEND_FLOOR,
    BLEND_MAX_WEIGHT,
    CONTEXT,
    MemoryBank,
    PULLBACK,
    calibrate_match_thresh,
    infer,
    load_model,
)
from src.aryan.components import MultiTaskLoss  # noqa: E402
from src.aryan.dataset import SPLITS_DIR, XMT_SPLITS_DIR, load_all_splits  # noqa: E402
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402

OUT_ROOT = ROOT / "results" / "ram_improve" / "lab_attack"
TFM_CONTEXT = 100
TFM_BATCH = 32

VARIANT_SPECS: dict[str, dict] = {
    "ary_frozen": dict(
        adapt=False, loss_type="mse", use_memory=False, key_space="raw", knn_k=1, match_thresh=None
    ),
    "ary_v8_ram": dict(
        adapt=False, loss_type="mse", use_memory=True, key_space="raw", knn_k=1, match_thresh="raw"
    ),
    "ary_v10_ram": dict(
        adapt=True, loss_type="multitask", use_memory=True, key_space="hidden", knn_k=3, match_thresh="hidden"
    ),
    "timesfm_ary_cls": dict(
        adapt=False, loss_type="mse", use_memory=False, key_space="raw", knn_k=1, match_thresh=None
    ),
}


def _metrics(p_atts: np.ndarray, tb_arr: np.ndarray, p_mits: list, tm_arr: np.ndarray, dyn_errs: list) -> dict:
    pred_bin = (p_atts > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(tb_arr, pred_bin, average="binary", zero_division=0)
    tn = int(((pred_bin == 0) & (tb_arr == 0)).sum())
    fp = int(((pred_bin == 1) & (tb_arr == 0)).sum())
    fpr = fp / max(fp + tn, 1)
    mit_pred = np.array([int(np.argmax(p)) for p in p_mits])
    mit_f1 = f1_score(tm_arr, mit_pred, average="macro", zero_division=0)
    return {
        "binary_f1": float(f1),
        "binary_precision": float(prec),
        "binary_recall": float(rec),
        "binary_fpr": float(fpr),
        "mitre_f1_macro": float(mit_f1),
        "dynamics_mse": float(np.mean(dyn_errs)) if dyn_errs else 0.0,
    }


def run_ary_traced(
    name: str,
    base_model: nn.Module,
    states: np.ndarray,
    bin_labels: np.ndarray,
    mit_labels: np.ndarray,
    *,
    adapt: bool,
    loss_type: str,
    use_memory: bool,
    key_space: str,
    knn_k: int,
    match_thresh: float,
) -> dict:
    t0 = time.time()
    online = copy.deepcopy(base_model)
    base_params = [p.clone().detach() for p in base_model.parameters()]
    opt = torch.optim.SGD(online.parameters(), lr=ADAPT_LR)
    mse = nn.MSELoss()
    mt_loss = MultiTaskLoss(lambda_dynamics=0.5, lambda_infiltration=1.2, lambda_mitre=1.0)

    bank = MemoryBank()
    n = len(states)
    trace_t, p_atts, tb_list, p_mits, tm_list, dyn_errs = [], [], [], [], [], []
    seqs_buf, next_buf, tb_buf, tm_buf = [], [], [], []

    online.eval()
    for t in range(CONTEXT - 1, n - 1):
        seq = states[t - CONTEXT + 1 : t + 1]
        out = infer(online, seq)
        p_att, p_mit, hidden, pred_state = out["p_att"], out["p_mit"], out["hidden"], out["pred_state"]
        true_next = states[t + 1]
        tb, tm = int(bin_labels[t + 1]), int(mit_labels[t + 1])
        key = hidden if key_space == "hidden" else seq.reshape(-1)

        if use_memory and len(bank) > 0:
            matches = bank.query(key, k=knn_k)
            close = [m for m in matches if m[2] < match_thresh]
            if close:
                weights = np.array([max(0.0, 1.0 - d / match_thresh) for _, _, d in close])
                weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(close)) / len(close)
                w_total = min(BLEND_MAX_WEIGHT, float(np.mean(weights))) * 0.7 + BLEND_FLOOR
                vote_bin = float(np.sum([w * b for (b, _, _), w in zip(close, weights)]))
                p_att = (1 - w_total) * p_att + w_total * vote_bin
                mit_onehot = np.zeros(7, dtype=np.float32)
                for (_, m, _), w in zip(close, weights):
                    mit_onehot[m] += w
                p_mit = (1 - w_total) * p_mit + w_total * mit_onehot

        trace_t.append(t + 1)
        p_atts.append(p_att)
        tb_list.append(tb)
        p_mits.append(p_mit.tolist())
        tm_list.append(tm)
        dyn_errs.append(float(np.nanmin([np.mean((pred_state - true_next) ** 2), 1e12])))

        if use_memory:
            bank.add(key, tb, tm)

        if adapt:
            seqs_buf.append(seq)
            next_buf.append(true_next)
            tb_buf.append(tb)
            tm_buf.append(tm)
            if len(seqs_buf) >= ADAPT_EVERY:
                online.train()
                seqs_t = torch.from_numpy(np.stack(seqs_buf).astype(np.float32))
                next_t = torch.from_numpy(np.stack(next_buf).astype(np.float32))
                tb_t = torch.tensor(tb_buf, dtype=torch.long)
                tm_t = torch.tensor(tm_buf, dtype=torch.long)
                for _ in range(ADAPT_STEPS):
                    opt.zero_grad()
                    out_a = online(seqs_t)
                    if loss_type == "mse":
                        loss = mse(out_a["pred_state_mean"], next_t)
                    else:
                        ld = mt_loss(
                            out_a["pred_state_mean"],
                            out_a["pred_state_logvar"],
                            next_t,
                            out_a["pred_binary"],
                            tb_t,
                            out_a["pred_mitre"],
                            tm_t,
                        )
                        loss = ld["total"]
                    reg = sum(((p - b) ** 2).sum() for p, b in zip(online.parameters(), base_params))
                    (loss + PULLBACK * reg).backward()
                    opt.step()
                seqs_buf, next_buf, tb_buf, tm_buf = [], [], [], []

    p_atts = np.array(p_atts)
    tb_arr = np.array(tb_list)
    tm_arr = np.array(tm_list)
    metrics = _metrics(p_atts, tb_arr, p_mits, tm_arr, dyn_errs)
    metrics["dynamics_mse"] = float(min(metrics["dynamics_mse"], 1e12))
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    metrics["n_steps"] = len(p_atts)

    return {
        "variant": name,
        "metrics": metrics,
        "trace": {
            "step": trace_t,
            "p_attack": p_atts.tolist(),
            "pred_binary": (p_atts > 0.5).astype(int).tolist(),
            "true_binary": tb_arr.tolist(),
            "true_mitre": tm_arr.tolist(),
            "pred_mitre": [int(np.argmax(p)) for p in p_mits],
            "dynamics_mse_step": dyn_errs,
        },
    }


def timesfm_one_step_batch(forecaster, states: np.ndarray, t: int) -> np.ndarray:
    """One-step forecast for all 242 features at timeline index t."""
    d = states.shape[1]
    preds = np.zeros(d, dtype=np.float32)
    for start in range(0, d, TFM_BATCH):
        end = min(start + TFM_BATCH, d)
        contexts = []
        for j in range(start, end):
            lo = max(0, t - TFM_CONTEXT + 1)
            ctx = states[lo : t + 1, j].astype(np.float32)
            if len(ctx) < TFM_CONTEXT:
                pad = np.full(TFM_CONTEXT - len(ctx), ctx[0] if len(ctx) else 0.0, dtype=np.float32)
                ctx = np.concatenate([pad, ctx])
            contexts.append(ctx)
        outs = forecaster.predict_batch(
            contexts, horizon=1, return_quantiles=False, use_symmetric_averaging=False
        )
        for j, o in zip(range(start, end), outs):
            preds[j] = float(np.asarray(o.forecast, dtype=np.float32)[0])
    return preds


def eval_timesfm_dynamics(forecaster, states: np.ndarray) -> dict:
    n, d = states.shape
    per_step_mse = []
    per_feat_err = np.zeros(d, dtype=np.float64)
    per_feat_n = 0
    t0 = time.time()
    t_start = CONTEXT - 1  # padding fills shorter TimesFM context
    for t in range(t_start, n - 1):
        pred = timesfm_one_step_batch(forecaster, states, t)
        true = states[t + 1]
        err = float(np.mean((pred - true) ** 2))
        per_step_mse.append(err)
        per_feat_err += (pred - true) ** 2
        per_feat_n += 1
    per_feat_mse = (per_feat_err / max(per_feat_n, 1)).astype(float)
    top_idx = np.argsort(per_feat_mse)[::-1][:10]
    return {
        "dynamics_mse_mean": float(np.mean(per_step_mse)) if per_step_mse else 0.0,
        "per_step_mse": per_step_mse,
        "top_features": [
            {"feature_idx": int(i), "feature_name": FEATURE_COLS_242[i], "mse": float(per_feat_mse[i])}
            for i in top_idx
        ],
        "elapsed_sec": round(time.time() - t0, 1),
        "n_steps": len(per_step_mse),
    }


def calibrate_thresholds(base_model, va_s, va_b) -> tuple[float, float]:
    val_raw_keys, val_hidden_keys = [], []
    for t in range(CONTEXT - 1, len(va_s) - 1):
        seq = va_s[t - CONTEXT + 1 : t + 1]
        out = infer(base_model, seq)
        val_raw_keys.append(seq.reshape(-1))
        val_hidden_keys.append(out["hidden"])
    val_raw_keys = np.stack(val_raw_keys)
    val_hidden_keys = np.stack(val_hidden_keys)
    val_bin = va_b[CONTEXT : len(va_s)]
    raw_thresh = calibrate_match_thresh(val_raw_keys, val_bin)
    hidden_thresh = calibrate_match_thresh(val_hidden_keys, val_bin)
    return raw_thresh, hidden_thresh


def eval_dataset(name: str, splits_dir: Path, forecaster, base_model) -> dict:
    splits = load_all_splits(splits_dir)
    te_s, te_b, te_m = splits["test"]
    va_s, va_b, _ = splits["val"]
    raw_thresh, hidden_thresh = calibrate_thresholds(base_model, va_s, va_b)

    print(f"\n=== dataset={name} test windows={len(te_s)} attack_rate={te_b.mean():.1%} ===")

    variants_out = {}
    for vname, spec in VARIANT_SPECS.items():
        if vname == "timesfm_ary_cls":
            continue  # same classifier trace as ary_frozen; metrics merged below
        mt = spec["match_thresh"]
        thresh = raw_thresh if mt == "raw" else hidden_thresh if mt == "hidden" else 30.0
        print(f"  {vname}...")
        variants_out[vname] = run_ary_traced(
            vname,
            base_model,
            te_s,
            te_b,
            te_m,
            adapt=spec["adapt"],
            loss_type=spec["loss_type"],
            use_memory=spec["use_memory"],
            key_space=spec["key_space"],
            knn_k=spec["knn_k"],
            match_thresh=thresh,
        )
        m = variants_out[vname]["metrics"]
        print(f"    BinF1={m['binary_f1']:.3f}  MitreF1={m['mitre_f1_macro']:.3f}  dynMSE={m['dynamics_mse']:.1f}")

    # TimesFM dynamics + ARY classifier (reuse frozen attack trace)
    print("  timesfm dynamics (242 features, 1-step)...")
    tfm = eval_timesfm_dynamics(forecaster, te_s)
    print(f"    TimesFM dynMSE={tfm['dynamics_mse_mean']:.1f}  ({tfm['elapsed_sec']}s)")

    frozen = variants_out["ary_frozen"]
    variants_out["timesfm_ary_cls"] = {
        "variant": "timesfm_ary_cls",
        "metrics": {
            **{k: frozen["metrics"][k] for k in ("binary_f1", "binary_precision", "binary_recall", "binary_fpr", "mitre_f1_macro")},
            "dynamics_mse": tfm["dynamics_mse_mean"],
            "ary_dynamics_mse": frozen["metrics"]["dynamics_mse"],
            "n_steps": frozen["metrics"]["n_steps"],
            "elapsed_sec": tfm["elapsed_sec"],
            "note": "Attack/MITRE from ARY.01 classifier on real causal windows; dynamics from TimesFM 1-step",
        },
        "trace": frozen["trace"],
        "timesfm": tfm,
    }

    payload = {
        "dataset": name,
        "split": "test",
        "n_windows": len(te_s),
        "attack_rate": float(te_b.mean()),
        "context_ary": CONTEXT,
        "context_timesfm": TFM_CONTEXT,
        "variants": variants_out,
        "summary_table": [
            {
                "variant": vname,
                **variants_out[vname]["metrics"],
            }
            for vname in ["ary_frozen", "ary_v8_ram", "ary_v10_ram", "timesfm_ary_cls"]
        ],
    }
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2))
    print(f"  -> {out_dir / 'eval.json'}")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aryan", "xmt", "both"], default="both")
    args = p.parse_args()

    base_model = load_model()
    print("Loading TimesFM-3...")
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    forecaster = TimesFM3Evaluator(
        ModelConfig(
            checkpoint_path="google/timesfm-3.0-pytorch",
            per_core_batch_size=TFM_BATCH,
            device="cuda",
        )
    )

    datasets = []
    if args.dataset in ("aryan", "both"):
        datasets.append(("aryan", SPLITS_DIR))
    if args.dataset in ("xmt", "both"):
        datasets.append(("xmt", XMT_SPLITS_DIR))

    summaries = []
    for name, path in datasets:
        summaries.append(eval_dataset(name, path, forecaster, base_model))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nDone. Open viewer: python scripts/falloff_viewer.py --mode lab")


if __name__ == "__main__":
    main()
