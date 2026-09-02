"""Compare ARY.01 (base / V8 / V10) against Google's TimesFM-3 on a synthetic
kill-chain timeline, with configurable receding-horizon context + prediction
length, and segregated output folders per run config and feature.

Defaults: context=100 steps taken, horizon=200 steps predicted per chunk.

ARY.01 was trained with lookback=20; when context > 20 the model still reads
only the last 20 revealed steps at each autoregressive step, but memory raw
keys and TimesFM inputs use the full context window.

Outputs (segregated):
  results/ram_improve/ctx{C}_hor{H}/feat{idx}_{name}/metrics.json
  results/ram_improve/ctx{C}_hor{H}/feat{idx}_{name}/bars.png
  results/ram_improve/ctx{C}_hor{H}/feat{idx}_{name}/timeline.png
  results/ram_improve/ctx{C}_hor{H}/run_summary.json
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ram_improve_eval import (  # noqa: E402
    ADAPT_LR,
    ADAPT_STEPS,
    BLEND_FLOOR,
    BLEND_MAX_WEIGHT,
    CONTEXT as ARY_LOOKBACK,
    PULLBACK,
    calibrate_match_thresh,
    infer,
    load_model,
)
from src.aryan.components import MultiTaskLoss  # noqa: E402
from src.aryan.constants import STAGE_ID_TO_ATTACK  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402
from src.aryan.timeline import build_timeline  # noqa: E402

TARGET_LEN = 1000
ATTACK_TO_STAGE = {v: k for k, v in STAGE_ID_TO_ATTACK.items()}


@dataclass(frozen=True)
class RunConfig:
    context: int
    horizon: int
    feature_idx: int
    feature_name: str
    gate_ram_on_attack: bool = False
    attack_gate: float = 0.5

    @property
    def config_dir(self) -> Path:
        suffix = "_attackgate" if self.gate_ram_on_attack else ""
        return ROOT / "results" / "ram_improve" / f"ctx{self.context}_hor{self.horizon}{suffix}"

    @property
    def feature_dir(self) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", self.feature_name).strip("_")
        return self.config_dir / f"feat{self.feature_idx:03d}_{safe}"


class ContMemory:
    def __init__(self):
        self.keys: list[np.ndarray] = []
        self.continuations: list[np.ndarray] = []

    def add(self, key, continuation):
        self.keys.append(key)
        self.continuations.append(continuation)

    def query(self, key, k=1):
        if not self.keys:
            return []
        M = np.stack(self.keys)
        d = np.linalg.norm(M - key[None, :], axis=1)
        idx = np.argsort(d)[:k]
        return [(self.continuations[i], float(d[i])) for i in idx]

    def __len__(self):
        return len(self.keys)


def labels_to_ids(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    bin_labels = np.array([0 if lab == "Benign" else 1 for lab in labels], dtype=np.int64)
    mit_labels = np.array([ATTACK_TO_STAGE.get(lab, 0) for lab in labels], dtype=np.int64)
    return bin_labels, mit_labels


def pick_feature_indices(tr_s: np.ndarray, *, feature_idx: int | None, feature_ranks: list[int]) -> list[int]:
    if feature_idx is not None:
        return [int(feature_idx)]
    order = np.argsort(tr_s.var(axis=0))[::-1]
    out = []
    for rank in feature_ranks:
        r = max(1, rank)
        out.append(int(order[min(r - 1, len(order) - 1)]))
    return sorted(set(out))


def ary_model_window(states: np.ndarray, t: int) -> list[np.ndarray]:
    """Last ARY_LOOKBACK states ending at t (inclusive)."""
    lo = max(0, t - ARY_LOOKBACK + 1)
    window = states[lo:t + 1]
    if len(window) < ARY_LOOKBACK:
        pad = np.repeat(window[:1], ARY_LOOKBACK - len(window), axis=0)
        window = np.concatenate([pad, window], axis=0)
    return [window[i] for i in range(len(window))]


def series_context_slice(series: np.ndarray, t: int, context: int) -> np.ndarray:
    lo = max(0, t - context + 1)
    window = series[lo:t + 1].astype(np.float32)
    if len(window) < context:
        pad = np.full(context - len(window), window[0], dtype=np.float32)
        window = np.concatenate([pad, window])
    return window


def context_slice(states: np.ndarray, t: int, context: int) -> np.ndarray:
    lo = max(0, t - context + 1)
    window = states[lo:t + 1]
    if len(window) < context:
        pad = np.repeat(window[:1], context - len(window), axis=0)
        window = np.concatenate([pad, window], axis=0)
    return window


def p_attack_at(model, states: np.ndarray, t: int) -> float:
    seq = np.stack(ary_model_window(states, t)).astype(np.float32)
    return infer(model, seq)["p_att"]


def ary_state_rollout(
    base_model,
    states,
    bin_labels,
    mit_labels,
    cfg: RunConfig,
    *,
    adapt: bool,
    loss_type: str,
    use_memory: bool,
    key_space: str,
    knn_k: int,
    match_thresh: float,
    anomaly_thresh: float,
    gate_on_attack: bool = False,
) -> tuple[np.ndarray, dict]:
    n, d = states.shape
    context, horizon = cfg.context, cfg.horizon
    online = copy.deepcopy(base_model)
    base_params = [p.clone().detach() for p in base_model.parameters()]
    opt = torch.optim.SGD(online.parameters(), lr=ADAPT_LR)
    mse = nn.MSELoss()
    mt_loss = MultiTaskLoss(lambda_dynamics=0.5, lambda_infiltration=1.2, lambda_mitre=1.0)

    bank = ContMemory()
    full_predicted = np.full((n, d), np.nan, dtype=np.float32)
    pending_match = None
    gate_stats = {"chunks": 0, "chunks_attack_pred": 0, "blends_applied": 0, "memory_writes": 0}

    t = context - 1
    while t + horizon < n:
        gate_stats["chunks"] += 1
        buf = ary_model_window(states, t)
        online.eval()
        p_att = p_attack_at(online, states, t)
        attack_pred = p_att >= cfg.attack_gate
        if attack_pred:
            gate_stats["chunks_attack_pred"] += 1

        raw_preds = []
        hidden_at_start = None
        with torch.no_grad():
            for _step in range(horizon):
                x = torch.from_numpy(np.stack(buf).astype(np.float32)).unsqueeze(0)
                out = online(x)
                if hidden_at_start is None:
                    hidden_at_start = out["hidden"][0].numpy()
                nxt = out["pred_state_mean"].squeeze(0).numpy()
                raw_preds.append(nxt)
                buf = buf[1:] + [nxt]
        raw_preds = np.stack(raw_preds)

        preds = raw_preds
        allow_ram = (not gate_on_attack) or attack_pred
        if use_memory and pending_match is not None and allow_ram:
            cont, dist = pending_match
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / match_thresh)) * 0.5 + BLEND_FLOOR
            k = min(len(cont), horizon)
            recalled = states[t][None, :] + cont[:k]
            preds = raw_preds.copy()
            preds[:k] = (1 - w) * raw_preds[:k] + w * recalled
            gate_stats["blends_applied"] += 1
        pending_match = None

        full_predicted[t + 1:t + 1 + horizon] = preds
        true_future = states[t + 1:t + 1 + horizon]

        if use_memory and allow_ram:
            step_err = np.mean((raw_preds - true_future) ** 2, axis=1)
            peak_idx = int(np.argmax(step_err))
            if step_err[peak_idx] > anomaly_thresh:
                center = t + 1 + peak_idx
                ctx = context_slice(states, t, context)
                query_key = hidden_at_start if key_space == "hidden" else ctx.reshape(-1)
                cont_end = min(n, center + 1 + horizon)
                continuation = states[center + 1:cont_end] - states[center]
                bank.add(query_key, continuation)
                gate_stats["memory_writes"] += 1
                if len(bank) > 1:
                    matches = bank.query(query_key, k=knn_k)
                    close = [(c, dd) for c, dd in matches if dd < match_thresh]
                    if close:
                        weights = np.array([max(0.0, 1.0 - dd / match_thresh) for _, dd in close])
                        weights = weights / weights.sum()
                        max_len = max(len(c) for c, _ in close)
                        acc = np.zeros((max_len, d), dtype=np.float32)
                        for (c, _), w2 in zip(close, weights):
                            acc[:len(c)] += w2 * c
                        avg_dist = float(np.mean([dd for _, dd in close]))
                        pending_match = (acc, avg_dist)

        if adapt:
            seqs_t = torch.from_numpy(
                np.stack([states[i - ARY_LOOKBACK + 1:i + 1] for i in range(t, t + horizon)]).astype(np.float32)
            )
            next_t = torch.from_numpy(true_future.astype(np.float32))
            tb_t = torch.tensor(bin_labels[t + 1:t + 1 + horizon], dtype=torch.long)
            tm_t = torch.tensor(mit_labels[t + 1:t + 1 + horizon], dtype=torch.long)
            online.train()
            for _ in range(ADAPT_STEPS):
                opt.zero_grad()
                out_a = online(seqs_t)
                if loss_type == "mse":
                    loss = mse(out_a["pred_state_mean"], next_t)
                else:
                    ld = mt_loss(
                        out_a["pred_state_mean"], out_a["pred_state_logvar"], next_t,
                        out_a["pred_binary"], tb_t, out_a["pred_mitre"], tm_t,
                    )
                    loss = ld["total"]
                reg = sum(((p - b) ** 2).sum() for p, b in zip(online.parameters(), base_params))
                (loss + PULLBACK * reg).backward()
                opt.step()

        t += horizon

    return full_predicted, gate_stats


def timesfm_rollout(
    forecaster,
    series: np.ndarray,
    cfg: RunConfig,
    use_memory: bool,
    *,
    gate_model=None,
    full_states: np.ndarray | None = None,
    gate_on_attack: bool = False,
) -> tuple[np.ndarray, dict]:
    n = len(series)
    context, horizon = cfg.context, cfg.horizon
    chunk_starts = [t for t in range(context - 1, n - 1, horizon) if t + horizon < n]
    contexts = [series[max(0, t - context + 1):t + 1].astype(np.float32) for t in chunk_starts]

    outputs = list(
        forecaster.predict_batch(contexts, horizon=horizon, return_quantiles=False, use_symmetric_averaging=False)
    )
    raw_by_chunk = [np.asarray(o.forecast, dtype=np.float32) for o in outputs]

    full_predicted = np.full(n, np.nan, dtype=np.float32)
    gate_stats = {"chunks": len(chunk_starts), "chunks_attack_pred": 0, "blends_applied": 0, "memory_writes": 0}
    if not use_memory:
        for t, raw in zip(chunk_starts, raw_by_chunk):
            full_predicted[t + 1:t + 1 + horizon] = raw
        return full_predicted, gate_stats

    bank = ContMemory()
    errs_seen: list[float] = []
    pending_match = None
    for t, raw in zip(chunk_starts, raw_by_chunk):
        attack_pred = True
        if gate_on_attack and gate_model is not None and full_states is not None:
            p_att = p_attack_at(gate_model, full_states, t)
            attack_pred = p_att >= cfg.attack_gate
            if attack_pred:
                gate_stats["chunks_attack_pred"] += 1

        preds = raw
        allow_ram = (not gate_on_attack) or attack_pred
        if pending_match is not None and allow_ram:
            cont, dist = pending_match
            thresh = max(np.std(series[:t + 1]) * 3.0, 1e-6)
            w = min(BLEND_MAX_WEIGHT, max(0.0, 1.0 - dist / thresh)) * 0.5 + BLEND_FLOOR
            k = min(len(cont), horizon)
            recalled = series[t] + cont[:k]
            preds = raw.copy()
            preds[:k] = (1 - w) * raw[:k] + w * recalled
            gate_stats["blends_applied"] += 1
        pending_match = None

        full_predicted[t + 1:t + 1 + horizon] = preds
        true_future = series[t + 1:t + 1 + horizon]
        step_err = (raw - true_future) ** 2
        errs_seen.extend(step_err.tolist())
        surprise_thresh = np.percentile(errs_seen, 90) if len(errs_seen) > 5 else 1e18
        peak_idx = int(np.argmax(step_err))
        if allow_ram and step_err[peak_idx] > surprise_thresh:
            center = t + 1 + peak_idx
            key = series_context_slice(series, center, context)
            cont_end = min(n, center + 1 + horizon)
            continuation = series[center + 1:cont_end] - series[center]
            bank.add(key, continuation)
            gate_stats["memory_writes"] += 1
            if len(bank) > 1:
                match_thresh = np.std(series[:center + 1]) * 3.0
                matches = bank.query(key, k=1)
                if matches:
                    cont2, dist = matches[0]
                    if dist < match_thresh:
                        pending_match = (cont2, dist)

    return full_predicted, gate_stats


def calibrate_thresholds(base_model, va_s, va_b, context: int) -> tuple[float, float, float]:
    val_raw_keys, val_hidden_keys, val_errs = [], [], []
    for t in range(ARY_LOOKBACK - 1, len(va_s) - 1):
        seq = va_s[t - ARY_LOOKBACK + 1:t + 1]
        out = infer(base_model, seq)
        ctx = context_slice(va_s, t, context)
        val_raw_keys.append(ctx.reshape(-1))
        val_hidden_keys.append(out["hidden"])
        val_errs.append(float(np.mean((out["pred_state"] - va_s[t + 1]) ** 2)))
    val_raw_keys = np.stack(val_raw_keys)
    val_hidden_keys = np.stack(val_hidden_keys)
    val_bin_labels = va_b[ARY_LOOKBACK:len(va_s)]
    raw_thresh = calibrate_match_thresh(val_raw_keys, val_bin_labels)
    hidden_thresh = calibrate_match_thresh(val_hidden_keys, val_bin_labels)
    anomaly_thresh = float(np.percentile(val_errs, 90))
    return raw_thresh, hidden_thresh, anomaly_thresh


def run_one_feature(
    cfg: RunConfig,
    *,
    full,
    labels,
    segments,
    bin_labels,
    mit_labels,
    base_model,
    forecaster,
    raw_thresh: float,
    hidden_thresh: float,
    anomaly_thresh: float,
) -> dict:
    attack_segments = [(lbl, a, b) for lbl, a, b in segments if lbl != "Benign"]
    n = len(full)
    series = full[:, cfg.feature_idx]
    cfg.feature_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== feature[{cfg.feature_idx}] {cfg.feature_name} | ctx={cfg.context} hor={cfg.horizon}"
          f"{' | attack-gated RAM' if cfg.gate_ram_on_attack else ''} ===")

    gate = cfg.gate_ram_on_attack
    variant_specs = [
        ("ary_base", dict(adapt=False, loss_type="mse", use_memory=False, key_space="raw", knn_k=1, gate_on_attack=False)),
        ("ary_V8", dict(adapt=False, loss_type="mse", use_memory=True, key_space="raw", knn_k=1, gate_on_attack=False)),
        ("ary_V10", dict(adapt=True, loss_type="multitask", use_memory=True, key_space="hidden", knn_k=3, gate_on_attack=False)),
    ]
    if gate:
        variant_specs += [
            ("ary_V8_gated", dict(adapt=False, loss_type="mse", use_memory=True, key_space="raw", knn_k=1, gate_on_attack=True)),
            ("ary_V10_gated", dict(adapt=True, loss_type="multitask", use_memory=True, key_space="hidden", knn_k=3, gate_on_attack=True)),
        ]

    results: dict[str, np.ndarray] = {}
    gate_info: dict[str, dict] = {}
    for name, kwargs in variant_specs:
        t0 = time.time()
        print(f"  {name}...")
        thresh = hidden_thresh if kwargs["key_space"] == "hidden" else raw_thresh
        pred, gstats = ary_state_rollout(
            base_model, full, bin_labels, mit_labels, cfg,
            match_thresh=thresh, anomaly_thresh=anomaly_thresh, **kwargs,
        )
        results[name] = pred
        gate_info[name] = gstats
        if kwargs.get("gate_on_attack"):
            print(f"    gate: {gstats['chunks_attack_pred']}/{gstats['chunks']} chunks, "
                  f"writes={gstats['memory_writes']} blends={gstats['blends_applied']}")
        print(f"    {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  timesfm...")
    tfm_base, _ = timesfm_rollout(forecaster, series, cfg, use_memory=False)
    print(f"    {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  timesfm_ram...")
    tfm_ram, gs_ram = timesfm_rollout(forecaster, series, cfg, use_memory=True)
    print(f"    {time.time() - t0:.1f}s")

    if gate:
        t0 = time.time()
        print("  timesfm_ram_gated...")
        tfm_ram_g, gs_ram_g = timesfm_rollout(
            forecaster, series, cfg, use_memory=True,
            gate_model=base_model, full_states=full, gate_on_attack=True,
        )
        print(f"    gate: {gs_ram_g['chunks_attack_pred']}/{gs_ram_g['chunks']} chunks, "
              f"writes={gs_ram_g['memory_writes']} blends={gs_ram_g['blends_applied']}")
        print(f"    {time.time() - t0:.1f}s")
        gate_info["timesfm_ram_gated"] = gs_ram_g
    else:
        tfm_ram_g = None
    gate_info["timesfm_ram"] = gs_ram

    series_by_variant = {
        "ary_base": results["ary_base"][:, cfg.feature_idx],
        "ary_V8": results["ary_V8"][:, cfg.feature_idx],
        "ary_V10": results["ary_V10"][:, cfg.feature_idx],
        "timesfm": tfm_base,
        "timesfm_ram": tfm_ram,
    }
    if tfm_ram_g is not None:
        series_by_variant["ary_V8_gated"] = results["ary_V8_gated"][:, cfg.feature_idx]
        series_by_variant["ary_V10_gated"] = results["ary_V10_gated"][:, cfg.feature_idx]
        series_by_variant["timesfm_ram_gated"] = tfm_ram_g

    metrics = []
    for name, pred in series_by_variant.items():
        mask = ~np.isnan(pred)
        mse = float(np.mean((pred[mask] - series[mask]) ** 2)) if mask.sum() else float("nan")
        mae = float(np.mean(np.abs(pred[mask] - series[mask]))) if mask.sum() else float("nan")
        row = {"variant": name, "mse": mse, "mae": mae, "n_covered": int(mask.sum())}
        if name in gate_info:
            row["gate_stats"] = gate_info[name]
        metrics.append(row)
        print(f"    {name:<18} MSE={mse:.4f}  MAE={mae:.4f}  covered={int(mask.sum())}/{n}")

    payload = {
        "context": cfg.context,
        "horizon": cfg.horizon,
        "ary_lookback": ARY_LOOKBACK,
        "gate_ram_on_attack": cfg.gate_ram_on_attack,
        "attack_gate": cfg.attack_gate,
        "feature_idx": cfg.feature_idx,
        "feature_name": cfg.feature_name,
        "target_len": n,
        "metrics": metrics,
    }
    (cfg.feature_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    plot_bars(metrics, cfg.feature_dir / "bars.png", cfg)
    plot_timeline(series, series_by_variant, attack_segments, cfg.feature_dir / "timeline.png", cfg)
    print(f"  -> {cfg.feature_dir}")
    return payload


def plot_bars(metrics, out_path: Path, cfg: RunConfig):
    names = [m["variant"] for m in metrics]
    mses = [m["mse"] for m in metrics]
    colors = ["#7f8c8d", "#27ae60", "#e67e22", "#3498db", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(names, mses, color=colors[:len(names)])
    for b, v in zip(bars, mses):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", color="white", fontsize=9)
    ax.set_title(
        f"MSE — feat[{cfg.feature_idx}] {cfg.feature_name} | ctx={cfg.context} hor={cfg.horizon}",
        color="white", fontsize=11,
    )
    ax.tick_params(colors="white")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_timeline(actual, series_by_variant, attack_segments, out_path: Path, cfg: RunConfig):
    names = list(series_by_variant.keys())
    colors = {
        "ary_base": "#7f8c8d", "ary_V8": "#27ae60", "ary_V10": "#e67e22",
        "ary_V8_gated": "#2ecc71", "ary_V10_gated": "#f39c12",
        "timesfm": "#3498db", "timesfm_ram": "#e74c3c", "timesfm_ram_gated": "#c0392b",
    }
    x = np.arange(len(actual))

    fig, axes = plt.subplots(len(names), 1, figsize=(16, 2.6 * len(names)), sharex=True)
    fig.patch.set_facecolor("#0d1117")

    for ax, name in zip(axes, names):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        for lbl, a, b in attack_segments:
            ax.axvspan(a, b, color="#f85149", alpha=0.15, lw=0)
        ax.plot(x, actual, color="black", lw=1.4, label="actual", zorder=5)
        ax.plot(x, series_by_variant[name], color=colors.get(name, "#58a6ff"), lw=1.0, label=name, alpha=0.9, zorder=4)
        ax.set_title(name, color="white", fontsize=10, loc="left")
        ax.tick_params(colors="white")
        ax.legend(loc="upper right", fontsize=7, facecolor="#161b22", labelcolor="white", framealpha=0.6)

    axes[-1].set_xlabel("timeline step", color="white")
    fig.suptitle(
        f"feat[{cfg.feature_idx}] {cfg.feature_name} | ctx={cfg.context} hor={cfg.horizon} "
        f"(black=truth, red=attacks)",
        color="white", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--context", type=int, default=100, help="History steps taken before each forecast chunk")
    p.add_argument("--horizon", type=int, default=200, help="Steps predicted per chunk")
    p.add_argument("--feature-idx", type=int, default=None, help="Single 242-d feature index")
    p.add_argument("--feature-ranks", type=str, default="1,2",
                   help="Comma-separated variance ranks when --feature-idx omitted (e.g. 1,2,3)")
    p.add_argument("--gate-ram-on-attack", action="store_true",
                   help="Also run attack-gated RAM variants (memory only when P(attack) >= threshold)")
    p.add_argument("--attack-gate", type=float, default=0.5, help="P(attack) threshold for gated RAM")
    args = p.parse_args()

    ranks = [int(x.strip()) for x in args.feature_ranks.split(",") if x.strip()]
    splits = load_all_splits()
    train, val, test = splits["train"], splits["val"], splits["test"]
    tr_s, _, _ = train
    va_s, va_b, _ = val

    feature_indices = pick_feature_indices(tr_s, feature_idx=args.feature_idx, feature_ranks=ranks)
    full, labels, segments = build_timeline(train, val, test, TARGET_LEN)
    bin_labels, mit_labels = labels_to_ids(labels)

    print(f"Timeline: {len(full)} steps | context={args.context} horizon={args.horizon} "
          f"| ARY lookback={ARY_LOOKBACK} | features={feature_indices}")

    base_model = load_model()
    print("Calibrating memory thresholds on val...")
    raw_thresh, hidden_thresh, anomaly_thresh = calibrate_thresholds(base_model, va_s, va_b, args.context)
    print(f"  raw={raw_thresh:.2f} hidden={hidden_thresh:.2f} anomaly={anomaly_thresh:.2f}")

    print("\nLoading TimesFM-3...")
    t0 = time.time()
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    forecaster = TimesFM3Evaluator(
        ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=16, device="cuda")
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    run_summaries = []
    for fidx in feature_indices:
        feat_name = FEATURE_COLS_242[fidx] if 0 <= fidx < len(FEATURE_COLS_242) else f"dim_{fidx}"
        cfg = RunConfig(
            context=args.context, horizon=args.horizon, feature_idx=fidx, feature_name=feat_name,
            gate_ram_on_attack=args.gate_ram_on_attack, attack_gate=args.attack_gate,
        )
        summary = run_one_feature(
            cfg,
            full=full, labels=labels, segments=segments,
            bin_labels=bin_labels, mit_labels=mit_labels,
            base_model=base_model, forecaster=forecaster,
            raw_thresh=raw_thresh, hidden_thresh=hidden_thresh, anomaly_thresh=anomaly_thresh,
        )
        run_summaries.append(summary)

    config_dir = RunConfig(
        args.context, args.horizon, feature_indices[0], "summary",
        gate_ram_on_attack=args.gate_ram_on_attack, attack_gate=args.attack_gate,
    ).config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run_summary.json").write_text(json.dumps({
        "context": args.context,
        "horizon": args.horizon,
        "ary_lookback": ARY_LOOKBACK,
        "gate_ram_on_attack": args.gate_ram_on_attack,
        "attack_gate": args.attack_gate,
        "features": run_summaries,
    }, indent=2))
    print(f"\nRun summary -> {config_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
