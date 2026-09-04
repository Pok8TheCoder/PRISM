#!/usr/bin/env python3
"""SHAP / gradient attribution analysis for PRISM model zoo.

Compares feature drivers across TP / TN / FP / FN buckets to surface root
causes of missed detections and false alarms.

Supported models (regular stack):
  ary5        ARY-5sV01 @ 242-d, 5s windows, lookback 20
  ary30       ARY.01 best @ 242-d, 30s windows, lookback 20
  shaun       Shaun PRISM V2 @ 292-d, 15s windows, lookback 30

Notes:
  - timesfm_ary_cls uses the same ARY classifier as ary5 for attack detection;
    run `--models ary5` for that path. TimesFM forecast heads are not covered here.
  - RAMX wrappers are not separate models — analyze the base checkpoint, then
    compare failure buckets on replay traces separately.

Usage:
  pip install shap   # listed in requirements.txt; gradient-only works without it
  python scripts/shap_model_analysis.py
  python scripts/shap_model_analysis.py --models ary5,shaun --max-per-bucket 30
  python scripts/shap_model_analysis.py --models ary5 --method gradient --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
sys.path.insert(0, str(ROOT))

from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.feature_schema242 import FEATURE_COLS_242  # noqa: E402
from src.explainability.feature_names import (  # noqa: E402
    ary_block_indices,
    shaun_block_indices,
    shaun_feature_names_292,
)
from src.explainability.shap_analysis import (  # noqa: E402
    LastStepAttackWrapper,
    ModelSpec,
    aggregate_blocks,
    assign_buckets,
    build_labeled_sequences,
    explain_samples,
    failure_delta,
    load_ary_model,
    load_shaun_bundle,
    sample_by_bucket,
    top_features,
    write_report_md,
    ary_attack_prob,
    shaun_attack_prob,
)

OUT_DIR = ROOT / "results" / "shap_analysis"
ARY5_CKPT = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
ARY30_CKPT = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"
SPLITS_5S = ROOT / "data" / "aryan_splits_5s"
SPLITS_30 = ROOT / "data" / "aryan_splits"


def model_registry() -> dict[str, ModelSpec]:
    return {
        "ary5": ModelSpec(
            key="ary5",
            label="ARY-5sV01",
            architecture="ary",
            ckpt=ARY5_CKPT,
            lookback=20,
            splits_dir=SPLITS_5S,
            feature_names=list(FEATURE_COLS_242),
            block_indices=ary_block_indices(),
            notes="TimesFM hybrid attack classifier reuses this same ARY base head.",
        ),
        "ary30": ModelSpec(
            key="ary30",
            label="ARY.01 (30s)",
            architecture="ary",
            ckpt=ARY30_CKPT,
            lookback=20,
            splits_dir=SPLITS_30,
            feature_names=list(FEATURE_COLS_242),
            block_indices=ary_block_indices(),
        ),
        "shaun": ModelSpec(
            key="shaun",
            label="Shaun PRISM V2",
            architecture="shaun",
            ckpt=SHAUN_ROOT / "weights" / "world_model.pt",
            lookback=30,
            shaun_root=SHAUN_ROOT,
            feature_names=shaun_feature_names_292(),
            block_indices=shaun_block_indices(),
            notes="Uses Shaun processed holdout states (attack_labels.npy). RAMX fusion not included.",
        ),
    }


def load_eval_sequences(spec: ModelSpec) -> list:
    from src.explainability.shap_analysis import SequenceSample

    if spec.architecture == "ary":
        assert spec.splits_dir is not None
        te_s, te_b, _ = load_all_splits(spec.splits_dir)["test"]
        return build_labeled_sequences(te_s, te_b, spec.lookback)

    states_path = spec.shaun_root / "data/processed/states.npy"
    labels_path = spec.shaun_root / "data/processed/attack_labels.npy"
    if not states_path.exists():
        raise FileNotFoundError(f"Missing Shaun data: {states_path}")
    states = np.load(states_path)
    labels = np.load(labels_path)
    # Chronological holdout: last 15% as eval (matches Shaun benchmark docs).
    cut = int(len(states) * 0.85)
    return build_labeled_sequences(states[cut:], labels[cut:], spec.lookback)


def load_background_last_steps(spec: ModelSpec, n: int = 32) -> np.ndarray:
    if spec.architecture == "ary":
        assert spec.splits_dir is not None
        va_s, va_b, _ = load_all_splits(spec.splits_dir)["val"]
        idx = np.where(va_b == 0)[0]
        idx = idx[idx >= spec.lookback - 1]
        pick = idx[:n]
        return np.stack([va_s[i] for i in pick]).astype(np.float32)

    states_path = spec.shaun_root / "data/processed/states.npy"
    labels_path = spec.shaun_root / "data/processed/attack_labels.npy"
    states = np.load(states_path)
    labels = np.load(labels_path)
    cut = int(len(states) * 0.85)
    train_states, train_labels = states[:cut], labels[:cut]
    idx = np.where(train_labels == 0)[0][:n]
    return train_states[idx].astype(np.float32)


def make_wrapper_factory(spec: ModelSpec, bundle: dict | None, ary_model: torch.nn.Module | None):
    def factory(sample):
        prefix = torch.from_numpy(sample.seq[:-1].astype(np.float32)).unsqueeze(0)
        if spec.architecture == "shaun":
            assert bundle is not None
            return LastStepAttackWrapper(
                "shaun",
                bundle["model"],
                prefix,
                shaun_mean=bundle["mean"],
                shaun_std=bundle["std"],
            )
        assert ary_model is not None
        return LastStepAttackWrapper("ary", ary_model, prefix)

    return factory


def plot_block_bars(block_means: dict[str, dict[str, float]], out_path: Path, title: str) -> None:
    buckets = [k for k in ("tp", "fp", "fn", "tn") if k in block_means]
    if not buckets:
        return
    blocks = list(next(iter(block_means.values())).keys())
    x = np.arange(len(blocks))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, bucket in enumerate(buckets):
        vals = [block_means[bucket].get(b, 0.0) for b in blocks]
        ax.bar(x + i * width, vals, width, label=bucket.upper())
    ax.set_xticks(x + width * (len(buckets) - 1) / 2)
    ax.set_xticklabels(blocks, rotation=30, ha="right")
    ax.set_ylabel("Mean |attribution|")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_top_features(
    fn_imp: np.ndarray,
    tp_imp: np.ndarray,
    feature_names: list[str],
    out_path: Path,
    k: int = 12,
) -> None:
    delta = fn_imp - tp_imp
    order = np.argsort(np.abs(delta))[::-1][:k]
    labels = [feature_names[i] if i < len(feature_names) else f"d{i}" for i in order]
    vals = delta[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("FN − TP mean |attribution|")
    ax.set_title("Failure root-cause candidates")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def analyze_model(spec: ModelSpec, args: argparse.Namespace) -> dict:
    if not spec.ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {spec.ckpt}")

    print(f"\n=== {spec.label} ({spec.key}) ===")
    sequences = load_eval_sequences(spec)
    print(f"  eval windows: {len(sequences)}")

    bundle = None
    ary_model = None
    if spec.architecture == "shaun":
        bundle = load_shaun_bundle(spec.shaun_root)
        scorer = lambda seq: shaun_attack_prob(bundle, seq)
    else:
        ary_model = load_ary_model(spec.ckpt, spec.lookback)
        scorer = lambda seq: ary_attack_prob(ary_model, seq)

    assign_buckets(sequences, scorer, threshold=args.threshold)
    counts = {k: sum(1 for s in sequences if s.bucket == k) for k in ("tp", "tn", "fp", "fn")}
    print(f"  buckets: {counts}")

    sampled = sample_by_bucket(sequences, args.max_per_bucket, seed=args.seed)
    background = load_background_last_steps(spec, n=args.background_size)
    wrapper_factory = make_wrapper_factory(spec, bundle, ary_model)

    block_means: dict[str, dict[str, float]] = {}
    bucket_top: dict[str, list] = {}
    bucket_rows: dict[str, list] = {}

    for bucket in ("tp", "tn", "fp", "fn"):
        rows = sampled[bucket]
        if not rows:
            continue
        mean_abs, detail = explain_samples(
            rows,
            wrapper_factory,
            background,
            spec.feature_names,
            method=args.method,
        )
        block_means[bucket] = aggregate_blocks(mean_abs, spec.block_indices)
        bucket_top[bucket] = top_features(mean_abs, spec.feature_names, k=20)
        bucket_rows[bucket] = detail
        print(f"  explained {bucket}: n={len(rows)}")

    failure_analysis = {}
    if "fn" in block_means and "tp" in block_means:
        fn_imp, _ = explain_samples(
            sampled["fn"], wrapper_factory, background, spec.feature_names, method=args.method,
        )
        tp_imp, _ = explain_samples(
            sampled["tp"], wrapper_factory, background, spec.feature_names, method=args.method,
        )
        if len(fn_imp) and len(tp_imp):
            failure_analysis = {
                "top_deltas": failure_delta(fn_imp, tp_imp, spec.feature_names, k=25),
                "fn_top": top_features(fn_imp, spec.feature_names, k=15),
                "tp_top": top_features(tp_imp, spec.feature_names, k=15),
            }

    out_model_dir = args.out_dir / spec.key
    out_model_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": {
            "key": spec.key,
            "label": spec.label,
            "architecture": spec.architecture,
            "ckpt": str(spec.ckpt),
            "lookback": spec.lookback,
            "notes": spec.notes,
        },
        "method": args.method,
        "threshold": args.threshold,
        "bucket_counts": counts,
        "sampled_counts": {k: len(v) for k, v in sampled.items()},
        "block_means": block_means,
        "top_features_by_bucket": bucket_top,
        "sample_details": bucket_rows,
        "failure_analysis": failure_analysis,
    }

    (out_model_dir / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report_md(payload, out_model_dir / "REPORT.md")

    if not args.no_plots:
        plot_block_bars(block_means, out_model_dir / "blocks_by_bucket.png", f"{spec.label} block attribution")
        if failure_analysis.get("top_deltas") and "fn" in sampled and "tp" in sampled:
            fn_imp, _ = explain_samples(
                sampled["fn"], wrapper_factory, background, spec.feature_names, method="gradient",
            )
            tp_imp, _ = explain_samples(
                sampled["tp"], wrapper_factory, background, spec.feature_names, method="gradient",
            )
            if len(fn_imp) and len(tp_imp):
                plot_top_features(fn_imp, tp_imp, spec.feature_names, out_model_dir / "fn_vs_tp_delta.png")

    print(f"  wrote {out_model_dir}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="SHAP attribution for PRISM models")
    ap.add_argument("--models", default="ary5,shaun", help="comma-separated: ary5,ary30,shaun")
    ap.add_argument("--max-per-bucket", type=int, default=25, help="samples per TP/TN/FP/FN")
    ap.add_argument("--background-size", type=int, default=32, help="SHAP background windows")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--method", choices=("gradient", "shap", "both"), default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if args.method in ("shap", "both"):
        try:
            import shap  # noqa: F401
        except ImportError:
            print("WARNING: shap not installed — falling back to gradient-only.", file=sys.stderr)
            print("  pip install shap", file=sys.stderr)
            args.method = "gradient"

    reg = model_registry()
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    missing = [k for k in keys if k not in reg]
    if missing:
        print(f"Unknown models: {missing}. Available: {list(reg)}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for key in keys:
        summaries.append(analyze_model(reg[key], args))

    index = {
        "models": keys,
        "out_dir": str(args.out_dir),
        "method": args.method,
        "summaries": [
            {
                "key": s["model"]["key"],
                "label": s["model"]["label"],
                "bucket_counts": s["bucket_counts"],
                "failure_top": (s.get("failure_analysis") or {}).get("top_deltas", [])[:5],
            }
            for s in summaries
        ],
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join([
            "# Model SHAP / Attribution Analysis",
            "",
            "Per-model outputs live in subfolders with:",
            "- `analysis.json` — full numeric results",
            "- `REPORT.md` — human summary",
            "- `blocks_by_bucket.png` — block-level |attribution| by outcome bucket",
            "- `fn_vs_tp_delta.png` — FN vs TP root-cause candidates (if both buckets exist)",
            "",
            "Re-run:",
            "```bash",
            "python scripts/shap_model_analysis.py --models ary5,ary30,shaun",
            "```",
        ]),
        encoding="utf-8",
    )
    print(f"\nDone. Index: {args.out_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
