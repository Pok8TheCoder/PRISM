#!/usr/bin/env python3
"""Lab timeline plots: actual vs P(attack) for ARY 30s/5s base + RAMX.

Fixes eval_ary5s label bug — PCAP ground truth comes from filename via
``resolve_pcap_class``, not from missing flow Label columns.

Usage:
  python scripts/plot_lab_ary_comparison.py
  python scripts/plot_lab_ary_comparison.py --pcaps live_http_flood_none.pcap r1_a1_T1110_ssh_bruteforce_none.pcap
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import pcap_to_states  # noqa: E402
from src.aryan.streaming_variants import (  # noqa: E402
    StreamingARY,
    StreamingARYRamxV01,
    calibrate_thresholds,
    load_model,
)
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402

CKPT_30 = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"
CKPT_5 = ROOT / "models" / "checkpoints" / "ary5s_v01.pt"
SPLITS_30 = ROOT / "data" / "aryan_splits"
SPLITS_5 = ROOT / "data" / "aryan_splits_5s"
OUT_DIR = ROOT / "results" / "lab_ary_comparison"
EVAL_DIR = OUT_DIR / "eval"
WARMUP_N = 20

DEFAULT_PCAPS = [
    "live_http_flood_none.pcap",
    "live_port_scan_sequential_none.pcap",
    "live_slow_loris_none.pcap",
    "r1_a1_T1110_ssh_bruteforce_none.pcap",
    "r3_a1_http_flood_none.pcap",
    "r27_a1_T1499_http_flood_none.pcap",
]

COLORS = {
    "actual": "#f85149",
    "ary30_base": "#58a6ff",
    "ary30_ramx": "#1f6feb",
    "ary5_base": "#3fb950",
    "ary5_ramx": "#238636",
}


def pcap_labels(pcap_name: str, n: int) -> list[int]:
    cls = resolve_pcap_class(pcap_name) or "Benign"
    attack = 0 if cls == "Benign" else 1
    return [attack] * n


def benign_warmup(splits_dir: Path, n: int) -> tuple[list[np.ndarray], list[int], list[int]]:
    va_s, va_b, va_m = load_all_splits(splits_dir)["val"]
    idx = np.where(va_b == 0)[0][:n]
    states = [np.asarray(va_s[i], dtype=np.float32) for i in idx]
    bins = [0] * len(states)
    mits = [int(va_m[i]) for i in idx]
    return states, bins, mits


def build_timeline(
    pcap: Path,
    window_sec: float,
    warmup_states: list[np.ndarray],
    warmup_bins: list[int],
    warmup_mits: list[int],
) -> tuple[list[np.ndarray], list[int], list[int], int]:
    states, _, _ = pcap_to_states(pcap, window_sec=window_sec)
    if len(states) == 0:
        return [], [], [], 0
    atk_states = [np.asarray(s, dtype=np.float32) for s in states]
    atk_bins = pcap_labels(pcap.name, len(atk_states))
    atk_mits = [0] * len(atk_bins)
    all_s = warmup_states + atk_states
    all_b = warmup_bins + atk_bins
    all_m = warmup_mits + atk_mits
    return all_s, all_b, all_m, len(warmup_states)


def replay(system_factory, states, bins, mits) -> list[float]:
    sys_obj = system_factory()
    p_atts = []
    for s, tb, tm in zip(states, bins, mits):
        out = sys_obj.step(s, true_bin=tb, true_mit=tm)
        p_atts.append(out["p_att"])
    return p_atts


def score_round(p_atts: list[float], bins: list[int], attack_start: int) -> dict:
    pred = [1 if p >= 0.5 else 0 for p in p_atts]
    tp = sum(1 for p, t in zip(pred, bins) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, bins) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, bins) if p == 0 and t == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ttd = None
    for i in range(attack_start, len(bins)):
        if pred[i] == 1 and bins[i] == 1:
            ttd = i - attack_start
            break
    harm = sum(1 for p, t in zip(pred[:attack_start], bins[:attack_start]) if t == 0 and p == 1)
    return {
        "f1": f1, "precision": prec, "recall": rec, "ttd": ttd,
        "detected": ttd is not None, "harm": harm,
        "max_p_attack": max(p_atts[attack_start:], default=0.0),
    }


def variant_payload(bins: list[int], p_atts: list[float], attack_start: int) -> dict:
    scores = score_round(p_atts, bins, attack_start)
    return {
        "metrics": scores,
        "attack_start": attack_start,
        "trace": {
            "step": list(range(len(bins))),
            "true_binary": bins,
            "p_attack": p_atts,
        },
    }


def save_eval_json(payload: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{payload['scenario_id']}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def plot_round(
    pcap_name: str,
    window_sec: float,
    bins: list[int],
    attack_start: int,
    series: dict[str, list[float]],
    out_path: Path,
) -> None:
    n = len(bins)
    x = np.arange(n)
    actual = np.array(bins, dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True, height_ratios=[1, 3])
    fig.patch.set_facecolor("#0d1117")

    ax0, ax1 = axes
    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.axvspan(attack_start, n, color="#f85149", alpha=0.10, lw=0)
        ax.tick_params(colors="white")

    ax0.step(x, actual, where="mid", color=COLORS["actual"], lw=2.0, label="actual (ground truth)")
    ax0.set_ylim(-0.15, 1.25)
    ax0.set_ylabel("label", color="white")
    ax0.set_title(f"Ground truth — attack windows = 1", color="white", fontsize=10, loc="left")
    ax0.legend(loc="upper right", fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    order = ["ary30_base", "ary30_ramx", "ary5_base", "ary5_ramx"]
    labels = {
        "ary30_base": "ARY.01 30s base",
        "ary30_ramx": "ARY.01 30s + RAMX",
        "ary5_base": "ARY 5s base",
        "ary5_ramx": "ARY 5s + RAMX",
    }
    for key in order:
        if key not in series:
            continue
        p = series[key]
        ax1.plot(x[: len(p)], p, color=COLORS[key], lw=1.4, label=labels[key], alpha=0.95)
    ax1.axhline(0.5, color="#c9d1d9", lw=0.8, ls="--", alpha=0.45)
    ax1.set_ylim(-0.05, 1.08)
    ax1.set_ylabel("P(attack)", color="white")
    ax1.set_xlabel("window step (benign warm-up → attack PCAP)", color="white")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white", ncol=2)
    ax1.set_title("Model predictions", color="white", fontsize=10, loc="left")

    cls = resolve_pcap_class(pcap_name) or "Benign"
    fig.suptitle(
        f"{pcap_name}  ·  class={cls}  ·  ingest={window_sec:.0f}s windows  ·  warm-up={attack_start} steps",
        color="white",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_pcap_comparison(
    pcap_name: str,
    b30: list[int],
    start30: int,
    series_30: dict[str, list[float]],
    b5: list[int],
    start5: int,
    series_5: dict[str, list[float]],
    out_path: Path,
) -> None:
    cls = resolve_pcap_class(pcap_name) or "Benign"
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=False)
    fig.patch.set_facecolor("#0d1117")

    panels = [
        ("30s ground truth", b30, start30, None, COLORS["actual"]),
        ("30s predictions", b30, start30, series_30, None),
        ("5s ground truth", b5, start5, None, COLORS["actual"]),
        ("5s predictions", b5, start5, series_5, None),
    ]
    pred_labels = {
        "ary30_base": "ARY.01 30s base",
        "ary30_ramx": "ARY.01 30s + RAMX",
        "ary5_base": "ARY 5s base",
        "ary5_ramx": "ARY 5s + RAMX",
    }

    for ax, (title, bins, attack_start, series, line_color) in zip(axes, panels):
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="white")
        n = len(bins)
        x = np.arange(n)
        ax.axvspan(attack_start, n, color="#f85149", alpha=0.10, lw=0)
        if series is None:
            ax.step(x, np.array(bins, dtype=float), where="mid", color=line_color, lw=2.0)
            ax.set_ylim(-0.15, 1.25)
            ax.set_ylabel("label", color="white")
        else:
            for key, p in series.items():
                ax.plot(x[: len(p)], p, color=COLORS[key], lw=1.4, label=pred_labels[key])
            ax.axhline(0.5, color="#c9d1d9", lw=0.8, ls="--", alpha=0.45)
            ax.set_ylim(-0.05, 1.08)
            ax.set_ylabel("P(attack)", color="white")
            ax.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="white")
        ax.set_title(title, color="white", fontsize=10, loc="left")
        ax.set_xlabel("window step (CIC warm-up → attack PCAP)", color="white")

    fig.suptitle(f"{pcap_name}  ·  class={cls}", color="white", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_live_round(round_path: Path, model_30, hidden_30, model_5, hidden_5, out_path: Path) -> dict | None:
    """Replay live_lab round.json trace (15s states) on both checkpoints."""
    data = json.loads(round_path.read_text())
    trace = data["trace"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    bins = [int(t["true_bin"]) for t in trace]
    mits = [int(t["true_mit"]) for t in trace]
    attack_start = next((i for i, b in enumerate(bins) if b == 1), len(bins))

    systems = {
        "ary30_base": lambda: StreamingARY("ary_base", base_model=model_30, hidden_thresh=hidden_30),
        "ary30_ramx": lambda: StreamingARYRamxV01(base_model=model_30, hidden_thresh=hidden_30),
        "ary5_base": lambda: StreamingARY("ary_base", base_model=model_5, hidden_thresh=hidden_5),
        "ary5_ramx": lambda: StreamingARYRamxV01(base_model=model_5, hidden_thresh=hidden_5),
    }
    series = {k: replay(f, states, bins, mits) for k, f in systems.items()}
    plot_round(
        f"{data['objective']}/{data.get('evasion', 'none')} (live round)",
        float(data.get("window_sec", 15)),
        bins,
        attack_start,
        series,
        out_path,
    )
    scores = {k: score_round(series[k], bins, attack_start) for k in systems}
    payload = {
        "scenario_id": f"live_{data['objective']}_{data.get('evasion', 'none')}",
        "kind": "live_round",
        "title": f"{data['objective']}/{data.get('evasion', 'none')} (live round)",
        "source": str(round_path.relative_to(ROOT)),
        "objective": data["objective"],
        "class_id": data.get("class_id"),
        "window_sec": float(data.get("window_sec", 15)),
        "attack_start": attack_start,
        "n_windows": len(bins),
        "variants": {
            k: variant_payload(bins, series[k], attack_start) for k in systems
        },
    }
    save_eval_json(payload, EVAL_DIR)
    return {
        "source": str(round_path.relative_to(ROOT)),
        "objective": data["objective"],
        "class_id": data.get("class_id"),
        "window_sec_trace": data.get("window_sec"),
        "n_windows": len(bins),
        "attack_start": attack_start,
        "scores": scores,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pcaps", nargs="*", default=DEFAULT_PCAPS)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--include-live-rounds", action="store_true", default=True)
    args = p.parse_args()

    print("Loading models...")
    model_30 = load_model(CKPT_30)
    model_5 = load_model(CKPT_5)
    _, hidden_30 = calibrate_thresholds(model_30, *load_all_splits(SPLITS_30)["val"][:2])
    _, hidden_5 = calibrate_thresholds(model_5, *load_all_splits(SPLITS_5)["val"][:2])
    print(f"  hidden_thresh 30s={hidden_30:.3f}  5s={hidden_5:.3f}")

    w30_s, w30_b, w30_m = benign_warmup(SPLITS_30, WARMUP_N)
    w5_s, w5_b, w5_m = benign_warmup(SPLITS_5, WARMUP_N)

    all_results: list[dict] = []

    for name in args.pcaps:
        pcap = SAVE_DIR / name
        if not pcap.exists():
            print(f"  skip missing {name}")
            continue
        cls = resolve_pcap_class(name) or "Benign"
        print(f"  PCAP {name} ({cls})")

        s30, b30, m30, start30 = build_timeline(pcap, 30.0, w30_s, w30_b, w30_m)
        s5, b5, m5, start5 = build_timeline(pcap, 5.0, w5_s, w5_b, w5_m)
        if not s30 or not s5:
            print(f"    empty states, skip")
            continue

        systems_30 = {
            "ary30_base": lambda: StreamingARY("ary_base", base_model=model_30, hidden_thresh=hidden_30),
            "ary30_ramx": lambda: StreamingARYRamxV01(base_model=model_30, hidden_thresh=hidden_30),
        }
        systems_5 = {
            "ary5_base": lambda: StreamingARY("ary_base", base_model=model_5, hidden_thresh=hidden_5),
            "ary5_ramx": lambda: StreamingARYRamxV01(base_model=model_5, hidden_thresh=hidden_5),
        }
        series = {}
        for k, f in systems_30.items():
            series[k] = replay(f, s30, b30, m30)
        for k, f in systems_5.items():
            series[k] = replay(f, s5, b5, m5)

        stem = Path(name).stem
        plot_pcap_comparison(name, b30, start30, {k: series[k] for k in ("ary30_base", "ary30_ramx")},
                             b5, start5, {k: series[k] for k in ("ary5_base", "ary5_ramx")},
                             args.out_dir / f"{stem}_timeline.png")

        scores = {}
        for k in series:
            bins_k = b30 if k.startswith("ary30") else b5
            start_k = start30 if k.startswith("ary30") else start5
            scores[k] = score_round(series[k], bins_k, start_k)

        payload = {
            "scenario_id": stem,
            "kind": "pcap",
            "title": name,
            "class_id": cls,
            "pcap": name,
            "panels": {
                "30s": {
                    "window_sec": 30.0,
                    "attack_start": start30,
                    "n_windows": len(b30),
                    "variants": {
                        "ary30_base": variant_payload(b30, series["ary30_base"], start30),
                        "ary30_ramx": variant_payload(b30, series["ary30_ramx"], start30),
                    },
                },
                "5s": {
                    "window_sec": 5.0,
                    "attack_start": start5,
                    "n_windows": len(b5),
                    "variants": {
                        "ary5_base": variant_payload(b5, series["ary5_base"], start5),
                        "ary5_ramx": variant_payload(b5, series["ary5_ramx"], start5),
                    },
                },
            },
            "scores": scores,
        }
        save_eval_json(payload, EVAL_DIR)

        row = {
            "pcap": name,
            "class_id": cls,
            "n_windows_30s": len(b30),
            "n_windows_5s": len(b5),
            "attack_start_30s": start30,
            "attack_start_5s": start5,
            "scores": scores,
        }
        all_results.append(row)
        print(f"    F1  30s base={scores['ary30_base']['f1']:.3f} ramx={scores['ary30_ramx']['f1']:.3f}  "
              f"5s base={scores['ary5_base']['f1']:.3f} ramx={scores['ary5_ramx']['f1']:.3f}")

    if args.include_live_rounds:
        live_dir = ROOT / "results" / "ram_improve" / "live_lab"
        for rf in sorted(live_dir.rglob("round.json")):
            if rf.parent.name != "none":
                continue
            rel = rf.relative_to(live_dir)
            out_png = args.out_dir / f"live_{rel.parent.parent.name}_none_timeline.png"
            print(f"  live round {rel.parent.parent.name}/none")
            row = plot_live_round(rf, model_30, hidden_30, model_5, hidden_5, out_png)
            if row:
                all_results.append(row)
                sc = row["scores"]
                print(f"    F1  30s base={sc['ary30_base']['f1']:.3f} ramx={sc['ary30_ramx']['f1']:.3f}  "
                      f"5s base={sc['ary5_base']['f1']:.3f} ramx={sc['ary5_ramx']['f1']:.3f}")

    summary_path = args.out_dir / "lab_comparison_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2))
    n_eval = len(list(EVAL_DIR.glob("*.json"))) if EVAL_DIR.exists() else 0
    print(f"\nSaved {len(list(args.out_dir.glob('*.png')))} plots -> {args.out_dir}")
    print(f"Viewer JSON ({n_eval} scenarios) -> {EVAL_DIR}")
    print(f"Open viewer: python scripts/falloff_viewer.py --mode ary_compare")
    print(f"Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
