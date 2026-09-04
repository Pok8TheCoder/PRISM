#!/usr/bin/env python3
"""Benchmark base vs RAMX across traffic-density tiers + native CIC holdout.

Tests the hypothesis that Shaun V2 base should match RAMX once lab traffic
reaches CIC training density (~450 flows/15s window), not literal millions
of flows per window (CIC aggregates into hundreds per window).

Usage:
  python scripts/bench_realistic_scale.py
  python scripts/bench_realistic_scale.py --max-pcaps 20 --tiers quiet_lab,cic_like,enterprise
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
sys.path.insert(0, str(ROOT))

from scripts.eval_ary5s import load_ckpt  # noqa: E402
from scripts.plot_lab_ary_comparison import benign_warmup, replay, score_round  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402
from src.aryan.streaming_variants import StreamingARY, StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.shaun.streaming import StreamingShaunBase, StreamingShaunRamxV2, load_shaun_bundle  # noqa: E402

from scripts.bench_scaled_lab import (  # noqa: E402
    build_scaled_timeline,
    eval_pcap,
    run_shaun_sidecar,
    summarize,
)

CKPT_5 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
SPLITS_5 = ROOT / "data" / "aryan_splits_5s"
OUT_DIR = ROOT / "results" / "realistic_scale_bench"
WARMUP_N = 20

SCALE_TIERS = [
    {
        "id": "quiet_lab",
        "scale_factor": 1.0,
        "replicate": 1,
        "note": "Docker lab default (~20–100 flows/window)",
    },
    {
        "id": "cic_like",
        "scale_factor": 10.0,
        "replicate": 5,
        "note": "CIC training density (~500 flows/window)",
    },
    {
        "id": "enterprise",
        "scale_factor": 50.0,
        "replicate": 10,
        "note": "High-volume enterprise (~5k flows/window)",
    },
    {
        "id": "extreme",
        "scale_factor": 100.0,
        "replicate": 20,
        "note": "Stress tier (~10k+ flows/window)",
    },
]


def attack_pcaps(max_n: int = 0) -> list[Path]:
    pcaps = sorted(SAVE_DIR.glob("*.pcap"))
    atk = [p for p in pcaps if (resolve_pcap_class(p.name) or "Benign") != "Benign"]
    if max_n > 0:
        atk = atk[:max_n]
    return atk


def eval_cic_native_holdout(max_windows: int = 200) -> dict:
    """Score Shaun base vs RAMX on native CIC attack windows (no PCAP scaling)."""
    states_path = SHAUN_ROOT / "data/processed/states.npy"
    labels_path = SHAUN_ROOT / "data/processed/attack_labels.npy"
    if not states_path.exists() or not labels_path.exists():
        return {"error": f"missing Shaun processed data under {SHAUN_ROOT / 'data/processed'}"}

    states = np.load(states_path)
    labels = np.load(labels_path)
    benign_idx = np.where(labels == 0)[0][:WARMUP_N]
    attack_idx = np.where(labels == 1)[0]
    if max_windows > 0:
        attack_idx = attack_idx[:max_windows]

    warm_states = [states[i].astype(np.float32) for i in benign_idx]
    bundle = load_shaun_bundle()
    base = StreamingShaunBase(bundle)
    ramx = StreamingShaunRamxV2(bundle, context_skip_steps=WARMUP_N)

    base_rows: list[dict] = []
    ramx_rows: list[dict] = []
    flows_med: list[float] = []

    for ai in attack_idx:
        atk = states[ai].astype(np.float32)
        flows_med.append(float(np.expm1(atk[276])))
        timeline = warm_states + [atk]
        bins = [0] * WARMUP_N + [1]
        start = WARMUP_N

        pb = replay(lambda: StreamingShaunBase(bundle), timeline, bins, [0] * len(bins))
        pr = replay(lambda: StreamingShaunRamxV2(bundle, context_skip_steps=WARMUP_N), timeline, bins, [0] * len(bins))
        base_rows.append(score_round(pb, bins, start))
        ramx_rows.append(score_round(pr, bins, start))
        base_rows[-1]["attack_max_p"] = float(np.max(pb[start:]))
        ramx_rows[-1]["attack_max_p"] = float(np.max(pr[start:]))
        base_rows[-1]["warmup_mean_p"] = float(np.mean(pb[:start]))
        ramx_rows[-1]["warmup_mean_p"] = float(np.mean(pr[:start]))

    def _agg(rows: list[dict]) -> dict:
        return {
            "n": len(rows),
            "mean_f1": float(np.mean([r["f1"] for r in rows])),
            "detection_rate": float(np.mean([1.0 if r["detected"] else 0.0 for r in rows])),
            "mean_attack_max_p": float(np.mean([r["attack_max_p"] for r in rows])),
            "mean_warmup_mean_p": float(np.mean([r["warmup_mean_p"] for r in rows])),
        }

    b = _agg(base_rows)
    r = _agg(ramx_rows)
    return {
        "source": "shaun_processed_holdout_attack_windows",
        "n_attack_windows": len(attack_idx),
        "flows_per_window_med": float(np.median(flows_med)) if flows_med else 0.0,
        "shaun_base": b,
        "shaun_ramx": r,
        "ramx_delta": {
            "f1": r["mean_f1"] - b["mean_f1"],
            "detection_rate": r["detection_rate"] - b["detection_rate"],
            "attack_max_p": r["mean_attack_max_p"] - b["mean_attack_max_p"],
        },
    }


def ramx_delta(summary: dict, base_key: str, ramx_key: str) -> dict:
    b, r = summary.get(base_key, {}), summary.get(ramx_key, {})
    if not b or not r:
        return {}
    return {
        "f1": r.get("mean_f1", 0) - b.get("mean_f1", 0),
        "detection_rate": r.get("detection_rate", 0) - b.get("detection_rate", 0),
        "attack_max_p": r.get("mean_attack_max_p", 0) - b.get("mean_attack_max_p", 0),
    }


def write_report(payload: dict, out_dir: Path) -> None:
    lines = [
        "# Realistic Scale Benchmark: Does RAMX matter at CIC-like density?",
        "",
        "Shaun V2 was trained on **millions of total flows** across CIC/IoT captures,",
        "but each **15s state window** aggregates to **~450 flows** (not millions per window).",
        "This bench sweeps synthetic scale tiers on lab PCAPs plus a **native CIC holdout** replay.",
        "",
        "## Scale tiers (lab PCAPs)",
        "",
        "| Tier | Scale | Replicate | Note |",
        "|---|---|---|---|",
    ]
    for t in SCALE_TIERS:
        lines.append(
            f"| `{t['id']}` | {t['scale_factor']:g}× | {t['replicate']}× | {t['note']} |"
        )

    lines.extend(["", "## Lab PCAP aggregates (attack captures only)", ""])
    for tier in payload["tiers"]:
        tid = tier["id"]
        s = tier["lab_summary"]
        lines.append(f"### {tid}")
        lines.append("")
        lines.append("| Model | Mean F1 | Det rate | Attack max P | Warmup P |")
        lines.append("|---|---|---|---|---|")
        for label, key in [
            ("ARY-5s base", "ary5_base"),
            ("ARY-5s + RAMX", "ary5_ramx"),
            ("Shaun base", "shaun_v2"),
            ("Shaun + RAMX", "shaun_v2_ramx"),
        ]:
            row = s.get(key, {})
            if row:
                lines.append(
                    f"| {label} | {row['mean_f1']:.3f} | {row['detection_rate']:.1%} | "
                    f"{row['mean_attack_max_p']:.3f} | {row['mean_warmup_mean_p']:.3f} |"
                )
        d = tier.get("ramx_delta", {})
        if d.get("shaun"):
            sd = d["shaun"]
            lines.append("")
            lines.append(
                f"**Shaun RAMX delta:** F1 {sd['f1']:+.3f}, det {sd['detection_rate']:+.1%}, "
                f"attack max P {sd['attack_max_p']:+.3f}"
            )
        lines.append("")

    cic = payload.get("cic_native", {})
    if cic and "shaun_base" in cic:
        lines.extend([
            "## Native CIC holdout (no PCAP scaling)",
            "",
            f"Attack windows evaluated: **{cic['n_attack_windows']}** | "
            f"Median flows/window: **{cic['flows_per_window_med']:.0f}**",
            "",
            "| Model | Mean F1 | Det rate | Attack max P | Warmup P |",
            "|---|---|---|---|---|",
        ])
        for label, key in [("Shaun base", "shaun_base"), ("Shaun + RAMX", "shaun_ramx")]:
            row = cic[key]
            lines.append(
                f"| {label} | {row['mean_f1']:.3f} | {row['detection_rate']:.1%} | "
                f"{row['mean_attack_max_p']:.3f} | {row['mean_warmup_mean_p']:.3f} |"
            )
        rd = cic["ramx_delta"]
        lines.extend([
            "",
            f"**Shaun RAMX delta on CIC native:** F1 {rd['f1']:+.3f}, "
            f"det {rd['detection_rate']:+.1%}, attack max P {rd['attack_max_p']:+.3f}",
            "",
            "## Interpretation",
            "",
            "- If your friend means **training-corpus scale**, the CIC native row is the honest test.",
            "- If they mean **millions of flows per 15s window**, that is **not** what CIC windows contain;",
            "  the `extreme` tier stress-tests beyond CIC density using replicated lab PCAPs.",
            "- **RAMX delta → 0** would support \"base is enough at realistic scale\"; a large delta means",
            "  adaptation still helps (warmup calibration / relative anomaly), even when absolute P is high.",
        ])

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_tier(
    tier: dict,
    pcaps: list[Path],
    model_5,
    hidden_5,
    warmup_5,
) -> dict:
    sf, rep = tier["scale_factor"], tier["replicate"]
    print(f"\n=== Tier {tier['id']}: {sf:g}× replicate {rep}× ({len(pcaps)} attack PCAPs) ===")
    shaun_sidecar = run_shaun_sidecar(pcaps, sf, rep)
    rows = []
    for i, pcap in enumerate(pcaps, 1):
        print(f"  [{i}/{len(pcaps)}] {pcap.name}")
        rows.append(eval_pcap(pcap, model_5, hidden_5, shaun_sidecar, warmup_5, sf, rep))

    lab_summary = {
        k: summarize(rows, k)
        for k in ("ary5_base", "ary5_ramx", "shaun_v2", "shaun_v2_ramx")
    }
    return {
        **tier,
        "n_pcaps": len(rows),
        "lab_rows": rows,
        "lab_summary": lab_summary,
        "ramx_delta": {
            "shaun": ramx_delta(lab_summary, "shaun_v2", "shaun_v2_ramx"),
            "ary5": ramx_delta(lab_summary, "ary5_base", "ary5_ramx"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pcaps", type=int, default=0, help="0 = all attack PCAPs")
    ap.add_argument("--tiers", default="quiet_lab,cic_like,enterprise", help="comma-separated tier ids")
    ap.add_argument("--skip-cic", action="store_true")
    ap.add_argument("--cic-windows", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if not CKPT_5.exists() or not (SHAUN_ROOT / "weights/world_model.pt").exists():
        print("Missing model checkpoint(s)", file=sys.stderr)
        return 1

    tier_ids = [t.strip() for t in args.tiers.split(",") if t.strip()]
    tiers = [t for t in SCALE_TIERS if t["id"] in tier_ids]
    if not tiers:
        print(f"No matching tiers in {tier_ids}", file=sys.stderr)
        return 1

    pcaps = attack_pcaps(args.max_pcaps)
    if not pcaps:
        print(f"No attack PCAPs under {SAVE_DIR}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_5 = load_ckpt(CKPT_5)
    va_s, va_b, _ = load_all_splits(SPLITS_5)["val"]
    _, hidden_5 = calibrate_thresholds(model_5, va_s, va_b)
    warmup_5 = benign_warmup(SPLITS_5, WARMUP_N)

    print(f"Realistic scale bench: {len(pcaps)} attack PCAPs, tiers={[t['id'] for t in tiers]}")
    tier_results = [run_tier(t, pcaps, model_5, hidden_5, warmup_5) for t in tiers]

    cic_native = {}
    if not args.skip_cic:
        print("\n=== Native CIC holdout (Shaun attack windows) ===")
        cic_native = eval_cic_native_holdout(args.cic_windows)
        if "error" not in cic_native:
            b, r = cic_native["shaun_base"], cic_native["shaun_ramx"]
            print(
                f"  flows/win~{cic_native['flows_per_window_med']:.0f} | "
                f"base F1={b['mean_f1']:.3f} det={b['detection_rate']:.1%} | "
                f"ramx F1={r['mean_f1']:.3f} det={r['detection_rate']:.1%} | "
                f"delta F1={cic_native['ramx_delta']['f1']:+.3f}"
            )
        else:
            print(f"  skipped: {cic_native['error']}")

    payload = {
        "n_attack_pcaps": len(pcaps),
        "warmup_steps": WARMUP_N,
        "tiers": tier_results,
        "cic_native": cic_native,
    }
    (args.out_dir / "bench.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(payload, args.out_dir)
    print(f"\nWrote {args.out_dir / 'bench.json'} and REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
