#!/usr/bin/env python3
"""Compare Shaun RAMX v2 vs v3 on scaled lab PCAPs (warmup harm + detection).

Uses PRISM streaming wrappers so import paths stay correct.

Usage:
  python scripts/bench_ramx_v3_lab.py
  python scripts/bench_ramx_v3_lab.py --max-pcaps 40 --scale-factor 10 --replicate 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
sys.path.insert(0, str(ROOT))

from scripts.lab_traffic_scale import scale_prism_rows  # noqa: E402
from scripts.plot_lab_ary_comparison import replay, score_round  # noqa: E402
from scripts.shaun_pcap_ingest import rows_to_shaun_states  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402
from src.pipeline.extract import pcap_to_rows  # noqa: E402
from src.shaun.streaming import (  # noqa: E402
    StreamingShaunRamxV2,
    StreamingShaunRamxV3,
    load_shaun_bundle,
)

WARMUP_N = 20
OUT_DIR = ROOT / "results" / "ramx_v3_lab_bench"


def shaun_warmup(n: int) -> list[np.ndarray]:
    states = np.load(SHAUN_ROOT / "data/processed/states.npy")
    labels = np.load(SHAUN_ROOT / "data/processed/attack_labels.npy")
    idx = np.where(labels == 0)[0][:n]
    return [states[i].astype(np.float32) for i in idx]


def build_timeline(
    pcap: Path,
    warmup: list[np.ndarray],
    *,
    scale_factor: float,
    replicate: int,
) -> tuple[list[np.ndarray], list[int], int] | None:
    rows = scale_prism_rows(pcap_to_rows(pcap), factor=scale_factor, replicate=replicate)
    if not rows:
        return None
    atk = rows_to_shaun_states(rows, shaun_root=SHAUN_ROOT, window_sec=15.0)
    if not len(atk):
        return None
    atk_states = [np.asarray(s, dtype=np.float32) for s in atk]
    cls = resolve_pcap_class(pcap.name) or "Benign"
    atk_bins = [0 if cls == "Benign" else 1] * len(atk_states)
    timeline = warmup + atk_states
    bins = [0] * len(warmup) + atk_bins
    return timeline, bins, len(warmup)


def eval_pcaps(
    pcaps: list[Path],
    *,
    scale_factor: float,
    replicate: int,
    warmup: list[np.ndarray],
    bundle: dict,
) -> dict:
    rows: list[dict] = []
    for pcap in pcaps:
        built = build_timeline(pcap, warmup, scale_factor=scale_factor, replicate=replicate)
        if built is None:
            rows.append({"pcap": pcap.name, "error": "empty ingest"})
            continue
        timeline, bins, start = built
        p2 = replay(lambda: StreamingShaunRamxV2(bundle, context_skip_steps=start), timeline, bins, [0] * len(bins))
        p3 = replay(lambda: StreamingShaunRamxV3(bundle, context_skip_steps=start), timeline, bins, [0] * len(bins))
        s2, s3 = score_round(p2, bins, start), score_round(p3, bins, start)
        rows.append({
            "pcap": pcap.name,
            "class": resolve_pcap_class(pcap.name) or "Benign",
            "n_windows": len(timeline),
            "v2": s2,
            "v3": s3,
            "v2_warmup_mean_p": float(np.mean(p2[:start])) if start else 0.0,
            "v3_warmup_mean_p": float(np.mean(p3[:start])) if start else 0.0,
        })

    scored = [r for r in rows if "v2" in r]
    atk = [r for r in scored if r["class"] != "Benign"]

    def _agg(key: str) -> dict:
        rs = [r[key] for r in scored if key in r]
        return {
            "mean_f1": float(np.mean([x["f1"] for x in rs])) if rs else 0.0,
            "detection_rate": float(np.mean([1.0 if x["detected"] else 0.0 for x in rs])) if rs else 0.0,
            "total_warmup_harm": int(sum(x["harm"] for x in rs)),
            "attack_det_rate": float(np.mean([1.0 if r[key]["detected"] else 0.0 for r in atk])) if atk else 0.0,
        }

    return {
        "n_pcaps": len(pcaps),
        "n_scored": len(scored),
        "scale_factor": scale_factor,
        "replicate": replicate,
        "warmup_steps": WARMUP_N,
        "v2": _agg("v2"),
        "v3": _agg("v3"),
        "delta": {
            "warmup_harm": _agg("v3")["total_warmup_harm"] - _agg("v2")["total_warmup_harm"],
            "attack_det_rate": _agg("v3")["attack_det_rate"] - _agg("v2")["attack_det_rate"],
            "mean_f1": _agg("v3")["mean_f1"] - _agg("v2")["mean_f1"],
        },
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pcaps", type=int, default=40)
    ap.add_argument("--scale-factor", type=float, default=10.0)
    ap.add_argument("--replicate", type=int, default=5)
    ap.add_argument("--include-benign", action="store_true")
    args = ap.parse_args()

    pcaps = sorted(SAVE_DIR.glob("*.pcap"))
    if not args.include_benign:
        pcaps = [p for p in pcaps if (resolve_pcap_class(p.name) or "Benign") != "Benign"]
    if args.max_pcaps > 0:
        pcaps = pcaps[: args.max_pcaps]

    bundle = load_shaun_bundle()
    warmup = shaun_warmup(WARMUP_N)
    payload = eval_pcaps(
        pcaps,
        scale_factor=args.scale_factor,
        replicate=args.replicate,
        warmup=warmup,
        bundle=bundle,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "v2_vs_v3_scaled_lab.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ("n_scored", "v2", "v3", "delta")}, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
