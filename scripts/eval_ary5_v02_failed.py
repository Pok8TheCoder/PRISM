#!/usr/bin/env python3
"""Compare ARY-5sV01 vs V02 (+ RAMX) on failed PCAPs and CIC test."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_ary5_vs_ary01 import (  # noqa: E402
    SPLITS_5,
    benign_warmup,
    eval_cic,
    load_ckpt,
    pcap_timeline,
    replay,
    score_seq,
)
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.streaming_variants import StreamingARYRamxV01, calibrate_thresholds  # noqa: E402

CKPT_V01 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
CKPT_V02 = ROOT / "models" / "checkpoints" / "ary5s_v02.pt"
FAILED = [
    "live_port_scan_sequential_none.pcap",
    "r44_a1_T1135_share_discovery_none.pcap",
]
WARMUP_N = 20


def eval_pcap(pcap_name: str, factory) -> dict:
    w5_s, w5_b, w5_m = benign_warmup(SPLITS_5, WARMUP_N)
    tl = pcap_timeline(SAVE_DIR / pcap_name, 5.0, w5_s, w5_b, w5_m)
    if tl is None:
        return {"error": "empty timeline"}
    s5, b5, m5, start, cls = tl
    p_atts = replay(factory, s5, b5, m5)
    sc = score_seq(p_atts, b5, start)
    return {"class_id": cls, **sc, "n_windows": len(b5), "attack_start": start}


def main() -> int:
    out = ROOT / "results" / "ary5_v02_failed_eval.json"
    results = {"failed_pcaps": {}, "cic_test": {}}

    for label, ckpt in [("v01", CKPT_V01), ("v02", CKPT_V02)]:
        if not ckpt.exists():
            print(f"Missing {ckpt}")
            continue
        model = load_ckpt(ckpt)
        va_s, va_b, _ = load_all_splits(SPLITS_5)["val"]
        _, h5 = calibrate_thresholds(model, va_s, va_b)
        factory = lambda m=model, h=h5: StreamingARYRamxV01(base_model=m, hidden_thresh=h)

        results["cic_test"][label] = eval_cic(model, SPLITS_5)
        results["failed_pcaps"][label] = {}
        for pcap in FAILED:
            results["failed_pcaps"][label][pcap] = eval_pcap(pcap, factory)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n=== Failed PCAPs (ARY-5s + RAMX) ===")
    for pcap in FAILED:
        print(f"\n  {pcap}")
        for label in ("v01", "v02"):
            if label not in results["failed_pcaps"]:
                continue
            r = results["failed_pcaps"][label].get(pcap, {})
            det = r.get("detected", False)
            print(f"    {label}: detected={det}  f1={r.get('f1', 0):.3f}  max_p={r.get('max_p_attack', 0):.4f}")

    print("\n=== CIC test (base model, no RAMX) ===")
    for label in ("v01", "v02"):
        if label in results["cic_test"]:
            c = results["cic_test"][label]
            print(f"  {label}: F1={c['f1']:.3f}  recall={c['recall']:.3f}")

    print(f"\nJSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
