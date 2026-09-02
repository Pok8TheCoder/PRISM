#!/usr/bin/env python3
"""Benchmark ARY-5sV01 + RAMX against all 222 MITRE ATT&CK techniques.

For each technique in ``data/mitre_network_catalog.json``:
  - host_only (161): not evaluable from network traffic alone
  - network_observable (61): eval via lab PCAPs mapped to ``mapped_class_id``

Detection pass: at least one lab PCAP for the mapped class triggers alert (F1>0).

Usage:
  python scripts/bench_mitre222_ary5_ramx.py
  python scripts/bench_mitre222_ary5_ramx.py --json results/mitre222_ary5_ramx/bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.compare_ary5_vs_ary01 import (  # noqa: E402
    CKPT_5,
    SPLITS_5,
    benign_warmup,
    load_ckpt,
    pcap_timeline,
    replay,
    score_seq,
)
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.streaming_variants import StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.model.attack_catalog import load_catalog, resolve_pcap_class  # noqa: E402

OUT_DIR = ROOT / "results" / "mitre222_ary5_ramx"
WARMUP_N = 20


def eval_all_pcaps(factory) -> dict[str, list[dict]]:
    """Score every lab PCAP; return rows grouped by class_id."""
    w5_s, w5_b, w5_m = benign_warmup(SPLITS_5, WARMUP_N)
    by_class: dict[str, list[dict]] = defaultdict(list)

    for pcap in sorted(SAVE_DIR.glob("*.pcap")):
        tl = pcap_timeline(pcap, 5.0, w5_s, w5_b, w5_m)
        if tl is None:
            continue
        s5, b5, m5, start, cls = tl
        p_atts = replay(factory, s5, b5, m5)
        sc = score_seq(p_atts, b5, start)
        by_class[cls].append({
            "pcap": pcap.name,
            "detected": sc["detected"],
            "f1": sc["f1"],
            "max_p_attack": sc["max_p_attack"],
        })
    return dict(by_class)


def class_result(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n_pcaps": 0, "detected": False, "all_detected": False,
            "det_rate": 0.0, "mean_f1": 0.0, "best_f1": 0.0,
            "missed_pcaps": [], "passed_pcaps": [],
        }
    det = sum(1 for r in rows if r["detected"])
    missed = [r for r in rows if not r["detected"]]
    passed = [r for r in rows if r["detected"]]
    return {
        "n_pcaps": len(rows),
        "detected": det > 0,
        "all_detected": det == len(rows),
        "det_rate": det / len(rows),
        "mean_f1": float(np.mean([r["f1"] for r in rows])),
        "best_f1": float(max(r["f1"] for r in rows)),
        "missed_pcaps": [r["pcap"] for r in missed],
        "passed_pcaps": [r["pcap"] for r in passed],
        "missed_details": [
            {"pcap": r["pcap"], "f1": r["f1"], "max_p_attack": r["max_p_attack"]} for r in missed
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=OUT_DIR / "mitre222_bench.json")
    p.add_argument("--failures-md", type=Path, default=OUT_DIR / "FAILURES.md")
    args = p.parse_args()

    print("Loading ARY-5sV01 + RAMX...")
    model = load_ckpt(CKPT_5)
    va_s, va_b, _ = load_all_splits(SPLITS_5)["val"]
    _, h5 = calibrate_thresholds(model, va_s, va_b)
    factory = lambda: StreamingARYRamxV01(base_model=model, hidden_thresh=h5)

    print(f"Scoring lab PCAPs in {SAVE_DIR}...")
    pcap_by_class = eval_all_pcaps(factory)
    print(f"  {sum(len(v) for v in pcap_by_class.values())} PCAPs across {len(pcap_by_class)} classes")

    catalog = load_catalog()
    techniques = catalog["techniques"]
    class_cache: dict[str, dict] = {}

    rows = []
    for tech in techniques:
        tid = tech["id"]
        bucket = tech["bucket"]
        mapped = tech.get("mapped_class_id")

        if bucket == "host_only":
            status = "host_only"
            detected = None
            detail = {"reason": "No network-observable traffic signature in catalog"}
        elif not mapped:
            status = "unmapped"
            detected = None
            detail = {}
        else:
            if mapped not in class_cache:
                class_cache[mapped] = class_result(pcap_by_class.get(mapped, []))
            cr = class_cache[mapped]
            if cr["n_pcaps"] == 0:
                status = "no_lab_pcap"
                detected = None
                detail = {"mapped_class_id": mapped, "reason": "Network-observable but no lab PCAP captured"}
            else:
                status = "tested"
                # Strict: fail if ANY lab PCAP for this class is missed
                detected = cr["all_detected"]
                detail = {
                    "mapped_class_id": mapped,
                    **cr,
                }

        rows.append({
            "technique_id": tid,
            "name": tech["name"],
            "tactics": tech.get("tactics", []),
            "bucket": bucket,
            "mapped_class_id": mapped,
            "family_id": tech.get("family_id"),
            "status": status,
            **detail,
            "detected": detected,
        })

    tested = [r for r in rows if r["status"] == "tested"]
    passed = [r for r in tested if r["detected"]]
    failed = [r for r in tested if not r["detected"]]
    partial_classes = [
        cid for cid, cr in class_cache.items()
        if cr.get("n_pcaps", 0) > 0 and cr.get("detected") and not cr.get("all_detected")
    ]
    host_only = [r for r in rows if r["status"] == "host_only"]
    no_pcap = [r for r in rows if r["status"] == "no_lab_pcap"]

    # Dedupe failure list by mapped class (many techniques share one bot)
    failed_classes: dict[str, dict] = {}
    for r in failed:
        cid = r["mapped_class_id"]
        if cid not in failed_classes:
            failed_classes[cid] = {
                "class_id": cid,
                "technique_ids": [],
                "n_pcaps": r.get("n_pcaps", 0),
                "mean_f1": r.get("mean_f1", 0),
                "missed_pcaps": r.get("missed_pcaps", []),
            }
        failed_classes[cid]["technique_ids"].append(r["technique_id"])

    report = {
        "model": "ARY-5sV01 + RAMX (oracle labels, 5s windows, lab PCAPs)",
        "checkpoint": str(CKPT_5),
        "summary": {
            "total_mitre_techniques": len(rows),
            "host_only_not_evaluable": len(host_only),
            "network_observable": len(rows) - len(host_only),
            "tested_with_lab_pcap": len(tested),
            "detected_pass_strict": len(passed),
            "detected_fail_strict": len(failed),
            "partial_class_misses": len(partial_classes),
            "no_lab_pcap": len(no_pcap),
            "pass_rate_of_tested": len(passed) / len(tested) if tested else 0.0,
            "lab_pcap_classes": len(pcap_by_class),
            "lab_pcap_count": sum(len(v) for v in pcap_by_class.values()),
        },
        "failed_classes": list(failed_classes.values()),
        "partial_fail_classes": [
            {"class_id": cid, **class_cache[cid]} for cid in sorted(partial_classes)
        ],
        "techniques": rows,
        "pcap_by_class": {k: v for k, v in sorted(pcap_by_class.items())},
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))

    # Markdown failure report
    lines = [
        "# ARY-5sV01 + RAMX — MITRE ATT&CK (222 techniques) benchmark",
        "",
        f"**Model:** ARY-5sV01 + RAMX (oracle eval, 5s windows, {report['summary']['lab_pcap_count']} lab PCAPs)",
        "",
        "## Summary",
        "",
        f"| Metric | Count |",
        f"|---|---:|",
        f"| Total MITRE techniques | {len(rows)} |",
        f"| Host-only (not network evaluable) | {len(host_only)} |",
        f"| Network-observable | {len(rows) - len(host_only)} |",
        f"| Tested (lab PCAP exists) | {len(tested)} |",
        f"| **Pass (all PCAPs detect)** | **{len(passed)}** |",
        f"| **Fail (any PCAP missed)** | **{len(failed)}** |",
        f"| Partial misses (some PCAPs) | {len(partial_classes)} classes |",
        f"| Network-observable, no PCAP yet | {len(no_pcap)} |",
        "",
        "## Detection failures (tested, missed)",
        "",
    ]
    if failed_classes:
        for fc in sorted(failed_classes.values(), key=lambda x: x["class_id"]):
            lines.append(f"### `{fc['class_id']}`")
            lines.append(f"- MITRE techniques ({len(fc['technique_ids'])}): {', '.join(sorted(fc['technique_ids']))}")
            lines.append(f"- Lab PCAPs: {fc['n_pcaps']} — all missed")
            lines.append(f"- Mean F1: {fc['mean_f1']:.3f}")
            for mp in fc.get("missed_pcaps", [])[:5]:
                lines.append(f"  - `{mp}`")
            if len(fc.get("missed_pcaps", [])) > 5:
                lines.append(f"  - ... +{len(fc['missed_pcaps']) - 5} more")
            lines.append("")
    else:
        lines.append("_None — all tested techniques passed on every lab PCAP._")
        lines.append("")

    if partial_classes:
        lines.extend(["", "## Partial failures (class detected on some PCAPs, missed on others)", ""])
        for cid in sorted(partial_classes):
            cr = class_cache[cid]
            lines.append(f"### `{cid}` — {cr['det_rate']:.0%} det rate ({len(cr['passed_pcaps'])}/{cr['n_pcaps']} PCAPs)")
            for md in cr.get("missed_details", []):
                lines.append(f"- MISS `{md['pcap']}` max_p={md['max_p_attack']:.4f}")
            lines.append("")

    lines.extend([
        "## Full technique failure list",
        "",
        "| Technique | Name | Mapped class | Status |",
        "|---|---|---|---|",
    ])
    for r in sorted(failed, key=lambda x: x["technique_id"]):
        lines.append(
            f"| {r['technique_id']} | {r['name'][:50]} | {r.get('mapped_class_id', '-')} | FAIL |"
        )

    lines.extend([
        "",
        "## Host-only techniques (161) — not network-evaluable",
        "",
        "These require host telemetry (process, registry, etc.); flow-based IDS cannot score them.",
        "",
    ])
    for r in sorted(host_only, key=lambda x: x["technique_id"])[:20]:
        lines.append(f"- {r['technique_id']}: {r['name']}")
    lines.append(f"- ... and {len(host_only) - 20} more (see JSON)")

    args.failures_md.write_text("\n".join(lines))

    print("\n=== MITRE 222 benchmark (ARY-5sV01 + RAMX) ===")
    s = report["summary"]
    print(f"  Total techniques:     {s['total_mitre_techniques']}")
    print(f"  Host-only (N/A):        {s['host_only_not_evaluable']}")
    print(f"  Tested with lab PCAP:   {s['tested_with_lab_pcap']}")
    print(f"  PASS (all PCAPs detect): {s['detected_pass_strict']}")
    print(f"  FAIL (any PCAP missed):   {s['detected_fail_strict']}")
    print(f"  Partial class misses:     {s['partial_class_misses']} classes")
    print(f"  Pass rate (tested):     {s['pass_rate_of_tested']:.1%}")
    if failed_classes:
        print("\n  Failed classes:")
        for fc in sorted(failed_classes.values(), key=lambda x: x["class_id"]):
            print(f"    {fc['class_id']}: {len(fc['technique_ids'])} technique(s), {fc['n_pcaps']} PCAP(s)")
    print(f"\n  JSON -> {args.json}")
    print(f"  Report -> {args.failures_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
