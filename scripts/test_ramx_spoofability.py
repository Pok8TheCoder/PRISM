#!/usr/bin/env python3
"""Spoofability probe: can traffic fool ARY-5sV01 + RAMX?

Tests:
  1. Benign-only streams — false alarms (attack on benign)?
  2. Lab PCAP warm-up region — harm during known-benign windows
  3. Attack evasion — missed detections (benign on attack)?
  4. Timing evasion PCAPs — slow/random_timing variants
  5. Memory priming — attack memory then replay benign CIC windows

Usage:
  python scripts/test_ramx_spoofability.py
  python scripts/test_ramx_spoofability.py --json results/ramx_spoofability.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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

WARMUP_N = 20
THRESH = 0.5


def replay_detailed(factory, states, bins, mits):
    sys_obj = factory()
    p_atts, p_raws, steps = [], [], []
    for i, (s, tb, tm) in enumerate(zip(states, bins, mits)):
        out = sys_obj.step(s, true_bin=tb, true_mit=tm)
        p_atts.append(out["p_att"])
        steps.append(i)
    return p_atts, sys_obj


def benign_only_stats(p_atts: list[float], thresh: float = THRESH) -> dict:
    pred = [1 if p >= thresh else 0 for p in p_atts]
    n = len(p_atts)
    fp = sum(pred)
    return {
        "n_windows": n,
        "false_alarms": fp,
        "false_alarm_rate": fp / n if n else 0.0,
        "max_p_attack": max(p_atts) if p_atts else 0.0,
        "mean_p_attack": float(np.mean(p_atts)) if p_atts else 0.0,
        "any_alarm": fp > 0,
    }


def region_stats(p_atts: list[float], bins: list[int], region_bin: int, thresh: float = THRESH) -> dict:
    idx = [i for i, b in enumerate(bins) if b == region_bin]
    sub = [p_atts[i] for i in idx]
    pred = [1 if p >= thresh else 0 for p in sub]
    if region_bin == 0:
        fp = sum(pred)
        return {
            "n_windows": len(sub),
            "false_alarms": fp,
            "false_alarm_rate": fp / len(sub) if sub else 0.0,
            "max_p_attack": max(sub) if sub else 0.0,
        }
    tp = sum(1 for p, i in zip(pred, idx) if p == 1)
    fn = len(sub) - tp
    return {
        "n_windows": len(sub),
        "detected_windows": tp,
        "missed_windows": fn,
        "det_rate": tp / len(sub) if sub else 0.0,
        "max_p_attack": max(sub) if sub else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=ROOT / "results" / "ramx_spoofability.json")
    p.add_argument("--benign-len", type=int, default=200, help="CIC val benign-only windows to replay")
    args = p.parse_args()

    print("Loading ARY-5sV01 + RAMX...")
    model = load_ckpt(CKPT_5)
    va_s, va_b, va_m = load_all_splits(SPLITS_5)["val"]
    _, h5 = calibrate_thresholds(model, va_s, va_b)
    factory = lambda: StreamingARYRamxV01(base_model=model, hidden_thresh=h5)

    report: dict = {"checkpoint": str(CKPT_5), "threshold": THRESH, "tests": {}}

    # --- 1. Long benign-only CIC val replay ---
    print("\n=== 1. Benign-only CIC val (attack-on-benign spoof?) ===")
    bidx = np.where(va_b == 0)[0][: args.benign_len]
    b_states = [np.asarray(va_s[i], dtype=np.float32) for i in bidx]
    b_bins = [0] * len(b_states)
    b_mits = [int(va_m[i]) for i in bidx]
    p_benign = replay(factory, b_states, b_bins, b_mits)
    t1 = benign_only_stats(p_benign)
    report["tests"]["cic_benign_only"] = t1
    print(f"  windows={t1['n_windows']} false_alarms={t1['false_alarms']} "
          f"rate={t1['false_alarm_rate']:.1%} max_p={t1['max_p_attack']:.3f}")

    # --- 2. Memory priming: attack PCAP warm-up then CIC benign ---
    print("\n=== 2. Memory priming then benign (post-attack benign alarm?) ===")
    w5_s, w5_b, w5_m = benign_warmup(SPLITS_5, WARMUP_N)
    priming_pcap = SAVE_DIR / "live_http_flood_none.pcap"
    primed = False
    if priming_pcap.exists():
        tl = pcap_timeline(priming_pcap, 5.0, w5_s, w5_b, w5_m)
        if tl:
            s5, b5, m5, start, _ = tl
            _ = replay(factory, s5, b5, m5)  # prime memory with attack session
            post = replay(factory, b_states[:80], b_bins[:80], b_mits[:80])
            t2 = benign_only_stats(post)
            t2["priming_pcap"] = priming_pcap.name
            report["tests"]["post_attack_benign_replay"] = t2
            primed = True
            print(f"  primed on {priming_pcap.name} -> benign replay: "
                  f"false_alarms={t2['false_alarms']}/{t2['n_windows']} max_p={t2['max_p_attack']:.3f}")
    if not primed:
        print("  skip — no priming PCAP")

    # --- 3. All lab PCAPs: warmup harm + attack miss ---
    print("\n=== 3. Lab PCAPs (141 captures) ===")
    w5_s, w5_b, w5_m = benign_warmup(SPLITS_5, WARMUP_N)
    pcaps = sorted(SAVE_DIR.glob("*.pcap"))
    warmup_harm = []
    attack_miss = []
    evasion_rows = []
    benign_pcaps = []

    for pcap in pcaps:
        tl = pcap_timeline(pcap, 5.0, w5_s, w5_b, w5_m)
        if tl is None:
            continue
        s5, b5, m5, start, cls = tl
        p_atts = replay(factory, s5, b5, m5)
        sc = score_seq(p_atts, b5, start)

        warm = region_stats(p_atts[:start], b5[:start], 0)
        atk = region_stats(p_atts[start:], b5[start:], 1)

        row = {
            "pcap": pcap.name,
            "class_id": cls,
            "evasion": pcap.stem.split("_")[-1] if "_" in pcap.stem else "unknown",
            "warmup_false_alarms": warm["false_alarms"],
            "warmup_max_p": warm["max_p_attack"],
            "attack_detected": sc["detected"],
            "attack_f1": sc["f1"],
            "attack_max_p": sc["max_p_attack"],
            "attack_miss_windows": atk["missed_windows"],
            "attack_n_windows": atk["n_windows"],
        }
        if warm["false_alarms"] > 0:
            warmup_harm.append(row)
        if cls == "Benign":
            benign_pcaps.append(row)
        if not sc["detected"] and cls != "Benign":
            attack_miss.append(row)
        if any(x in pcap.name for x in ("slow_timing", "random_timing", "randomize_port")):
            evasion_rows.append(row)

    report["tests"]["lab_pcaps"] = {
        "n_pcaps": len(pcaps),
        "warmup_false_alarm_pcaps": len(warmup_harm),
        "warmup_harm_cases": warmup_harm[:20],
        "attack_missed_pcaps": len(attack_miss),
        "attack_miss_cases": attack_miss,
        "benign_pcap_count": len(benign_pcaps),
        "benign_pcaps": benign_pcaps,
        "evasion_pcaps": evasion_rows,
        "summary": {
            "det_rate": 1.0 - len(attack_miss) / max(len(pcaps) - len(benign_pcaps), 1),
            "warmup_harm_rate": len(warmup_harm) / max(len(pcaps), 1),
        },
    }
    print(f"  pcaps={len(pcaps)} warmup_harm={len(warmup_harm)} attack_missed={len(attack_miss)} "
          f"benign_pcaps={len(benign_pcaps)} evasion_pcaps={len(evasion_rows)}")
    if attack_miss:
        for r in attack_miss:
            print(f"    MISS {r['pcap']} max_p={r['attack_max_p']:.3f}")
    if warmup_harm:
        for r in warmup_harm[:5]:
            print(f"    WARMUP FP {r['pcap']} alarms={r['warmup_false_alarms']} max_p={r['warmup_max_p']:.3f}")

    # --- 4. Live lab round.json replays (all evasions) ---
    print("\n=== 4. Live lab round.json replays ===")
    live_dir = ROOT / "results" / "ram_improve" / "live_lab"
    live_rows = []
    for rf in sorted(live_dir.rglob("round.json")):
        data = json.loads(rf.read_text())
        states = [np.array(t["state"], dtype=np.float32) for t in data["trace"]]
        bins = [int(t["true_bin"]) for t in data["trace"]]
        mits = [int(t["true_mit"]) for t in data["trace"]]
        start = next((i for i, b in enumerate(bins) if b == 1), len(bins))
        p_atts = replay(factory, states, bins, mits)
        sc = score_seq(p_atts, bins, start)
        warm = region_stats(p_atts[:start], bins[:start], 0)
        live_rows.append({
            "id": f"{data['objective']}/{data.get('evasion', rf.parent.name)}",
            "evasion": data.get("evasion", rf.parent.name),
            "detected": sc["detected"],
            "f1": sc["f1"],
            "harm": sc["harm"],
            "warmup_false_alarms": warm["false_alarms"],
            "max_p_attack": sc["max_p_attack"],
        })
    report["tests"]["live_lab_rounds"] = {
        "n": len(live_rows),
        "rows": live_rows,
        "missed": [r for r in live_rows if not r["detected"]],
        "warmup_harm": [r for r in live_rows if r["warmup_false_alarms"] > 0],
    }
    missed_live = [r for r in live_rows if not r["detected"]]
    harm_live = [r for r in live_rows if r["warmup_false_alarms"] > 0]
    print(f"  rounds={len(live_rows)} missed={len(missed_live)} warmup_harm={len(harm_live)}")
    for r in live_rows:
        flag = "MISS" if not r["detected"] else "OK  "
        harm = f" harm={r['warmup_false_alarms']}" if r["warmup_false_alarms"] else ""
        print(f"    {flag} {r['id']:<35} F1={r['f1']:.2f} max_p={r['max_p_attack']:.3f}{harm}")

    # --- 5. Label poisoning: attack traffic, memory only sees benign labels ---
    print("\n=== 5. Label poisoning (attack traffic, bins forced 0) ===")
    poison_cases = []
    poison_samples = [
        "live_http_flood_none.pcap",
        "live_http_flood_random_timing.pcap",
        "r1_a3_ssh_bruteforce_slow_timing.pcap",
        "live_port_scan_sequential_none.pcap",
        "r44_a1_T1135_share_discovery_none.pcap",
    ]
    for name in poison_samples:
        pcap = SAVE_DIR / name
        if not pcap.exists():
            continue
        tl = pcap_timeline(pcap, 5.0, w5_s, w5_b, w5_m)
        if tl is None:
            continue
        s5, b5, m5, start, cls = tl
        fake_b = [0] * len(b5)
        fake_m = [0] * len(m5)
        p_atts = replay(factory, s5, fake_b, fake_m)
        sc = score_seq(p_atts, b5, start)
        row = {
            "pcap": name,
            "true_detected_with_labels": sc["detected"],
            "max_p_attack": sc["max_p_attack"],
            "f1_vs_truth": sc["f1"],
        }
        poison_cases.append(row)
        print(f"  {name}: detected={sc['detected']} max_p={sc['max_p_attack']:.3f} f1={sc['f1']:.3f}")
    report["tests"]["label_poisoning"] = poison_cases

    # --- 6. Live-realistic: never pass ground-truth labels to memory/TTT ---
    print("\n=== 6. Live-realistic (no labels passed to RAMX) ===")
    live_realistic_miss = []
    oracle_vs_live = []
    for pcap in pcaps:
        tl = pcap_timeline(pcap, 5.0, w5_s, w5_b, w5_m)
        if tl is None:
            continue
        s5, b5, m5, start, cls = tl
        if cls == "Benign":
            continue

        sys_oracle = factory()
        sys_live = factory()
        p_oracle, p_live = [], []
        for s, tb, tm in zip(s5, b5, m5):
            p_oracle.append(sys_oracle.step(s, true_bin=tb, true_mit=tm)["p_att"])
        for s in s5:
            p_live.append(sys_live.step(s, true_bin=None, true_mit=None)["p_att"])
        sc_o = score_seq(p_oracle, b5, start)
        sc_l = score_seq(p_live, b5, start)
        row = {
            "pcap": pcap.name,
            "oracle_detected": sc_o["detected"],
            "live_detected": sc_l["detected"],
            "oracle_max_p": sc_o["max_p_attack"],
            "live_max_p": sc_l["max_p_attack"],
        }
        oracle_vs_live.append(row)
        if not sc_l["detected"]:
            live_realistic_miss.append(row)

    regressions = [r for r in oracle_vs_live if r["oracle_detected"] and not r["live_detected"]]
    report["tests"]["live_realistic"] = {
        "n_attack_pcaps": len(oracle_vs_live),
        "oracle_miss": sum(1 for r in oracle_vs_live if not r["oracle_detected"]),
        "live_miss": len(live_realistic_miss),
        "regressions": regressions[:30],
        "live_miss_cases": live_realistic_miss[:30],
    }
    print(f"  attack_pcaps={len(oracle_vs_live)} oracle_miss={report['tests']['live_realistic']['oracle_miss']} "
          f"live_miss={len(live_realistic_miss)} regressions={len(regressions)}")
    for r in regressions[:8]:
        print(f"    REGRESS {r['pcap']}: oracle_max={r['oracle_max_p']:.3f} live_max={r['live_max_p']:.3f}")

    # --- Verdict ---
    easily_spoofed_attack = len(attack_miss) >= 5 or len(live_realistic_miss) >= 20 or any(
        r["evasion"] in ("slow_timing", "random_timing") and not r["attack_detected"] for r in evasion_rows
    )
    easily_spoofed_benign = t1["false_alarm_rate"] > 0.05 or len(warmup_harm) > 10
    live_realistic = report["tests"].get("live_realistic", {})
    live_miss_n = live_realistic.get("live_miss", 0)
    live_n = live_realistic.get("n_attack_pcaps", 1)

    report["verdict"] = {
        "attack_evasion_easy_oracle_eval": easily_spoofed_attack,
        "attack_evasion_easy_live_realistic": live_miss_n > live_n * 0.2,
        "benign_false_alarm_easy": easily_spoofed_benign,
        "live_realistic_det_rate": 1.0 - live_miss_n / max(live_n, 1),
        "notes": [
            "oracle eval passes true_bin into step() — inflates detection via memory label writes",
            "live-realistic passes true_bin=None — only suspicious_thresh (0.01) gates dynamic memory",
        ],
    }

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    print(f"\nJSON -> {args.json}")

    print("\n=== VERDICT ===")
    if easily_spoofed_benign:
        print("  BENIGN SPOOF: relatively easy to trigger false alarms")
    else:
        print("  BENIGN SPOOF: NOT easy — very few false alarms on benign traffic")
    print(f"  ATTACK EVASION (oracle eval): {len(attack_miss)}/{len(pcaps)} missed")
    print(f"  ATTACK EVASION (live-realistic, no labels): {live_miss_n}/{live_n} missed "
          f"({report['verdict']['live_realistic_det_rate']:.1%} det rate)")
    if report["verdict"]["attack_evasion_easy_live_realistic"]:
        print("  -> Live-realistic evasion is MUCH easier than oracle eval suggests")
    elif len(regressions) > 0:
        print(f"  -> {len(regressions)} PCAPs only detect with oracle labels (eval artifact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
