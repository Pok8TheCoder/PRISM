#!/usr/bin/env python3
"""Zero-day / OOD attack eval: ARY base, ARY+RAMX_V.01, SHNV.01 base, SHNV.01+RAMX.

ARY.01 was trained on CIC-IDS-2018 labels only (see ``src/aryan/constants.py``).
SHNV.01 (PRISM_MODEL_PACKAGE) was trained on CIC-IDS + UNSW-NB15 + CTU-13 in
native 110-d / 60s windows — catalog bots and lab HTTP objectives are OOD for
both models.

Usage:
  python scripts/test_zeroday_ramx.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.lab_scoring import compute_round_scores  # noqa: E402
from src.aryan.streaming_variants import (  # noqa: E402
    StreamingARY,
    StreamingARYRamxV01,
    calibrate_thresholds,
    load_model,
)

from src.shnv01.streaming import (  # noqa: E402
    SHNV_PKG_ROOT,
    StreamingSHNV01,
    calibrate_shnv01_hidden_thresh,
    lab242_windows_to_friend110_minutes,
    load_shnv01_bundle,
)

# Exact CIC-IDS-2018 attack names ARY.01 was trained on (constants.py).
CIC_TRAINED = {
    "Benign",
    "FTP-BruteForce",
    "SSH-Bruteforce",
    "Bot",
    "Infilteration",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "DoS attacks-GoldenEye",
    "DDoS attacks-LOIC-HTTP",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
}

# Live-lab HTTP objectives — explicitly NOT in CIC training.
LAB_OBJECTIVES = {
    "T1491_web_defacement",
    "T1552_key_theft",
    "T1555_sqli_cred_theft",
}

CLASS_RE = re.compile(r"(T\d+_[A-Za-z0-9_]+)")


def class_from_pcap(path: str) -> str:
    m = CLASS_RE.search(Path(path).stem)
    return m.group(1) if m else Path(path).stem


def is_ood(class_id: str) -> bool:
    if class_id in LAB_OBJECTIVES:
        return True
    if class_id in CIC_TRAINED:
        return False
    # Catalog bots / MITRE ids are all OOD relative to CIC label set.
    if class_id.startswith("T") and class_id not in CIC_TRAINED:
        return True
    return class_id not in CIC_TRAINED


def replay_system(sys_obj, states, bins, mits) -> list[float]:
    p_atts = []
    for s, tb, tm in zip(states, bins, mits):
        out = sys_obj.step(s, true_bin=tb, true_mit=tm)
        p_atts.append(out["p_att"])
    return p_atts


def score_sequence(p_atts: list[float], true_bins: list[int], attack_start: int) -> dict:
    pred = [1 if p >= 0.5 else 0 for p in p_atts]
    tb = true_bins
    tp = sum(1 for p, t in zip(pred, tb) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, tb) if p == 1 and t == 0)
    fn = sum(1 for p, t in zip(pred, tb) if p == 0 and t == 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    ttd = None
    for i in range(attack_start, len(tb)):
        if pred[i] == 1 and tb[i] == 1:
            ttd = i - attack_start
            break
    max_p_attack = max(p_atts[attack_start:], default=0.0)
    harm = sum(1 for p, t in zip(pred[:attack_start], tb[:attack_start]) if t == 0 and p == 1)
    return {
        "f1": f1, "precision": prec, "recall": rec, "ttd": ttd,
        "detected": ttd is not None, "max_p_attack": max_p_attack, "harm": harm,
    }


def ary242_to_shnv110(
    states_242: list[np.ndarray],
    true_bins: list[int],
    true_mits: list[int],
    windows_per_minute: int,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    return lab242_windows_to_friend110_minutes(states_242, true_bins, true_mits, windows_per_minute)


def main() -> int:
    print("Loading ARY.01...")
    base_model = load_model()
    splits = load_all_splits()
    va_s, va_b, va_m = splits["val"]
    raw_thresh, hidden_thresh = calibrate_thresholds(base_model, va_s, va_b)

    print("Loading SHNV.01...")
    shnv_bundle = load_shnv01_bundle()
    shnv_states = np.load(SHNV_PKG_ROOT / "data" / "processed" / "states.npy")
    shnv_atks = np.load(SHNV_PKG_ROOT / "data" / "processed" / "attack_labels.npy")
    shnv_mits = np.load(SHNV_PKG_ROOT / "data" / "processed" / "mitre_labels.npy")
    shnv_benign_idx = np.where(shnv_atks == 0)[0]
    shnv_warmup = shnv_states[shnv_benign_idx[: shnv_bundle["lookback"]]]
    shnv_warmup_bins = [0] * len(shnv_warmup)
    shnv_warmup_mits = [0] * len(shnv_warmup)
    va_start = int(len(shnv_states) * 0.70)
    shnv_hidden_thresh = calibrate_shnv01_hidden_thresh(
        shnv_bundle, shnv_states[va_start : va_start + 400], shnv_atks[va_start : va_start + 400],
    )

    # Benign warm-up pool from CIC val split (in-distribution normal for ARY).
    benign_idx = np.where(va_b == 0)[0]
    warmup = va_s[benign_idx[:20]]
    warmup_bins = [0] * len(warmup)
    warmup_mits = [0] * len(warmup)

    systems = {
        "ary_base": lambda: StreamingARY("ary_base", base_model=base_model, raw_thresh=raw_thresh, hidden_thresh=hidden_thresh),
        "ary_ramx_v01": lambda: StreamingARYRamxV01(base_model=base_model, hidden_thresh=hidden_thresh),
        "shnv_01_base": lambda: StreamingSHNV01(shnv_bundle, match_thresh=shnv_hidden_thresh, use_ram=False),
        "shnv_01_ramx": lambda: StreamingSHNV01(shnv_bundle, match_thresh=shnv_hidden_thresh, use_ram=True),
    }

    print("\n=== Attacks ARY.01 WAS trained on (CIC-IDS-2018) ===")
    for name in sorted(CIC_TRAINED - {"Benign"}):
        print(f"  {name}")
    print(f"  (+ Benign baseline)\n")

    print("=== Zero-day / OOD sources for this test ===")
    print("  All catalog Docker bots in xmt_splits/test (none in CIC label set)")
    print("  Live-lab objectives: T1491 defacement, T1552 key theft, T1555 SQLi cred theft")
    print("  Note: CIC has 'SQL Injection' / 'SSH-Bruteforce' but NOT our lab bots or catalog class_ids\n")

    # --- XMT held-out pcaps grouped by attack class ---
    xmt = np.load(ROOT / "data" / "xmt_splits" / "test.npz", allow_pickle=True)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, pcap in enumerate(xmt["source_pcap"]):
        by_class[class_from_pcap(str(pcap))].append(i)

    header = f"{'attack (OOD)':<28} {'system':<14} {'F1':>6} {'recall':>7} {'TTD':>5} {'maxP':>6} {'harm':>5} {'det':>5}"
    print("=== XMT test captures (benign warm-up + OOD attack windows) ===")
    print("  ARY: 20x CIC-242 30s windows | SHNV: 20x native 110-d 60s minutes + 242->110 adapter (2 win/min)")
    print(header)
    print("-" * len(header))

    agg: dict[str, list[float]] = {k: [] for k in systems}
    for cls in sorted(by_class):
        if not is_ood(cls):
            continue
        idxs = sorted(by_class[cls], key=lambda i: str(xmt["source_pcap"][i]))
        atk_states = xmt["states"][idxs]
        atk_bins = xmt["labels_binary"][idxs].tolist()
        atk_mits = xmt["labels_mitre"][idxs].tolist()

        # ARY path (native 242-d 30s)
        ary_states = [np.asarray(s, dtype=np.float32) for s in np.vstack([warmup, atk_states])]
        ary_bins = warmup_bins + atk_bins
        ary_mits = warmup_mits + atk_mits
        ary_attack_start = len(warmup)

        # SHNV path (110-d 60s): native benign minutes + adapted attack minutes
        atk242 = [np.asarray(s, dtype=np.float32) for s in atk_states]
        shnv_atk_s, shnv_atk_b, shnv_atk_m = ary242_to_shnv110(atk242, atk_bins, atk_mits, windows_per_minute=2)
        shnv_states = [np.asarray(s, dtype=np.float32) for s in np.vstack([shnv_warmup, shnv_atk_s])]
        shnv_bins = shnv_warmup_bins + shnv_atk_b
        shnv_mits = shnv_warmup_mits + shnv_atk_m
        shnv_attack_start = len(shnv_warmup)

        for sid, factory in systems.items():
            if sid.startswith("shnv"):
                sys_obj = factory()
                p_atts = replay_system(sys_obj, shnv_states, shnv_bins, shnv_mits)
                sc = score_sequence(p_atts, shnv_bins, shnv_attack_start)
            else:
                sys_obj = factory()
                p_atts = replay_system(sys_obj, ary_states, ary_bins, ary_mits)
                sc = score_sequence(p_atts, ary_bins, ary_attack_start)
            agg[sid].append(sc["f1"])
            print(
                f"{cls:<28} {sid:<14} {sc['f1']:>6.3f} {sc['recall']:>7.3f} "
                f"{str(sc['ttd']):>5} {sc['max_p_attack']:>6.3f} {sc['harm']:>5} {str(sc['detected']):>5}"
            )

    # --- Live lab objective rounds (pure zero-day HTTP exploits) ---
    print("\n=== Live lab objective rounds (HTTP zero-days, captured traffic) ===")
    print("  ARY: native 242-d 15s | SHNV: 242->110 adapter (4 win/min -> 60s)")
    print(header)
    print("-" * len(header))
    lab_agg: dict[str, list[float]] = {k: [] for k in systems}
    for rf in sorted((ROOT / "results" / "ram_improve" / "live_lab").rglob("round.json")):
        if rf.parent.name != "none":
            continue
        data = json.loads(rf.read_text())
        trace = data["trace"]
        events = data["events"]
        states242 = [np.array(t["state"], dtype=np.float32) for t in trace]
        bins242 = [int(t["true_bin"]) for t in trace]
        mits242 = [int(t["true_mit"]) for t in trace]

        shnv_s, shnv_b, shnv_m = ary242_to_shnv110(states242, bins242, mits242, windows_per_minute=4)
        shnv_replay_trace = [
            {"window_idx": i, "t_start": float(i * 60), "true_bin": tb, "systems": {}}
            for i, tb in enumerate(shnv_b)
        ]

        for sid, factory in systems.items():
            sys_obj = factory()
            if sid.startswith("shnv"):
                for i, (s, tb, tm) in enumerate(zip(shnv_s, shnv_b, shnv_m)):
                    out = sys_obj.step(s, true_bin=tb, true_mit=tm)
                    shnv_replay_trace[i]["systems"][sid] = {"p_att": out["p_att"]}
                sc = compute_round_scores(shnv_replay_trace, events, [sid])[sid]
            else:
                replay_trace = [
                    {"window_idx": t["window_idx"], "t_start": t["t_start"], "true_bin": tb, "systems": {}}
                    for t, tb in zip(trace, bins242)
                ]
                for i, (s, tb, tm) in enumerate(zip(states242, bins242, mits242)):
                    out = sys_obj.step(s, true_bin=tb, true_mit=tm)
                    replay_trace[i]["systems"][sid] = {"p_att": out["p_att"]}
                sc = compute_round_scores(replay_trace, events, [sid])[sid]
            lab_agg[sid].append(sc["binary_f1"])
            print(
                f"{data['objective']+'/none':<28} {sid:<14} {sc['binary_f1']:>6.3f} {sc['recall']:>7.3f} "
                f"{str(sc['ttd_windows']):>5} {'-':>6} {sc['benign_harm_count']:>5} {str(sc['detected']):>5}"
            )

    print("\n=== Summary ===")
    for sid in systems:
        xmt_mean = np.mean(agg[sid]) if agg[sid] else 0.0
        lab_mean = np.mean(lab_agg[sid]) if lab_agg[sid] else 0.0
        print(f"  {sid:<14}  xmt_ood_mean_f1={xmt_mean:.3f} (n={len(agg[sid])})  lab_zero_day_mean_f1={lab_mean:.3f} (n={len(lab_agg[sid])})")

    # Headline: RAMX rescued base (ARY and SHNV separately)
    for base_id, ramx_id in [("ary_base", "ary_ramx_v01"), ("shnv_01_base", "shnv_01_ramx")]:
        rescued = 0
        total = 0
        for cls in sorted(by_class):
            if not is_ood(cls):
                continue
            idxs = sorted(by_class[cls], key=lambda i: str(xmt["source_pcap"][i]))
            atk_states = xmt["states"][idxs]
            atk_bins = xmt["labels_binary"][idxs].tolist()
            atk_mits = xmt["labels_mitre"][idxs].tolist()
            if base_id.startswith("shnv"):
                atk242 = [np.asarray(s, dtype=np.float32) for s in atk_states]
                sa, ba, ma = ary242_to_shnv110(atk242, atk_bins, atk_mits, 2)
                states = [np.asarray(s, dtype=np.float32) for s in np.vstack([shnv_warmup, sa])]
                bins = shnv_warmup_bins + ba
                mits = shnv_warmup_mits + ma
                attack_start = len(shnv_warmup)
            else:
                states = [np.asarray(s, dtype=np.float32) for s in np.vstack([warmup, atk_states])]
                bins = warmup_bins + atk_bins
                mits = warmup_mits + atk_mits
                attack_start = len(warmup)
            pb = score_sequence(replay_system(systems[base_id](), states, bins, mits), bins, attack_start)
            pr = score_sequence(replay_system(systems[ramx_id](), states, bins, mits), bins, attack_start)
            total += 1
            if not pb["detected"] and pr["detected"]:
                rescued += 1
        print(f"  {ramx_id} rescued {base_id} on {rescued}/{total} OOD xmt groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
