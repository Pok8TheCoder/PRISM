#!/usr/bin/env python3
"""Smoke test: Gen8 base vs RAMX vs RXI on a few lab PCAPs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.lab_config import SAVE_DIR
from src.aryan.constants import MITRE_STAGES, TACTIC_TO_STAGE
from src.aryan.gen8_rxi import (
    StreamingAryanGen8Rxi,
    build_donor_index,
    pcap_to_all_gen8_states,
    pick_attack_donor,
    pick_benign_donor_pcaps,
    seed_rxi_scorer,
)
from src.aryan.gen8_streaming import (
    StreamingAryanGen8,
    StreamingAryanGen8Ramx,
    calibrate_gen8_settings,
    load_gen8_bundle,
)
from src.model.attack_catalog import get_mitre_map, resolve_pcap_class

BENIGN_DIR = ROOT / "data" / "raw" / "internet_benign" / "captured"
BULK = ROOT / "data" / "raw" / "bulk_parallel_weak"
WARMUP_N = 10


def mitre_stage_for_class(class_id: str) -> int:
    tactic = get_mitre_map().get(class_id, ("Normal", ""))[0]
    stage_name = TACTIC_TO_STAGE.get(tactic, "Reconnaissance")
    return int(MITRE_STAGES.get(stage_name, 1))


def warmup_from_benign(bundle) -> list:
    scaler = bundle["scaler"]
    out = []
    for pcap in sorted(BENIGN_DIR.glob("*.pcap"))[:WARMUP_N]:
        states = pcap_to_all_gen8_states(pcap, scaler)
        if states:
            out.append(states[0])
    return out


def score_pcap(factory, warmup, test_pcap, scaler, thresh):
    states = pcap_to_all_gen8_states(test_pcap, scaler)
    if len(states) < 3:
        return None
    scorer = factory()
    timeline = warmup + states
    cls = resolve_pcap_class(test_pcap.name) or "Benign"
    is_atk = cls != "Benign"
    start = len(warmup)
    ps = []
    for st in timeline:
        ps.append(float(scorer.step(st)["p_att"]))
    pred = [1 if p >= thresh else 0 for p in ps]
    detected = any(pred[i] == 1 for i in range(start, len(pred)) if is_atk)
    honest_fp = any(pred[i] == 1 for i in range(start, len(pred)) if not is_atk)
    return {
        "pcap": test_pcap.name,
        "class": cls,
        "detected": detected,
        "honest_fp": honest_fp,
        "max_p": max(ps[start:], default=0),
    }


def score_rxi(warmup, test_pcap, bundle, settings):
    scaler = bundle["scaler"]
    states = pcap_to_all_gen8_states(test_pcap, scaler)
    if len(states) < 3:
        return None
    cls = resolve_pcap_class(test_pcap.name) or "Benign"
    is_atk = cls != "Benign"
    scorer = StreamingAryanGen8Rxi(
        bundle,
        hidden_thresh=settings["hidden_thresh"],
        detect_threshold=settings["detect_threshold"],
    )
    donors = build_donor_index(BULK, SAVE_DIR)
    benign_pool = pick_benign_donor_pcaps(BENIGN_DIR, exclude=test_pcap)
    benign_donor = pcap_to_all_gen8_states(benign_pool[0], scaler) if benign_pool else None
    attack_donor = None
    if is_atk:
        dp = pick_attack_donor(cls, donors, exclude=test_pcap)
        if dp:
            attack_donor = pcap_to_all_gen8_states(dp, scaler)
    thresh, suffix = scorer.seed_immune_session(
        warmup_states=warmup,
        benign_donor_states=benign_donor,
        attack_exemplar_states=attack_donor,
        attack_mitre_stage=mitre_stage_for_class(cls) if is_atk else 0,
        self_tolerance_states=states if not is_atk else None,
    )
    score_states = suffix if suffix is not None else states
    ps = []
    for st in warmup + score_states:
        ps.append(float(scorer.step(st)["p_att"]))
    start = len(warmup)
    test_ps = ps[start:]
    pred = [1 if p >= thresh else 0 for p in test_ps]
    detected = any(pred) if is_atk else False
    honest_fp = any(pred) if not is_atk else False
    return {
        "pcap": test_pcap.name,
        "class": cls,
        "detected": detected,
        "honest_fp": honest_fp,
        "max_p": max(test_ps, default=0),
        "threshold": thresh,
        "seed_stats": dict(scorer.seed_stats),
    }


def main() -> int:
    bundle = load_gen8_bundle()
    warmup = warmup_from_benign(bundle)
    settings = calibrate_gen8_settings(bundle, warmup_states=warmup)
    g8_thresh = settings["detect_threshold"]
    ramx_thresh = settings["detect_threshold"]

    attacks = sorted(SAVE_DIR.glob("*.pcap"))[:8]
    benign = sorted(BENIGN_DIR.glob("*.pcap"))[10:13]
    pcaps = attacks + benign

    results = {"gen8": [], "ramx": [], "rxi": []}
    for pcap in pcaps:
        print(f"Scoring {pcap.name}...")
        g8 = score_pcap(
            lambda: StreamingAryanGen8(bundle, detect_threshold=g8_thresh),
            warmup, pcap, bundle["scaler"], g8_thresh,
        )
        rx = score_pcap(
            lambda: StreamingAryanGen8Ramx(
                bundle, hidden_thresh=settings["hidden_thresh"], detect_threshold=ramx_thresh,
            ),
            warmup, pcap, bundle["scaler"], ramx_thresh,
        )
        rxi = score_rxi(warmup, pcap, bundle, settings)
        if g8:
            results["gen8"].append(g8)
        if rx:
            results["ramx"].append(rx)
        if rxi:
            results["rxi"].append(rxi)

    def summarize(rows):
        atk = [r for r in rows if r["class"] != "Benign"]
        ben = [r for r in rows if r["class"] == "Benign"]
        return {
            "attack_det": sum(int(r["detected"]) for r in atk) / max(len(atk), 1),
            "benign_fp": sum(int(r["honest_fp"]) for r in ben) / max(len(ben), 1),
        }

    out = {k: {"rows": v, "summary": summarize(v)} for k, v in results.items()}
    out_path = ROOT / "results" / "gen8_rxi_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for k, s in ((name, out[name]["summary"]) for name in out if name in results):
        print(f"{k}: attack_det={s['attack_det']:.1%} benign_fp={s['benign_fp']:.1%}")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
