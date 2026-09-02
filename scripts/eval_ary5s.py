#!/usr/bin/env python3
"""Evaluate ARY checkpoint on CIC test + lab PCAPs (base + RAMX).

Usage:
  python scripts/eval_ary5s.py --ckpt models/checkpoints/ary5s_v01.pt --window-sec 5
  python scripts/eval_ary5s.py --ckpt models/checkpoints/aryan_world_model_best.pt --window-sec 30
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

from scripts.ram_improve_eval import CONTEXT, calibrate_match_thresh, infer  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import pcap_to_states  # noqa: E402
from src.aryan.streaming_variants import (  # noqa: E402
    StreamingARY,
    StreamingARYRamxV01,
    calibrate_thresholds,
)
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402


def load_ckpt(path: Path) -> TemporalTransformerWorldModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    model.load_state_dict(sd)
    model.eval()
    return model


def eval_cic_test(model, splits_dir: Path) -> dict:
    te_s, te_b, te_m = load_all_splits(splits_dir)["test"]
    p_atts, tb = [], []
    for t in range(CONTEXT - 1, len(te_s) - 1):
        seq = te_s[t - CONTEXT + 1 : t + 1]
        out = infer(model, seq)
        p_atts.append(out["p_att"])
        tb.append(int(te_b[t + 1]))
    pred = (np.array(p_atts) > 0.5).astype(int)
    tb_arr = np.array(tb)
    tp = int(((pred == 1) & (tb_arr == 1)).sum())
    fp = int(((pred == 1) & (tb_arr == 0)).sum())
    fn = int(((pred == 0) & (tb_arr == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "fpr": fp / max(fp + int((pred == 0).sum()), 1)}


def replay_system(sys_obj, states, bins, mits):
    p_atts = []
    for s, tb, tm in zip(states, bins, mits):
        out = sys_obj.step(s, true_bin=tb, true_mit=tm)
        p_atts.append(out["p_att"])
    return p_atts


def score_round(p_atts, bins, attack_start: int) -> dict:
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
    return {"f1": f1, "precision": prec, "recall": rec, "ttd": ttd, "detected": ttd is not None, "harm": harm}


def eval_lab_pcaps(model, window_sec: float, hidden_thresh: float) -> dict:
    pcaps = sorted(SAVE_DIR.glob("*.pcap"))
    if not pcaps:
        return {"error": f"no pcaps in {SAVE_DIR}"}

    systems = {
        "base": lambda: StreamingARY("ary_base", base_model=model, hidden_thresh=hidden_thresh),
        "ramx": lambda: StreamingARYRamxV01(base_model=model, hidden_thresh=hidden_thresh),
    }
    rows = {k: [] for k in systems}

    for pcap in pcaps:
        states, labels, _ = pcap_to_states(pcap, window_sec=window_sec)
        if len(states) == 0:
            continue
        bins = [0 if lab == "Benign" else 1 for lab in labels]
        mits = [0] * len(bins)
        attack_start = next((i for i, b in enumerate(bins) if b == 1), len(bins))
        for name, factory in systems.items():
            p = replay_system(factory(), [np.asarray(s) for s in states], bins, mits)
            rows[name].append({"pcap": pcap.name, **score_round(p, bins, attack_start)})

    out = {}
    for name, entries in rows.items():
        if not entries:
            continue
        out[name] = {
            "n_pcaps": len(entries),
            "mean_f1": float(np.mean([e["f1"] for e in entries])),
            "detection_rate": float(np.mean([1.0 if e["detected"] else 0.0 for e in entries])),
            "mean_harm": float(np.mean([e["harm"] for e in entries])),
            "entries": entries[:5],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "aryan_splits_5s")
    p.add_argument("--window-sec", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "ary5s_eval.json")
    args = p.parse_args()

    model = load_ckpt(args.ckpt)
    va_s, va_b, _ = load_all_splits(args.splits_dir)["val"]
    raw_t, hidden_t = calibrate_thresholds(model, va_s, va_b)

    cic = eval_cic_test(model, args.splits_dir)
    lab = eval_lab_pcaps(model, args.window_sec, hidden_t)

    result = {
        "ckpt": str(args.ckpt),
        "splits_dir": str(args.splits_dir),
        "window_sec": args.window_sec,
        "hidden_thresh": hidden_t,
        "cic_test": cic,
        "lab_pcaps": lab,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
