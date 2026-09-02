#!/usr/bin/env python3
"""Head-to-head: ARY.01 30s vs ARY 5s — base + RAMX on all benchmarks.

Usage:
  python scripts/compare_ary5_vs_ary01.py
  python scripts/compare_ary5_vs_ary01.py --json results/ary5_vs_ary01_full.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ram_improve_eval import CONTEXT, infer  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import pcap_to_states  # noqa: E402
from src.aryan.lab_scoring import compute_round_scores  # noqa: E402
from src.aryan.streaming_variants import (  # noqa: E402
    StreamingARY,
    StreamingARYRamxV01,
    calibrate_thresholds,
    load_model,
)
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402

CKPT_30 = ROOT / "models" / "checkpoints" / "aryan_world_model_best.pt"
CKPT_5 = ROOT / "models" / "checkpoints" / "ary5s_v01.pt"
SPLITS_30 = ROOT / "data" / "aryan_splits"
SPLITS_5 = ROOT / "data" / "aryan_splits_5s"
WARMUP_N = 20

SYSTEMS = ("ary30_base", "ary30_ramx", "ary5_base", "ary5_ramx")
CLASS_RE = re.compile(r"(T\d+_[A-Za-z0-9_]+)")


def load_ckpt(path: Path) -> TemporalTransformerWorldModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    model.load_state_dict(sd)
    model.eval()
    return model


def eval_cic(model, splits_dir: Path) -> dict:
    te_s, te_b, _ = load_all_splits(splits_dir)["test"]
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
    tn = int(((pred == 0) & (tb_arr == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "f1": f1, "precision": prec, "recall": rec,
        "fpr": fp / max(fp + tn, 1), "n": len(tb_arr),
        "detected": rec > 0,
    }


def replay(factory, states, bins, mits) -> list[float]:
    sys_obj = factory()
    out = []
    for s, tb, tm in zip(states, bins, mits):
        out.append(sys_obj.step(s, true_bin=tb, true_mit=tm)["p_att"])
    return out


def score_seq(p_atts: list[float], bins: list[int], attack_start: int) -> dict:
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


def benign_warmup(splits_dir: Path, n: int):
    va_s, va_b, va_m = load_all_splits(splits_dir)["val"]
    idx = np.where(va_b == 0)[0][:n]
    return (
        [np.asarray(va_s[i], dtype=np.float32) for i in idx],
        [0] * len(idx),
        [int(va_m[i]) for i in idx],
    )


def pcap_timeline(pcap: Path, window_sec: float, warmup_s, warmup_b, warmup_m):
    states, _, _ = pcap_to_states(pcap, window_sec=window_sec)
    if len(states) == 0:
        return None
    cls = resolve_pcap_class(pcap.name) or "Benign"
    atk = 0 if cls == "Benign" else 1
    atk_s = [np.asarray(s, dtype=np.float32) for s in states]
    atk_b = [atk] * len(atk_s)
    atk_m = [0] * len(atk_b)
    all_s = warmup_s + atk_s
    all_b = warmup_b + atk_b
    all_m = warmup_m + atk_m
    start = len(warmup_s)
    return all_s, all_b, all_m, start, cls


def class_from_pcap(path: str) -> str:
    m = CLASS_RE.search(Path(path).stem)
    return m.group(1) if m else Path(path).stem


def head_to_head(rows: list[dict], key_a: str, key_b: str, metric: str = "f1") -> dict:
    a_wins = b_wins = ties = 0
    a_vals, b_vals = [], []
    losses_for_b: list[str] = []
    for row in rows:
        sa = row["scores"][key_a].get(metric, 0)
        sb = row["scores"][key_b].get(metric, 0)
        a_vals.append(sa)
        b_vals.append(sb)
        if sa > sb:
            a_wins += 1
            losses_for_b.append(row.get("id", "?"))
        elif sb > sa:
            b_wins += 1
        else:
            ties += 1
    return {
        "a": key_a, "b": key_b, "metric": metric,
        "a_mean": float(np.mean(a_vals)) if a_vals else 0.0,
        "b_mean": float(np.mean(b_vals)) if b_vals else 0.0,
        "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "n": len(rows),
        "b_loses_to_a": losses_for_b,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, default=ROOT / "results" / "ary5_vs_ary01_full.json")
    args = p.parse_args()

    print("Loading models...")
    model_30 = load_ckpt(CKPT_30)
    model_5 = load_ckpt(CKPT_5)
    _, h30 = calibrate_thresholds(model_30, *load_all_splits(SPLITS_30)["val"][:2])
    _, h5 = calibrate_thresholds(model_5, *load_all_splits(SPLITS_5)["val"][:2])

    factories = {
        "ary30_base": lambda: StreamingARY("ary_base", base_model=model_30, hidden_thresh=h30),
        "ary30_ramx": lambda: StreamingARYRamxV01(base_model=model_30, hidden_thresh=h30),
        "ary5_base": lambda: StreamingARY("ary_base", base_model=model_5, hidden_thresh=h5),
        "ary5_ramx": lambda: StreamingARYRamxV01(base_model=model_5, hidden_thresh=h5),
    }

    result: dict = {"systems": list(SYSTEMS), "benchmarks": {}}

    # --- CIC test (each model on its own split) ---
    print("\n=== CIC test (in-distribution) ===")
    cic = {
        "ary30_base": eval_cic(model_30, SPLITS_30),
        "ary30_ramx": eval_cic(model_30, SPLITS_30),  # RAMX needs streaming; use base ckpt proxy below
    }
    # RAMX on CIC batch: replay val->test style from val calib split
    va_s, va_b, va_m = load_all_splits(SPLITS_30)["val"]
    te_s, te_b, te_m = load_all_splits(SPLITS_30)["test"]
    states30 = [np.asarray(s, dtype=np.float32) for s in te_s]
    bins30 = te_b.tolist()
    mits30 = te_m.tolist()
    attack_start30 = next((i for i, b in enumerate(bins30) if b == 1), len(bins30))
    cic_stream = {}
    for sid in SYSTEMS:
        p_atts = replay(factories[sid], states30, bins30, mits30)
        cic_stream[sid] = score_seq(p_atts, bins30, attack_start30)
    cic["ary30_base"] = cic_stream["ary30_base"]
    cic["ary30_ramx"] = cic_stream["ary30_ramx"]

    states5 = [np.asarray(s, dtype=np.float32) for s in load_all_splits(SPLITS_5)["test"][0]]
    bins5 = load_all_splits(SPLITS_5)["test"][1].tolist()
    mits5 = load_all_splits(SPLITS_5)["test"][2].tolist()
    attack_start5 = next((i for i, b in enumerate(bins5) if b == 1), len(bins5))
    for sid in ("ary5_base", "ary5_ramx"):
        p_atts = replay(factories[sid], states5, bins5, mits5)
        cic_stream[sid] = score_seq(p_atts, bins5, attack_start5)
    cic.update({k: cic_stream[k] for k in ("ary5_base", "ary5_ramx")})
    result["benchmarks"]["cic_test"] = {"rows": [{"id": "cic_test", "scores": cic}], "summary": cic}
    for sid in SYSTEMS:
        m = cic[sid]
        print(f"  {sid:<14} F1={m['f1']:.3f} prec={m['precision']:.3f} rec={m['recall']:.3f} harm={m.get('harm', 0)}")

    # --- Lab PCAPs (all in SAVE_DIR) ---
    print("\n=== Lab PCAPs (OOD, CIC warm-up + PCAP @ native window) ===")
    w30_s, w30_b, w30_m = benign_warmup(SPLITS_30, WARMUP_N)
    w5_s, w5_b, w5_m = benign_warmup(SPLITS_5, WARMUP_N)
    pcap_rows = []
    pcaps = sorted(SAVE_DIR.glob("*.pcap"))
    for pcap in pcaps:
        t30 = pcap_timeline(pcap, 30.0, w30_s, w30_b, w30_m)
        t5 = pcap_timeline(pcap, 5.0, w5_s, w5_b, w5_m)
        if t30 is None or t5 is None:
            continue
        s30, b30, m30, start30, cls = t30
        s5, b5, m5, start5, _ = t5
        scores = {}
        for sid in ("ary30_base", "ary30_ramx"):
            scores[sid] = score_seq(replay(factories[sid], s30, b30, m30), b30, start30)
        for sid in ("ary5_base", "ary5_ramx"):
            scores[sid] = score_seq(replay(factories[sid], s5, b5, m5), b5, start5)
        pcap_rows.append({"id": pcap.name, "class_id": cls, "scores": scores})

    result["benchmarks"]["lab_pcaps"] = {
        "n": len(pcap_rows),
        "rows": pcap_rows,
        "mean_f1": {sid: float(np.mean([r["scores"][sid]["f1"] for r in pcap_rows])) for sid in SYSTEMS},
        "det_rate": {sid: float(np.mean([1.0 if r["scores"][sid]["detected"] else 0.0 for r in pcap_rows])) for sid in SYSTEMS},
        "mean_harm": {sid: float(np.mean([r["scores"][sid]["harm"] for r in pcap_rows])) for sid in SYSTEMS},
    }
    print(f"  n_pcaps={len(pcap_rows)}")
    for sid in SYSTEMS:
        mf = result["benchmarks"]["lab_pcaps"]["mean_f1"][sid]
        dr = result["benchmarks"]["lab_pcaps"]["det_rate"][sid]
        print(f"  {sid:<14} mean_F1={mf:.3f} det_rate={dr:.1%}")

    # --- Live lab rounds ---
    print("\n=== Live lab zero-day rounds (none evasion) ===")
    live_rows = []
    live_dir = ROOT / "results" / "ram_improve" / "live_lab"
    for rf in sorted(live_dir.rglob("round.json")):
        if rf.parent.name != "none":
            continue
        data = json.loads(rf.read_text())
        states = [np.array(t["state"], dtype=np.float32) for t in data["trace"]]
        bins = [int(t["true_bin"]) for t in data["trace"]]
        mits = [int(t["true_mit"]) for t in data["trace"]]
        start = next((i for i, b in enumerate(bins) if b == 1), len(bins))
        scores = {sid: score_seq(replay(factories[sid], states, bins, mits), bins, start) for sid in SYSTEMS}
        live_rows.append({
            "id": f"{data['objective']}/none",
            "class_id": data.get("class_id"),
            "scores": scores,
        })
    result["benchmarks"]["live_lab"] = {
        "n": len(live_rows),
        "rows": live_rows,
        "mean_f1": {sid: float(np.mean([r["scores"][sid]["f1"] for r in live_rows])) for sid in SYSTEMS},
        "det_rate": {sid: float(np.mean([1.0 if r["scores"][sid]["detected"] else 0.0 for r in live_rows])) for sid in SYSTEMS},
    }
    for sid in SYSTEMS:
        mf = result["benchmarks"]["live_lab"]["mean_f1"][sid]
        dr = result["benchmarks"]["live_lab"]["det_rate"][sid]
        print(f"  {sid:<14} mean_F1={mf:.3f} det_rate={dr:.1%}")

    # --- XMT OOD catalog ---
    print("\n=== XMT test (OOD catalog bots, 30s ingest) ===")
    xmt_path = ROOT / "data" / "xmt_splits" / "test.npz"
    xmt_rows = []
    if xmt_path.exists():
        xmt = np.load(xmt_path, allow_pickle=True)
        by_class: dict[str, list[int]] = defaultdict(list)
        for i, src in enumerate(xmt["source_pcap"]):
            by_class[class_from_pcap(str(src))].append(i)
        va_s, va_b, va_m = load_all_splits(SPLITS_30)["val"]
        bidx = np.where(va_b == 0)[0][:WARMUP_N]
        warmup = [np.asarray(va_s[i], dtype=np.float32) for i in bidx]
        wb, wm = [0] * len(warmup), [0] * len(warmup)
        for cls, idxs in sorted(by_class.items()):
            if cls == "Benign":
                continue
            atk_s = [np.asarray(xmt["states"][i], dtype=np.float32) for i in idxs]
            atk_b = xmt["labels_binary"][idxs].tolist()
            atk_m = xmt["labels_mitre"][idxs].tolist()
            s30 = warmup + atk_s
            b30 = wb + atk_b
            m30 = wm + atk_m
            start = len(warmup)
            s5 = s30  # same 242-d states; 5s model still runs (window geometry mismatch noted)
            scores = {}
            for sid in ("ary30_base", "ary30_ramx"):
                scores[sid] = score_seq(replay(factories[sid], s30, b30, m30), b30, start)
            for sid in ("ary5_base", "ary5_ramx"):
                scores[sid] = score_seq(replay(factories[sid], s5, b30, m30), b30, start)
            xmt_rows.append({"id": cls, "scores": scores})
        result["benchmarks"]["xmt_ood"] = {
            "n": len(xmt_rows),
            "rows": xmt_rows,
            "mean_f1": {sid: float(np.mean([r["scores"][sid]["f1"] for r in xmt_rows])) for sid in SYSTEMS},
            "det_rate": {sid: float(np.mean([1.0 if r["scores"][sid]["detected"] else 0.0 for r in xmt_rows])) for sid in SYSTEMS},
        }
        for sid in SYSTEMS:
            mf = result["benchmarks"]["xmt_ood"]["mean_f1"][sid]
            dr = result["benchmarks"]["xmt_ood"]["det_rate"][sid]
            print(f"  {sid:<14} mean_F1={mf:.3f} det_rate={dr:.1%} (n_classes={len(xmt_rows)})")
    else:
        print("  skip — no xmt_splits/test.npz")

    # --- Head-to-head ---
    print("\n=== Head-to-head (does ARY-5s lose to ARY.01 30s?) ===")
    h2h = {}
    for bench in ("cic_test", "lab_pcaps", "live_lab", "xmt_ood"):
        if bench not in result["benchmarks"]:
            continue
        rows = result["benchmarks"][bench].get("rows", [])
        if not rows:
            continue
        h2h[bench] = {
            "base_30_vs_base_5": head_to_head(rows, "ary30_base", "ary5_base"),
            "ramx_30_vs_ramx_5": head_to_head(rows, "ary30_ramx", "ary5_ramx"),
            "base_30_vs_ramx_5": head_to_head(rows, "ary30_base", "ary5_ramx"),
            "ramx_30_vs_ramx_5_det": head_to_head(rows, "ary30_ramx", "ary5_ramx", "detected"),
        }
        for name, h in h2h[bench].items():
            winner = h["a"] if h["a_mean"] > h["b_mean"] else (h["b"] if h["b_mean"] > h["a_mean"] else "tie")
            print(f"  [{bench}] {name}: 30s mean={h['a_mean']:.3f} vs 5s mean={h['b_mean']:.3f} "
                  f"(wins {h['a_wins']}/{h['b_wins']}/{h['ties']}) -> {winner}")

    # Overall where 5s loses to 30s (RAMX vs RAMX F1)
    all_rows = []
    for bench in result["benchmarks"].values():
        all_rows.extend(bench.get("rows", []))
    overall_ramx = head_to_head(all_rows, "ary30_ramx", "ary5_ramx")
    overall_base = head_to_head(all_rows, "ary30_base", "ary5_base")
    result["head_to_head"] = {"overall_base": overall_base, "overall_ramx": overall_ramx, "by_benchmark": h2h}

    print("\n=== OVERALL (all scenarios pooled) ===")
    for label, h in [("base", overall_base), ("RAMX", overall_ramx)]:
        print(f"  {label}: 30s mean F1={h['a_mean']:.3f}  5s mean F1={h['b_mean']:.3f}  "
              f"30s wins {h['a_wins']}  5s wins {h['b_wins']}  ties {h['ties']}")
        if h["b_loses_to_a"]:
            print(f"    5s loses to 30s on ({len(h['b_loses_to_a'])}): {', '.join(h['b_loses_to_a'][:15])}"
                  + ("..." if len(h["b_loses_to_a"]) > 15 else ""))

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2))
    print(f"\nFull JSON -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
