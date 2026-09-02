#!/usr/bin/env python3
"""Compare ARY-5sV01 vs Shaun V2 (base + RAMX) on CIC-scaled lab PCAPs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
sys.path.insert(0, str(ROOT))

from scripts.eval_ary5s import load_ckpt  # noqa: E402
from scripts.lab_traffic_scale import (  # noqa: E402
    DEFAULT_REPLICATE,
    DEFAULT_SCALE_FACTOR,
    scale_prism_rows,
)
from scripts.plot_lab_ary_comparison import benign_warmup, replay, score_round  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import _states_from_flow_df  # noqa: E402
from src.aryan.streaming_variants import StreamingARY, StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402
from src.pipeline.extract import pcap_to_rows  # noqa: E402

CKPT_5 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
SHAUN_CKPT = SHAUN_ROOT / "weights" / "world_model.pt"
SPLITS_5 = ROOT / "data" / "aryan_splits_5s"
OUT_DIR = ROOT / "results" / "scaled_lab_bench"
WARMUP_N = 20


def pcap_labels(name: str, n: int) -> list[int]:
    cls = resolve_pcap_class(name) or "Benign"
    return [0 if cls == "Benign" else 1] * n


def build_scaled_timeline(
    pcap: Path,
    window_sec: float,
    warmup_states: list[np.ndarray],
    warmup_bins: list[int],
    warmup_mits: list[int],
    scale_factor: float,
    replicate: int,
) -> tuple[list[np.ndarray], list[int], list[int], int]:
    rows = scale_prism_rows(pcap_to_rows(pcap), factor=scale_factor, replicate=replicate)
    if not rows:
        return [], [], [], 0
    states, _, _ = _states_from_flow_df(pd.DataFrame(rows), window_sec=window_sec)
    if len(states) == 0:
        return [], [], [], 0
    atk_states = [np.asarray(s, dtype=np.float32) for s in states]
    atk_bins = pcap_labels(pcap.name, len(atk_states))
    atk_mits = [0] * len(atk_bins)
    all_s = warmup_states + atk_states
    all_b = warmup_bins + atk_bins
    all_m = warmup_mits + atk_mits
    return all_s, all_b, all_m, len(warmup_states)


def run_shaun_sidecar(pcaps: list[Path], scale_factor: float, replicate: int) -> dict:
    if not pcaps:
        return {}
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "shaun_eval_sidecar.py"),
        f"--scale-factor={scale_factor}",
        f"--replicate={replicate}",
        *[str(p.resolve()) for p in pcaps],
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"shaun sidecar failed: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout)


def eval_pcap(
    pcap: Path,
    model_5,
    hidden_5: float,
    shaun_sidecar: dict,
    warmup_5,
    scale_factor: float,
    replicate: int,
) -> dict:
    cls = resolve_pcap_class(pcap.name) or "Benign"
    s5, b5, m5, start5 = build_scaled_timeline(
        pcap, 5.0, *warmup_5, scale_factor=scale_factor, replicate=replicate
    )
    if not s5:
        return {"pcap": pcap.name, "class": cls, "error": "empty scaled 5s ingest"}

    systems = {
        "ary5_base": lambda: StreamingARY("ary_base", base_model=model_5, hidden_thresh=hidden_5),
        "ary5_ramx": lambda: StreamingARYRamxV01(base_model=model_5, hidden_thresh=hidden_5),
    }
    p5 = {k: replay(f, s5, b5, m5) for k, f in systems.items()}

    side = shaun_sidecar.get(str(pcap.resolve())) or shaun_sidecar.get(str(pcap))
    if side and side.get("p_attack_base"):
        start_s = int(side["attack_start"])
        bs = [0] * start_s + pcap_labels(pcap.name, len(side["p_attack_base"]) - start_s)

        def _shaun_scores(key: str, p_key: str) -> dict:
            p = side[p_key]
            return {**score_round(p, bs, start_s), "raw_summary": side.get(key, {})}

        shaun_base = _shaun_scores("base", "p_attack_base")
        shaun_ramx = _shaun_scores("ramx", "p_attack_ramx")
    else:
        shaun_base = shaun_ramx = {"error": "shaun sidecar missing or empty ingest"}
        start_s = WARMUP_N

    row = {
        "pcap": pcap.name,
        "class": cls,
        "attack_start_5s": start5,
        "attack_start_shaun_15s": start_s,
        "ary5_base": {**score_round(p5["ary5_base"], b5, start5), "raw_p_attack": p5["ary5_base"]},
        "ary5_ramx": {**score_round(p5["ary5_ramx"], b5, start5), "raw_p_attack": p5["ary5_ramx"]},
        "shaun_v2": shaun_base,
        "shaun_v2_ramx": shaun_ramx,
    }
    for k in ("ary5_base", "ary5_ramx"):
        r = row[k].pop("raw_p_attack")
        row[k]["raw_summary"] = {
            "warmup_mean": float(np.mean(r[:start5])),
            "attack_mean": float(np.mean(r[start5:])),
            "attack_max": float(np.max(r[start5:])),
        }
    return row


def summarize(rows: list[dict], key: str) -> dict:
    ok = [r for r in rows if key in r and "f1" in r.get(key, {})]
    if not ok:
        return {}
    return {
        "n": len(ok),
        "mean_f1": float(np.mean([r[key]["f1"] for r in ok])),
        "detection_rate": float(np.mean([1.0 if r[key]["detected"] else 0.0 for r in ok])),
        "mean_attack_max_p": float(np.mean([r[key]["raw_summary"]["attack_max"] for r in ok])),
        "mean_warmup_mean_p": float(np.mean([r[key]["raw_summary"]["warmup_mean"] for r in ok])),
    }


def write_report(rows: list[dict], out_dir: Path, scale_factor: float, replicate: int) -> None:
    lines = [
        "# Scaled Lab Bench: ARY-5sV01 vs Shaun V2 (base + RAMX)",
        "",
        f"Traffic scaling: **{scale_factor:g}×** rate/count features, **{replicate}×** flow replication",
        f"(targets CIC-like density; quiet lab median ~20 flows → ~{20 * replicate} flows/capture)",
        "",
        f"ARY-5s: `{CKPT_5}` @ 5s windows | ARY RAMX = hidden-key bank (oracle labels)",
        f"Shaun V2: `{SHAUN_CKPT}` @ 15s SchemaAligner ingest | Shaun RAMX v2 = context-gated WarmupBaselineCalibrator",
        "",
        "## Aggregate (attack PCAPs)",
        "",
        "| Model | Mean F1 | Det rate | Mean attack max P | Mean warmup P |",
        "|---|---|---|---|---|",
    ]
    for name, key in [
        ("ARY-5s base", "ary5_base"),
        ("ARY-5s + RAMX", "ary5_ramx"),
        ("Shaun V2 base", "shaun_v2"),
        ("Shaun V2 + RAMX", "shaun_v2_ramx"),
    ]:
        s = summarize(rows, key)
        if s:
            lines.append(
                f"| {name} | {s['mean_f1']:.3f} | {s['detection_rate']:.1%} | "
                f"{s['mean_attack_max_p']:.3f} | {s['mean_warmup_mean_p']:.3f} |"
            )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-pcaps", type=int, default=0)
    p.add_argument("--pcaps", nargs="*", default=None)
    p.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    p.add_argument("--replicate", type=int, default=DEFAULT_REPLICATE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if not CKPT_5.exists() or not SHAUN_CKPT.exists():
        print("Missing checkpoint(s)", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_5 = load_ckpt(CKPT_5)
    va_s, va_b, _ = load_all_splits(SPLITS_5)["val"]
    _, hidden_5 = calibrate_thresholds(model_5, va_s, va_b)
    warmup_5 = benign_warmup(SPLITS_5, WARMUP_N)

    pcaps = [SAVE_DIR / n for n in args.pcaps] if args.pcaps else sorted(SAVE_DIR.glob("*.pcap"))
    if args.max_pcaps > 0:
        pcaps = pcaps[: args.max_pcaps]
    pcaps = [p for p in pcaps if p.exists()]

    print(
        f"Scaled lab bench: {len(pcaps)} PCAPs, scale={args.scale_factor:g}×, replicate={args.replicate}×"
    )
    shaun_sidecar = run_shaun_sidecar(pcaps, args.scale_factor, args.replicate)

    lab_rows = []
    for i, pcap in enumerate(pcaps, 1):
        print(f"[{i}/{len(pcaps)}] {pcap.name}")
        lab_rows.append(
            eval_pcap(
                pcap, model_5, hidden_5, shaun_sidecar, warmup_5,
                args.scale_factor, args.replicate,
            )
        )

    payload = {
        "scale_factor": args.scale_factor,
        "replicate": args.replicate,
        "shaun_ingest": "schema_aligner",
        "shaun_ramx_impl": shaun_sidecar.get("ramx_impl"),
        "lab_pcaps": lab_rows,
        "lab_summary": {
            k: summarize(lab_rows, k)
            for k in ("ary5_base", "ary5_ramx", "shaun_v2", "shaun_v2_ramx")
        },
        "quiet_lab_reference": str(ROOT / "results" / "shaun_vs_ary5" / "bench.json"),
    }
    (args.out_dir / "bench.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(lab_rows, args.out_dir, args.scale_factor, args.replicate)
    print(f"Wrote {args.out_dir / 'bench.json'}")
    s = payload["lab_summary"]
    for k in ("ary5_base", "ary5_ramx", "shaun_v2", "shaun_v2_ramx"):
        if s.get(k):
            print(f"  {k}: F1={s[k]['mean_f1']:.3f} det={s[k]['detection_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
