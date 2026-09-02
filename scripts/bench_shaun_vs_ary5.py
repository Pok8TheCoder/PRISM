#!/usr/bin/env python3
"""Compare ARY-5sV01 (base + RAMX) vs Shaun PRISM V2 on lab PCAPs and live Docker rounds."""

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
from scripts.plot_lab_ary_comparison import benign_warmup, build_timeline, replay, score_round  # noqa: E402
from src.adversarial.lab_config import SAVE_DIR  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.streaming_variants import StreamingARY, StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402

CKPT_5 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
SHAUN_CKPT = SHAUN_ROOT / "weights" / "world_model.pt"
SPLITS_5 = ROOT / "data" / "aryan_splits_5s"
OUT_DIR = ROOT / "results" / "shaun_vs_ary5"
WARMUP_N = 20

LIVE_ROUNDS = [
    ROOT / "results/ram_improve/live_lab/cred_theft/none/round.json",
    ROOT / "results/ram_improve/live_lab/key_theft/none/round.json",
    ROOT / "results/ram_improve/live_lab/defacement/none/round.json",
]


def pcap_labels(name: str, n: int) -> list[int]:
    cls = resolve_pcap_class(name) or "Benign"
    return [0 if cls == "Benign" else 1] * n


def run_shaun_sidecar(pcaps: list[Path]) -> dict:
    if not pcaps:
        return {}
    cmd = [sys.executable, str(ROOT / "scripts" / "shaun_eval_sidecar.py"), *[str(p.resolve()) for p in pcaps]]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"shaun sidecar failed: {r.stderr.strip() or r.stdout.strip()}")
    return json.loads(r.stdout)


def eval_pcap(pcap: Path, model_5, hidden_5: float, shaun_sidecar: dict, warmup_5) -> dict:
    cls = resolve_pcap_class(pcap.name) or "Benign"
    s5, b5, m5, start5 = build_timeline(pcap, 5.0, *warmup_5)
    if not s5:
        return {"pcap": pcap.name, "class": cls, "error": "empty 5s ingest"}

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
            return {
                **score_round(p, bs, start_s),
                "raw_summary": side.get(key, {}),
            }

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


def eval_live_round(round_path: Path, model_5, hidden_5: float) -> dict | None:
    if not round_path.exists():
        return None
    data = json.loads(round_path.read_text(encoding="utf-8"))
    trace = data["trace"]
    states = [np.array(t["state"], dtype=np.float32) for t in trace]
    bins = [int(t["true_bin"]) for t in trace]
    mits = [int(t["true_mit"]) for t in trace]
    attack_start = next((i for i, b in enumerate(bins) if b == 1), len(bins))
    systems = {
        "ary5_base": lambda: StreamingARY("ary_base", base_model=model_5, hidden_thresh=hidden_5),
        "ary5_ramx": lambda: StreamingARYRamxV01(base_model=model_5, hidden_thresh=hidden_5),
    }
    series = {k: replay(f, states, bins, mits) for k, f in systems.items()}
    out = {
        "scenario": f"{data['objective']}/{data.get('evasion', 'none')}",
        "source": str(round_path.relative_to(ROOT)),
        "window_sec_trace": data.get("window_sec"),
        "n_windows": len(bins),
        "attack_start": attack_start,
        "note": "Live Docker trace @ 30s windows (242-d). Shaun not comparable on this trace.",
    }
    for k in ("ary5_base", "ary5_ramx"):
        s = series[k]
        out[k] = {
            **score_round(s, bins, attack_start),
            "raw_summary": {
                "warmup_mean": float(np.mean(s[:attack_start])) if attack_start else 0.0,
                "attack_mean": float(np.mean(s[attack_start:])) if attack_start < len(s) else 0.0,
                "attack_max": float(np.max(s[attack_start:])) if attack_start < len(s) else 0.0,
            },
        }
    return out


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


def write_report(lab_rows: list[dict], live_rows: list[dict], out_dir: Path) -> None:
    lines = [
        "# Shaun PRISM V2 vs ARY-5sV01 (base + RAMX)",
        "",
        f"Shaun: `{SHAUN_CKPT}` (`origin/shaun`, 292D @ 15s, L=30)",
        f"ARY-5s: `{CKPT_5}` (242D @ 5s, RAMX uses oracle labels)",
        "",
        "Shaun lab ingest: **fair path** — PCAP → CIC-style rows → `SchemaAligner` → `StateBuilder` (15s).",
        "Shaun RAMX: **v2 context-gated** — baseline from context buffer, fusion alerts only on lab PCAP (fixes warmup FP).",
        "ARY RAMX: hidden-key memory bank (`RAMX_V.01`, oracle labels).",
        "",
        "## Lab PCAP aggregate",
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
        s = summarize(lab_rows, key)
        if s:
            lines.append(
                f"| {name} | {s['mean_f1']:.3f} | {s['detection_rate']:.1%} | "
                f"{s['mean_attack_max_p']:.3f} | {s['mean_warmup_mean_p']:.3f} |"
            )
    lines.extend(["", "## Live Docker rounds", ""])
    for row in live_rows:
        lines.append(f"### {row['scenario']}")
        lines.append(f"- windows: {row['n_windows']} @ {row.get('window_sec_trace')}s")
        for k in ("ary5_base", "ary5_ramx"):
            s = row[k]
            lines.append(
                f"- **{k}**: F1={s['f1']:.3f} det={s['detected']} TTD={s['ttd']} "
                f"warmup_P={s['raw_summary']['warmup_mean']:.3f} attack_max_P={s['raw_summary']['attack_max']:.3f}"
            )
        lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-pcaps", type=int, default=0)
    p.add_argument("--pcaps", nargs="*", default=None)
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

    print(f"Running Shaun sidecar on {len(pcaps)} PCAPs...")
    shaun_sidecar = run_shaun_sidecar(pcaps)

    lab_rows = []
    for i, pcap in enumerate(pcaps, 1):
        print(f"[{i}/{len(pcaps)}] {pcap.name}")
        lab_rows.append(eval_pcap(pcap, model_5, hidden_5, shaun_sidecar, warmup_5))

    live_rows = [r for rp in LIVE_ROUNDS if (r := eval_live_round(rp, model_5, hidden_5))]
    for row in live_rows:
        print(f"live {row['scenario']}: base F1={row['ary5_base']['f1']:.3f} ramx F1={row['ary5_ramx']['f1']:.3f}")

    payload = {
        "shaun_branch": "origin/shaun @ 2b54a38",
        "shaun_ingest": "schema_aligner",
        "shaun_ramx_impl": shaun_sidecar.get("ramx_impl"),
        "shaun_ramx_warmup_steps": shaun_sidecar.get("ramx_warmup_steps"),
        "lab_pcaps": lab_rows,
        "live_docker": live_rows,
        "lab_summary": {k: summarize(lab_rows, k) for k in ("ary5_base", "ary5_ramx", "shaun_v2", "shaun_v2_ramx")},
        "shaun_hidden_thresh": shaun_sidecar.get("hidden_thresh"),
    }
    (args.out_dir / "bench.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(lab_rows, live_rows, args.out_dir)
    print(f"Wrote {args.out_dir / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
