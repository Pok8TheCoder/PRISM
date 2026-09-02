#!/usr/bin/env python3
"""Quick IPS compare: ARY-5s+RAMX vs Shaun SN2RXv2 (+ sn_base zero-day shadow).

Short defaults (~2-3 min total). Scaled traffic (10x + 5x replicate + busier
benign client). Cred-theft is a live zero-day vs CIC training.

Usage:
  python scripts/compare_ips_sn2rx_ary.py
  python scripts/compare_ips_sn2rx_ary.py --duration 30 --speed 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.lab_config import IPS_OUT_ROOT  # noqa: E402

LIVE = ROOT / "scripts" / "live_ips_multi.py"
OUT = IPS_OUT_ROOT / "sn2rx_compare"


def run_backend(backend: str, args: argparse.Namespace, out_dir: Path) -> dict:
    cmd = [
        sys.executable, str(LIVE),
        "--backend", backend,
        "--duration", str(args.duration),
        "--warmup-sec", str(args.warmup_sec),
        "--speed", str(args.speed),
        "--scale-factor", str(args.scale_factor),
        "--replicate", str(args.replicate),
        "--out-dir", str(out_dir),
    ]
    if backend == "sn2rx":
        cmd.extend(["--log-systems", "sn_base"])
    print(f"\n{'=' * 60}\nIPS run: {backend}\n{'=' * 60}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"live_ips_multi failed for {backend}")
    tag = "ary5_ramx" if backend == "ary" else "sn2rx"
    return json.loads((out_dir / tag / "round.json").read_text(encoding="utf-8"))


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}"
    return str(v)


def max_attack_p(trace: list, sid: str) -> float:
    vals = [
        float(w["systems"].get(sid, {}).get("p_att", 0))
        for w in trace
        if w.get("phase") == "attack"
    ]
    return max(vals) if vals else 0.0


def write_report(out_dir: Path, ary: dict, sn: dict) -> Path:
    sa = ary.get("scores", {})
    ss = sn.get("scores", {})
    sb = sn.get("scores_by_system", {}).get("sn_base", {})
    trace = sn.get("trace", [])
    sb_max = max_attack_p(trace, "sn_base")
    sn_max = max_attack_p(trace, "sn2rx")
    lines = [
        "# IPS Quick Compare: ARY-5s+RAMX vs SN2RXv2 (scaled live lab)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Traffic: PCAP row scale 10x + 5x replicate; benign-client 10 workers @ 0.5s.",
        "Attack: live SQLi cred theft (zero-day vs CIC).",
        "",
        "## IPS blocking (primary)",
        "",
        "| System | Outcome | Prevented theft | F1 | TTD | Benign harm | Blocked at |",
        "|---|---|---|---|---|---|---|",
        f"| ARY-5s+RAMX | {sa.get('outcome')} | {fmt(sa.get('prevented_theft'))} | "
        f"{fmt(sa.get('binary_f1'))} | {fmt(sa.get('ttd_sec'))} | {fmt(sa.get('benign_harm'))} | "
        f"{fmt(sa.get('blocked_at_sec'))} |",
        f"| **SN2RXv2** | {ss.get('outcome')} | {fmt(ss.get('prevented_theft'))} | "
        f"{fmt(ss.get('binary_f1'))} | {fmt(ss.get('ttd_sec'))} | {fmt(ss.get('benign_harm'))} | "
        f"{fmt(ss.get('blocked_at_sec'))} |",
        "",
        "## Zero-day side test (same SN2RX run, shadow scores)",
        "",
        "| System | F1 | Detected | Max P(attack) | TTD | Benign harm |",
        "|---|---|---|---|---|---|",
        f"| Shaun V2 base (sn_base) | {fmt(sb.get('binary_f1'))} | {fmt(sb.get('detected'))} | "
        f"{fmt(sb_max)} | {fmt(sb.get('ttd_sec'))} | {fmt(sb.get('benign_harm'))} |",
        f"| **SN2RXv2** | {fmt(ss.get('binary_f1'))} | {fmt(ss.get('detected'))} | "
        f"{fmt(sn_max)} | {fmt(ss.get('ttd_sec'))} | {fmt(ss.get('benign_harm'))} |",
        "",
        "## Notes",
        "",
        "- `sn_base` vs `sn2rx` on the **same** scaled live trace tests zero-day detection without extra Docker time.",
        "- ARY RAMX uses oracle labels for memory; Shaun RAMX v2 does not.",
        "",
    ]
    path = out_dir / "COMPARE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--warmup-sec", type=float, default=15.0)
    p.add_argument("--speed", type=float, default=20.0)
    p.add_argument("--scale-factor", type=float, default=10.0)
    p.add_argument("--replicate", type=int, default=5)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ary = run_backend("ary", args, args.out_dir)
    sn = run_backend("sn2rx", args, args.out_dir)

    summary = {"ary5_ramx": ary, "sn2rx": sn}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_report(args.out_dir, ary, sn)
    print(f"\nReport -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
