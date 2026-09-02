#!/usr/bin/env python3
"""Run IPS red-team experiments for ARY-5sV01+RAMX vs ARY-5sV02+RAMX.

Resets the lab between runs and writes a comparison report.

Usage:
  python scripts/compare_ips_v01_v02.py
  python scripts/compare_ips_v01_v02.py --duration 1200 --speed 10
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

LIVE = ROOT / "scripts" / "live_ips_redteam.py"


def run_model(model: str, duration: float, speed: float, warmup_sec: float, out_dir: Path) -> dict:
    cmd = [
        sys.executable, str(LIVE),
        "--model", model,
        "--duration", str(duration),
        "--speed", str(speed),
        "--warmup-sec", str(warmup_sec),
        "--out-dir", str(out_dir),
    ]
    print(f"\n{'=' * 60}\nStarting IPS red-team run: {model}\n{'=' * 60}")
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"live_ips_redteam.py failed for {model} (exit {r.returncode})")
    round_path = out_dir / model / "round.json"
    return json.loads(round_path.read_text(encoding="utf-8"))


def fmt_bool(v: bool | None) -> str:
    if v is None:
        return "n/a"
    return "yes" if v else "no"


def fmt_sec(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}s"


def write_compare_md(out_dir: Path, results: dict[str, dict]) -> Path:
    lines = [
        "# IPS Red-Team Compare: ARY-5sV01+RAMX vs ARY-5sV02+RAMX",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Model | Outcome | Creds stolen | Stolen at | IPS blocked | Blocked at | Prevented theft | F1 | TTD | Benign harm |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in ("v01", "v02"):
        r = results.get(model, {})
        s = r.get("scores", {})
        lines.append(
            f"| {model} | {s.get('outcome', 'n/a')} | {fmt_bool(s.get('creds_stolen'))} | "
            f"{fmt_sec(s.get('stolen_at_sec'))} | {fmt_bool(s.get('ips_blocked'))} | "
            f"{fmt_sec(s.get('blocked_at_sec'))} | {fmt_bool(s.get('prevented_theft'))} | "
            f"{s.get('binary_f1', 0):.3f} | {fmt_sec(s.get('ttd_sec'))} | {s.get('benign_harm', 0)} |"
        )
    lines.extend(["", "## Notes", ""])
    for model in ("v01", "v02"):
        r = results.get(model, {})
        if not r:
            continue
        lines.append(f"### {model}")
        lines.append(f"- Round: `{r.get('round_id')}`")
        lines.append(f"- Warmup: {r.get('warmup_sec')}s ({r.get('warmup_windows')} windows)")
        lines.append(f"- Attack windows: {r.get('n_windows')} @ {r.get('state_window_sec')}s state / {r.get('wall_window_sec')}s wall")
        lines.append(f"- Checkpoint: `{r.get('checkpoint')}`")
        lines.append("")

    path = out_dir / "COMPARE.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--warmup-sec", type=float, default=30.0)
    p.add_argument("--out-dir", type=Path, default=IPS_OUT_ROOT)
    p.add_argument("--skip-up", action="store_true", help="Do not run lab_ctl up first")
    args = p.parse_args()

    if not args.skip_up:
        print("Ensuring lab is up...")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "up"], cwd=str(ROOT))
        if r.returncode != 0:
            return r.returncode

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for model in ("v01", "v02"):
        results[model] = run_model(model, args.duration, args.speed, args.warmup_sec, args.out_dir)

    summary_path = args.out_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_path = write_compare_md(args.out_dir, results)
    print(f"\nComparison written to {md_path}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
