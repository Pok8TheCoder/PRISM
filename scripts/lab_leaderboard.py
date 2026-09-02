#!/usr/bin/env python3
"""Aggregate all `round.json` files from scripts/live_attack_lab.py into a
single leaderboard (plan section 5).

Run:
  python scripts/lab_leaderboard.py
  python scripts/lab_leaderboard.py --results-dir results/ram_improve/live_lab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.lab_scoring import aggregate_leaderboard  # noqa: E402

OUT_ROOT = ROOT / "results" / "ram_improve" / "live_lab"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(OUT_ROOT))
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    round_files = sorted(results_dir.glob("*/*/round.json"))
    if not round_files:
        print(f"No round.json files found under {results_dir}", file=sys.stderr)
        return 1

    payload = aggregate_leaderboard(round_files)
    out_path = results_dir / "leaderboard.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(f"Aggregated {len(round_files)} round(s) -> {out_path}\n")
    hdr = f"{'System':<20}{'Rounds':>7}{'Wins':>6}{'Losses':>8}{'DetRate':>9}{'MeanF1':>8}{'MeanTTD(s)':>12}{'BenignHarm':>12}"
    print(hdr)
    print("-" * len(hdr))
    for row in payload["leaderboard"]:
        ttd = f"{row['mean_ttd_sec']:.1f}" if row["mean_ttd_sec"] is not None else "n/a"
        print(
            f"{row['system']:<20}{row['rounds']:>7}{row['wins']:>6}{row['losses']:>8}"
            f"{row['detection_rate']:>9.1%}{row['mean_f1']:>8.3f}{ttd:>12}{row['total_benign_harm']:>12}"
        )
    if payload["leaderboard"]:
        winner = payload["leaderboard"][0]
        print(f"\nRound winner (most wins, fewest losses, best mean F1): {winner['system']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
