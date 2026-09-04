#!/usr/bin/env python3
"""Long-running Docker lab: slow-rising threat vs RAMX / Shaun (5s windows).

Backends: sn2rx3, ary5_ramx, sn_base (Shaun V2 classifier, w5s).
Runs control (immediate strike) vs poison (slow ramp + strike). ML only — no IPS block.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.live_ips_multi import (  # noqa: E402
    BACKEND_CONFIG,
    W5S_CKPT,
    _json_safe,
    build_systems,
    capture_one_window,
    clear_redteam_events,
    ensure_redteam_running,
    reset_lab,
    stop_redteam_event_stream,
)
from src.adversarial.bots.lab_objective_base import new_round_id  # noqa: E402
from src.adversarial.lab_clock import LabClock  # noqa: E402
from src.adversarial.lab_config import REDTEAM_CONTAINER, REDTEAM_EVENTS_CONTAINER_PATH  # noqa: E402
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.aryan.constants import MITRE_STAGES, TACTIC_TO_STAGE  # noqa: E402
from src.shaun.streaming import StreamingShaunBase, load_shaun_bundle  # noqa: E402

MITRE_TACTIC = "Credential Access"
DETECT_THRESHOLD = 0.5
TARGET_CONTAINER = "target-server"
REPORTS_ROOT = ROOT / "reports"


def ingest_backend(backend: str) -> str:
    if backend in ("sn2rx3", "sn_base"):
        return "sn2rx3"
    return backend


def resolve_systems(backend: str, shaun_ckpt: Path | None) -> tuple[dict[str, object], str]:
    if backend == "sn_base":
        bundle = load_shaun_bundle(ckpt_path=shaun_ckpt)
        return {"sn_base": StreamingShaunBase(bundle)}, "sn_base"
    return build_systems(backend, [], shaun_ckpt=shaun_ckpt)


def start_redteam_script(script: str, round_id: str, duration_sec: float) -> None:
    clear_redteam_events()
    cmd = [
        "docker", "exec", "-d", REDTEAM_CONTAINER,
        "python", f"/agent/{script}",
        "--target-host", "target-server",
        "--duration-sec", str(duration_sec),
        "--events", REDTEAM_EVENTS_CONTAINER_PATH,
        "--round-id", round_id,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed to start {script}: {r.stderr.strip()}")


def summarize_strike(trace: list[dict], block_id: str) -> dict[str, Any]:
    strike = [w for w in trace if w["phase"] == "strike"]
    if not strike:
        return {"n_windows": 0, "detected": False, "max_p_att": 0.0, "ttd_sec": None}
    p_vals = [float(w["systems"][block_id]["p_att"]) for w in strike]
    raw_vals = [float(w["systems"][block_id].get("raw_p_att", p)) for w, p in zip(strike, p_vals)]
    t0 = strike[0]["t_start"]
    ttd_sec = None
    for w in strike:
        if float(w["systems"][block_id]["p_att"]) >= DETECT_THRESHOLD:
            ttd_sec = float(w["t_start"]) - t0
            break
    return {
        "n_windows": len(strike),
        "detected": any(p >= DETECT_THRESHOLD for p in p_vals),
        "max_p_att": max(p_vals),
        "max_raw_p_att": max(raw_vals),
        "mean_p_att": float(np.mean(p_vals)),
        "ttd_sec": ttd_sec,
        "first3_p": p_vals[:3],
    }


def summarize_ramp(trace: list[dict], block_id: str) -> dict[str, Any]:
    ramp = [w for w in trace if w["phase"] == "ramp"]
    if not ramp:
        return {"n_windows": 0}
    p_vals = [float(w["systems"][block_id]["p_att"]) for w in ramp]
    raw_vals = [float(w["systems"][block_id].get("raw_p_att", p)) for w, p in zip(ramp, p_vals)]
    offsets = [float(w["systems"][block_id].get("raw_offset", 0) or 0) for w in ramp]
    return {
        "n_windows": len(ramp),
        "max_p_att": max(p_vals),
        "mean_p_att": float(np.mean(p_vals)),
        "max_raw_p_att": max(raw_vals),
        "raw_offset_start": offsets[0] if offsets else 0.0,
        "raw_offset_end": offsets[-1] if offsets else 0.0,
        "any_alert_during_ramp": any(p >= DETECT_THRESHOLD for p in p_vals),
    }


def write_report_md(
    *,
    backend: str,
    results: dict[str, Any],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Slow-Rising Poison Test — `{backend}` (5s windows)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Setup",
        "",
        "- Docker adversarial lab (scaled benign 10×, replicate 5×)",
        "- ML detection only (no IPS block)",
        "- **Control:** warmup → immediate full cred-theft agent",
        "- **Poison:** warmup → 5 min slow-rising ramp (scored benign) → strike",
        "",
    ]
    if "comparison" in results:
        c = results["comparison"]
        lines.extend([
            "## Comparison",
            "",
            "| Metric | Control | Poison |",
            "|--------|---------|--------|",
            f"| Detected at strike | {c['control_detected']} | {c['poison_detected']} |",
            f"| Max P(attack) | {c['control_max_p']:.3f} | {c['poison_max_p']:.3f} |",
            f"| Delta max P | — | {c['delta_max_p']:+.3f} |",
            f"| Poison suppressed detection | — | {c['poison_suppressed_detection']} |",
            "",
        ])
        if "ramp_raw_offset_drift" in c:
            lines.append(f"- Raw offset drift (poison ramp): {c['ramp_raw_offset_drift']:.3f}")
            lines.append("")
    for key in ("control", "poison"):
        if key not in results or "strike" not in results[key]:
            continue
        s = results[key]["strike"]
        r = results[key].get("ramp_summary", {})
        lines.extend([
            f"## {key.title()} run",
            "",
            f"- Strike detected: **{s['detected']}**  max_P={s['max_p_att']:.3f}  TTD={s.get('ttd_sec')}",
            f"- Ramp windows: {r.get('n_windows', 0)}  max_P={r.get('max_p_att', 0):.3f}",
            "",
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_condition(
    *,
    name: str,
    mode: str,
    backend: str,
    block_id: str,
    systems: dict[str, object],
    warmup_sec: float,
    ramp_sec: float,
    strike_sec: float,
    speed: float,
    scale_factor: float,
    replicate: int,
    window_sec: float,
    out_dir: Path,
) -> dict[str, Any]:
    ensure_redteam_running()
    reset_lab(scale_benign=True)

    round_id = new_round_id()
    round_dir = out_dir / name
    round_dir.mkdir(parents=True, exist_ok=True)

    cap_backend = ingest_backend(backend)
    clock = LabClock(speed=speed, base_window_sec=window_sec)
    capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps")
    atk_capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps_active")

    if hasattr(systems.get(block_id), "begin_live_phase"):
        systems[block_id].begin_live_phase()

    mit_stage = MITRE_STAGES.get(TACTIC_TO_STAGE.get(MITRE_TACTIC, "Benign"), 0)
    trace: list[dict] = []

    print(f"\n=== {backend} / {name} ({mode}) ===")
    print(f"  window={window_sec}s  warmup={warmup_sec:.0f}s  ramp={ramp_sec:.0f}s  strike={strike_sec:.0f}s")

    state_box: dict[str, np.ndarray | None] = {"last": None}
    window_idx = 0
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        state, t_start, t_end = capture_one_window(
            capture, f"w{window_idx:05d}.pcap", clock, state_box["last"],
            backend=cap_backend, window_sec=window_sec,
            scale_factor=scale_factor, replicate=replicate,
        )
        state_box["last"] = state
        sys_out = {}
        for sid, sys_obj in systems.items():
            out = sys_obj.step(state, true_bin=0, true_mit=0)
            sys_out[sid] = _json_safe(out)
        trace.append({
            "window_idx": window_idx, "t_start": t_start, "t_end": t_end,
            "phase": "warmup", "true_bin": 0, "true_mit": 0,
            "capture_window_sec": window_sec, "systems": sys_out,
        })
        window_idx += 1
    warmup_windows = window_idx
    print(f"  warmup: {warmup_windows} windows")

    if hasattr(systems.get(block_id), "set_context_skip"):
        systems[block_id].set_context_skip(warmup_windows)

    agent_duration = ramp_sec + strike_sec if mode == "poison" else strike_sec
    agent_script = "redteam_agent_slow.py" if mode == "poison" else "redteam_agent.py"
    start_redteam_script(agent_script, round_id, agent_duration)

    if mode == "poison" and ramp_sec > 0:
        ramp_start = time.time()
        while time.time() - ramp_start < ramp_sec:
            state, t_start, t_end = capture_one_window(
                atk_capture, f"w{window_idx:05d}.pcap", clock, state_box["last"],
                backend=cap_backend, window_sec=window_sec,
                scale_factor=scale_factor, replicate=replicate,
            )
            state_box["last"] = state
            sys_out = {}
            for sid, sys_obj in systems.items():
                out = sys_obj.step(state, true_bin=0, true_mit=0)
                sys_out[sid] = _json_safe(out)
            trace.append({
                "window_idx": window_idx, "t_start": t_start, "t_end": t_end,
                "phase": "ramp", "true_bin": 0, "true_mit": 0,
                "capture_window_sec": window_sec, "systems": sys_out,
            })
            window_idx += 1
        print(f"  ramp: {window_idx - warmup_windows} windows")

    strike_start = time.time()
    while time.time() - strike_start < strike_sec:
        state, t_start, t_end = capture_one_window(
            atk_capture, f"w{window_idx:05d}.pcap", clock, state_box["last"],
            backend=cap_backend, window_sec=window_sec,
            scale_factor=scale_factor, replicate=replicate,
        )
        state_box["last"] = state
        sys_out = {}
        for sid, sys_obj in systems.items():
            out = sys_obj.step(state, true_bin=1, true_mit=mit_stage)
            sys_out[sid] = _json_safe(out)
        trace.append({
            "window_idx": window_idx, "t_start": t_start, "t_end": t_end,
            "phase": "strike", "true_bin": 1, "true_mit": mit_stage,
            "capture_window_sec": window_sec, "systems": sys_out,
        })
        window_idx += 1

    stop_redteam_event_stream()

    strike_sum = summarize_strike(trace, block_id)
    ramp_sum = summarize_ramp(trace, block_id)
    print(f"  strike: detected={strike_sum['detected']} max_P={strike_sum['max_p_att']:.3f} TTD={strike_sum.get('ttd_sec')}")

    payload = {
        "backend": backend,
        "block_id": block_id,
        "condition": name,
        "mode": mode,
        "round_id": round_id,
        "window_sec": window_sec,
        "warmup_sec": warmup_sec,
        "ramp_sec": ramp_sec,
        "strike_sec": strike_sec,
        "warmup_windows": warmup_windows,
        "n_windows": len(trace),
        "ramp_summary": ramp_sum,
        "strike": strike_sum,
        "trace": trace,
    }
    out_path = round_dir / "round.json"
    out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"  -> {out_path}")
    return payload


def run_backend_suite(
    backend: str,
    *,
    warmup_sec: float,
    ramp_sec: float,
    strike_sec: float,
    speed: float,
    scale_factor: float,
    replicate: int,
    window_sec: float,
    shaun_ckpt: Path | None,
    out_dir: Path,
    control_only: bool,
    poison_only: bool,
) -> dict[str, Any]:
    systems, block_id = resolve_systems(backend, shaun_ckpt)
    backend_out = out_dir / backend
    backend_out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"backend": backend, "block_id": block_id}

    if not poison_only:
        results["control"] = run_condition(
            name="control", mode="control", backend=backend, block_id=block_id,
            systems=systems, warmup_sec=warmup_sec, ramp_sec=0.0, strike_sec=strike_sec,
            speed=speed, scale_factor=scale_factor, replicate=replicate,
            window_sec=window_sec, out_dir=backend_out,
        )
    if not control_only:
        results["poison"] = run_condition(
            name="poison_slow_rise", mode="poison", backend=backend, block_id=block_id,
            systems=systems, warmup_sec=warmup_sec, ramp_sec=ramp_sec, strike_sec=strike_sec,
            speed=speed, scale_factor=scale_factor, replicate=replicate,
            window_sec=window_sec, out_dir=backend_out,
        )

    if "control" in results and "poison" in results:
        c = results["control"]["strike"]
        p = results["poison"]["strike"]
        results["comparison"] = {
            "control_detected": c["detected"],
            "poison_detected": p["detected"],
            "control_max_p": c["max_p_att"],
            "poison_max_p": p["max_p_att"],
            "delta_max_p": p["max_p_att"] - c["max_p_att"],
            "poison_suppressed_detection": (
                (not p["detected"] and c["detected"]) or (p["max_p_att"] < c["max_p_att"] - 0.05)
            ),
            "ramp_raw_offset_drift": (
                results["poison"]["ramp_summary"].get("raw_offset_end", 0)
                - results["poison"]["ramp_summary"].get("raw_offset_start", 0)
            ),
        }
        (backend_out / "comparison.json").write_text(
            json.dumps(results["comparison"], indent=2), encoding="utf-8"
        )

    slim = {k: {kk: vv for kk, vv in v.items() if kk != "trace"} if isinstance(v, dict) else v for k, v in results.items()}
    (backend_out / "summary.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    report_dir = REPORTS_ROOT / "lab" / "poison-slow-rise" / backend
    write_report_md(backend=backend, results=results, report_path=report_dir / "REPORT.md")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("sn2rx3", "ary", "sn_base", "all"), default="all")
    ap.add_argument("--warmup-sec", type=float, default=60.0)
    ap.add_argument("--ramp-sec", type=float, default=300.0)
    ap.add_argument("--strike-sec", type=float, default=60.0)
    ap.add_argument("--window-sec", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--scale-factor", type=float, default=10.0)
    ap.add_argument("--replicate", type=int, default=5)
    ap.add_argument("--shaun-ckpt", type=Path, default=W5S_CKPT)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "ips_redteam" / "poison_slow_rise")
    ap.add_argument("--control-only", action="store_true")
    ap.add_argument("--poison-only", action="store_true")
    args = ap.parse_args()

    backends = ["sn2rx3", "ary", "sn_base"] if args.backend == "all" else [args.backend]
    if args.backend in ("sn2rx3", "sn_base") and not args.shaun_ckpt.exists():
        print(f"Missing checkpoint: {args.shaun_ckpt}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for backend in backends:
        ckpt = args.shaun_ckpt if backend in ("sn2rx3", "sn_base") else None
        all_results[backend] = run_backend_suite(
            backend,
            warmup_sec=args.warmup_sec,
            ramp_sec=args.ramp_sec,
            strike_sec=args.strike_sec,
            speed=args.speed,
            scale_factor=args.scale_factor,
            replicate=args.replicate,
            window_sec=args.window_sec,
            shaun_ckpt=ckpt,
            out_dir=args.out_dir,
            control_only=args.control_only,
            poison_only=args.poison_only,
        )

    (args.out_dir / "all_backends_summary.json").write_text(
        json.dumps({k: v.get("comparison", v) for k, v in all_results.items()}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
