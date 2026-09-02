#!/usr/bin/env python3
"""Live IPS + red-team: ARY-5s+RAMX or Shaun SN2RXv2 (scaled traffic).

Scores one blocking system per run; optional shadow scorers (e.g. sn_base on
the same trace for zero-day comparison).

Usage:
  python scripts/live_ips_multi.py --backend ary --duration 45 --speed 20
  python scripts/live_ips_multi.py --backend sn2rx --log-systems sn_base --scale-factor 10 --replicate 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lab_traffic_scale import DEFAULT_REPLICATE, DEFAULT_SCALE_FACTOR, scale_prism_rows  # noqa: E402
from scripts.ram_improve_eval import CONTEXT  # noqa: E402
from src.adversarial.bots.lab_objective_base import new_round_id  # noqa: E402
from src.adversarial.ips_controller import block_container, clear_blocks, get_container_ip  # noqa: E402
from src.adversarial.lab_clock import LabClock  # noqa: E402
from src.adversarial.lab_config import (  # noqa: E402
    IPS_OUT_ROOT,
    REDTEAM_CONTAINER,
    REDTEAM_EVENTS_CONTAINER_PATH,
    TARGET_CONTAINER,
)
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.aryan.constants import MITRE_STAGES, TACTIC_TO_STAGE  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import TARGET_DIM, _states_from_flow_df  # noqa: E402
from src.aryan.ips_scoring import compute_ips_scores  # noqa: E402
from src.aryan.streaming_variants import StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402
from src.pipeline.extract import pcap_to_rows  # noqa: E402
from src.shaun.streaming import (  # noqa: E402
    StreamingShaunBase,
    StreamingShaunRamxV2,
    load_shaun_bundle,
    pcap_to_shaun_state,
)

LAB_CTL = ROOT / "scripts" / "lab_ctl.py"
SPLITS_5S = ROOT / "data" / "aryan_splits_5s"
CKPT_V01 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
MITRE_TACTIC = "Credential Access"
EARLY_ALERT_SEC = 120.0
DETECT_THRESHOLD = 0.5


def _json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


BACKEND_CONFIG = {
    "ary": {"window_sec": 5.0, "block_id": "ary5_ramx", "tag": "ary5_ramx"},
    "sn2rx": {"window_sec": 15.0, "block_id": "sn2rx", "tag": "sn2rx"},
}


def load_ary_ckpt(path: Path) -> TemporalTransformerWorldModel:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    model.load_state_dict(sd)
    model.eval()
    return model


def reset_lab(*, scale_benign: bool = True) -> None:
    env = os.environ.copy()
    if scale_benign:
        env.setdefault("BENIGN_WORKERS", "10")
        env.setdefault("BENIGN_INTERVAL", "0.5")
    subprocess.run([sys.executable, str(LAB_CTL), "reset"], cwd=str(ROOT), check=True, env=env)
    if scale_benign:
        subprocess.run(
            ["docker", "compose", "-f", str(ROOT / "docker" / "docker-compose.yml"), "-p", "prism",
             "up", "-d", "--force-recreate", "benign-client"],
            cwd=str(ROOT), check=True, env=env,
        )
    clear_blocks()


def state_from_pcap(
    pcap_path: Path,
    *,
    backend: str,
    window_sec: float,
    scale_factor: float,
    replicate: int,
    last_state: np.ndarray | None,
) -> np.ndarray:
    if backend == "sn2rx":
        return pcap_to_shaun_state(
            pcap_path, last_state=last_state, scale_factor=scale_factor, replicate=replicate,
        )
    from src.pipeline.extract import pcap_to_rows
    rows = pcap_to_rows(pcap_path)
    if scale_factor != 1.0 or replicate != 1:
        rows = scale_prism_rows(rows, factor=scale_factor, replicate=replicate)
    if not rows:
        if last_state is not None:
            return last_state.copy()
        return np.zeros(TARGET_DIM, dtype=np.float32)
    states, _, _ = _states_from_flow_df(pd.DataFrame(rows), window_sec=window_sec)
    if len(states):
        return states[0].astype(np.float32)
    if last_state is not None:
        return last_state.copy()
    return np.zeros(TARGET_DIM, dtype=np.float32)


def capture_one_window(
    capture: TrafficCapture,
    filename: str,
    clock: LabClock,
    last_state: np.ndarray | None,
    *,
    backend: str,
    window_sec: float,
    scale_factor: float,
    replicate: int,
) -> tuple[np.ndarray, float, float]:
    t_start = time.time()
    capture.start_capture(filename)
    clock.sleep(clock.window_sec())
    pcap_path = capture.stop_capture()
    t_end = time.time()

    state = None
    if pcap_path is not None and pcap_path.exists():
        try:
            state = state_from_pcap(
                pcap_path, backend=backend, window_sec=window_sec,
                scale_factor=scale_factor, replicate=replicate, last_state=last_state,
            )
        except Exception as exc:
            print(f"WARNING: ingest failed on {pcap_path}: {exc}", file=sys.stderr)
        try:
            pcap_path.unlink()
        except OSError:
            pass
    if state is None:
        dim = 292 if backend == "sn2rx" else TARGET_DIM
        state = last_state.copy() if last_state is not None else np.zeros(dim, dtype=np.float32)
    return state.astype(np.float32), t_start, t_end


def read_redteam_events() -> list[dict]:
    r = subprocess.run(
        ["docker", "exec", REDTEAM_CONTAINER, "cat", REDTEAM_EVENTS_CONTAINER_PATH],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    events = []
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def clear_redteam_events() -> None:
    subprocess.run(
        ["docker", "exec", REDTEAM_CONTAINER, "sh", "-c", f": > {REDTEAM_EVENTS_CONTAINER_PATH}"],
        capture_output=True,
    )


def start_redteam_agent(round_id: str, duration_sec: float) -> None:
    clear_redteam_events()
    cmd = [
        "docker", "exec", "-d", REDTEAM_CONTAINER,
        "python", "/agent/redteam_agent.py",
        "--target-host", "target-server",
        "--duration-sec", str(duration_sec),
        "--events", REDTEAM_EVENTS_CONTAINER_PATH,
        "--round-id", round_id,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"failed to start redteam agent: {r.stderr.strip()}")


def ensure_redteam_running() -> None:
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", REDTEAM_CONTAINER], capture_output=True, text=True)
    if r.returncode != 0 or (r.stdout or "").strip().lower() != "true":
        raise RuntimeError("redteam container is not running — run: python scripts/lab_ctl.py up")


def poll_early_theft(events: list[dict], t0: float, alerted: set[str]) -> None:
    for e in events:
        if e.get("stage") != "post_exploit" or not e.get("success"):
            continue
        rid = e.get("round_id", "")
        if rid in alerted:
            continue
        elapsed = float(e["ts"]) - t0
        if elapsed <= EARLY_ALERT_SEC:
            print(f"\n*** EARLY CRED ALERT ({elapsed:.1f}s): verified password hashes stolen ***\n")
        else:
            print(f"\n*** CRED THEFT SUCCESS at {elapsed:.1f}s ***\n")
        alerted.add(rid)


def build_systems(
    backend: str,
    log_systems: list[str],
) -> tuple[dict[str, object], str]:
    cfg = BACKEND_CONFIG[backend]
    block_id = cfg["block_id"]
    systems: dict[str, object] = {}

    if backend == "ary":
        model = load_ary_ckpt(CKPT_V01)
        splits = load_all_splits(SPLITS_5S)
        va_s, va_b, _ = splits["val"]
        _, hidden_thresh = calibrate_thresholds(model, va_s, va_b)
        systems[block_id] = StreamingARYRamxV01(base_model=model, hidden_thresh=hidden_thresh)
    else:
        bundle = load_shaun_bundle()
        # Gate all steps until warmup completes (updated after warmup loop).
        systems[block_id] = StreamingShaunRamxV2(bundle, context_skip_steps=10_000)
        systems.setdefault("sn_base", StreamingShaunBase(bundle))

    for sid in log_systems:
        if sid == "sn_base" and sid not in systems:
            systems[sid] = StreamingShaunBase(load_shaun_bundle())
    return systems, block_id


def run_experiment(
    *,
    backend: str,
    duration_sec: float,
    speed: float,
    warmup_sec: float,
    out_dir: Path,
    scale_factor: float,
    replicate: int,
    log_systems: list[str],
    scale_benign: bool = True,
) -> dict:
    ensure_redteam_running()
    reset_lab(scale_benign=scale_benign)

    cfg = BACKEND_CONFIG[backend]
    window_sec = cfg["window_sec"]
    model_tag = cfg["tag"]
    round_id = new_round_id()
    round_dir = out_dir / model_tag
    round_dir.mkdir(parents=True, exist_ok=True)

    clock = LabClock(speed=speed, base_window_sec=window_sec)
    capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps")
    mit_stage = MITRE_STAGES.get(TACTIC_TO_STAGE.get(MITRE_TACTIC, "Benign"), 0)

    try:
        redteam_ip = get_container_ip(REDTEAM_CONTAINER)
        print(f"  redteam IP: {redteam_ip}")
    except RuntimeError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        redteam_ip = ""

    print(
        f"  backend={backend}  scale={scale_factor}x replicate={replicate}  "
        f"benign_scaled={scale_benign}  window={window_sec}s  speed={speed}x"
    )
    print(f"  warm-up: {warmup_sec:.0f}s benign-only (red team idle)")

    systems, block_id = build_systems(backend, log_systems)
    all_ids = list(systems.keys())

    trace: list[dict] = []
    state_box: dict[str, np.ndarray | None] = {"last": None}
    ips_blocked = False
    blocked_at_sec: float | None = None
    blocked_at_window: int | None = None
    t0 = time.time()
    alerted: set[str] = set()
    last_poll = 0.0
    warmup_windows = 0

    def do_window(true_bin: int, phase: str, window_idx: int) -> None:
        nonlocal ips_blocked, blocked_at_sec, blocked_at_window, warmup_windows
        state, t_start, t_end = capture_one_window(
            capture, f"w{window_idx:05d}.pcap", clock, state_box["last"],
            backend=backend, window_sec=window_sec,
            scale_factor=scale_factor, replicate=replicate,
        )
        state_box["last"] = state
        true_mit = mit_stage if true_bin else 0

        sys_out: dict[str, dict] = {}
        for sid, sys_obj in systems.items():
            out = sys_obj.step(state, true_bin=true_bin, true_mit=true_mit)
            sys_out[sid] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in out.items()}

        p_block = float(sys_out[block_id]["p_att"])
        if phase == "attack" and p_block >= DETECT_THRESHOLD and not ips_blocked:
            br = block_container(REDTEAM_CONTAINER)
            if br.blocked:
                ips_blocked = True
                blocked_at_sec = time.time() - t0
                blocked_at_window = window_idx
                print(f"  IPS BLOCK ({block_id}) at window {window_idx} ({blocked_at_sec:.1f}s): {br.source_ip}")
            else:
                print(f"  WARNING: IPS block failed at window {window_idx}: {br.message}", file=sys.stderr)

        trace.append({
            "window_idx": window_idx,
            "t_start": t_start,
            "t_end": t_end,
            "phase": phase,
            "true_bin": true_bin,
            "true_mit": true_mit,
            "ips_blocked": ips_blocked,
            "systems": sys_out,
        })
        if phase == "warmup":
            warmup_windows += 1

    window_idx = 0
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        do_window(true_bin=0, phase="warmup", window_idx=window_idx)
        window_idx += 1
    print(f"  warmup done: {warmup_windows} windows over {time.time() - warmup_start:.1f}s")

    if backend == "sn2rx" and block_id in systems:
        systems[block_id].set_context_skip(warmup_windows)
        print(f"  SN2RX context gate set to {warmup_windows} warmup windows")

    start_redteam_agent(round_id, duration_sec)
    attack_start_wall = time.time()
    print(f"  attack phase: {duration_sec}s wall @ {clock.window_sec():.2f}s/window (zero-day cred theft)")
    while time.time() - attack_start_wall < duration_sec:
        do_window(true_bin=1, phase="attack", window_idx=window_idx)
        window_idx += 1
        now = time.time()
        if now - last_poll >= 2.0:
            poll_early_theft(read_redteam_events(), t0, alerted)
            last_poll = now

    events = read_redteam_events()
    poll_early_theft(events, t0, alerted)

    scores_by_system = {
        sid: compute_ips_scores(
            trace, events, system_id=sid,
            ips_blocked=ips_blocked if sid == block_id else False,
            blocked_at_sec=blocked_at_sec if sid == block_id else None,
            blocked_at_window=blocked_at_window if sid == block_id else None,
        )
        for sid in all_ids
    }
    scores = scores_by_system[block_id]

    payload = {
        "backend": backend,
        "model": model_tag,
        "round_id": round_id,
        "duration_sec": duration_sec,
        "speed": speed,
        "state_window_sec": window_sec,
        "scale_factor": scale_factor,
        "replicate": replicate,
        "scale_benign": scale_benign,
        "wall_window_sec": clock.window_sec(),
        "warmup_sec": warmup_sec,
        "warmup_windows": warmup_windows,
        "n_windows": len(trace),
        "redteam_ip": redteam_ip,
        "block_system_id": block_id,
        "log_systems": all_ids,
        "zero_day_objective": "T1555_sqli_cred_theft",
        "events": events,
        "trace": trace,
        "scores": scores,
        "scores_by_system": scores_by_system,
        **{k: scores[k] for k in (
            "creds_stolen", "stolen_at_sec", "ips_blocked", "blocked_at_sec",
            "blocked_at_window", "prevented_theft", "outcome",
        )},
    }

    out_path = round_dir / "round.json"
    out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(f"\n=== {model_tag} IPS result ===")
    print(f"  outcome: {scores['outcome']}  prevented: {scores['prevented_theft']}")
    print(f"  F1={scores['binary_f1']:.3f}  TTD={scores['ttd_sec']}  benign_harm={scores['benign_harm']}")
    for sid in all_ids:
        if sid == block_id:
            continue
        s = scores_by_system[sid]
        atk_p = [float(w["systems"].get(sid, {}).get("p_att", 0)) for w in trace if w["phase"] == "attack"]
        max_p = max(atk_p) if atk_p else 0.0
        print(
            f"  shadow {sid}: F1={s['binary_f1']:.3f} det={s['detected']} "
            f"max_P={max_p:.3f} harm={s['benign_harm']}"
        )
    print(f"  -> {out_path}")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=tuple(BACKEND_CONFIG), default="ary")
    p.add_argument("--duration", type=float, default=30.0, help="Wall-clock attack phase seconds")
    p.add_argument("--warmup-sec", type=float, default=15.0)
    p.add_argument("--speed", type=float, default=20.0)
    p.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    p.add_argument("--replicate", type=int, default=DEFAULT_REPLICATE)
    p.add_argument("--no-scale-benign", action="store_true", help="Keep default benign-client rate")
    p.add_argument("--log-systems", default="", help="Comma-separated shadow scorers, e.g. sn_base")
    p.add_argument("--out-dir", type=Path, default=IPS_OUT_ROOT / "sn2rx_compare")
    args = p.parse_args()

    if args.backend == "sn2rx" and not (ROOT.parent / "PRISM-shaun").exists():
        print("Missing PRISM-shaun worktree for Shaun backend", file=sys.stderr)
        return 1
    if args.backend == "ary" and not CKPT_V01.exists():
        print(f"Missing checkpoint: {CKPT_V01}", file=sys.stderr)
        return 1

    log_systems = [s.strip() for s in args.log_systems.split(",") if s.strip()]
    if args.backend == "sn2rx" and "sn_base" not in log_systems:
        log_systems.append("sn_base")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(
        backend=args.backend,
        duration_sec=args.duration,
        speed=args.speed,
        warmup_sec=args.warmup_sec,
        out_dir=args.out_dir,
        scale_factor=args.scale_factor,
        replicate=args.replicate,
        log_systems=log_systems,
        scale_benign=not args.no_scale_benign,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
