#!/usr/bin/env python3
"""Live IPS + isolated red-team experiment (ARY-5s + RAMX).

Runs a 20-minute cred-theft attempt from the isolated ``redteam`` container
while the host orchestrator captures 5s traffic windows, scores with
``StreamingARYRamxV01``, and blocks the red-team IP on ``target-server`` when
P(attack) >= 0.5.

Usage:
  python scripts/live_ips_redteam.py --model v01 --duration 1200 --speed 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.aryan.streaming_variants import CONTEXT  # noqa: E402
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
from src.aryan.ingest import TARGET_DIM, pcap_to_states  # noqa: E402
from src.aryan.ips_scoring import compute_ips_scores  # noqa: E402
from src.aryan.streaming_variants import StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402
from src.models.streaming_gen8 import StreamingGen8WorldModel, load_gen8_checkpoint  # noqa: E402
from src.models.gen8_world_model import Gen8DecoupledMLPWorldModel  # noqa: E402

LAB_CTL = ROOT / "scripts" / "lab_ctl.py"
SPLITS_5S = ROOT / "data" / "splits_universal_gen8_5s"
SPLITS_ARYAN_5S = ROOT / "data" / "aryan_splits_5s"
SPLITS_1S = ROOT / "data" / "aryan_splits_1s"
CKPT_GEN8 = ROOT / "weights" / "universal_gen8" / "world_model_best.pt"
CKPT_V01 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
CKPT_V02 = ROOT / "models" / "checkpoints" / "ary_5sv02.pt"
CKPT_1S = ROOT / "models" / "checkpoints" / "ary_1sv01.pt"
STATE_WINDOW_SEC = 5.0
SYSTEM_ID = "gen8_world_model"
MITRE_TACTIC = "Credential Access"
EARLY_ALERT_SEC = 120.0
DETECT_THRESHOLD = 0.5


def load_ckpt(path: Path) -> nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("model_class") == "Gen8DecoupledMLPWorldModel" or "universal_gen8" in str(path):
        return load_gen8_checkpoint(path)
    sd = ck.get("model_state_dict", ck)
    d_state = sd["embedding.proj.weight"].shape[1]
    model = TemporalTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=CONTEXT)
    model.load_state_dict(sd)
    model.eval()
    return model


def reset_lab() -> None:
    subprocess.run([sys.executable, str(LAB_CTL), "reset"], cwd=str(ROOT), check=True)
    clear_blocks()


def capture_one_window(
    capture: TrafficCapture,
    filename: str,
    clock: LabClock,
    last_state: np.ndarray | None,
    *,
    state_window_sec: float,
) -> tuple[np.ndarray, float, float]:
    t_start = time.time()
    capture.start_capture(filename)
    clock.sleep(clock.window_sec())
    pcap_path = capture.stop_capture()
    t_end = time.time()

    state = None
    if pcap_path is not None and pcap_path.exists():
        try:
            states, _, _ = pcap_to_states(pcap_path, window_sec=state_window_sec)
        except Exception as exc:
            print(f"WARNING: pcap_to_states failed on {pcap_path}: {exc}", file=sys.stderr)
            states = np.zeros((0, TARGET_DIM), dtype=np.float32)
        if len(states):
            state = states[0]
        try:
            pcap_path.unlink()
        except OSError:
            pass
    if state is None:
        state = last_state.copy() if last_state is not None else np.zeros(TARGET_DIM, dtype=np.float32)
    return state.astype(np.float32), t_start, t_end


def read_redteam_events() -> list[dict]:
    r = subprocess.run(
        ["docker", "exec", REDTEAM_CONTAINER, "cat", REDTEAM_EVENTS_CONTAINER_PATH],
        capture_output=True,
        text=True,
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


def run_experiment(
    *,
    model_tag: str,
    ckpt_path: Path,
    splits_dir: Path,
    state_window_sec: float,
    duration_sec: float,
    speed: float,
    warmup_sec: float,
    out_dir: Path,
) -> dict:
    ensure_redteam_running()
    reset_lab()

    round_id = new_round_id()
    round_dir = out_dir / model_tag
    round_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint {ckpt_path} ...")
    model = load_ckpt(ckpt_path)
    if isinstance(model, Gen8DecoupledMLPWorldModel):
        system = StreamingGen8WorldModel(base_model=model, detect_thresh=DETECT_THRESHOLD)
    else:
        splits = load_all_splits(splits_dir)
        va_s, va_b, _ = splits["val"]
        _, hidden_thresh = calibrate_thresholds(model, va_s, va_b)
        system = StreamingARYRamxV01(base_model=model, hidden_thresh=hidden_thresh)

    clock = LabClock(speed=speed, base_window_sec=state_window_sec)
    capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps")
    mit_stage = MITRE_STAGES.get(TACTIC_TO_STAGE.get(MITRE_TACTIC, "Benign"), 0)

    try:
        redteam_ip = get_container_ip(REDTEAM_CONTAINER)
        print(f"  redteam IP: {redteam_ip}")
    except RuntimeError as exc:
        print(f"WARNING: {exc}", file=sys.stderr)
        redteam_ip = ""

    print(f"  warm-up: {warmup_sec:.0f}s benign-only (red team idle)")

    trace: list[dict] = []
    state_box: dict[str, np.ndarray | None] = {"last": None}
    ips_blocked = False
    blocked_at_sec: float | None = None
    blocked_at_window: int | None = None
    t0 = time.time()
    alerted: set[str] = set()
    last_poll = 0.0

    def do_window(true_bin: int, phase: str, window_idx: int) -> None:
        nonlocal ips_blocked, blocked_at_sec, blocked_at_window
        state, t_start, t_end = capture_one_window(
            capture, f"w{window_idx:05d}.pcap", clock, state_box["last"], state_window_sec=state_window_sec,
        )
        state_box["last"] = state
        true_mit = mit_stage if true_bin else 0
        out = system.step(state, true_bin=true_bin, true_mit=true_mit)
        p_att = float(out["p_att"])

        if p_att >= DETECT_THRESHOLD and not ips_blocked:
            br = block_container(REDTEAM_CONTAINER)
            if br.blocked:
                ips_blocked = True
                blocked_at_sec = time.time() - t0
                blocked_at_window = window_idx
                print(f"  IPS BLOCK at window {window_idx} ({blocked_at_sec:.1f}s): {br.source_ip}")
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
            "systems": {SYSTEM_ID: {"p_att": p_att}},
        })

    window_idx = 0
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        do_window(true_bin=0, phase="warmup", window_idx=window_idx)
        window_idx += 1
    print(f"  warmup done: {window_idx} windows over {time.time() - warmup_start:.1f}s")

    start_redteam_agent(round_id, duration_sec)
    attack_start_wall = time.time()
    print(f"  attack phase: {duration_sec}s wall clock @ {clock.window_sec():.2f}s/window")
    while time.time() - attack_start_wall < duration_sec:
        do_window(true_bin=1, phase="attack", window_idx=window_idx)
        window_idx += 1
        now = time.time()
        if now - last_poll >= 2.0:
            poll_early_theft(read_redteam_events(), t0, alerted)
            last_poll = now

    events = read_redteam_events()
    poll_early_theft(events, t0, alerted)

    scores = compute_ips_scores(
        trace,
        events,
        system_id=SYSTEM_ID,
        ips_blocked=ips_blocked,
        blocked_at_sec=blocked_at_sec,
        blocked_at_window=blocked_at_window,
    )

    payload = {
        "model": model_tag,
        "checkpoint": str(ckpt_path),
        "round_id": round_id,
        "duration_sec": duration_sec,
        "speed": speed,
        "state_window_sec": state_window_sec,
        "splits_dir": str(splits_dir),
        "wall_window_sec": clock.window_sec(),
        "warmup_sec": warmup_sec,
        "warmup_windows": sum(1 for w in trace if w["phase"] == "warmup"),
        "n_windows": len(trace),
        "redteam_ip": redteam_ip,
        "system_id": SYSTEM_ID,
        "events": events,
        "trace": trace,
        "scores": scores,
        "creds_stolen": scores["creds_stolen"],
        "stolen_at_sec": scores["stolen_at_sec"],
        "ips_blocked": scores["ips_blocked"],
        "blocked_at_sec": scores["blocked_at_sec"],
        "blocked_at_window": scores["blocked_at_window"],
        "prevented_theft": scores["prevented_theft"],
        "outcome": scores["outcome"],
    }

    out_path = round_dir / "round.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n=== {model_tag} result ===")
    print(f"  outcome: {scores['outcome']}")
    print(f"  creds_stolen: {scores['creds_stolen']}  stolen_at_sec: {scores['stolen_at_sec']}")
    print(f"  ips_blocked: {scores['ips_blocked']}  blocked_at_sec: {scores['blocked_at_sec']}")
    print(f"  prevented_theft: {scores['prevented_theft']}")
    print(f"  F1={scores['binary_f1']:.3f}  TTD={scores['ttd_sec']}  benign_harm={scores['benign_harm']}")
    print(f"  -> {out_path}")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("gen8", "v01", "v02", "1s"), default="gen8")
    p.add_argument("--window-sec", type=float, default=None, help="StateBuilder window (default: 5, or 1 for --model 1s)")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--splits-dir", type=Path, default=None)
    p.add_argument("--duration", type=float, default=120.0, help="Wall-clock attack phase seconds")
    p.add_argument("--warmup-sec", type=float, default=30.0, help="Benign-only warmup before red team starts")
    p.add_argument("--speed", type=float, default=10.0)
    p.add_argument("--out-dir", type=Path, default=IPS_OUT_ROOT)
    args = p.parse_args()

    if args.model == "gen8":
        ckpt = args.ckpt or CKPT_GEN8
        splits_dir = args.splits_dir or SPLITS_5S
        window_sec = args.window_sec or 5.0
        model_tag = "gen8"
    elif args.model == "v01":
        ckpt = args.ckpt or (CKPT_GEN8 if not CKPT_V01.exists() else CKPT_V01)
        splits_dir = args.splits_dir or SPLITS_5S
        window_sec = args.window_sec or 5.0
        model_tag = "gen8" if ckpt == CKPT_GEN8 else "v01"
    elif args.model == "v02":
        ckpt = args.ckpt or CKPT_V02
        splits_dir = args.splits_dir or SPLITS_5S
        window_sec = args.window_sec or 5.0
        model_tag = "v02"
    else:
        ckpt = args.ckpt or CKPT_1S
        splits_dir = args.splits_dir or SPLITS_1S
        window_sec = args.window_sec or 1.0
        model_tag = "1s"

    if not ckpt.exists():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(
        model_tag=model_tag,
        ckpt_path=ckpt,
        splits_dir=splits_dir,
        state_window_sec=window_sec,
        duration_sec=args.duration,
        speed=args.speed,
        warmup_sec=args.warmup_sec,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
