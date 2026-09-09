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
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lab_traffic_scale import DEFAULT_REPLICATE, DEFAULT_SCALE_FACTOR, scale_prism_rows  # noqa: E402
from src.aryan.streaming_variants import CONTEXT  # noqa: E402
from src.adversarial.bots.lab_objective_base import new_round_id  # noqa: E402
from src.adversarial.ips_controller import block_container, clear_blocks, get_container_ip  # noqa: E402
from src.adversarial.lab_clock import LabClock  # noqa: E402
from src.adversarial.lab_config import (  # noqa: E402
    IPS_OUT_ROOT,
    REDTEAM_CONTAINER,
    REDTEAM_EVENTS_CONTAINER_PATH,
    REDTEAM_EVENTS_HOST_PATH,
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
try:
    from src.shaun.streaming import (  # noqa: E402
        StreamingShaunBase,
        StreamingShaunRamxV2,
        StreamingShaunRamxV3,
        load_shaun_bundle,
        pcap_to_shaun_state,
    )
except ImportError:
    StreamingShaunBase = None
    StreamingShaunRamxV2 = None
    StreamingShaunRamxV3 = None
    load_shaun_bundle = None
    pcap_to_shaun_state = None
from src.hx.causal import StreamingHXC, load_hxc_bundle  # noqa: E402
from src.aryan.world_model import TemporalTransformerWorldModel  # noqa: E402
from src.models.streaming_gen8 import StreamingGen8WorldModel, load_gen8_checkpoint  # noqa: E402
from src.models.gen8_world_model import Gen8DecoupledMLPWorldModel  # noqa: E402

LAB_CTL = ROOT / "scripts" / "lab_ctl.py"
SPLITS_5S = ROOT / "data" / "splits_universal_gen8_5s"
SPLITS_ARYAN_5S = ROOT / "data" / "aryan_splits_5s"
CKPT_GEN8 = ROOT / "weights" / "universal_gen8" / "world_model_best.pt"
CKPT_V01 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
MITRE_TACTIC = "Credential Access"
EARLY_ALERT_SEC = 120.0
DETECT_THRESHOLD = 0.5
DEFAULT_ATTACK_WINDOW_SEC = 5.0
W5S_CKPT = ROOT.parent / "PRISM-shaun" / "weights" / "w5s" / "world_model.pt"


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
    "gen8": {"window_sec": 5.0, "block_id": "gen8_world_model", "tag": "gen8_world_model"},
    "ary": {"window_sec": 5.0, "block_id": "gen8_world_model", "tag": "gen8_world_model"},
    "sn2rx": {"window_sec": 15.0, "block_id": "sn2rx", "tag": "sn2rx"},
    "sn2rx3": {"window_sec": 15.0, "block_id": "sn2rx3", "tag": "sn2rx3"},
    "hx": {"window_sec": 5.0, "block_id": "hx", "tag": "hx"},
    "hx_c": {"window_sec": 5.0, "block_id": "hx_c", "tag": "hx_c"},
}

SHAUN_LIKE = ("sn2rx", "sn2rx3", "hx", "hx_c")


def load_ary_ckpt(path: Path) -> nn.Module:
    p = path if path.exists() else CKPT_GEN8
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("model_class") == "Gen8DecoupledMLPWorldModel" or "universal_gen8" in str(p):
        return load_gen8_checkpoint(p)
    sd = ck.get("model_state_dict", ck)
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
    if backend in SHAUN_LIKE:
        return pcap_to_shaun_state(
            pcap_path, last_state=last_state, scale_factor=scale_factor, replicate=replicate,
            window_sec=window_sec,
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


def _sleep_with_poll(
    clock: LabClock,
    base_seconds: float,
    poll_fn: Callable[[], None] | None,
    poll_interval: float | None = None,
) -> None:
    """Sleep for simulated lab time, polling red-team events during capture."""
    if base_seconds <= 0:
        if poll_fn:
            poll_fn()
        return
    if poll_fn is None:
        clock.sleep(base_seconds)
        return
    wall_total = base_seconds / clock.speed
    interval = poll_interval if poll_interval is not None else max(0.02, wall_total / 10.0)
    deadline = time.monotonic() + wall_total
    while True:
        poll_fn()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(interval, remaining))


def _run_with_poll(
    work_fn: Callable[[], object],
    poll_fn: Callable[[], None] | None,
    poll_interval: float = 0.05,
) -> object:
    """Run slow work while polling for kill-chain events."""
    if poll_fn is None:
        return work_fn()
    result: list[object] = []
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(work_fn())
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while thread.is_alive():
        poll_fn()
        thread.join(timeout=poll_interval)
    if error:
        raise error[0]
    return result[0]


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
    poll_fn: Callable[[], None] | None = None,
) -> tuple[np.ndarray, float, float]:
    t_start = time.time()
    capture.start_capture(filename)
    _sleep_with_poll(clock, window_sec, poll_fn)
    pcap_path = capture.stop_capture()

    state = None
    if pcap_path is not None and pcap_path.exists():
        try:
            def ingest() -> np.ndarray:
                return state_from_pcap(
                    pcap_path, backend=backend, window_sec=window_sec,
                    scale_factor=scale_factor, replicate=replicate, last_state=last_state,
                )

            state = _run_with_poll(ingest, poll_fn)  # type: ignore[assignment]
        except Exception as exc:
            print(f"WARNING: ingest failed on {pcap_path}: {exc}", file=sys.stderr)
        try:
            pcap_path.unlink()
        except OSError:
            pass
    t_end = time.time()
    if state is None:
        dim = 292 if backend in SHAUN_LIKE else TARGET_DIM
        state = last_state.copy() if last_state is not None else np.zeros(dim, dtype=np.float32)
    return state.astype(np.float32), t_start, t_end


class RedteamEventStream:
    """Tail red-team JSONL inside the container (avoids slow host bind-mount sync on Windows)."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        self.stop()
        self._proc = subprocess.Popen(
            [
                "docker", "exec", REDTEAM_CONTAINER,
                "tail", "-F", "-n", "0", REDTEAM_EVENTS_CONTAINER_PATH,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            with self._lock:
                self._events.append(event)

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()


_EVENT_STREAM: RedteamEventStream | None = None


def read_redteam_events() -> list[dict]:
    if _EVENT_STREAM is not None:
        return _EVENT_STREAM.events()
    host_path = REDTEAM_EVENTS_HOST_PATH
    if host_path.is_file():
        text = host_path.read_text(encoding="utf-8")
    else:
        r = subprocess.run(
            ["docker", "exec", REDTEAM_CONTAINER, "cat", REDTEAM_EVENTS_CONTAINER_PATH],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            return []
        text = r.stdout or ""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def clear_redteam_events() -> None:
    global _EVENT_STREAM
    if _EVENT_STREAM is not None:
        _EVENT_STREAM.stop()
        _EVENT_STREAM = None
    host_path = REDTEAM_EVENTS_HOST_PATH
    host_path.parent.mkdir(parents=True, exist_ok=True)
    host_path.write_text("", encoding="utf-8")
    subprocess.run(
        ["docker", "exec", REDTEAM_CONTAINER, "sh", "-c", f": > {REDTEAM_EVENTS_CONTAINER_PATH}"],
        capture_output=True,
    )


def start_redteam_event_stream() -> None:
    global _EVENT_STREAM
    if _EVENT_STREAM is not None:
        _EVENT_STREAM.stop()
    _EVENT_STREAM = RedteamEventStream()
    _EVENT_STREAM.start()


def stop_redteam_event_stream() -> None:
    global _EVENT_STREAM
    if _EVENT_STREAM is not None:
        _EVENT_STREAM.stop()
        _EVENT_STREAM = None


def start_redteam_agent(round_id: str, duration_sec: float, *, clear_events: bool = True) -> None:
    if clear_events:
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


def try_ips_block(
    block_id: str,
    window_idx: int,
    t0: float,
    ips_state: dict,
    reason: str,
) -> bool:
    """Block red-team container if not already blocked. Returns True if newly blocked."""
    if ips_state["blocked"]:
        return False
    br = block_container(REDTEAM_CONTAINER)
    if br.blocked:
        ips_state["blocked"] = True
        ips_state["blocked_at_sec"] = time.time() - t0
        ips_state["blocked_at_window"] = window_idx
        ips_state["block_reason"] = reason
        print(
            f"  IPS BLOCK ({block_id}) at window {window_idx} "
            f"({ips_state['blocked_at_sec']:.1f}s) [{reason}]: {br.source_ip}"
        )
        return True
    print(f"  WARNING: IPS block failed at window {window_idx}: {br.message}", file=sys.stderr)
    return False


def poll_events_and_block(
    events: list[dict],
    t0: float,
    *,
    block_stages: set[str],
    block_id: str,
    window_idx: int,
    ips_state: dict,
    seen_event_keys: set[tuple],
) -> None:
    """Block on red-team kill-chain stages before post_exploit (theft) completes."""
    for e in events:
        stage = str(e.get("stage", ""))
        if stage not in block_stages or not e.get("success"):
            continue
        key = (e.get("round_id", ""), stage, float(e.get("ts", 0)))
        if key in seen_event_keys:
            continue
        seen_event_keys.add(key)
        elapsed = float(e["ts"]) - t0
        print(f"  red-team event: {stage} success @ {elapsed:.1f}s")
        try_ips_block(block_id, window_idx, t0, ips_state, reason=f"event:{stage}")


def build_systems(
    backend: str,
    log_systems: list[str],
    shaun_ckpt: Path | None = None,
) -> tuple[dict[str, object], str]:
    cfg = BACKEND_CONFIG[backend]
    block_id = cfg["block_id"]
    systems: dict[str, object] = {}
    ckpt = shaun_ckpt.resolve() if shaun_ckpt else None

    if backend in ("gen8", "ary"):
        model = load_ary_ckpt(CKPT_GEN8)
        if isinstance(model, Gen8DecoupledMLPWorldModel):
            systems[block_id] = StreamingGen8WorldModel(base_model=model, detect_thresh=DETECT_THRESHOLD)
        else:
            splits = load_all_splits(SPLITS_5S)
            va_s, va_b, _ = splits["val"]
            _, hidden_thresh = calibrate_thresholds(model, va_s, va_b)
            systems[block_id] = StreamingARYRamxV01(base_model=model, hidden_thresh=hidden_thresh)
    elif backend == "hx":
        bundle = load_hx_bundle()
        systems[block_id] = StreamingHX(bundle, context_skip_steps=10_000)
    elif backend == "hx_c":
        bundle = load_hxc_bundle()
        systems[block_id] = StreamingHXC(bundle, context_skip_steps=10_000)
    elif backend == "sn2rx3":
        bundle = load_shaun_bundle(ckpt_path=ckpt)
        systems[block_id] = StreamingShaunRamxV3(bundle, context_skip_steps=10_000)
        systems.setdefault("sn_base", StreamingShaunBase(bundle))
    else:
        bundle = load_shaun_bundle(ckpt_path=ckpt)
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
    attack_window_sec: float | None = None,
    event_block_stages: list[str] | None = None,
    shaun_ckpt: Path | None = None,
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

    systems, block_id = build_systems(backend, log_systems, shaun_ckpt=shaun_ckpt)
    all_ids = list(systems.keys())
    atk_window_sec = attack_window_sec if attack_window_sec is not None else window_sec
    block_stages = set(event_block_stages or ["recon"])
    if backend in SHAUN_LIKE and block_id in systems and hasattr(systems[block_id], "begin_live_phase"):
        systems[block_id].begin_live_phase()

    trace: list[dict] = []
    state_box: dict[str, np.ndarray | None] = {"last": None}
    ips_state = {
        "blocked": False,
        "blocked_at_sec": None,
        "blocked_at_window": None,
        "block_reason": "",
    }
    t0 = time.time()
    alerted: set[str] = set()
    seen_events: set[tuple] = set()
    warmup_windows = 0

    def do_window(
        true_bin: int,
        phase: str,
        window_idx: int,
        *,
        capture_window_sec: float | None = None,
        use_clock: LabClock | None = None,
        use_capture: TrafficCapture | None = None,
    ) -> None:
        cap_sec = capture_window_sec if capture_window_sec is not None else window_sec
        cap = use_capture or capture
        clk = use_clock or clock

        def poll_during_capture() -> None:
            if phase != "attack" or ips_state["blocked"]:
                return
            poll_events_and_block(
                read_redteam_events(), t0,
                block_stages=block_stages,
                block_id=block_id,
                window_idx=window_idx,
                ips_state=ips_state,
                seen_event_keys=seen_events,
            )

        if phase == "attack":
            poll_during_capture()

        state, t_start, t_end = capture_one_window(
            cap, f"w{window_idx:05d}.pcap", clk, state_box["last"],
            backend=backend, window_sec=cap_sec,
            scale_factor=scale_factor, replicate=replicate,
            poll_fn=poll_during_capture if phase == "attack" else None,
        )
        state_box["last"] = state
        true_mit = mit_stage if true_bin else 0

        poll_during_capture()

        sys_out: dict[str, dict] = {}

        def score_systems() -> None:
            for sid, sys_obj in systems.items():
                out = sys_obj.step(state, true_bin=true_bin, true_mit=true_mit)
                sys_out[sid] = {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in out.items()
                }

        if phase == "attack" and not ips_state["blocked"]:
            _run_with_poll(score_systems, poll_during_capture)
        else:
            score_systems()

        poll_during_capture()

        p_block = float(sys_out[block_id]["p_att"])
        if phase == "attack" and not ips_state["blocked"] and p_block >= DETECT_THRESHOLD:
            try_ips_block(block_id, window_idx, t0, ips_state, reason="ml_score")

        trace.append({
            "window_idx": window_idx,
            "t_start": t_start,
            "t_end": t_end,
            "phase": phase,
            "true_bin": true_bin,
            "true_mit": true_mit,
            "capture_window_sec": cap_sec,
            "ips_blocked": ips_state["blocked"],
            "block_reason": ips_state.get("block_reason", ""),
            "systems": sys_out,
        })
        if phase == "warmup":
            nonlocal warmup_windows
            warmup_windows += 1

    window_idx = 0
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        do_window(true_bin=0, phase="warmup", window_idx=window_idx)
        window_idx += 1
    print(f"  warmup done: {warmup_windows} windows over {time.time() - warmup_start:.1f}s")

    if backend in SHAUN_LIKE and block_id in systems:
        systems[block_id].set_context_skip(warmup_windows)
        print(f"  {backend} context gate set to {warmup_windows} warmup windows")

    attack_clock = LabClock(speed=speed, base_window_sec=atk_window_sec)
    attack_capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps_attack")
    print(f"  attack capture: {atk_window_sec}s windows ({attack_clock.window_sec():.2f}s wall each)")

    clear_redteam_events()
    start_redteam_event_stream()
    start_redteam_agent(round_id, duration_sec, clear_events=False)
    attack_start_wall = time.time()
    print(f"  attack phase: {duration_sec}s wall (zero-day cred theft)")

    def attack_event_watcher() -> None:
        while (
            not ips_state["blocked"]
            and time.time() - attack_start_wall < duration_sec
        ):
            poll_events_and_block(
                read_redteam_events(), t0,
                block_stages=block_stages,
                block_id=block_id,
                window_idx=window_idx,
                ips_state=ips_state,
                seen_event_keys=seen_events,
            )
            time.sleep(0.05)

    threading.Thread(target=attack_event_watcher, daemon=True).start()
    while time.time() - attack_start_wall < duration_sec:
        do_window(
            true_bin=1, phase="attack", window_idx=window_idx,
            capture_window_sec=atk_window_sec,
            use_clock=attack_clock,
            use_capture=attack_capture,
        )
        window_idx += 1
        poll_early_theft(read_redteam_events(), t0, alerted)

    events = read_redteam_events()
    poll_early_theft(events, t0, alerted)
    stop_redteam_event_stream()

    ips_blocked = ips_state["blocked"]
    blocked_at_sec = ips_state["blocked_at_sec"]
    blocked_at_window = ips_state["blocked_at_window"]

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
        "attack_window_sec": atk_window_sec,
        "attack_wall_window_sec": attack_clock.window_sec(),
        "event_block_stages": sorted(block_stages),
        "block_reason": ips_state.get("block_reason", ""),
        "shaun_ckpt": str(shaun_ckpt) if shaun_ckpt else "",
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
    p.add_argument("--out-dir", type=Path, default=IPS_OUT_ROOT / "theft_prevention")
    p.add_argument("--attack-window-sec", type=float, default=DEFAULT_ATTACK_WINDOW_SEC,
                   help="Shorter capture/score windows during attack phase (default 5s)")
    p.add_argument("--event-block-stages", default="recon",
                   help="Comma-separated red-team stages that trigger IPS block (before post_exploit)")
    p.add_argument("--shaun-ckpt", type=Path, default=W5S_CKPT if W5S_CKPT.exists() else None,
                   help="Shaun checkpoint (default w5s if present)")
    args = p.parse_args()

    if args.backend in SHAUN_LIKE and not (ROOT.parent / "PRISM-shaun").exists():
        print("Missing PRISM-shaun worktree for Shaun backend", file=sys.stderr)
        return 1
    if args.backend == "ary" and not CKPT_V01.exists():
        print(f"Missing checkpoint: {CKPT_V01}", file=sys.stderr)
        return 1

    log_systems = [s.strip() for s in args.log_systems.split(",") if s.strip()]
    if args.backend in ("sn2rx", "sn2rx3") and "sn_base" not in log_systems:
        log_systems.append("sn_base")

    stages = [s.strip() for s in args.event_block_stages.split(",") if s.strip()]
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
        attack_window_sec=args.attack_window_sec,
        event_block_stages=stages,
        shaun_ckpt=args.shaun_ckpt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
