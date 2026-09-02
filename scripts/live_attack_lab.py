#!/usr/bin/env python3
"""Live adversarial attack lab orchestrator (plan section 3).

Runs one or more (objective x evasion-version) rounds against the isolated
Docker lab:

  lab_ctl.py reset -> warm-up (benign-only windows, model lookback context)
  -> concurrently: [attacker-bot: recon -> exploit -> post-exploit] and
     [per-window loop: capture -> 242-d state -> step all 6 streaming
      systems -> 100-step forecast -> log]
  -> cooldown windows -> proxy scoring -> round.json

The attacker bot itself runs *inside* the attacker-bot container (it needs
docker-internal DNS to reach `target-server`, which the isolated lab network
doesn't expose to the host) via a blocking `docker exec`
(`src/adversarial/objective_attack_script.py`) on a background thread; the
host-side loop uses that thread's liveness as ground truth for "is this
window during the attack" (recon + exploit + post-exploit all count).
Real per-window traffic capture reuses `TrafficCapture` (`docker exec
tcpdump`) unmodified, same as the rest of the adversarial pipeline.

Run:
  python scripts/live_attack_lab.py --objective defacement --evasion none --speed 10
  python scripts/live_attack_lab.py --objective all --evasion all --speed 10
  python scripts/lab_leaderboard.py   # after 1+ rounds, aggregates round.json -> leaderboard.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.bots.lab_objective_base import new_round_id  # noqa: E402
from src.adversarial.lab_clock import LabClock  # noqa: E402
from src.adversarial.lab_config import ATTACKER_CONTAINER, TARGET_CONTAINER, TARGET_HOST  # noqa: E402
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.aryan.constants import MITRE_STAGES, TACTIC_TO_STAGE  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import TARGET_DIM, pcap_to_states  # noqa: E402
from src.aryan.lab_scoring import compute_round_scores  # noqa: E402
from src.aryan.streaming_variants import build_all_systems, calibrate_thresholds, load_model  # noqa: E402

OUT_ROOT = ROOT / "results" / "ram_improve" / "live_lab"
OBJECTIVES_YAML = ROOT / "configs" / "lab_objectives.yaml"
LAB_CTL = ROOT / "scripts" / "lab_ctl.py"

# num_flows, num_unique_dst_ports, port_entropy -- see src/aryan/feature_schema242.py
DEFAULT_TFM_FEATURES_BASE = (0, 3, 4)


class NaiveForecaster:
    """Persistence fallback for when the real TimesFM-3 package isn't
    installed in this environment (it's a large external model, not
    vendored here -- see scripts/timesfm_ary_lab_eval.py for the real
    `timesfm3.TimesFM3Evaluator` usage). Forecasts every future step as the
    last observed context value. `StreamingTimesFMHybrid` only needs a
    `.predict_batch(contexts, horizon, **kwargs) -> [obj with .forecast]`
    interface, so swapping in the real forecaster requires no other changes
    to this script."""

    class _Out:
        def __init__(self, forecast: np.ndarray):
            self.forecast = forecast

    def predict_batch(self, contexts, horizon, **kwargs):
        outs = []
        for ctx in contexts:
            last = float(ctx[-1]) if len(ctx) else 0.0
            outs.append(self._Out(np.full(horizon, last, dtype=np.float32)))
        return outs


def load_forecaster(device: str = "cuda"):
    try:
        from timesfm3 import ModelConfig, TimesFM3Evaluator
    except ImportError:
        print(
            "WARNING: timesfm3 not installed in this environment -- using a NaiveForecaster "
            "(persistence) fallback for all TimesFM+ARY hybrid dynamics/forecast output. "
            "Install the real TimesFM-3 package for results that reflect the actual model.",
            file=sys.stderr,
        )
        return NaiveForecaster()
    try:
        return TimesFM3Evaluator(
            ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=16, device=device)
        )
    except Exception as exc:  # pragma: no cover - depends on external hardware/model availability
        print(f"WARNING: failed to load TimesFM-3 ({exc}) -- using NaiveForecaster fallback.", file=sys.stderr)
        return NaiveForecaster()


def load_objectives() -> list[dict]:
    cfg = yaml.safe_load(OBJECTIVES_YAML.read_text(encoding="utf-8"))
    return cfg["objectives"]


def pick_tfm_features(train_states: np.ndarray, n_extra: int = 5) -> list[int]:
    order = np.argsort(train_states.var(axis=0))[::-1]
    extra = [int(i) for i in order[:n_extra] if int(i) not in DEFAULT_TFM_FEATURES_BASE]
    return sorted(set(list(DEFAULT_TFM_FEATURES_BASE) + extra))


def reset_lab() -> None:
    subprocess.run([sys.executable, str(LAB_CTL), "reset"], cwd=str(ROOT), check=True)


def capture_one_window(
    capture: TrafficCapture, filename: str, clock: LabClock, last_state: "np.ndarray | None"
) -> tuple[np.ndarray, float, float]:
    t_start = time.time()
    capture.start_capture(filename)
    clock.sleep(clock.window_sec())
    pcap_path = capture.stop_capture()
    t_end = time.time()

    state = None
    if pcap_path is not None and pcap_path.exists():
        try:
            states, _, _ = pcap_to_states(pcap_path)
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
        # No packets captured this window -- repeat the last known state (a
        # reasonable "nothing changed" proxy) instead of an artificial
        # all-zero vector that could itself look like a real anomaly.
        state = last_state.copy() if last_state is not None else np.zeros(TARGET_DIM, dtype=np.float32)
    return state.astype(np.float32), t_start, t_end


def run_bot_thread(
    objective: dict, evasion: str, speed: float, round_id: str, target_ip: str,
    event_log_container_path: str, result_holder: dict,
) -> None:
    recon_arg = ",".join(objective.get("recon_bots", []))
    cmd = [
        "docker", "exec", ATTACKER_CONTAINER,
        "python3", "-m", "src.adversarial.objective_attack_script",
        objective["class_id"], evasion, round_id,
        "--speed", str(speed), "--recon", recon_arg,
        "--target", target_ip, "--event-log", event_log_container_path,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    result_holder["returncode"] = r.returncode
    result_holder["stdout"] = r.stdout
    result_holder["stderr"] = r.stderr
    result_holder["elapsed_sec"] = time.time() - t0


def fetch_event_log(event_log_container_path: str, dest: Path) -> list[dict]:
    subprocess.run(
        ["docker", "cp", f"{ATTACKER_CONTAINER}:{event_log_container_path}", str(dest)],
        capture_output=True,
    )
    subprocess.run(
        ["docker", "exec", ATTACKER_CONTAINER, "rm", "-f", event_log_container_path],
        capture_output=True,
    )
    events = []
    if dest.exists():
        for line in dest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def run_round(
    objective: dict,
    evasion: str,
    *,
    speed: float,
    warmup_windows: int,
    cooldown_windows: int,
    max_windows: int,
    horizon: int,
    forecast_features: list[int],
    forecast_every: int,
    systems_factory,
    out_dir: Path,
) -> dict:
    print(f"\n=== round: objective={objective['id']} evasion={evasion} ===")
    reset_lab()

    round_id = new_round_id()
    clock = LabClock(speed=speed)
    round_dir = out_dir / objective["id"] / evasion
    round_dir.mkdir(parents=True, exist_ok=True)
    capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps")

    systems = systems_factory()
    mit_stage = MITRE_STAGES.get(TACTIC_TO_STAGE.get(objective["mitre_tactic"], "Benign"), 0)

    trace: list[dict] = []
    state_box = {"last": None}
    window_box = {"idx": 0}

    def do_window(true_bin: int, phase: str) -> None:
        state, t_start, t_end = capture_one_window(
            capture, f"w{window_box['idx']:05d}.pcap", clock, state_box["last"],
        )
        state_box["last"] = state
        true_mit = mit_stage if true_bin else 0
        systems_out = {}
        for sid, sys_obj in systems.items():
            out = sys_obj.step(state, true_bin=true_bin, true_mit=true_mit)
            entry = {"p_att": out["p_att"], "dynamics_mse": getattr(sys_obj, "last_dynamics_mse", None)}
            if window_box["idx"] % max(1, forecast_every) == 0:
                try:
                    fc = sys_obj.forecast(horizon=horizon)
                    entry["forecast"] = {int(f): fc[:, f].tolist() for f in forecast_features if f < fc.shape[1]}
                except Exception as exc:
                    entry["forecast_error"] = str(exc)
            systems_out[sid] = entry
        trace.append({
            "window_idx": window_box["idx"],
            "t_start": t_start,
            "t_end": t_end,
            "phase": phase,
            "true_bin": true_bin,
            "true_mit": true_mit,
            "state": state.tolist(),
            "systems": systems_out,
        })
        window_box["idx"] += 1

    print(f"  warm-up: {warmup_windows} windows @ {clock.window_sec():.1f}s/window")
    for _ in range(warmup_windows):
        do_window(true_bin=0, phase="warmup")

    event_log_path = f"/tmp/lab_events_{round_id}.jsonl"
    result_holder: dict = {}
    bot_thread = threading.Thread(
        target=run_bot_thread,
        args=(objective, evasion, speed, round_id, TARGET_HOST, event_log_path, result_holder),
        daemon=True,
    )
    print("  starting attacker bot thread (recon -> exploit -> post-exploit)...")
    bot_thread.start()

    cooldown_left = None
    while window_box["idx"] < max_windows:
        alive = bot_thread.is_alive()
        do_window(true_bin=1 if alive else 0, phase="attack" if alive else "cooldown")
        if not alive:
            cooldown_left = cooldown_windows - 1 if cooldown_left is None else cooldown_left - 1
            if cooldown_left <= 0:
                break

    bot_thread.join(timeout=30)
    if bot_thread.is_alive():
        print("  WARNING: attacker bot thread still running past round end -- force-killing.", file=sys.stderr)
        subprocess.run(
            ["docker", "exec", ATTACKER_CONTAINER, "pkill", "-f", "objective_attack_script"],
            capture_output=True,
        )
        bot_thread.join(timeout=10)

    events = fetch_event_log(event_log_path, round_dir / "events.jsonl")
    scores = compute_round_scores(trace, events, list(systems.keys()))

    payload = {
        "objective": objective["id"],
        "class_id": objective["class_id"],
        "mitre_tactic": objective["mitre_tactic"],
        "evasion": evasion,
        "round_id": round_id,
        "speed": speed,
        "n_windows": len(trace),
        "window_sec": clock.window_sec(),
        "forecast_horizon": horizon,
        "forecast_features": forecast_features,
        "system_ids": list(systems.keys()),
        "bot_result": {
            "returncode": result_holder.get("returncode"),
            "elapsed_sec": result_holder.get("elapsed_sec"),
        },
        "events": events,
        "trace": trace,
        "scores": scores,
    }
    out_path = round_dir / "round.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  -> {out_path}  ({len(trace)} windows)")
    for sid, sc in scores.items():
        print(
            f"    {sid:<20} F1={sc['binary_f1']:.3f}  TTD={sc['ttd_windows']}  "
            f"lost={sc['lost_to_attacker']}  benign_harm={sc['benign_harm_count']}"
        )
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objective", default="all", help="defacement|key_theft|cred_theft|all")
    p.add_argument("--evasion", default="all", help="none|random_timing|slow_timing|all")
    p.add_argument("--speed", type=float, default=10.0, help="Lab pacing multiplier (1.0 = realistic real-time)")
    p.add_argument("--warmup-windows", type=int, default=5)
    p.add_argument("--cooldown-windows", type=int, default=5)
    p.add_argument("--max-windows", type=int, default=120, help="Safety cap on windows per round")
    p.add_argument("--horizon", type=int, default=100, help="Forecast steps ahead, refreshed every window")
    p.add_argument("--gate-threshold", type=float, default=0.15, help="P(attack) gate for fmary_gated_ram")
    p.add_argument("--forecast-every", type=int, default=1, help="Store a forecast every N windows (1=every window)")
    p.add_argument("--tfm-features", default=None, help="Comma-separated feature indices; default: auto-picked")
    p.add_argument("--out-dir", default=str(OUT_ROOT))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    objectives = load_objectives()
    if args.objective != "all":
        objectives = [o for o in objectives if o["id"] == args.objective]
        if not objectives:
            print(f"Unknown objective: {args.objective}", file=sys.stderr)
            return 1

    print("Loading ARY.01 checkpoint...")
    base_model = load_model()
    print("Loading TimesFM-3 (or fallback)...")
    forecaster = load_forecaster(device=args.device)

    print("Calibrating memory thresholds on aryan val split...")
    splits = load_all_splits()
    tr_s, _, _ = splits["train"]
    va_s, va_b, _ = splits["val"]
    raw_thresh, hidden_thresh = calibrate_thresholds(base_model, va_s, va_b)
    print(f"  raw={raw_thresh:.2f} hidden={hidden_thresh:.2f}")

    if args.tfm_features:
        tfm_features = [int(x) for x in args.tfm_features.split(",")]
    else:
        tfm_features = pick_tfm_features(tr_s)
    print(f"  TimesFM curated features: {tfm_features}")

    def systems_factory():
        return build_all_systems(
            base_model, forecaster, raw_thresh, hidden_thresh,
            tfm_features=tfm_features, gate_threshold=args.gate_threshold,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = []
    for objective in objectives:
        versions = objective["evasion_versions"] if args.evasion == "all" else [args.evasion]
        for evasion in versions:
            run_round(
                objective, evasion,
                speed=args.speed, warmup_windows=args.warmup_windows,
                cooldown_windows=args.cooldown_windows, max_windows=args.max_windows,
                horizon=args.horizon, forecast_features=tfm_features, forecast_every=args.forecast_every,
                systems_factory=systems_factory, out_dir=out_dir,
            )
            all_rounds.append(str(out_dir / objective["id"] / evasion / "round.json"))

    (out_dir / "run_manifest.json").write_text(json.dumps({"rounds": all_rounds}, indent=2))
    print(f"\nDone. {len(all_rounds)} round(s). Build the leaderboard with:")
    print("  python scripts/lab_leaderboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
