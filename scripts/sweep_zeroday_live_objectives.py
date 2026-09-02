#!/usr/bin/env python3
"""Zero-day sweep: sn_base vs SN2RXv2 vs ARY-5s base/RAMX on 3 lab HTTP objectives.

Runs one short scaled live round per objective (cred_theft, key_theft,
defacement) and scores all systems on the *same* traffic timeline.

Also replays stored ``results/ram_improve/live_lab/*/none/round.json`` traces
for ARY-5s (242-d @ ~15s) as a historical baseline reference.

Usage:
  python scripts/sweep_zeroday_live_objectives.py
  python scripts/sweep_zeroday_live_objectives.py --offline-only
  python scripts/sweep_zeroday_live_objectives.py --speed 25 --warmup-sec 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_ary5s import load_ckpt  # noqa: E402
from scripts.lab_traffic_scale import DEFAULT_REPLICATE, DEFAULT_SCALE_FACTOR, scale_prism_rows  # noqa: E402
from scripts.plot_lab_ary_comparison import score_round  # noqa: E402
from src.adversarial.bots.lab_objective_base import new_round_id  # noqa: E402
from src.adversarial.lab_config import ATTACKER_CONTAINER, TARGET_CONTAINER  # noqa: E402
from src.adversarial.lab_clock import LabClock  # noqa: E402
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.aryan.constants import MITRE_STAGES, TACTIC_TO_STAGE  # noqa: E402
from src.aryan.dataset import load_all_splits  # noqa: E402
from src.aryan.ingest import TARGET_DIM, _states_from_flow_df  # noqa: E402
from src.aryan.lab_scoring import compute_round_scores  # noqa: E402
from src.aryan.streaming_variants import StreamingARY, StreamingARYRamxV01, calibrate_thresholds  # noqa: E402
from src.pipeline.extract import pcap_to_rows  # noqa: E402
from src.shaun.streaming import (  # noqa: E402
    StreamingShaunBase,
    StreamingShaunRamxV2,
    load_shaun_bundle,
    pcap_to_shaun_state,
)

LAB_CTL = ROOT / "scripts" / "lab_ctl.py"
SPLITS_5S = ROOT / "data" / "aryan_splits_5s"
CKPT_5 = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
OUT_DIR = ROOT / "results" / "zeroday_live_sweep"
STORED_ROUNDS = {
    "cred_theft": ROOT / "results/ram_improve/live_lab/cred_theft/none/round.json",
    "key_theft": ROOT / "results/ram_improve/live_lab/key_theft/none/round.json",
    "defacement": ROOT / "results/ram_improve/live_lab/defacement/none/round.json",
}
OBJECTIVE_IDS = ("cred_theft", "key_theft", "defacement")
WINDOW_SEC = 15.0
SYSTEMS = ("ary5_base", "ary5_ramx", "sn_base", "sn2rx")


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


def load_objectives() -> list[dict]:
    cfg = yaml.safe_load((ROOT / "configs" / "lab_objectives.yaml").read_text(encoding="utf-8"))
    by_id = {o["id"]: o for o in cfg["objectives"]}
    return [by_id[i] for i in OBJECTIVE_IDS]


def reset_lab(scale_benign: bool = True) -> None:
    import os

    env = os.environ.copy()
    if scale_benign:
        env.setdefault("BENIGN_WORKERS", "10")
        env.setdefault("BENIGN_INTERVAL", "0.5")
    subprocess.run([sys.executable, str(LAB_CTL), "reset"], cwd=str(ROOT), check=True, env=env)
    if scale_benign:
        subprocess.run(
            [
                "docker", "compose", "-f", str(ROOT / "docker" / "docker-compose.yml"), "-p", "prism",
                "up", "-d", "--force-recreate", "benign-client",
            ],
            cwd=str(ROOT), check=True, env=env,
        )


def ary_state_from_pcap(
    pcap: Path,
    *,
    scale_factor: float,
    replicate: int,
    last_state: np.ndarray | None,
) -> np.ndarray:
    rows = pcap_to_rows(pcap)
    if scale_factor != 1.0 or replicate != 1:
        rows = scale_prism_rows(rows, factor=scale_factor, replicate=replicate)
    if not rows:
        if last_state is not None:
            return last_state.copy()
        return np.zeros(TARGET_DIM, dtype=np.float32)
    states, _, _ = _states_from_flow_df(pd.DataFrame(rows), window_sec=WINDOW_SEC)
    if len(states):
        return states[0].astype(np.float32)
    if last_state is not None:
        return last_state.copy()
    return np.zeros(TARGET_DIM, dtype=np.float32)


def fetch_event_log(container_path: str, host_path: Path) -> list[dict]:
    r = subprocess.run(
        ["docker", "cp", f"{ATTACKER_CONTAINER}:{container_path}", str(host_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        return []
    events = []
    for line in host_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def run_bot_thread(objective: dict, evasion: str, speed: float, round_id: str, event_log: str, holder: dict) -> None:
    recon_arg = ",".join(objective.get("recon_bots", []))
    cmd = [
        "docker", "exec", ATTACKER_CONTAINER,
        "python3", "-m", "src.adversarial.objective_attack_script",
        objective["class_id"], evasion, round_id,
        "--speed", str(speed), "--recon", recon_arg,
        "--target", "target-server", "--event-log", event_log,
    ]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    holder["returncode"] = r.returncode
    holder["stdout"] = r.stdout
    holder["stderr"] = r.stderr
    holder["elapsed_sec"] = time.time() - t0


def capture_window(
    capture: TrafficCapture,
    filename: str,
    clock: LabClock,
    ary_last: np.ndarray | None,
    shaun_last: np.ndarray | None,
    scale_factor: float,
    replicate: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    t_start = time.time()
    capture.start_capture(filename)
    clock.sleep(clock.window_sec())
    pcap_path = capture.stop_capture()
    t_end = time.time()
    ary_state = shaun_state = None
    if pcap_path is not None and pcap_path.exists():
        try:
            ary_state = ary_state_from_pcap(
                pcap_path, scale_factor=scale_factor, replicate=replicate, last_state=ary_last,
            )
            shaun_state = pcap_to_shaun_state(
                pcap_path, last_state=shaun_last, scale_factor=scale_factor, replicate=replicate,
            )
        except Exception as exc:
            print(f"WARNING: ingest failed on {pcap_path}: {exc}", file=sys.stderr)
        try:
            pcap_path.unlink()
        except OSError:
            pass
    if ary_state is None:
        ary_state = ary_last.copy() if ary_last is not None else np.zeros(TARGET_DIM, dtype=np.float32)
    if shaun_state is None:
        shaun_state = shaun_last.copy() if shaun_last is not None else np.zeros(292, dtype=np.float32)
    return ary_state.astype(np.float32), shaun_state.astype(np.float32), t_start, t_end


def build_scorers(model_5, hidden_5: float, shaun_bundle: dict) -> dict[str, object]:
    return {
        "ary5_base": StreamingARY("ary_base", base_model=model_5, hidden_thresh=hidden_5),
        "ary5_ramx": StreamingARYRamxV01(base_model=model_5, hidden_thresh=hidden_5),
        "sn_base": StreamingShaunBase(shaun_bundle),
        "sn2rx": StreamingShaunRamxV2(shaun_bundle, context_skip_steps=10_000),
    }


def run_live_objective_round(
    objective: dict,
    *,
    model_5,
    hidden_5: float,
    shaun_bundle: dict,
    speed: float,
    warmup_sec: float,
    scale_factor: float,
    replicate: int,
    out_dir: Path,
    max_windows: int = 40,
) -> dict:
    reset_lab(scale_benign=True)
    round_id = new_round_id()
    round_dir = out_dir / "live_rescore" / objective["id"]
    round_dir.mkdir(parents=True, exist_ok=True)

    clock = LabClock(speed=speed, base_window_sec=WINDOW_SEC)
    capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=round_dir / "pcaps")
    scorers = build_scorers(model_5, hidden_5, shaun_bundle)
    mit_stage = MITRE_STAGES.get(TACTIC_TO_STAGE.get(objective["mitre_tactic"], "Benign"), 0)

    trace: list[dict] = []
    ary_last: np.ndarray | None = None
    shaun_last: np.ndarray | None = None
    warmup_windows = 0

    def do_window(true_bin: int, phase: str, window_idx: int) -> None:
        nonlocal ary_last, shaun_last, warmup_windows
        ary_state, shaun_state, t_start, t_end = capture_window(
            capture, f"w{window_idx:05d}.pcap", clock, ary_last, shaun_last, scale_factor, replicate,
        )
        ary_last, shaun_last = ary_state, shaun_state
        true_mit = mit_stage if true_bin else 0
        sys_out: dict[str, dict] = {}
        for sid, sys_obj in scorers.items():
            state = shaun_state if sid.startswith("sn") else ary_state
            out = sys_obj.step(state, true_bin=true_bin, true_mit=true_mit)
            sys_out[sid] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in out.items()}
        trace.append({
            "window_idx": window_idx,
            "t_start": t_start,
            "t_end": t_end,
            "phase": phase,
            "true_bin": true_bin,
            "true_mit": true_mit,
            "systems": sys_out,
        })
        if phase == "warmup":
            warmup_windows += 1

    window_idx = 0
    warmup_start = time.time()
    while time.time() - warmup_start < warmup_sec:
        do_window(0, "warmup", window_idx)
        window_idx += 1

    scorers["sn2rx"].set_context_skip(warmup_windows)
    event_log = f"/tmp/lab_events_{round_id}.jsonl"
    holder: dict = {}
    bot_thread = threading.Thread(
        target=run_bot_thread,
        args=(objective, "none", speed, round_id, event_log, holder),
        daemon=True,
    )
    bot_thread.start()

    cooldown_left = None
    while window_idx < max_windows:
        alive = bot_thread.is_alive()
        do_window(1 if alive else 0, "attack" if alive else "cooldown", window_idx)
        window_idx += 1
        if not alive:
            cooldown_left = 2 if cooldown_left is None else cooldown_left - 1
            if cooldown_left <= 0:
                break

    bot_thread.join(timeout=20)
    events = fetch_event_log(event_log, round_dir / "events.jsonl")
    scores = compute_round_scores(trace, events, list(SYSTEMS))

    payload = {
        "source": "live_rescore",
        "objective": objective["id"],
        "class_id": objective["class_id"],
        "evasion": "none",
        "round_id": round_id,
        "speed": speed,
        "warmup_sec": warmup_sec,
        "warmup_windows": warmup_windows,
        "scale_factor": scale_factor,
        "replicate": replicate,
        "window_sec": WINDOW_SEC,
        "n_windows": len(trace),
        "events": events,
        "trace": trace,
        "scores": scores,
    }
    (round_dir / "round.json").write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return payload


def replay_stored_ary_round(round_path: Path, model_5, hidden_5: float) -> dict | None:
    if not round_path.exists():
        return None
    data = json.loads(round_path.read_text(encoding="utf-8"))
    states = [np.array(t["state"], dtype=np.float32) for t in data["trace"]]
    bins = [int(t["true_bin"]) for t in data["trace"]]
    mits = [int(t["true_mit"]) for t in data["trace"]]
    attack_start = next((i for i, b in enumerate(bins) if b == 1), len(bins))

    def replay(factory) -> list[float]:
        sys_obj = factory()
        out = []
        for s, tb, tm in zip(states, bins, mits):
            out.append(float(sys_obj.step(s, true_bin=tb, true_mit=tm)["p_att"]))
        return out

    series = {
        "ary5_base": replay(lambda: StreamingARY("ary_base", base_model=model_5, hidden_thresh=hidden_5)),
        "ary5_ramx": replay(lambda: StreamingARYRamxV01(base_model=model_5, hidden_thresh=hidden_5)),
    }
    result = {
        "source": "stored_live_lab",
        "objective": data["objective"],
        "class_id": data["class_id"],
        "path": str(round_path.relative_to(ROOT)),
        "n_windows": len(bins),
        "attack_start": attack_start,
        "note": "242-d stored trace @ ~15s; Shaun not comparable on stored captures.",
    }
    for sid, p in series.items():
        result[sid] = {
            **score_round(p, bins, attack_start),
            "attack_max_p": float(max(p[attack_start:], default=0.0)),
            "warmup_mean_p": float(np.mean(p[:attack_start])) if attack_start else 0.0,
        }
    return result


def attack_max_p(trace: list[dict], sid: str) -> float:
    vals = [
        float(w["systems"].get(sid, {}).get("p_att", 0))
        for w in trace
        if w.get("phase") == "attack"
    ]
    return max(vals) if vals else 0.0


def write_report(stored: list[dict], live: list[dict], out_dir: Path) -> Path:
    lines = [
        "# Zero-Day Live Objective Sweep",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "HTTP lab objectives (never in CIC-IDS-2018 training):",
        "- `T1555_sqli_cred_theft`",
        "- `T1552_key_theft`",
        "- `T1491_web_defacement`",
        "",
        "## Live rescore (all 4 systems, same scaled traffic)",
        "",
        "Traffic: PCAP 10× + 5× replicate, benign 10 workers @ 0.5s, 15s windows.",
        "",
        "| Objective | System | F1 | Det | Max P (attack) | TTD | Warmup harm |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in live:
        obj = row["objective"]
        trace = row["trace"]
        for sid in SYSTEMS:
            sc = row["scores"][sid]
            ttd = sc["ttd_sec"]
            lines.append(
                f"| {obj} | {sid} | {sc['binary_f1']:.3f} | "
                f"{'yes' if sc['detected'] else 'no'} | {attack_max_p(trace, sid):.3f} | "
                f"{ttd if ttd is not None else 'n/a'} | {sc['benign_harm_count']} |"
            )
    lines.extend([
        "",
        "## Stored live_lab replay (ARY-5s only, historical captures)",
        "",
        "| Objective | System | F1 | Det | Max P | Warmup mean P |",
        "|---|---|---:|---|---:|---:|",
    ])
    for row in stored:
        if not row:
            continue
        for sid in ("ary5_base", "ary5_ramx"):
            sc = row[sid]
            lines.append(
                f"| {row['objective']} | {sid} | {sc['f1']:.3f} | "
                f"{'yes' if sc['detected'] else 'no'} | {sc['attack_max_p']:.3f} | {sc['warmup_mean_p']:.3f} |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Stored `live_lab` rounds embed 242-d ARY states only — Shaun requires fair PCAP ingest (live rescore).",
        "- Zero-day = lab objective class IDs not in CIC training set.",
        "",
    ])
    path = out_dir / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline-only", action="store_true", help="Skip live Docker rescore (ARY stored only)")
    p.add_argument("--speed", type=float, default=25.0)
    p.add_argument("--warmup-sec", type=float, default=10.0)
    p.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    p.add_argument("--replicate", type=int, default=DEFAULT_REPLICATE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if not CKPT_5.exists():
        print(f"Missing {CKPT_5}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading ARY-5s + Shaun bundles...")
    model_5 = load_ckpt(CKPT_5)
    va_s, va_b, _ = load_all_splits(SPLITS_5S)["val"]
    _, hidden_5 = calibrate_thresholds(model_5, va_s, va_b)
    shaun_bundle = None if args.offline_only else load_shaun_bundle()

    stored_rows = [replay_stored_ary_round(STORED_ROUNDS[oid], model_5, hidden_5) for oid in OBJECTIVE_IDS]
    live_rows: list[dict] = []

    if not args.offline_only:
        if not (ROOT.parent / "PRISM-shaun").exists():
            print("PRISM-shaun missing; use --offline-only for ARY-only", file=sys.stderr)
            return 1
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", TARGET_CONTAINER],
            capture_output=True, text=True,
        )
        if r.returncode != 0 or (r.stdout or "").strip().lower() != "true":
            print("Docker lab not running — start with: python scripts/lab_ctl.py up", file=sys.stderr)
            return 1
        for objective in load_objectives():
            print(f"\n=== Live rescore: {objective['id']} ({objective['class_id']}) ===")
            live_rows.append(
                run_live_objective_round(
                    objective,
                    model_5=model_5,
                    hidden_5=hidden_5,
                    shaun_bundle=shaun_bundle,
                    speed=args.speed,
                    warmup_sec=args.warmup_sec,
                    scale_factor=args.scale_factor,
                    replicate=args.replicate,
                    out_dir=args.out_dir,
                )
            )

    summary = {"stored_ary": stored_rows, "live_rescore": live_rows}
    (args.out_dir / "sweep.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    report = write_report(stored_rows, live_rows, args.out_dir)
    print(f"\nReport -> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
