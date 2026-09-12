"""Rich mock payloads for UI/UX demo mode — no Docker, models, or scorer required."""

from __future__ import annotations

import math
import time
from typing import Any

_START = time.time()
_LOG_SEQ = 1
_LOG_LINES: list[dict[str, Any]] = [
    {"ts": 1, "kind": "phase", "text": "UI demo mode — stub BFF (no real lab/scorer)"},
    {"ts": 2, "kind": "info", "text": "LIVE — scoring simulated; IPS arms for enum/spray/loot only."},
    {"ts": 3, "kind": "scorer", "text": "w001 warmup   hx=0.012  cone=0.18  v3=0.021  blocked=False"},
    {"ts": 4, "kind": "scorer", "text": "w002 warmup   hx=0.015  cone=0.22  v3=0.019  blocked=False"},
    {"ts": 5, "kind": "phase", "text": "LIVE — scoring on; benign Harborline traffic baseline."},
]
_POLICY = "ips"
_LAB_CONFIG: dict[str, Any] = {
    "horizonSec": 60.0,
    "attackDelaySec": 45.0,
    "attackDelayMinSec": 30,
    "attackDelayMaxSec": 60,
}
_SCHEDULED_ATTACKS: list[dict[str, Any]] = []


def _series(n: int, base: float, amp: float, phase: float = 0, ramp_at: float | None = None) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for i in range(n):
        y = base + math.sin((i + phase) / 6) * amp
        if ramp_at is not None and i > ramp_at:
            y += (i - ramp_at) * 0.012
        out.append({"t": float(i), "y": round(max(0.0, min(0.99, y)), 4)})
    return out


def _append_log(text: str, kind: str = "info") -> None:
    global _LOG_SEQ
    _LOG_SEQ += 1
    _LOG_LINES.append({"ts": _LOG_SEQ, "kind": kind, "text": text})


def dashboard() -> dict[str, Any]:
    elapsed = time.time() - _START
    return {
        "containers": [
            {"name": "harborline-site", "status": "running", "image": "prism/harborline:latest"},
            {"name": "attacker-bot", "status": "running", "image": "prism/attacker:latest"},
            {"name": "suricata", "status": "running", "image": "jasonish/suricata:latest"},
            {"name": "benign-client", "status": "running", "image": "prism/benign:latest"},
            {"name": "redteam", "status": "running", "image": "prism/redteam:latest"},
        ],
        "training": {
            "active": elapsed % 120 < 8,
            "job": "hx_c_lab_adapt",
            "progress": round((elapsed % 8) / 8, 2) if elapsed % 120 < 8 else None,
            "message": "Scorer live — HX-C lab-adapt in progress" if elapsed % 120 < 8 else "Scorer live — window streaming",
        },
        "datasets": [
            {"name": "lab_events buffer", "path": "data/lab_events/forecast_adapt_buffer.npz", "samples": 2840, "sizeMb": 14.2},
            {"name": "adversarial missed", "path": "data/raw/adversarial/missed/", "samples": 37, "sizeMb": 2.1},
            {"name": "CIC-IDS val (5s)", "path": "data/processed/aryan_cic_5s/", "samples": 42000, "sizeMb": 890},
            {"name": "Harborline PCAPs", "path": "data/lab_events/demo_forecast_pcaps/", "samples": 512, "sizeMb": 48.6},
        ],
        "device": {"cuda": True, "deviceName": "NVIDIA GeForce RTX 4070 SUPER", "vramGb": 12},
        "networks": [
            {"name": "lab_front", "cidr": "172.28.0.0/24", "protected": True},
            {"name": "lab_back", "cidr": "172.29.0.0/24", "protected": True},
            {"name": "internet_sim", "cidr": "10.0.0.0/24", "protected": False},
        ],
        "models": [
            {"id": "shaun_v3", "name": "Shaun v3", "version": "w5s", "loaded": True},
            {"id": "hx_c", "name": "HX-C", "version": "lab", "loaded": True},
            {"id": "ary_5s", "name": "ARY 5s + RAMX", "version": "ramx-v2", "loaded": True},
            {"id": "xmt_01", "name": "XMT.01", "version": "v1", "loaded": False},
        ],
        "metricsTimeseries": [
            {
                "t": i,
                "flowRate": 140 + math.sin(i / 7) * 35 + (20 if i > 25 else 0),
                "pAttack": round(min(0.92, max(0.02, 0.06 + math.sin(i / 5) * 0.04 + (i - 28) * 0.025 if i > 28 else 0)), 3),
            }
            for i in range(60)
        ],
        "scorerRunning": True,
        "forecastPhase": "live",
    }


def models_registry() -> list[dict[str, Any]]:
    return [
        {
            "id": "shaun_v3",
            "name": "Shaun v3",
            "version": "w5s",
            "checkpointPath": "PRISM-shaun/weights/w5s/world_model.pt",
            "sizeMb": 48.2,
            "parameters": 1_240_000,
            "classes": ["benign", "T1046_service_scan", "T1110_ssh_bruteforce", "T1190_web_exploit_probe"],
            "featureDim": 292,
            "seqLen": 12,
            "lastTrained": "2026-08-15",
            "tags": ["forecast", "ips", "lab"],
            "exists": True,
        },
        {
            "id": "hx_c",
            "name": "HX-C",
            "version": "lab",
            "checkpointPath": "Automode/train/checkpoints/hx_c_w5s_lab.pt",
            "sizeMb": 22.6,
            "parameters": 580_000,
            "classes": ["benign", "recon", "enum", "spray", "loot"],
            "featureDim": 128,
            "seqLen": 8,
            "lastTrained": "2026-09-06",
            "tags": ["forecast", "kill-chain", "lab-adapt"],
            "exists": True,
        },
        {
            "id": "aryan_gen8",
            "name": "Aryan Gen8",
            "version": "universal-5s",
            "checkpointPath": "PRISM-aryan/weights/universal_gen8/world_model_best.pt",
            "sizeMb": 63.8,
            "parameters": 3_800_000,
            "classes": ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "C2", "Exfiltration", "Impact"],
            "featureDim": 242,
            "seqLen": 20,
            "lastTrained": "2026-09-08",
            "tags": ["forecast", "ips", "gen8"],
            "exists": True,
        },
        {
            "id": "aryan_gen8_ramx",
            "name": "Aryan Gen8 + RAMX",
            "version": "universal-5s",
            "checkpointPath": "PRISM-aryan/weights/universal_gen8/world_model_best.pt",
            "sizeMb": 63.8,
            "parameters": 3_800_000,
            "classes": ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "C2", "Exfiltration", "Impact"],
            "featureDim": 242,
            "seqLen": 20,
            "lastTrained": "2026-09-08",
            "tags": ["forecast", "ips", "ramx", "gen8"],
            "exists": True,
        },
        {
            "id": "aryan_gen8_rxi",
            "name": "Aryan Gen8 + RXI",
            "version": "immune-v2",
            "checkpointPath": "PRISM-aryan/weights/universal_gen8/world_model_best.pt",
            "sizeMb": 63.8,
            "parameters": 3_800_000,
            "classes": ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "C2", "Exfiltration", "Impact"],
            "featureDim": 242,
            "seqLen": 20,
            "lastTrained": "2026-09-10",
            "tags": ["forecast", "ips", "ramx", "gen8", "immune", "rxi"],
            "exists": True,
        },
        {
            "id": "ary_5s",
            "name": "ARY 5s + RAMX",
            "version": "ramx-v2",
            "checkpointPath": "models/checkpoints/ary_5s_ramx.pt",
            "sizeMb": 31.4,
            "parameters": 890_000,
            "classes": [f"T{1000 + i}" for i in range(12)],
            "featureDim": 242,
            "seqLen": 10,
            "lastTrained": "2026-07-20",
            "tags": ["multiclass", "cic-ids", "ramx"],
            "exists": True,
        },
        {
            "id": "xmt_01",
            "name": "XMT.01",
            "version": "v1",
            "checkpointPath": "models/checkpoints/xmt_world_model_best.pt",
            "sizeMb": 28.1,
            "parameters": 720_000,
            "classes": ["benign", "recon", "enum", "spray", "loot", "exfil"],
            "featureDim": 128,
            "seqLen": 10,
            "lastTrained": "2026-06-12",
            "tags": ["forecast", "lab-trained"],
            "exists": False,
        },
    ]


def scripts() -> list[dict[str, str]]:
    return [
        {"id": "killchain-recon", "label": "Kill-chain: recon", "description": "Port scan phase"},
        {"id": "killchain-enum", "label": "Kill-chain: enum", "description": "Directory bust"},
        {"id": "killchain-spray", "label": "Kill-chain: spray", "description": "Credential spray"},
        {"id": "killchain-loot", "label": "Kill-chain: loot", "description": "Payroll download"},
        {"id": "killchain-all", "label": "Kill-chain: all", "description": "Full chain"},
        {"id": "auto-attack", "label": "Auto-attack", "description": "Simulated kill-chain after LIVE"},
        {"id": "lab-up", "label": "Lab up", "description": "Stub — no Docker"},
        {"id": "lab-down", "label": "Lab down", "description": "Stub — no Docker"},
        {"id": "bench-fair-ids", "label": "Fair IDS bench", "description": "Stub offline bench"},
        {"id": "forecast-record", "label": "Record session", "description": "Stub WebM + manifest"},
        {"id": "scorer-start", "label": "Start scorer", "description": "Stub — already live in demo"},
        {"id": "scorer-stop", "label": "Stop scorer", "description": "Stub — demo keeps streaming"},
    ]


def _gt_regions() -> list[dict[str, Any]]:
    regions = [
        {"start": 18, "end": 28, "kind": "ground_truth", "label": "RECON", "classId": "T1046"},
        {"start": 30, "end": 42, "kind": "ground_truth", "label": "ENUM", "classId": "T1190"},
        {"start": 44, "end": 55, "kind": "ground_truth", "label": "SPRAY", "classId": "T1110"},
    ]
    for atk in _SCHEDULED_ATTACKS:
        regions.append(
            {
                "start": atk["start"],
                "end": atk["end"],
                "kind": "ground_truth",
                "label": atk["label"],
                "classId": atk.get("classId"),
            }
        )
    return regions


def _memory_regions(playhead: float) -> list[dict[str, Any]]:
    if playhead < 12:
        return []
    return [
        {"start": 12, "end": min(playhead, 24), "kind": "episodic", "label": "RAMX episodic", "classId": "hx_c"},
    ]


def _model_regions(offset: int, playhead: float) -> list[dict[str, Any]]:
    regions = [
        {"start": 19 + offset, "end": 27 + offset, "kind": "suspicious", "label": "T1046_service_scan", "classId": "T1046"},
        {
            "start": 32 + offset,
            "end": 40 + offset,
            "kind": "attack",
            "label": "T1190_web_exploit",
            "classId": "T1190",
            "resolved": playhead > 40 + offset,
            "correct": True,
        },
        {
            "start": 46 + offset,
            "end": 52 + offset,
            "kind": "suspicious",
            "label": "T1110_ssh_bruteforce",
            "classId": "T1110",
            "resolved": playhead > 52 + offset,
            "correct": False,
        },
    ]
    return [r for r in regions if r["start"] <= playhead + 8]


def session(mode: str = "recorded") -> dict[str, Any]:
    window_sec = 1.0
    if mode == "live":
        elapsed = time.time() - _START
        playhead = max(0.0, elapsed - 3.0)  # short warmup
        duration = max(playhead + 18, 30.0)
        n = int(playhead) + 20
    else:
        playhead = 38.0
        duration = 70.0
        n = 70

    models = [
        {
            "id": "shaun_v3",
            "name": "Shaun v3",
            "predicted": _series(n, 0.12, 0.04, 0, 28),
            "regions": _model_regions(0, playhead),
            "accuracy": {"lineMae": 0.042, "regionPrecision": 0.78, "regionRecall": 0.71},
        },
        {
            "id": "hx_c",
            "name": "HX-C",
            "predicted": _series(n, 0.10, 0.05, 2, 26),
            "regions": _model_regions(1, playhead),
            "accuracy": {"lineMae": 0.038, "regionPrecision": 0.82, "regionRecall": 0.75},
        },
    ]

    return {
        "id": "ui-demo",
        "playheadSec": round(playhead, 2),
        "elapsedSec": round(playhead, 2),
        "windowSec": window_sec,
        "durationSec": duration,
        "mode": mode,
        "policyMode": _POLICY,
        "ips": {
            "armed": playhead > 30,
            "blocker": "hx_c" if playhead > 42 else None,
            "blockAtSec": 42.0 if playhead > 42 else None,
            "streaks": {"shaun_v3": 1 if playhead > 35 else 0, "hx_c": 2 if playhead > 40 else 0},
        },
        "actual": _series(n, 0.11, 0.035, 1, 27),
        "models": models,
        "groundTruthRegions": _gt_regions(),
        "memoryRegions": _memory_regions(playhead),
        "phase": "live" if mode == "live" else "recorded",
        "attackPhase": "enum" if 30 <= playhead < 44 else ("spray" if playhead >= 44 else "recon"),
        "scorerRunning": mode == "live",
    }


def logs(since: int = 0) -> list[dict[str, Any]]:
    return [ln for ln in _LOG_LINES if int(ln.get("ts", 0)) >= since]


def run_script(script_id: str, delay_sec: float | None = None) -> dict[str, Any]:
    delay = delay_sec if delay_sec is not None else 45.0
    labels = {
        "killchain-recon": ("RECON", "T1046", 18),
        "killchain-enum": ("ENUM", "T1190", 14),
        "killchain-spray": ("SPRAY", "T1110", 16),
        "killchain-loot": ("LOOT", "T1041", 14),
        "killchain-all": ("KILLCHAIN", "T1190", 50),
        "auto-attack": ("AUTO-ATTACK", "T1190", 45),
    }
    if script_id in labels:
        label, class_id, dur = labels[script_id]
        start = (time.time() - _START) + delay
        _SCHEDULED_ATTACKS.append(
            {"start": start, "end": start + dur, "label": label, "classId": class_id}
        )
        _append_log(f"[attack] scheduled {script_id} in {delay:.0f}s (session t≈{start:.0f}s)", "alert")
        _append_log(f"[attack] stub — {label} will appear on chart as ground truth", "phase")
    elif script_id == "scorer-start":
        _append_log("[script] scorer already live in UI demo mode", "phase")
    elif script_id == "scorer-stop":
        _append_log("[script] stub — scorer keeps streaming in demo", "info")
    else:
        _append_log(f"[script] stub queued {script_id} (no-op)", "phase")
    return {"job_id": f"stub-{script_id}-{int(time.time())}", "status": "queued"}


def set_policy(mode: str) -> str:
    global _POLICY
    _POLICY = "ips" if mode == "ips" else "ids"
    _append_log(f"Policy → {_POLICY.upper()}", "phase")
    return _POLICY


def get_config() -> dict[str, Any]:
    return dict(_LAB_CONFIG)


def patch_config(**updates: Any) -> dict[str, Any]:
    _LAB_CONFIG.update({k: v for k, v in updates.items() if v is not None})
    return get_config()
