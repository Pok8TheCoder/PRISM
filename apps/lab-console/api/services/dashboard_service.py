"""Build dashboard payload for Lab Console."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MOCKS = ROOT / "apps" / "lab-console" / "mocks"
EVENTS = ROOT / "data" / "lab_events"
STATE_PATH = EVENTS / "demo_forecast.json"
HIST_PATH = EVENTS / "demo_forecast.jsonl"

CONTAINER_IMAGES = {
    "target-server": "prism/harborline:latest",
    "attacker-bot": "prism/attacker:latest",
    "benign-client": "prism/benign:latest",
    "suricata": "jasonish/suricata:latest",
    "redteam": "prism/redteam:latest",
}


def _load_mock() -> dict[str, Any]:
    path = MOCKS / "dashboard.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_forecast_state() -> dict[str, Any] | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _metrics_from_history(limit: int = 60) -> list[dict[str, Any]]:
    if not HIST_PATH.exists():
        return []
    lines: list[dict[str, Any]] = []
    try:
        for raw in HIST_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
            if raw.strip():
                lines.append(json.loads(raw))
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for i, snap in enumerate(lines):
        out.append(
            {
                "t": i,
                "flowRate": float(snap.get("window", i)) * 4.0,
                "pAttack": float(max(snap.get("p_hx_c", 0.0), snap.get("p_sn2rx3", 0.0))),
            }
        )
    return out


def _dataset_stats() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    candidates = [
        ("lab_events buffer", EVENTS / "forecast_adapt_buffer.npz"),
        ("adversarial missed", ROOT / "data" / "raw" / "adversarial" / "missed"),
        ("CIC-IDS val (5s)", ROOT / "data" / "processed" / "aryan_cic_5s"),
    ]
    for name, path in candidates:
        if not path.exists():
            continue
        if path.is_dir():
            files = list(path.rglob("*"))
            size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
            samples = sum(1 for f in files if f.suffix in {".json", ".pcap", ".npz"})
            datasets.append({"name": name, "path": str(path.relative_to(ROOT)), "samples": samples, "sizeMb": round(size_mb, 1)})
        else:
            size_mb = path.stat().st_size / (1024 * 1024)
            datasets.append({"name": name, "path": str(path.relative_to(ROOT)), "sizeMb": round(size_mb, 1)})
    return datasets


def _model_summaries() -> list[dict[str, Any]]:
    from .models_service import list_model_registry

    return [
        {
            "id": m["id"],
            "name": m["name"],
            "version": m["version"],
            "loaded": m.get("exists", False),
        }
        for m in list_model_registry()
    ]


def build_dashboard() -> dict[str, Any]:
    mock = _load_mock()
    forecast = _read_forecast_state()

    containers: list[dict[str, Any]] = []
    try:
        from src.adversarial.lab_manager import lab_status

        status = lab_status()
        for name, running in status.items():
            containers.append(
                {
                    "name": name,
                    "status": "running" if running else "stopped",
                    "image": CONTAINER_IMAGES.get(name),
                }
            )
        # Optional extra containers if docker is reachable
        for extra in ("suricata", "redteam"):
            if extra not in status:
                try:
                    from src.adversarial.lab_manager import _container_running

                    containers.append(
                        {
                            "name": extra,
                            "status": "running" if _container_running(extra) else "stopped",
                            "image": CONTAINER_IMAGES.get(extra),
                        }
                    )
                except Exception:
                    pass
    except Exception:
        containers = mock.get("containers", [])

    device = dict(mock.get("device", {}))
    try:
        from src.ui.lab_controller import get_device, get_gpu_name

        dev = get_device()
        device = {
            "cuda": dev.type == "cuda",
            "deviceName": get_gpu_name() or ("CPU" if dev.type != "cuda" else "CUDA"),
            "vramGb": None,
        }
        if dev.type == "cuda":
            import torch

            props = torch.cuda.get_device_properties(0)
            device["vramGb"] = round(props.total_memory / (1024**3))
            device["deviceName"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    training = dict(mock.get("training", {"active": False, "message": "Idle"}))
    if forecast and forecast.get("phase") == "live":
        training["message"] = f"Scorer live — window {forecast.get('window', 0)}"

    metrics = _metrics_from_history()
    if not metrics:
        metrics = mock.get("metricsTimeseries", [])

    return {
        "containers": containers or mock.get("containers", []),
        "training": training,
        "datasets": _dataset_stats() or mock.get("datasets", []),
        "device": device,
        "networks": mock.get(
            "networks",
            [
                {"name": "lab_front", "cidr": "172.28.0.0/24", "protected": True},
                {"name": "lab_back", "cidr": "172.29.0.0/24", "protected": True},
                {"name": "internet_sim", "cidr": "10.0.0.0/24", "protected": False},
            ],
        ),
        "models": _model_summaries() or mock.get("models", []),
        "metricsTimeseries": metrics,
        "scorerRunning": forecast is not None and forecast.get("phase") in {"live", "warmup"},
        "forecastPhase": forecast.get("phase") if forecast else None,
    }
