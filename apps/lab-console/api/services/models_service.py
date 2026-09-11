"""Model registry for Adversarial Lab tab."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
MOCKS = ROOT / "apps" / "lab-console" / "mocks"

KNOWN_MODELS: list[dict[str, Any]] = [
    {
        "id": "laplace_world_model",
        "name": "PRISM Laplace Model (Tier 1 & Tier 2 Combined)",
        "version": "universal-laplace-5s",
        "checkpointPath": "weights/universal_gen10/world_model_best.pt",
        "parameters": 5_721_499,
        "classes": [
            "Benign",
            "Reconnaissance",
            "Initial Access",
            "Lateral Movement",
            "Command & Control",
            "Exfiltration",
            "Impact",
        ],
        "featureDim": 249,
        "seqLen": 20,
        "tags": ["laplace", "world_model", "spatiotemporal_gnn", "tier1_tier2_combined", "live_lab"],
    },
    {
        "id": "shaun_v3",
        "name": "Shaun v3",
        "version": "w5s",
        "checkpointPath": "PRISM-shaun/weights/w5s/world_model.pt",
        "parameters": 1_240_000,
        "classes": ["benign", "T1046_service_scan", "T1110_ssh_bruteforce", "T1190_web_exploit_probe"],
        "featureDim": 292,
        "seqLen": 12,
        "tags": ["forecast", "ips", "lab"],
    },
    {
        "id": "hx_c",
        "name": "HX-C",
        "version": "lab",
        "checkpointPath": "Automode/train/checkpoints/hx_c_w5s_lab.pt",
        "parameters": 580_000,
        "classes": ["benign", "recon", "enum", "spray", "loot"],
        "featureDim": 128,
        "seqLen": 8,
        "tags": ["forecast", "kill-chain", "lab-adapt"],
    },
    {
        "id": "gen9_world_model",
        "name": "PRISM Gen 9 Hierarchical World Model",
        "version": "universal-5s",
        "checkpointPath": "weights/universal_gen9/world_model_best.pt",
        "parameters": 5_670_611,
        "classes": [
            "Benign",
            "Reconnaissance",
            "Initial Access",
            "Lateral Movement",
            "Command & Control",
            "Exfiltration",
            "Impact",
        ],
        "featureDim": 242,
        "seqLen": 20,
        "tags": ["world_model", "hierarchical_routing", "supcon", "gen9", "live_lab"],
    },
    {
        "id": "gen8_world_model",
        "name": "PRISM Gen 8 World Model",
        "version": "universal-5s",
        "checkpointPath": "weights/universal_gen8/world_model_best.pt",
        "parameters": 5_602_611,
        "classes": [
            "Benign",
            "Reconnaissance",
            "Initial Access",
            "Lateral Movement",
            "Command & Control",
            "Exfiltration",
            "Impact",
        ],
        "featureDim": 242,
        "seqLen": 20,
        "tags": ["world_model", "temporal_transformer", "mitre_expert", "gen8", "live_lab"],
    },
    {
        "id": "xmt_01",
        "name": "XMT.01",
        "version": "v1",
        "checkpointPath": "models/checkpoints/xmt_world_model_best.pt",
        "parameters": 720_000,
        "classes": ["benign", "recon", "enum", "spray", "loot", "exfil"],
        "featureDim": 128,
        "seqLen": 10,
        "tags": ["forecast", "lab-trained"],
    },
]


def _resolve_checkpoint(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    # PRISM-shaun sits beside repo root
    if rel.startswith("PRISM-shaun"):
        return ROOT.parent / rel
    if rel.startswith("Automode"):
        return ROOT.parent / rel
    return ROOT / rel


def _enrich(entry: dict[str, Any]) -> dict[str, Any]:
    ckpt = _resolve_checkpoint(entry["checkpointPath"])
    exists = ckpt.is_file()
    size_mb = round(ckpt.stat().st_size / (1024 * 1024), 1) if exists else entry.get("sizeMb", 0)
    last_trained = None
    if exists:
        last_trained = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d")
    return {
        **entry,
        "sizeMb": size_mb,
        "exists": exists,
        "lastTrained": last_trained or entry.get("lastTrained"),
        "checkpointPath": str(ckpt.relative_to(ROOT.parent)) if exists and ROOT.parent in ckpt.parents else entry["checkpointPath"],
    }


def list_model_registry() -> list[dict[str, Any]]:
    mock_path = MOCKS / "models.json"
    if mock_path.exists():
        try:
            data = json.loads(mock_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return [_enrich(m) for m in data]
        except Exception:
            pass
    return [_enrich(m) for m in KNOWN_MODELS]
