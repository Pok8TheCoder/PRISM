#!/usr/bin/env python3
"""Run Shaun V2 base + native RAMXPredictor on lab PCAP paths (sidecar)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PRISM_ROOT = Path(__file__).resolve().parent.parent
SHAUN_ROOT = Path(__file__).resolve().parent.parent.parent / "PRISM-shaun"
sys.path.insert(0, str(PRISM_ROOT))

import importlib.util

_extract_path = PRISM_ROOT / "src" / "pipeline" / "extract.py"
_spec = importlib.util.spec_from_file_location("prism_extract", _extract_path)
_prism_extract = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prism_extract)
prism_pcap_to_rows = _prism_extract.pcap_to_rows

_ingest_path = PRISM_ROOT / "scripts" / "shaun_pcap_ingest.py"
_ingest_spec = importlib.util.spec_from_file_location("shaun_pcap_ingest", _ingest_path)
_ingest_mod = importlib.util.module_from_spec(_ingest_spec)
assert _ingest_spec.loader is not None
_ingest_spec.loader.exec_module(_ingest_mod)
rows_to_shaun_states = _ingest_mod.rows_to_shaun_states

_scale_path = PRISM_ROOT / "scripts" / "lab_traffic_scale.py"
_scale_spec = importlib.util.spec_from_file_location("lab_traffic_scale", _scale_path)
_scale_mod = importlib.util.module_from_spec(_scale_spec)
assert _scale_spec.loader is not None
_scale_spec.loader.exec_module(_scale_mod)
scale_prism_rows = _scale_mod.scale_prism_rows

os.chdir(SHAUN_ROOT)
for _k in list(sys.modules):
    if _k == "src" or _k.startswith("src."):
        del sys.modules[_k]
sys.path = [str(SHAUN_ROOT)] + [p for p in sys.path if Path(p).resolve() != PRISM_ROOT.resolve()]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.models.world_model import StateTransformerWorldModel  # noqa: E402

LOOKBACK = 30
WARMUP_N = 20
RAMX_WARMUP_STEPS = 15


def pcap_states(pcap: Path, scale_factor: float = 1.0, replicate: int = 1) -> np.ndarray:
    rows = prism_pcap_to_rows(pcap)
    if scale_factor != 1.0 or replicate != 1:
        rows = scale_prism_rows(rows, factor=scale_factor, replicate=replicate)
    return rows_to_shaun_states(rows, shaun_root=SHAUN_ROOT, window_sec=15.0)


def load_model():
    ck = torch.load("weights/world_model.pt", map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    d_state = sd["input_embed.0.weight"].shape[1]
    pe_len = sd.get("pos_encoder.pe", torch.zeros(1, 50, 256)).shape[1]
    model = StateTransformerWorldModel(d_state=d_state, d_model=256, nhead=8, num_layers=4, max_seq_len=pe_len)
    model.load_state_dict(sd)
    model.eval()
    mean = np.asarray(ck.get("scaler_mean", np.zeros((1, d_state))), dtype=np.float32)
    std = np.asarray(ck.get("scaler_std", np.ones((1, d_state))), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return model, mean, std


def warmup(n: int) -> list[np.ndarray]:
    states = np.load("data/processed/states.npy")
    labels = np.load("data/processed/attack_labels.npy")
    idx = np.where(labels == 0)[0][:n]
    return [states[i].astype(np.float32) for i in idx]


def _padded_window(buf: list[np.ndarray]) -> np.ndarray:
    window = buf[-LOOKBACK:]
    if len(window) < LOOKBACK:
        window = [window[0]] * (LOOKBACK - len(window)) + window
    return np.stack(window)


def forward_hidden(model, seq_norm: np.ndarray, device: torch.device) -> float:
    x = torch.from_numpy(seq_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.input_embed(x)
        emb = model.pos_encoder(emb)
        for layer in model.layers:
            emb, _ = layer(emb)
        h = model.layer_norm(emb)[:, -1, :]
        logits = model.attack_head(h)
    return float(torch.softmax(logits, dim=-1)[0, 1].item())


def replay_base(model, mean, std, states: list[np.ndarray], device) -> list[float]:
    m, s = mean.reshape(-1), std.reshape(-1)
    buf: list[np.ndarray] = []
    out: list[float] = []
    for raw in states:
        buf.append(((raw - m) / s).astype(np.float32))
        seq = _padded_window(buf)
        out.append(forward_hidden(model, seq, device))
    return out


def replay_shaun_ramx(
    model,
    mean,
    std,
    states: list[np.ndarray],
    device: torch.device,
    context_skip: int = WARMUP_N,
) -> list[float]:
    """RAMX v2: learn baseline during context buffer, fuse only on lab PCAP steps."""
    from src.prediction.ramx import RAMXPredictor, RAMX_VERSION

    predictor = RAMXPredictor(
        model,
        mean,
        std,
        warmup_steps=RAMX_WARMUP_STEPS,
        context_skip_steps=context_skip,
        enable_ttt=False,
    )
    buf: list[np.ndarray] = []
    out: list[float] = []
    for raw in states:
        raw = np.asarray(raw, dtype=np.float32)
        buf.append(raw)
        traj = _padded_window(buf)
        res = predictor.predict_state(traj, device=str(device))
        out.append(float(res["p_attack"]))
    assert res["ramx_version"] == RAMX_VERSION
    return out


def summarize_p(p: list[float], start: int) -> dict:
    if not p:
        return {"warmup_mean": 0.0, "attack_mean": 0.0, "attack_max": 0.0}
    return {
        "warmup_mean": float(np.mean(p[:start])) if start else 0.0,
        "attack_mean": float(np.mean(p[start:])) if len(p) > start else 0.0,
        "attack_max": float(np.max(p[start:])) if len(p) > start else 0.0,
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-factor", type=float, default=1.0)
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("pcaps", nargs="+")
    args = ap.parse_args()

    pcaps = [Path(p) for p in args.pcaps]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = load_model()
    model = model.to(device)
    warm = warmup(WARMUP_N)
    results = {
        "ingest": "schema_aligner",
        "ramx_impl": "shaun_ramx_v2_context_gated",
        "ramx_context_skip": WARMUP_N,
        "ramx_warmup_steps": RAMX_WARMUP_STEPS,
        "ramx_ttt": False,
        "scale_factor": args.scale_factor,
        "replicate": args.replicate,
    }
    for pcap in pcaps:
        atk = pcap_states(pcap, scale_factor=args.scale_factor, replicate=args.replicate)
        all_s = warm + [np.asarray(x, dtype=np.float32) for x in atk]
        start = len(warm)
        p_base = replay_base(model, mean, std, all_s, device)
        p_ramx = replay_shaun_ramx(model, mean, std, all_s, device)
        results[str(pcap)] = {
            "n_windows": len(p_base),
            "attack_start": start,
            "p_attack_base": p_base,
            "p_attack_ramx": p_ramx,
            "base": summarize_p(p_base, start),
            "ramx": summarize_p(p_ramx, start),
        }
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
