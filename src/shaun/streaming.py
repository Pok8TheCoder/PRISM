"""Step-by-step Shaun V2 base + RAMX v2 scorers for live Docker IPS."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
LOOKBACK = 30
RAMX_WARMUP_STEPS = 15
SHAUN_WINDOW_SEC = 15.0
SHAUN_DIM = 292


def _clear_src_modules() -> None:
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]


def _ensure_prism_path() -> None:
    root = str(ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_prism_imports():
    _clear_src_modules()
    _ensure_prism_path()
    from src.pipeline.extract import pcap_to_rows
    from scripts.lab_traffic_scale import scale_prism_rows
    from scripts.shaun_pcap_ingest import rows_to_shaun_states
    return pcap_to_rows, scale_prism_rows, rows_to_shaun_states


def _ensure_shaun_path() -> Path:
    shaun = SHAUN_ROOT.resolve()
    if not shaun.exists():
        raise FileNotFoundError(f"PRISM-shaun worktree not found: {shaun}")
    return shaun


def _load_shaun_model_modules():
    shaun = _ensure_shaun_path()
    prev_cwd = os.getcwd()
    prev_path = sys.path.copy()
    os.chdir(shaun)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path = [str(shaun)] + [p for p in sys.path if Path(p).resolve() != ROOT.resolve()]
    try:
        from src.models.world_model import StateTransformerWorldModel
        from src.prediction.ramx import RAMXPredictor
    finally:
        os.chdir(prev_cwd)
        sys.path = prev_path
    return StateTransformerWorldModel, RAMXPredictor


def load_shaun_bundle(
    shaun_root: Path | None = None,
    ckpt_path: Path | None = None,
) -> dict[str, Any]:
    """Load Shaun V2 checkpoint, scaler, and model class."""
    shaun = (shaun_root or SHAUN_ROOT).resolve()
    ck = (ckpt_path or shaun / "weights" / "world_model.pt").resolve()
    if not ck.exists():
        raise FileNotFoundError(f"Missing Shaun checkpoint: {ck}")

    StateTransformerWorldModel, RAMXPredictor = _load_shaun_model_modules()
    ckpt = torch.load(str(ck), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    d_state = sd["input_embed.0.weight"].shape[1]
    pe_len = sd.get("pos_encoder.pe", torch.zeros(1, 50, 256)).shape[1]
    model = StateTransformerWorldModel(
        d_state=d_state, d_model=256, nhead=8, num_layers=4, max_seq_len=pe_len,
    )
    model.load_state_dict(sd)
    model.eval()
    mean = np.asarray(ckpt.get("scaler_mean", np.zeros((1, d_state))), dtype=np.float32)
    std = np.asarray(ckpt.get("scaler_std", np.ones((1, d_state))), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "device": device,
        "shaun_root": shaun,
        "ckpt_path": ck,
        "RAMXPredictor": RAMXPredictor,
    }


def pcap_to_shaun_state(
    pcap: Path,
    *,
    last_state: np.ndarray | None = None,
    scale_factor: float = 1.0,
    replicate: int = 1,
    shaun_root: Path | None = None,
    window_sec: float = SHAUN_WINDOW_SEC,
) -> np.ndarray:
    """Extract one 292-d state from a live capture PCAP (optionally scaled)."""
    pcap_to_rows, scale_prism_rows, rows_to_shaun_states = _ensure_prism_imports()
    rows = pcap_to_rows(pcap)
    if scale_factor != 1.0 or replicate != 1:
        rows = scale_prism_rows(rows, factor=scale_factor, replicate=replicate)
    states = rows_to_shaun_states(rows, shaun_root=shaun_root or SHAUN_ROOT, window_sec=window_sec)
    if len(states):
        return states[-1].astype(np.float32)
    if last_state is not None:
        return last_state.copy()
    return np.zeros(SHAUN_DIM, dtype=np.float32)


def _padded_raw(buf: list[np.ndarray]) -> np.ndarray:
    window = buf[-LOOKBACK:]
    if len(window) < LOOKBACK:
        window = [window[0]] * (LOOKBACK - len(window)) + window
    return np.stack(window)


def _forward_base(model, mean, std, buf: list[np.ndarray], device: torch.device) -> float:
    m, s = mean.reshape(-1), std.reshape(-1)
    seq = (( _padded_raw(buf) - m) / s).astype(np.float32)
    x = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.input_embed(x)
        emb = model.pos_encoder(emb)
        for layer in model.layers:
            emb, _ = layer(emb)
        h = model.layer_norm(emb)[:, -1, :]
        logits = model.attack_head(h)
    return float(torch.softmax(logits, dim=-1)[0, 1].item())


class StreamingShaunBase:
    """Shaun V2 classifier only (no RAMX)."""

    system_id = "sn_base"

    def __init__(self, bundle: dict[str, Any] | None = None):
        bundle = bundle or load_shaun_bundle()
        self.model = bundle["model"]
        self.mean = bundle["mean"]
        self.std = bundle["std"]
        self.device = bundle["device"]
        self.buf: list[np.ndarray] = []

    def step(self, raw_state: np.ndarray, true_bin: int = 0, true_mit: int = 0) -> dict[str, float]:
        raw = np.asarray(raw_state, dtype=np.float32)
        self.buf.append(raw)
        p_att = _forward_base(self.model, self.mean, self.std, self.buf, self.device)
        return {"p_att": p_att}


class StreamingShaunRamxV2:
    """Shaun V2 + context-gated RAMX v2 (SN2RXv2)."""

    system_id = "sn2rx"

    def __init__(self, bundle: dict[str, Any] | None = None, context_skip_steps: int = 0):
        bundle = bundle or load_shaun_bundle()
        self.model = bundle["model"]
        self.mean = bundle["mean"]
        self.std = bundle["std"]
        self.device = bundle["device"]
        RAMXPredictor = bundle["RAMXPredictor"]
        self.predictor = RAMXPredictor(
            self.model,
            self.mean,
            self.std,
            warmup_steps=RAMX_WARMUP_STEPS,
            context_skip_steps=context_skip_steps,
            enable_ttt=False,
        )
        self.buf: list[np.ndarray] = []

    def set_context_skip(self, steps: int) -> None:
        self.predictor.context_skip_steps = steps

    def step(self, raw_state: np.ndarray, true_bin: int = 0, true_mit: int = 0) -> dict[str, float]:
        raw = np.asarray(raw_state, dtype=np.float32)
        self.buf.append(raw)
        traj = _padded_raw(self.buf)
        res = self.predictor.predict_state(traj, device=str(self.device))
        return {
            "p_att": float(res["p_attack"]),
            "raw_p_att": float(res["raw_p_attack"]),
            "relative_anomaly": float(res["relative_anomaly"]),
            "context_gated": bool(res.get("context_gated", False)),
        }


class StreamingShaunRamxV3:
    """Shaun V2 + RAMX v3 — alert suppression during warmup + raw local calibration."""

    system_id = "sn2rx3"

    def __init__(self, bundle: dict[str, Any] | None = None, context_skip_steps: int = 0):
        bundle = bundle or load_shaun_bundle()
        self.model = bundle["model"]
        self.mean = bundle["mean"]
        self.std = bundle["std"]
        self.device = bundle["device"]
        _, RAMXPredictorV3 = _load_shaun_ramx_v3()
        self.predictor = RAMXPredictorV3(
            self.model,
            self.mean,
            self.std,
            warmup_steps=RAMX_WARMUP_STEPS,
            context_skip_steps=context_skip_steps,
            enable_ttt=False,
        )
        self.buf: list[np.ndarray] = []

    def set_context_skip(self, steps: int) -> None:
        self.predictor.set_context_skip(steps)

    def begin_live_phase(self) -> None:
        self.predictor.begin_live_phase()

    def step(self, raw_state: np.ndarray, true_bin: int = 0, true_mit: int = 0) -> dict[str, float]:
        raw = np.asarray(raw_state, dtype=np.float32)
        self.buf.append(raw)
        traj = _padded_raw(self.buf)
        res = self.predictor.predict_state(traj, device=str(self.device))
        return {
            "p_att": float(res["p_attack"]),
            "raw_p_att": float(res["raw_p_attack"]),
            "calibrated_raw_p_att": float(res.get("calibrated_raw_p_attack", res["raw_p_attack"])),
            "relative_anomaly": float(res["relative_anomaly"]),
            "context_gated": bool(res.get("context_gated", False)),
            "alert_suppressed": bool(res.get("alert_suppressed", False)),
            "raw_offset": float(res.get("raw_offset", 0.0)),
            "ramx_version": str(res.get("ramx_version", "3.0")),
        }


def _load_shaun_ramx_v3():
    shaun = _ensure_shaun_path()
    prev_cwd = os.getcwd()
    prev_path = sys.path.copy()
    os.chdir(shaun)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path = [str(shaun)] + [p for p in sys.path if Path(p).resolve() != ROOT.resolve()]
    try:
        from src.prediction.ramx_v3 import RAMXPredictorV3
        from src.prediction.ramx import RAMXPredictor
        return RAMXPredictor, RAMXPredictorV3
    finally:
        os.chdir(prev_cwd)
        sys.path = prev_path
