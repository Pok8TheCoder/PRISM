"""Streaming inference for Aryan Gen8 world model (base + RAMX) on 242-d 5s windows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch
import torch.nn as nn

from src.aryan.constants import MITRE_STAGES_INV
from src.aryan.streaming_variants import (
    RAMXMemoryBank,
    StagedRAMXMemoryBank,
    _classify_blend,
    _classify_blend_gated,
    _classify_blend_gen8_led,
    _classify_blend_staged,
    _classify_blend_staged_soft,
    _classify_blend_targeted,
    calibrate_match_thresh,
)

ROOT = Path(__file__).resolve().parent.parent.parent
ARYAN_ROOT = ROOT.parent / "PRISM-aryan"
GEN8_CKPT = ARYAN_ROOT / "weights" / "universal_gen8" / "world_model_best.pt"
GEN8_SCALER = ARYAN_ROOT / "weights" / "universal_gen8_5s_scaler.pkl"


def _ensure_aryan_path() -> None:
    """Prefer PRISM-aryan `src` over PRISM `src` for Gen8 imports."""
    prism = str(ROOT)
    aryan = str(ARYAN_ROOT)
    if aryan not in sys.path:
        sys.path.insert(0, aryan)
    # Move aryan before prism so `import src.models` resolves to PRISM-aryan.
    sys.path[:] = [p for p in sys.path if p not in (prism, aryan)]
    sys.path.insert(0, aryan)
    if prism not in sys.path:
        sys.path.append(prism)


def _load_aryan_modules():
    """Import Gen8 builder from PRISM-aryan without shadowing PRISM's `src` package."""
    prism_src_keys = [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]
    saved = {k: sys.modules.pop(k) for k in prism_src_keys}
    _ensure_aryan_path()
    try:
        from src.models.world_model import build_world_model, load_checkpoint  # noqa: WPS433
        from src.utils.config import load_config  # noqa: WPS433
        return build_world_model, load_checkpoint, load_config
    finally:
        aryan_src_keys = [k for k in list(sys.modules) if k == "src" or k.startswith("src.")]
        for k in aryan_src_keys:
            sys.modules.pop(k, None)
        for k, mod in saved.items():
            sys.modules[k] = mod
        _ensure_aryan_path()


def preprocess_state(raw: np.ndarray, scaler: Any | None) -> np.ndarray:
    x = np.sign(raw) * np.log1p(np.abs(raw.astype(np.float64)))
    if scaler is not None:
        x = scaler.transform(x.reshape(1, -1))[0]
    return x.astype(np.float32)


def infer_gen8(model: nn.Module, seq: np.ndarray, device: torch.device) -> dict[str, Any]:
    x = torch.from_numpy(seq[None].astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x)
    p_mit = torch.softmax(out["pred_mitre"], dim=-1)[0].cpu().numpy()
    mit_idx = int(p_mit.argmax())
    hidden = out.get("mlp_features")
    if hidden is None:
        hidden = out["latent_h"]
    return {
        "p_att": float(torch.softmax(out["pred_binary"], dim=-1)[0, 1].item()),
        "p_mit": p_mit,
        "mitre_stage": mit_idx,
        "mitre_stage_name": MITRE_STAGES_INV.get(mit_idx, "Benign"),
        "hidden": hidden[0].cpu().numpy(),
        "pred_state": out["pred_state_mean"][0].cpu().numpy(),
    }


def load_gen8_bundle(
    ckpt_path: Path | None = None,
    scaler_path: Path | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    build_world_model, load_checkpoint, load_config = _load_aryan_modules()

    ckpt = Path(ckpt_path or GEN8_CKPT)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Gen8 checkpoint not found: {ckpt}")

    cfg = load_config(str(ARYAN_ROOT / "configs" / "train.yaml"))
    lookback = int(getattr(cfg.data, "lookback", 20))
    model = build_world_model(cfg.model)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(model, str(ckpt), device=str(dev))
    model.to(dev)
    model.eval()

    sp = Path(scaler_path or GEN8_SCALER)
    scaler = joblib.load(sp) if sp.is_file() else None

    return {
        "model": model,
        "scaler": scaler,
        "lookback": lookback,
        "device": dev,
        "ckpt": ckpt,
        "version": "gen8",
    }


def calibrate_gen8_detect_thresh(
    bundle: dict[str, Any],
    warmup_states: list[np.ndarray] | None = None,
    val_states: np.ndarray | None = None,
    val_bin: np.ndarray | None = None,
    margin: float = 1e-4,
    default: float = 0.5,
) -> float:
    """Benign-warmup threshold: max P(attack) on held-out benign windows + margin.

    Gen8's binary head is saturated (~0.99) on lab PCAP features; calibrating just
    above the warmup ceiling yields usable precision on offline lab benches.
    """
    states = list(warmup_states or [])
    scaler = bundle["scaler"]
    if not states and val_states is not None and val_bin is not None:
        for i, b in enumerate(val_bin):
            if not b:
                states.append(preprocess_state(np.asarray(val_states[i], dtype=np.float32), scaler))
    if not states:
        return default
    scorer = StreamingAryanGen8(bundle, detect_threshold=default)
    ps = [float(scorer.step(np.asarray(s, dtype=np.float32))["p_att"]) for s in states[:500]]
    if not ps:
        return default
    return float(np.percentile(ps, 99.5) + margin)


def calibrate_gen8_settings(
    bundle: dict[str, Any],
    warmup_states: list[np.ndarray] | None = None,
    val_states: np.ndarray | None = None,
    val_bin: np.ndarray | None = None,
) -> dict[str, float]:
    """One-shot lab + RAMX calibration bundle."""
    hidden_thresh = 30.0
    if val_states is not None and val_bin is not None and len(val_states) > bundle["lookback"] + 1:
        hidden_thresh = calibrate_gen8_hidden_thresh(bundle, val_states, val_bin)
    detect_thresh = calibrate_gen8_detect_thresh(
        bundle,
        warmup_states=warmup_states,
        val_states=val_states,
        val_bin=val_bin,
    )
    return {"detect_threshold": detect_thresh, "hidden_thresh": hidden_thresh}


def calibrate_gen8_hidden_thresh(
    bundle: dict[str, Any],
    val_states: np.ndarray,
    val_bin: np.ndarray,
) -> float:
    model = bundle["model"]
    scaler = bundle["scaler"]
    lookback = bundle["lookback"]
    device = bundle["device"]
    keys: list[np.ndarray] = []
    for t in range(lookback - 1, len(val_states) - 1):
        seq = np.stack(
            [preprocess_state(val_states[i], scaler) for i in range(t - lookback + 1, t + 1)]
        )
        keys.append(infer_gen8(model, seq, device)["hidden"])
    if not keys:
        return 30.0
    val_hidden = np.stack(keys)
    val_labels = val_bin[lookback : len(val_states)]
    return calibrate_match_thresh(val_hidden, val_labels)


class StreamingAryanGen8:
    """Gen8 base scorer — no episodic RAMX blend."""

    system_id = "aryan_gen8"

    def __init__(self, bundle: dict[str, Any], detect_threshold: float = 0.5):
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.lookback = int(bundle["lookback"])
        self.device = bundle["device"]
        self.detect_threshold = float(detect_threshold)
        self.buffer: list[np.ndarray] = []
        self._pending: Optional[dict] = None
        self.n_steps = 0
        self.last_p_att = 0.0
        self.last_dynamics_mse: Optional[float] = None

    def begin_live_phase(self) -> None:
        pass

    def set_context_skip(self, steps: int) -> None:
        pass

    def _padded_seq(self) -> np.ndarray:
        if len(self.buffer) < self.lookback:
            pad = [np.zeros_like(self.buffer[0])] * (self.lookback - len(self.buffer))
            seq = pad + self.buffer
        else:
            seq = self.buffer[-self.lookback :]
        return np.stack(seq)

    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict[str, Any]:
        state = preprocess_state(np.asarray(state, dtype=np.float32), self.scaler)
        self.n_steps += 1
        if self._pending is not None:
            self._finalize_pending(state)

        self.buffer.append(state)
        if len(self.buffer) > self.lookback:
            self.buffer.pop(0)
        full_context = len(self.buffer) >= self.lookback
        seq = self._padded_seq()
        out = infer_gen8(self.model, seq, self.device)
        self.last_p_att = float(out["p_att"])
        self._pending = {
            "pred_state": out["pred_state"],
            "full_context": full_context,
        }
        return {
            "p_att": self.last_p_att,
            "p_mit": out["p_mit"],
            "mitre_stage": out["mitre_stage"],
            "mitre_stage_name": out["mitre_stage_name"],
            "hidden": out["hidden"],
            "memory_written": False,
            "alert": self.last_p_att >= self.detect_threshold,
        }

    def _finalize_pending(self, state: np.ndarray) -> None:
        pend = self._pending
        self._pending = None
        if pend is None or not pend["full_context"]:
            return
        raw_mse = float(np.mean((pend["pred_state"] - state) ** 2))
        self.last_dynamics_mse = float(min(raw_mse, 1e12)) if np.isfinite(raw_mse) else 1e12

    def rollout(self, horizon: int) -> dict[str, Any]:
        p = self.last_p_att
        return {
            "forecast_max": p,
            "points": [{"p": p, "w": i + 1} for i in range(horizon)],
        }


class StreamingAryanGen8Ramx(StreamingAryanGen8):
    """Gen8 + RAMX episodic classify-blend (hidden-key k=3)."""

    system_id = "aryan_gen8_ramx"

    def __init__(
        self,
        bundle: dict[str, Any],
        hidden_thresh: float | None = None,
        detect_threshold: float = 0.5,
        rotation_interval: int = 100,
        dynamic_capacity_pre: int = 20,
        dynamic_capacity_post: int = 40,
        suspicious_thresh: float = 0.01,
        memory_mode: str = "unified",
        blend_mode: str = "unified",
        blend_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(bundle, detect_threshold=detect_threshold)
        self.match_thresh = hidden_thresh if hidden_thresh is not None else 30.0
        self.knn_k = 3
        self.memory_mode = memory_mode
        self.blend_mode = blend_mode
        self.blend_kwargs = dict(blend_kwargs or {})
        self._ramx_kwargs = {
            "rotation_interval": rotation_interval,
            "dynamic_capacity_pre": dynamic_capacity_pre,
            "dynamic_capacity_post": dynamic_capacity_post,
            "suspicious_thresh": suspicious_thresh,
        }
        self.bank = self._make_bank()

    def _make_bank(self) -> RAMXMemoryBank | StagedRAMXMemoryBank:
        if self.memory_mode == "staged":
            return StagedRAMXMemoryBank(**self._ramx_kwargs)
        return RAMXMemoryBank(**self._ramx_kwargs)

    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict[str, Any]:
        state = preprocess_state(np.asarray(state, dtype=np.float32), self.scaler)
        self.n_steps += 1
        memory_written = False
        if self._pending is not None:
            memory_written = self._finalize_pending_ramx(state, true_bin, true_mit)

        self.buffer.append(state)
        if len(self.buffer) > self.lookback:
            self.buffer.pop(0)
        full_context = len(self.buffer) >= self.lookback
        seq = self._padded_seq()
        out = infer_gen8(self.model, seq, self.device)
        p_att_raw = float(out["p_att"])
        key = out["hidden"]
        if self.blend_mode == "none":
            p_att, p_mit = p_att_raw, out["p_mit"]
        elif self.blend_mode == "gated":
            p_att, p_mit = _classify_blend_gated(
                self.bank, key, p_att_raw, out["p_mit"], self.knn_k, self.match_thresh,
                **self.blend_kwargs,
            )
        elif self.blend_mode == "targeted":
            p_att, p_mit = _classify_blend_targeted(
                self.bank, key, p_att_raw, out["p_mit"], self.knn_k, self.match_thresh,
                **self.blend_kwargs,
            )
        elif self.blend_mode == "gen8_led":
            p_att, p_mit = _classify_blend_gen8_led(
                self.bank, key, p_att_raw, out["p_mit"], self.knn_k, self.match_thresh,
                **self.blend_kwargs,
            )
        elif self.blend_mode == "staged_soft":
            p_att, p_mit = _classify_blend_staged_soft(
                self.bank, key, p_att_raw, out["p_mit"], self.knn_k, self.match_thresh,
                **self.blend_kwargs,
            )
        elif self.blend_mode == "staged":
            p_att, p_mit = _classify_blend_staged(
                self.bank, key, p_att_raw, out["p_mit"], self.knn_k, self.match_thresh,
            )
        else:
            p_att, p_mit = _classify_blend(
                self.bank, key, p_att_raw, out["p_mit"], self.knn_k, self.match_thresh,
            )
        self.last_p_att = float(p_att)
        self._pending = {
            "key": key,
            "pred_state": out["pred_state"],
            "full_context": full_context,
            "p_att_raw": p_att_raw,
            "true_bin_slot": true_bin,
            "true_mit_slot": true_mit,
        }
        mit_idx = int(np.argmax(p_mit))
        return {
            "p_att": self.last_p_att,
            "p_att_raw": p_att_raw,
            "p_mit": p_mit,
            "mitre_stage": mit_idx,
            "mitre_stage_name": MITRE_STAGES_INV.get(mit_idx, "Benign"),
            "hidden": key,
            "memory_written": memory_written,
            "alert": self.last_p_att >= self.detect_threshold,
        }

    def _finalize_pending_ramx(
        self,
        state: np.ndarray,
        true_bin: Optional[int],
        true_mit: Optional[int],
    ) -> bool:
        pend = self._pending
        self._pending = None
        if pend is None:
            return False
        raw_mse = float(np.mean((pend["pred_state"] - state) ** 2))
        self.last_dynamics_mse = float(min(raw_mse, 1e12)) if np.isfinite(raw_mse) else 1e12
        if true_bin is None or not pend["full_context"]:
            return False
        self.bank.ingest(pend["key"], int(true_bin), int(true_mit or 0), pend["p_att_raw"])
        return True


def pcap_to_gen8_state(
    pcap_path: Path | None,
    *,
    last_state: np.ndarray | None = None,
    window_sec: float = 5.0,
    scale_factor: float = 1.0,
    replicate: int = 1,
) -> np.ndarray:
    """242-d state from a PCAP window (same ingest path as ARY IPS backends)."""
    import pandas as pd

    from src.aryan.ingest import TARGET_DIM, _states_from_flow_df
    from src.pipeline.extract import pcap_to_rows

    if pcap_path is None or not Path(pcap_path).exists():
        if last_state is not None:
            return last_state.copy()
        return np.zeros(TARGET_DIM, dtype=np.float32)
    rows = pcap_to_rows(Path(pcap_path))
    if scale_factor != 1.0 or replicate != 1:
        from scripts.lab_traffic_scale import scale_prism_rows

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
