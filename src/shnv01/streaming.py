"""SHNV.01 — streaming inference for the 110-d world model (PRISM_MODEL_PACKAGE).

Base (``shnv_01_base``): frozen ``StateTransformerWorldModel`` classifier only.
RAM (``shnv_01_ramx``): same model + RAMX_V.01 classify-blend on the 256-d
transformer context vector.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SHNV_PKG_ROOT = ROOT / "_friend_pkg"  # extracted PRISM_MODEL_PACKAGE.zip
FRIEND_ROOT = SHNV_PKG_ROOT  # backwards-compatible alias


def _purge_src_modules() -> None:
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            if key.startswith("src.shnv01"):
                continue
            del sys.modules[key]


def _is_local_src_path(path: str) -> bool:
    try:
        rp = Path(path).resolve()
    except OSError:
        return False
    if rp == ROOT.resolve():
        return True
    if (rp / "src" / "__init__.py").exists() and (rp / "src" / "aryan").exists():
        return True
    return False


def _load_friend_world_model_class():
    saved = sys.path[:]
    _purge_src_modules()
    filtered = [p for p in saved if not _is_local_src_path(p)]
    sys.path[:] = [str(SHNV_PKG_ROOT)] + filtered
    try:
        from src.models.world_model import StateTransformerWorldModel
        return StateTransformerWorldModel
    finally:
        sys.path[:] = saved


def _load_ramx_symbols():
    saved = sys.path[:]
    _purge_src_modules()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        spec = importlib.util.spec_from_file_location(
            "prism_aryan_streaming_variants",
            ROOT / "src" / "aryan" / "streaming_variants.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.BLEND_FLOOR, mod.BLEND_MAX_WEIGHT, mod.RAMXMemoryBank, mod.calibrate_match_thresh
    finally:
        sys.path[:] = saved


StateTransformerWorldModel: Any = None
BLEND_FLOOR: float = 0.15
BLEND_MAX_WEIGHT: float = 0.6
RAMXMemoryBank: Any = None
calibrate_match_thresh: Any = None


def _ensure_deps() -> None:
    global StateTransformerWorldModel, BLEND_FLOOR, BLEND_MAX_WEIGHT, RAMXMemoryBank, calibrate_match_thresh
    if StateTransformerWorldModel is None:
        StateTransformerWorldModel = _load_friend_world_model_class()
    if RAMXMemoryBank is None:
        BLEND_FLOOR, BLEND_MAX_WEIGHT, RAMXMemoryBank, calibrate_match_thresh = _load_ramx_symbols()

BLEND_K = 3
DETECT_THRESHOLD = 0.5


def load_shnv01_bundle(ckpt_path: Optional[Path] = None) -> dict[str, Any]:
    _ensure_deps()
    ckpt_path = ckpt_path or (SHNV_PKG_ROOT / "weights" / "world_model.pt")
    chk = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pe = chk["model_state_dict"].get("pos_encoder.pe", torch.zeros(1, 20, 256))
    lookback = int(pe.shape[1])
    model = StateTransformerWorldModel(
        d_state=110, d_model=256, nhead=8, num_layers=4, max_seq_len=lookback,
    )
    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    scaler_mean = np.asarray(chk.get("scaler_mean", np.zeros((1, 110))), dtype=np.float32)
    scaler_std = np.asarray(chk.get("scaler_std", np.ones((1, 110))), dtype=np.float32)
    scaler_std = np.where(scaler_std == 0, 1.0, scaler_std).astype(np.float32)
    return {
        "model": model,
        "lookback": lookback,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }


@torch.no_grad()
def _forward(model: nn.Module, norm_seq: np.ndarray) -> dict:
    x = torch.from_numpy(norm_seq[None].astype(np.float32))
    emb = model.input_embed(x)
    emb = model.pos_encoder(emb)
    attn = None
    for layer in model.layers:
        emb, attn = layer(emb)
    emb = model.layer_norm(emb)
    h_t = emb[:, -1, :]
    pred_mean = model.dynamics_mean_head(h_t)
    pred_logvar = torch.clamp(model.dynamics_logvar_head(h_t), min=-6.0, max=3.0)
    pred_attack = model.attack_head(h_t)
    pred_mitre = model.mitre_head(h_t)
    pred_frac = model.fraction_head(h_t)
    p_att = torch.softmax(pred_attack, dim=-1)[0, 1].item()
    p_mit = torch.softmax(pred_mitre, dim=-1)[0].numpy()
    return {
        "p_att": float(p_att),
        "p_mit": p_mit,
        "hidden": h_t[0].numpy(),
        "pred_state": pred_mean[0].numpy(),
    }


def _classify_blend(
    bank: RAMXMemoryBank,
    key: np.ndarray,
    p_att: float,
    p_mit: np.ndarray,
    knn_k: int,
    match_thresh: float,
) -> tuple[float, np.ndarray]:
    if len(bank) == 0:
        return p_att, p_mit
    matches = bank.query(key, k=knn_k)
    close = [m for m in matches if m[2] < match_thresh]
    if not close:
        return p_att, p_mit
    weights = np.array([max(0.0, 1.0 - d / match_thresh) for _, _, d in close])
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(close)) / len(close)
    w_total = min(BLEND_MAX_WEIGHT, float(np.mean(weights))) * 0.7 + BLEND_FLOOR
    vote_bin = float(np.sum([w * b for (b, _, _), w in zip(close, weights)]))
    p_att_out = (1 - w_total) * p_att + w_total * vote_bin
    mit_onehot = np.zeros(len(p_mit), dtype=np.float32)
    for (_, m, _), w in zip(close, weights):
        mit_onehot[m] += w
    p_mit_out = (1 - w_total) * p_mit + w_total * mit_onehot
    return p_att_out, p_mit_out


class StreamingSHNV01:
    """SHNV.01 world model — ``use_ram=False`` for base, ``True`` for RAMX."""

    def __init__(self, bundle: dict[str, Any], match_thresh: float = 30.0, use_ram: bool = False):
        self.model = bundle["model"]
        self.lookback = bundle["lookback"]
        self.scaler_mean = bundle["scaler_mean"]
        self.scaler_std = bundle["scaler_std"]
        self.use_ram = use_ram
        self.match_thresh = match_thresh
        self.bank = RAMXMemoryBank() if use_ram else None
        self.buffer: list[np.ndarray] = []
        self._pending: Optional[dict] = None
        self.n_steps = 0

    def _norm(self, state: np.ndarray) -> np.ndarray:
        mean = self.scaler_mean.reshape(-1)
        std = self.scaler_std.reshape(-1)
        return ((state.reshape(-1) - mean) / std).astype(np.float32)

    def _padded_seq(self) -> np.ndarray:
        window = self.buffer[-self.lookback:]
        if len(window) < self.lookback:
            pad = [window[0]] * (self.lookback - len(window)) + window
            window = pad
        return np.stack(window)

    def step(self, state: np.ndarray, true_bin: Optional[int] = None, true_mit: Optional[int] = None) -> dict:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != 110:
            raise ValueError(f"expected 110-d state, got shape {state.shape}")
        self.n_steps += 1
        if self._pending is not None:
            self._finalize_pending(true_bin, true_mit)

        self.buffer.append(self._norm(state))
        full_context = len(self.buffer) >= self.lookback
        seq = self._padded_seq()
        out = _forward(self.model, seq)
        p_att, p_mit = out["p_att"], out["p_mit"]
        if self.use_ram and self.bank is not None:
            p_att, p_mit = _classify_blend(
                self.bank, out["hidden"], p_att, p_mit, BLEND_K, self.match_thresh,
            )
        self._pending = {"key": out["hidden"], "full_context": full_context, "p_att_raw": out["p_att"]}
        return {"p_att": float(p_att), "p_mit": p_mit}

    def _finalize_pending(self, true_bin: Optional[int], true_mit: Optional[int]) -> None:
        pend = self._pending
        self._pending = None
        if not self.use_ram or self.bank is None or true_bin is None or not pend["full_context"]:
            return
        self.bank.ingest(pend["key"], int(true_bin), int(true_mit) if true_mit is not None else 0, pend["p_att_raw"])


def lab242_windows_to_friend110_minutes(
    states_242: list[np.ndarray],
    true_bins: list[int],
    true_mits: list[int],
    windows_per_minute: int = 4,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """Aggregate 15s ARY lab windows (~4/min) into 60s 110-d states for friend's model."""
    minutes_s, minutes_b, minutes_m = [], [], []
    for i in range(0, len(states_242), windows_per_minute):
        chunk = states_242[i : i + windows_per_minute]
        if not chunk:
            continue
        tb = [true_bins[j] for j in range(i, min(i + windows_per_minute, len(true_bins)))]
        tm = [true_mits[j] for j in range(i, min(i + windows_per_minute, len(true_mits)))]
        minutes_s.append(_convert_chunk242_to110(chunk))
        minutes_b.append(1 if any(x == 1 for x in tb) else 0)
        minutes_m.append(max(tm) if any(x == 1 for x in tb) else 0)
    return minutes_s, minutes_b, minutes_m


def _convert_chunk242_to110(windows: list[np.ndarray]) -> np.ndarray:
    ws = [np.asarray(w, dtype=np.float32) for w in windows]
    num_flows = float(sum(w[0] for w in ws))
    means35 = np.mean([w[5:40] for w in ws], axis=0)
    stds35 = np.mean([w[114:149] for w in ws], axis=0)
    maxs35 = np.max([w[5:40] for w in ws], axis=0)
    port_ent = float(np.mean([w[4] for w in ws]))
    proto_tcp = float(np.mean([w[224] for w in ws]))
    proto_udp = float(np.mean([w[225] for w in ws]))
    proto_icmp = float(np.mean([w[226] for w in ws]))
    return np.concatenate([
        means35, stds35, maxs35,
        [np.log1p(num_flows), proto_tcp, proto_udp, proto_icmp, port_ent],
    ]).astype(np.float32)


def calibrate_shnv01_hidden_thresh(bundle: dict, val_states: np.ndarray, val_bins: np.ndarray) -> float:
    model = bundle["model"]
    lookback = bundle["lookback"]
    mean, std = bundle["scaler_mean"], bundle["scaler_std"]
    norm = (val_states - mean) / std
    keys = []
    labels = []
    for t in range(lookback - 1, len(norm) - 1):
        seq = norm[t - lookback + 1 : t + 1]
        out = _forward(model, seq)
        keys.append(out["hidden"])
        labels.append(int(val_bins[t]))
    if len(keys) < 10:
        return 30.0
    return calibrate_match_thresh(np.stack(keys), np.array(labels))


# Backwards-compatible aliases (older scripts used "friend_*" names)
StreamingFriendPRISM = StreamingSHNV01
load_friend_bundle = load_shnv01_bundle
calibrate_friend_hidden_thresh = calibrate_shnv01_hidden_thresh
