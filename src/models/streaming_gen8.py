"""Streaming adapter for PRISM Gen 8 Decoupled World Model.

Enables real-time sliding-window inference, continuous state dynamics forecasting,
threat infiltration detection, and MITRE ATT&CK stage disambiguation in live lab
and IPS red-team environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pickle

import numpy as np
import torch
import torch.nn as nn

from src.models.gen8_world_model import Gen8DecoupledMLPWorldModel, NUM_MITRE_STAGES

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CKPT = ROOT / "weights" / "universal_gen8" / "world_model_best.pt"
DEFAULT_SCALER = ROOT / "weights" / "universal_gen8_5s_scaler.pkl"

MITRE_STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
    "Impact",
]


def load_gen8_checkpoint(
    ckpt_path: Optional[Path | str] = None,
    device: str | torch.device = "cpu",
) -> Gen8DecoupledMLPWorldModel:
    """Load trained PRISM Gen 8 World Model weights."""
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not path.is_file():
        raise FileNotFoundError(f"Gen 8 checkpoint not found at: {path}")

    model = Gen8DecoupledMLPWorldModel(
        d_state=242,
        d_model=256,
        n_layers=4,
        n_heads=8,
        lookback=20,
        num_mitre_classes=NUM_MITRE_STAGES,
        mlp_hidden=512,
        residual_dynamics=True,
    )
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class StreamingGen8WorldModel:
    """Stateful streaming interface for PRISM Gen 8 Decoupled World Model.

    Maintains a temporal buffer of state vectors (lookback=20), performing online
    threat detection, next-state dynamics prediction, and autoregressive forecasting.
    """

    def __init__(
        self,
        base_model: Optional[Gen8DecoupledMLPWorldModel] = None,
        ckpt_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
        device: str | torch.device = "cpu",
        context_len: int = 20,
        detect_thresh: float = 0.5,
    ):
        self.device = torch.device(device)
        self.context_len = context_len
        self.detect_thresh = detect_thresh

        if base_model is not None:
            self.model = base_model.to(self.device)
            self.model.eval()
        else:
            self.model = load_gen8_checkpoint(ckpt_path, device=self.device)

        self.scaler = None
        s_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER
        if s_path.is_file():
            try:
                with open(s_path, "rb") as f:
                    self.scaler = pickle.load(f)
            except Exception:
                self.scaler = None

        self.buffer: list[np.ndarray] = []
        self.n_steps: int = 0
        self.latest_prediction: Optional[dict[str, Any]] = None

    def reset(self) -> None:
        """Clear temporal buffer and reset step counter."""
        self.buffer.clear()
        self.n_steps = 0
        self.latest_prediction = None

    def _get_context_tensor(self) -> torch.Tensor:
        """Extract padded/windowed sequence of shape (1, context_len, 242)."""
        if not self.buffer:
            raise ValueError("Cannot construct context tensor from empty buffer.")

        arr = np.array(self.buffer, dtype=np.float32)
        if len(arr) < self.context_len:
            pad_count = self.context_len - len(arr)
            pad = np.repeat(arr[:1], pad_count, axis=0)
            arr = np.concatenate([pad, arr], axis=0)
        else:
            arr = arr[-self.context_len :]

        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def step(
        self,
        state: np.ndarray,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
    ) -> dict[str, Any]:
        """Process one incoming telemetry window.

        Parameters
        ----------
        state : np.ndarray
            Current network state vector of shape (242,).
        true_bin : Optional[int]
            Ground-truth binary label if known.
        true_mit : Optional[int]
            Ground-truth MITRE stage index if known.

        Returns
        -------
        dict[str, Any]
            Inference outputs including p_att, p_mit, pred_state, hidden, and mitre_label.
        """
        raw_state = np.asarray(state, dtype=np.float32).ravel()
        if len(raw_state) < 242:
            raw_state = np.pad(raw_state, (0, 242 - len(raw_state)))
        elif len(raw_state) > 242:
            raw_state = raw_state[:242]

        self.buffer.append(raw_state)
        if len(self.buffer) > self.context_len * 2:
            self.buffer.pop(0)
        self.n_steps += 1

        seq_tensor = self._get_context_tensor()
        out = self.model(seq_tensor)

        p_att = float(torch.softmax(out["pred_binary"], dim=-1)[0, 1].item())
        p_mit = torch.softmax(out["pred_mitre"], dim=-1)[0].cpu().numpy()
        pred_state = out["pred_state_mean"][0].cpu().numpy()
        hidden = out["latent_h"][0].cpu().numpy()
        contrastive_z = out["contrastive_z"][0].cpu().numpy()
        mit_stage = int(np.argmax(p_mit))
        mit_label = MITRE_STAGE_NAMES[mit_stage] if mit_stage < len(MITRE_STAGE_NAMES) else f"Stage_{mit_stage}"

        result = {
            "p_att": p_att,
            "p_mit": p_mit,
            "pred_state": pred_state,
            "hidden": hidden,
            "contrastive_z": contrastive_z,
            "mitre_stage": mit_stage,
            "mitre_label": mit_label,
            "is_threat": bool(p_att >= self.detect_thresh),
            "step": self.n_steps,
        }
        self.latest_prediction = result
        return result

    @torch.no_grad()
    def forecast(self, horizon: int = 20) -> dict[str, Any]:
        """Autoregressively roll forward world model dynamics from current context.

        Parameters
        ----------
        horizon : int
            Number of future 5-second steps to forecast.

        Returns
        -------
        dict[str, Any]
            Dictionary containing forecasted states (H, 242), p_att list, and p_mit array (H, 7).
        """
        if not self.buffer:
            raise ValueError("Cannot forecast without prior state history in buffer.")

        sim_buffer = list(self.buffer[-self.context_len :])
        while len(sim_buffer) < self.context_len:
            sim_buffer.insert(0, sim_buffer[0])

        future_states: list[np.ndarray] = []
        future_p_att: list[float] = []
        future_p_mit: list[np.ndarray] = []

        for _ in range(horizon):
            cur_seq = np.array(sim_buffer[-self.context_len :], dtype=np.float32)
            seq_t = torch.from_numpy(cur_seq).unsqueeze(0).to(self.device)
            out = self.model(seq_t)

            next_s = out["pred_state_mean"][0].cpu().numpy()
            p_a = float(torch.softmax(out["pred_binary"], dim=-1)[0, 1].item())
            p_m = torch.softmax(out["pred_mitre"], dim=-1)[0].cpu().numpy()

            future_states.append(next_s)
            future_p_att.append(p_a)
            future_p_mit.append(p_m)
            sim_buffer.append(next_s)

        return {
            "states": np.stack(future_states, axis=0),
            "p_att": future_p_att,
            "p_mit": np.stack(future_p_mit, axis=0),
            "horizon": horizon,
        }
