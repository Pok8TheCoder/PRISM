"""StreamingHX: 5s Shaun core + RAMX-HX + optional 15s confirm + technique abstain."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
SHAUN_ROOT = ROOT.parent / "PRISM-shaun"
HX_CKPT = ROOT.parent / "Automode" / "train" / "checkpoints" / "hx_w5s.pt"
W5S_CKPT = SHAUN_ROOT / "weights" / "w5s" / "world_model.pt"
W15S_CKPT = SHAUN_ROOT / "weights" / "world_model.pt"
LOOKBACK = 30
SHAUN_DIM = 292
TECH_ABSTAIN = 0.7
DETECT_THRESHOLD = 0.5

from src.hx.heads import HXCore  # noqa: E402
from src.hx.ramx_hx import RAMXHXPredictor  # noqa: E402
from src.aryan.constants import MITRE_STAGES_INV, TACTIC_TO_STAGE  # noqa: E402
from src.model.attack_catalog import get_class_names, get_mitre_map  # noqa: E402


def _padded_raw(buf: list[np.ndarray]) -> np.ndarray:
    window = buf[-LOOKBACK:]
    if len(window) < LOOKBACK:
        window = [window[0]] * (LOOKBACK - len(window)) + window
    return np.stack(window)


def _load_shaun_class():
    shaun = SHAUN_ROOT.resolve()
    if not shaun.exists():
        raise FileNotFoundError(f"PRISM-shaun not found: {shaun}")
    prev_cwd = os.getcwd()
    prev_path = sys.path.copy()
    os.chdir(shaun)
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path = [str(shaun)] + [p for p in sys.path if Path(p).resolve() != ROOT.resolve()]
    try:
        from src.models.world_model import StateTransformerWorldModel
        return StateTransformerWorldModel
    finally:
        os.chdir(prev_cwd)
        sys.path = prev_path
        sys.path.insert(0, str(ROOT))


def _load_shaun_ckpt(ckpt_path: Path) -> dict[str, Any]:
    StateTransformerWorldModel = _load_shaun_class()
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
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
    return {"model": model, "mean": mean, "std": std, "d_state": d_state}


def load_hx_bundle(
    *,
    w5s_ckpt: Path | None = None,
    w15s_ckpt: Path | None = None,
    hx_ckpt: Path | None = None,
) -> dict[str, Any]:
    w5 = Path(w5s_ckpt or W5S_CKPT)
    if not w5.exists():
        raise FileNotFoundError(f"Missing Shaun w5s checkpoint: {w5}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    five = _load_shaun_ckpt(w5)
    names = get_class_names()
    core = HXCore(five["model"], n_tech=len(names))
    hx_path = Path(hx_ckpt or HX_CKPT)
    if hx_path.exists():
        extra = torch.load(str(hx_path), map_location="cpu", weights_only=False)
        if "technique_head" in extra:
            core.technique_head.load_state_dict(extra["technique_head"])
            core.technique_trained = True
        if "shaun_state_dict" in extra:
            core.shaun.load_state_dict(extra["shaun_state_dict"], strict=False)
        if "scaler_mean" in extra:
            five["mean"] = np.asarray(extra["scaler_mean"], dtype=np.float32)
            five["std"] = np.asarray(extra["scaler_std"], dtype=np.float32)
        names = extra.get("class_names", names)
    core = core.to(device)
    core.eval()

    confirm = None
    w15 = Path(w15s_ckpt or W15S_CKPT)
    if w15.exists():
        confirm = _load_shaun_ckpt(w15)
        confirm["model"] = confirm["model"].to(device)

    return {
        "core": core,
        "mean": five["mean"],
        "std": five["std"],
        "device": device,
        "class_names": names,
        "confirm": confirm,
        "hx_ckpt": hx_path if hx_path.exists() else None,
    }


def _forward_core(core: HXCore, mean, std, buf: list[np.ndarray], device) -> dict[str, Any]:
    m, s = mean.reshape(-1), std.reshape(-1)
    seq = ((_padded_raw(buf) - m) / s).astype(np.float32)
    x = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = core(x)
        p_att = float(torch.softmax(out["attack_logits"], dim=-1)[0, 1].item())
        mitre = torch.softmax(out["mitre_logits"], dim=-1)[0].cpu().numpy()
        tech = torch.softmax(out["tech_logits"], dim=-1)[0].cpu().numpy()
        hidden = out["hidden"][0].detach().cpu().numpy()
    return {
        "p_att": p_att,
        "mitre_probs": mitre,
        "tech_probs": tech,
        "hidden": hidden,
        "traj": seq,
    }


def _forward_shaun_attack(model, mean, std, states: list[np.ndarray], device) -> float:
    m, s = mean.reshape(-1), std.reshape(-1)
    seq = ((_padded_raw(states) - m) / s).astype(np.float32)
    x = torch.from_numpy(seq).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.input_embed(x)
        emb = model.pos_encoder(emb)
        for layer in model.layers:
            emb, _ = layer(emb)
        h = model.layer_norm(emb)[:, -1, :]
        logits = model.attack_head(h)
    return float(torch.softmax(logits, dim=-1)[0, 1].item())


class StreamingHX:
    """Attempt-1 live/PCAP scorer. Ignores true_bin (no oracle)."""

    system_id = "hx"

    def __init__(
        self,
        bundle: dict[str, Any] | None = None,
        context_skip_steps: int = 0,
        enable_confirm: bool = True,
    ):
        bundle = bundle or load_hx_bundle()
        self.core: HXCore = bundle["core"]
        self.mean = bundle["mean"]
        self.std = bundle["std"]
        self.device = bundle["device"]
        self.class_names: list[str] = list(bundle["class_names"])
        self.confirm_bundle = bundle.get("confirm") if enable_confirm else None
        self.mitre_map = get_mitre_map()
        self.ramx = RAMXHXPredictor(context_skip_steps=context_skip_steps)
        self.buf: list[np.ndarray] = []
        self.prev_fused = 0.0

    def set_context_skip(self, steps: int) -> None:
        self.ramx.set_context_skip(steps)

    def begin_live_phase(self) -> None:
        self.ramx.begin_live_phase()

    def _confirm_p(self) -> float | None:
        if self.confirm_bundle is None or len(self.buf) < 3:
            return None
        # Proxy 15s state: mean of last three 5s windows.
        proxy = [np.mean(np.stack(self.buf[-3:]), axis=0).astype(np.float32)]
        cb = self.confirm_bundle
        return _forward_shaun_attack(cb["model"], cb["mean"], cb["std"], proxy, self.device)

    def _emit_technique(self, tech_probs: np.ndarray, confirmed: bool) -> tuple[str | None, float]:
        if not self.core.technique_trained or not confirmed:
            return None, 0.0
        idx = int(np.argmax(tech_probs))
        p = float(tech_probs[idx])
        name = self.class_names[idx] if idx < len(self.class_names) else "Benign"
        if name == "Benign" or p < TECH_ABSTAIN:
            return None, p
        return name, p

    def step(self, raw_state: np.ndarray, true_bin: int | None = None, true_mit: int | None = None) -> dict:
        del true_bin, true_mit
        raw = np.asarray(raw_state, dtype=np.float32)
        self.buf.append(raw)
        fwd = _forward_core(self.core, self.mean, self.std, self.buf, self.device)
        raw_p = fwd["p_att"]
        stage = int(np.argmax(fwd["mitre_probs"]))

        self.ramx.step_count += 1
        if self.ramx.in_live_phase:
            self.ramx.live_step_count += 1
        self.ramx.calibrator.update(raw)
        rel = self.ramx.calibrator.get_relative_anomaly_score(raw)
        calibrated_raw = self.ramx.raw_calibrator.adjust(raw_p)
        self.ramx.raw_calibrator.update(raw_p)

        in_gate = self.ramx.step_count <= self.ramx.context_skip_steps
        in_cd = self.ramx.cooldown_left > 0
        if in_cd:
            self.ramx.cooldown_left -= 1

        fused, suppressed = self.ramx.fuse(
            raw_p=calibrated_raw,
            relative_anomaly=rel,
            in_context_gate=in_gate,
            in_cooldown=in_cd,
        )

        confirm_p = self._confirm_p()
        consecutive_ok = self.prev_fused >= DETECT_THRESHOLD
        proposed = fused >= DETECT_THRESHOLD
        if confirm_p is not None:
            confirmed = proposed and confirm_p >= DETECT_THRESHOLD
        else:
            confirmed = proposed and consecutive_ok

        if proposed and not confirmed:
            fused = min(fused, self.ramx.alert_cap_during_gate)
            confirmed = False

        traj = fwd["traj"]
        fused = self.ramx.maybe_memory(
            traj, fused, confirmed, stage, in_gate, in_cd,
        )
        if fused >= DETECT_THRESHOLD and confirmed and stage == 0:
            stage = 1  # at least recon if binary fires
        self.ramx.after_alert(confirmed, fused)
        self.prev_fused = fused

        tech, tech_p = self._emit_technique(fwd["tech_probs"], confirmed and fused >= DETECT_THRESHOLD)
        stage_name = MITRE_STAGES_INV.get(stage, "Benign")
        if tech:
            tactic = self.mitre_map.get(tech, ("", ""))[0]
            mapped = TACTIC_TO_STAGE.get(tactic)
            if mapped:
                stage_name = mapped

        return {
            "p_att": float(fused),
            "raw_p_att": float(raw_p),
            "calibrated_raw_p_att": float(calibrated_raw),
            "relative_anomaly": float(rel),
            "confirm_p": confirm_p,
            "confirmed": bool(confirmed and fused >= DETECT_THRESHOLD),
            "suspect": bool(proposed and not confirmed),
            "context_gated": in_gate,
            "alert_suppressed": suppressed or in_cd,
            "p_mit": fwd["mitre_probs"].tolist(),
            "mitre_stage": stage,
            "mitre_stage_name": stage_name,
            "technique": tech,
            "technique_p": tech_p,
            "ramx_version": "hx-1.0",
        }
