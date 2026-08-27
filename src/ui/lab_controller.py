"""Backend controller for the PRISM Streamlit dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.adversarial.lab_config import (
    CHECKPOINT_PATH,
    DEFAULT_ATTACK_CLASSES,
    DETECTION_THRESHOLD,
    EPOCHS_PER_CYCLE,
    MISSED_DIR,
    ROOT,
    SAVE_DIR,
)
from src.adversarial.lab_manager import ensure_lab_running, lab_status, verify_lab_connectivity
from src.adversarial.traffic_capture import TrafficCapture
from src.adversarial.training_loop import (
    load_model_and_scaler,
    predict_attack,
    retrain_model,
    run_attack_in_container,
    save_missed_sample,
)
from src.model.attack_catalog import (
    get_bot_class_ids,
    get_evasion_chains,
    get_mitre_map,
    load_catalog,
)
from src.model.world_model_multiclass import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    FEATURE_COLS,
    SEQ_LEN,
    pcap_to_rows,
    rows_to_matrix,
)

K_ROLLOUT = 5
MITRE_TACTICS = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command And Control",
    "Exfiltration",
    "Impact",
]


@dataclass
class AttackRunResult:
    success: bool
    class_id: str
    evasion: str
    flows_generated: int
    pcap_path: str | None
    pcap_bytes: int
    flow_count: int
    features: np.ndarray | None
    prediction: dict | None
    attack_result: dict
    elapsed_sec: float
    message: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = None
        if self.prediction and "probs" in self.prediction:
            d["prediction"] = {
                k: v for k, v in self.prediction.items() if k != "probs"
            }
            d["prediction"]["class_probs"] = {
                CLASS_NAMES[i]: float(self.prediction["probs"][i])
                for i in range(len(CLASS_NAMES))
            }
        return d


@dataclass
class LabSnapshot:
    containers: dict[str, bool]
    all_running: bool
    checkpoint_exists: bool
    checkpoint_compatible: bool
    device: str
    gpu_name: str | None
    missed_count: int
    capture_count: int


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_name() -> str | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


def checkpoint_compatible() -> bool:
    if not CHECKPOINT_PATH.exists():
        return False
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state = ckpt.get("model", {})
        w = state.get("classify_head.weight")
        inp = state.get("input_proj.weight")
        from src.pipeline.features import NUM_FEATURES
        return (
            w is not None
            and inp is not None
            and w.shape[0] == len(CLASS_NAMES)
            and inp.shape[1] == NUM_FEATURES
        )
    except Exception:
        return False


def get_lab_snapshot() -> LabSnapshot:
    containers = lab_status()
    missed = sum(1 for _ in MISSED_DIR.rglob("*.json")) if MISSED_DIR.exists() else 0
    captures = len(list(SAVE_DIR.glob("*.pcap"))) if SAVE_DIR.exists() else 0
    return LabSnapshot(
        containers=containers,
        all_running=all(containers.values()),
        checkpoint_exists=CHECKPOINT_PATH.exists(),
        checkpoint_compatible=checkpoint_compatible(),
        device=str(get_device()),
        gpu_name=get_gpu_name(),
        missed_count=missed,
        capture_count=captures,
    )


def start_lab() -> tuple[bool, str]:
    ok = ensure_lab_running()
    if not ok:
        return False, "Failed to start Docker lab. Is Docker Desktop running?"
    if not verify_lab_connectivity():
        return False, "Lab started but connectivity probe failed."
    return True, "Lab online — isolated prism-lab network active."


def stop_lab() -> tuple[bool, str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "lab_ctl.py"), "down"]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr or "Failed to stop lab."
    return True, "Lab stopped."


def run_attack_session(
    class_id: str,
    evasion: str = "none",
    round_num: int = 1,
    attempt: int = 1,
    model=None,
    scaler=None,
    device: torch.device | None = None,
) -> AttackRunResult:
    """Capture traffic, run bot, score with world model."""
    t0 = time.time()
    pcap_name = f"ui_r{round_num}_a{attempt}_{class_id}_{evasion}.pcap"
    capture = TrafficCapture()

    capture.start_capture(pcap_name)
    attack_result = run_attack_in_container(class_id, evasion)
    time.sleep(1)
    pcap_path = capture.stop_capture()

    if pcap_path is None or not pcap_path.exists() or pcap_path.stat().st_size == 0:
        return AttackRunResult(
            success=False,
            class_id=class_id,
            evasion=evasion,
            flows_generated=attack_result.get("flows", 0),
            pcap_path=None,
            pcap_bytes=0,
            flow_count=0,
            features=None,
            prediction=None,
            attack_result=attack_result,
            elapsed_sec=time.time() - t0,
            message="No traffic captured. Check target-server tcpdump.",
        )

    rows = pcap_to_rows(pcap_path)
    features = rows_to_matrix(rows)
    if len(features) == 0:
        return AttackRunResult(
            success=False,
            class_id=class_id,
            evasion=evasion,
            flows_generated=attack_result.get("flows", 0),
            pcap_path=str(pcap_path),
            pcap_bytes=pcap_path.stat().st_size,
            flow_count=0,
            features=None,
            prediction=None,
            attack_result=attack_result,
            elapsed_sec=time.time() - t0,
            message="PCAP captured but no flows extracted.",
        )

    prediction = None
    if model is not None and scaler is not None and device is not None:
        prediction = predict_attack(model, scaler, features, device)

    return AttackRunResult(
        success=True,
        class_id=class_id,
        evasion=evasion,
        flows_generated=attack_result.get("flows", 0),
        pcap_path=str(pcap_path),
        pcap_bytes=pcap_path.stat().st_size,
        flow_count=len(features),
        features=features,
        prediction=prediction,
        attack_result=attack_result,
        elapsed_sec=time.time() - t0,
        message="Attack session complete.",
    )


def run_evasion_chain(
    class_id: str,
    model,
    scaler,
    device: torch.device,
    max_attempts: int = 4,
) -> list[AttackRunResult]:
    """Run attack with automatic evasion escalation when correctly detected."""
    chains = get_evasion_chains()
    chain = chains.get(class_id, [])
    evasion = "none"
    evasion_idx = 0
    results: list[AttackRunResult] = []

    for attempt in range(1, max_attempts + 2):
        result = run_attack_session(
            class_id, evasion, round_num=1, attempt=attempt,
            model=model, scaler=scaler, device=device,
        )
        results.append(result)
        if not result.success or result.prediction is None:
            break

        correct = result.prediction["pred_class"] == class_id
        if result.prediction["detected"] and correct:
            if evasion_idx >= len(chain):
                break
            evasion = chain[evasion_idx]
            evasion_idx += 1
            continue
        break

    return results


def forecast_attack_probability(
    model,
    scaler,
    features: np.ndarray,
    device: torch.device,
    k: int = K_ROLLOUT,
) -> list[float]:
    """K-step autoregressive rollout of attack probability."""
    if len(features) == 0:
        return [0.0] * k

    window = features.copy()
    if len(window) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(window), window.shape[1]), dtype=np.float32)
        window = np.vstack([pad, window])
    window = window[-SEQ_LEN:]
    norm = ((window - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)

    benign_idx = CLASS_TO_IDX["Benign"]
    probs_out: list[float] = []

    current = norm.copy()
    model.eval()
    with torch.no_grad():
        for _ in range(k):
            x = torch.from_numpy(current).unsqueeze(0).to(device)
            next_state, logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            probs_out.append(float(1.0 - probs[benign_idx]))
            next_row = next_state.cpu().numpy()[0]
            current = np.vstack([current[1:], next_row.reshape(1, -1)])

    return probs_out


def compute_feature_attribution(
    features: np.ndarray,
    scaler,
    top_n: int = 8,
) -> list[tuple[str, float]]:
    """Z-score based attribution proxy for the latest flow window."""
    if len(features) == 0:
        return []
    last = features[-1]
    z = np.abs((last - scaler.mean_) / (scaler.scale_ + 1e-8))
    z = z / (z.sum() + 1e-9)
    order = np.argsort(z)[::-1][:top_n]
    return [(FEATURE_COLS[i], float(z[i])) for i in order]


def aggregate_tactic_probs(probs: np.ndarray) -> dict[str, float]:
    """Roll class probabilities up to MITRE tactic level."""
    mitre_map = get_mitre_map()
    tactic_scores: dict[str, float] = {t: 0.0 for t in MITRE_TACTICS}

    for i, class_name in enumerate(CLASS_NAMES):
        if class_name == "Benign":
            continue
        tactic, _ = mitre_map.get(class_name, ("", ""))
        if tactic in tactic_scores:
            tactic_scores[tactic] = max(tactic_scores[tactic], float(probs[i]))
        elif tactic:
            tactic_scores[tactic] = tactic_scores.get(tactic, 0.0) + float(probs[i])

    return tactic_scores


def threat_level(attack_prob: float) -> str:
    if attack_prob >= 0.85:
        return "CRITICAL"
    if attack_prob >= 0.6:
        return "ELEVATED"
    if attack_prob >= 0.35:
        return "GUARDED"
    return "NORMAL"


def list_missed_samples() -> list[dict]:
    samples = []
    if not MISSED_DIR.exists():
        return samples
    for meta_path in sorted(MISSED_DIR.rglob("*.json"), reverse=True):
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta["_path"] = str(meta_path)
            samples.append(meta)
        except Exception:
            continue
    return samples


def save_result_to_missed(
    result: AttackRunResult,
    reason: str = "manual_save",
    round_num: int = 0,
    attempt: int = 0,
) -> Path | None:
    if not result.success or result.features is None or result.prediction is None:
        return None
    pcap = Path(result.pcap_path)
    return save_missed_sample(
        pcap,
        result.features,
        result.class_id,
        result.prediction,
        reason,
        result.evasion,
        round_num,
        attempt,
    )


def retrain_from_missed(
    model,
    scaler,
    device: torch.device,
    epochs: int = EPOCHS_PER_CYCLE,
) -> tuple[bool, str, int]:
    feature_batches: list[np.ndarray] = []
    label_indices: list[int] = []

    for meta in list_missed_samples():
        class_id = meta.get("true_class_id")
        if class_id not in CLASS_TO_IDX:
            continue
        meta_dir = Path(meta["_path"]).parent
        feat_file = meta_dir / meta.get("features", "")
        if not feat_file.exists():
            continue
        feats = np.load(feat_file)
        if len(feats) >= SEQ_LEN:
            feature_batches.append(feats.astype(np.float32))
            label_indices.append(CLASS_TO_IDX[class_id])

    if not feature_batches:
        return False, "No usable missed samples for retraining.", 0

    retrain_model(model, scaler, feature_batches, label_indices, device, epochs=epochs)
    return True, f"Retrained on {len(feature_batches)} missed samples.", len(feature_batches)


def build_forensic_report(
    timeline: list[dict],
    alerts: list[dict],
    lab: LabSnapshot,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lab": asdict(lab),
        "timeline": timeline,
        "alerts": alerts,
        "missed_samples": list_missed_samples(),
        "stats_path": str(SAVE_DIR / "adversarial_stats.json"),
    }


def get_flow_preview_table(features: np.ndarray, limit: int = 12) -> list[dict]:
    if len(features) == 0:
        return []
    rows = []
    for i, row in enumerate(features[-limit:]):
        rows.append({
            "flow": len(features) - limit + i + 1 if len(features) > limit else i + 1,
            "dst_port": int(row[0]),
            "protocol": int(row[1]),
            "duration_us": float(row[2]),
            "fwd_pkts": int(row[3]),
            "syn_flags": int(row[15]),
            "iat_std": float(row[14]),
        })
    return rows


def get_catalog_summary() -> dict:
    cat = load_catalog()
    return cat.get("summary", {})
