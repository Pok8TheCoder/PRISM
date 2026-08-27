"""Adversarial training loop orchestrator for the isolated PRISM Docker lab.

Architecture:
  - attacker-bot -> target-server over internal DNS (prism-lab network)
  - target-server runs tcpdump; host copies PCAPs for GPU inference
  - benign-client generates background traffic on the same network

Loop:
  1. Ensure isolated lab is running (docker compose, internal network)
  2. Start tcpdump on target-server
  3. Run catalog attack bot inside attacker-bot
  4. Multi-class world model scores traffic
  5. On detection -> escalate evasion; on evasion/misclass -> save to missed/
  6. Retrain on accumulated missed samples
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.adversarial.lab_config import (
    ATTACKER_CONTAINER,
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEFAULT_ATTACK_CLASSES,
    DETECTION_THRESHOLD,
    EPOCHS_PER_CYCLE,
    MAX_EVASION_ATTEMPTS,
    MISSED_DIR,
    RETRAIN_BATCHES,
    SAVE_DIR,
    TARGET_HOST,
)
from src.adversarial.lab_manager import ensure_lab_running, lab_status, verify_lab_connectivity
from src.adversarial.traffic_capture import TrafficCapture
from src.model.attack_catalog import get_evasion_chains, get_mitre_map
from src.model.world_model_multiclass import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    MultiClassWorldModel,
    NUM_FEATURES,
    SEQ_LEN,
    pcap_to_rows,
    predict_live,
    rows_to_matrix,
)


def run_attack_in_container(class_id: str, evasion_name: str) -> dict:
    """Run attack script inside the attacker-bot Docker container."""
    cmd = [
        "docker", "exec", ATTACKER_CONTAINER,
        "python3", "-m", "src.adversarial.attack_script",
        TARGET_HOST, class_id, evasion_name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = (result.stdout or "") + (result.stderr or "")
        flows = 0
        resolved_class = class_id
        for line in output.splitlines():
            if line.startswith("DONE|"):
                parts = dict(p.split("=", 1) for p in line.split("|")[1:] if "=" in p)
                flows = int(parts.get("flows", 0))
                resolved_class = parts.get("class", class_id)
        return {
            "flows": flows,
            "class_id": resolved_class,
            "success": result.returncode == 0 and flows >= 0,
            "output": output.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"flows": 0, "class_id": class_id, "success": False, "error": "timeout"}
    except Exception as exc:
        return {"flows": 0, "class_id": class_id, "success": False, "error": str(exc)}


def load_model_and_scaler(device: torch.device):
    model = MultiClassWorldModel().to(device)
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(NUM_FEATURES, dtype=np.float64)
    scaler.scale_ = np.ones(NUM_FEATURES, dtype=np.float64)

    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        state = ckpt["model"]
        model_state = model.state_dict()
        compatible = (
            "input_proj.weight" in state
            and "classify_head.weight" in state
            and state["input_proj.weight"].shape == model_state["input_proj.weight"].shape
            and state["classify_head.weight"].shape == model_state["classify_head.weight"].shape
        )
        if compatible:
            model.load_state_dict(state)
            mean = np.array(ckpt["scaler_mean"], dtype=np.float64)
            scale = np.array(ckpt["scaler_std"], dtype=np.float64)
            if len(mean) == NUM_FEATURES:
                scaler.mean_ = mean
                scaler.scale_ = scale
            print(f"Loaded multi-class checkpoint: {CHECKPOINT_PATH}")
        else:
            print(
                f"WARNING: Checkpoint shape mismatch "
                f"(features={model.input_proj.in_features}, "
                f"classes={model.classify_head.out_features}). "
                "Using random weights — retrain with "
                "`venv\\Scripts\\python.exe src/model/world_model_multiclass.py`."
            )
    else:
        print(f"WARNING: No checkpoint at {CHECKPOINT_PATH}. Using random weights.")

    return model, scaler


def predict_attack(model, scaler, features: np.ndarray, device: torch.device):
    pred_idx, probs, _ = predict_live(model, scaler, features, device)
    benign_idx = CLASS_TO_IDX["Benign"]
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    attack_prob = float(1.0 - probs[benign_idx])
    detected = pred_idx != benign_idx and confidence >= DETECTION_THRESHOLD
    return {
        "pred_idx": pred_idx,
        "pred_class": pred_class,
        "confidence": confidence,
        "attack_prob": attack_prob,
        "detected": detected,
        "probs": probs,
    }


def save_missed_sample(
    pcap_path: Path,
    features: np.ndarray,
    true_class_id: str,
    prediction: dict,
    reason: str,
    evasion: str,
    round_num: int,
    attempt: int,
) -> Path:
    """Persist PCAP + features for dashboard retrain (Step 3)."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = MISSED_DIR / true_class_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"r{round_num}_a{attempt}_{true_class_id}_{evasion}_{ts}"
    dst_pcap = out_dir / f"{stem}.pcap"
    shutil.copy2(pcap_path, dst_pcap)

    np.save(out_dir / f"{stem}.npy", features)

    meta = {
        "timestamp": ts,
        "true_class_id": true_class_id,
        "predicted_class": prediction["pred_class"],
        "confidence": prediction["confidence"],
        "attack_prob": prediction["attack_prob"],
        "reason": reason,
        "evasion": evasion,
        "round": round_num,
        "attempt": attempt,
        "pcap": dst_pcap.name,
        "features": f"{stem}.npy",
        "class_probs": {
            CLASS_NAMES[i]: float(prediction["probs"][i])
            for i in range(len(CLASS_NAMES))
        },
    }
    meta_path = out_dir / f"{stem}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved missed sample -> {meta_path}")
    return meta_path


def retrain_model(
    model,
    scaler: StandardScaler,
    feature_batches: list[np.ndarray],
    label_indices: list[int],
    device: torch.device,
    epochs: int = EPOCHS_PER_CYCLE,
):
    sequences = []
    labels = []

    for feats, class_idx in zip(feature_batches, label_indices):
        if len(feats) < SEQ_LEN:
            continue
        norm = ((feats - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)
        for i in range(len(norm) - SEQ_LEN):
            sequences.append(norm[i:i + SEQ_LEN])
            labels.append(class_idx)

    if not sequences:
        print("  Not enough sequence data for retraining. Skipping.")
        return

    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n = X_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            _, logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"  Retrain Epoch {epoch + 1}/{epochs} | Loss: {total_loss / n:.4f}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.scale_,
        },
        CHECKPOINT_PATH,
    )
    print(f"  Updated checkpoint -> {CHECKPOINT_PATH}")


def run_adversarial_loop(num_rounds: int = 20, attack_classes: list[str] | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = attack_classes or DEFAULT_ATTACK_CLASSES
    evasion_chains = get_evasion_chains()
    mitre_map = get_mitre_map()

    print(f"Adversarial Training Loop | Device: {device}")
    print(f"Target host: {TARGET_HOST} (isolated prism-lab DNS)")
    print(f"Attack rotation: {len(classes)} catalog classes")
    print("=" * 70)

    if not ensure_lab_running():
        print("ERROR: Could not start Docker lab. Is Docker Desktop running?")
        return

    status = lab_status()
    print("Container status:", status)
    if not verify_lab_connectivity():
        print("ERROR: Lab connectivity verification failed.")
        return

    model, scaler = load_model_and_scaler(device)
    capture = TrafficCapture()

    missed_features: list[np.ndarray] = []
    missed_labels: list[int] = []
    stats = {
        "total_attacks": 0,
        "detected": 0,
        "evaded": 0,
        "misclassified": 0,
        "retrain_cycles": 0,
        "missed_saved": 0,
    }

    for round_num in range(1, num_rounds + 1):
        class_id = classes[(round_num - 1) % len(classes)]
        chain = evasion_chains.get(class_id, [])
        evasion = "none"
        evasion_idx = 0
        attempt = 0

        print(f"\n--- Round {round_num}/{num_rounds} | Class: {class_id} ---")

        while attempt <= MAX_EVASION_ATTEMPTS:
            attempt += 1
            stats["total_attacks"] += 1
            pcap_name = f"r{round_num}_a{attempt}_{class_id}_{evasion}.pcap"

            print(f"  Attempt {attempt} | Evasion: {evasion}")

            capture.start_capture(pcap_name)
            attack_result = run_attack_in_container(class_id, evasion)
            print(f"  Attack result: {attack_result}")

            time.sleep(1)
            pcap_path = capture.stop_capture()

            if pcap_path is None or not pcap_path.exists() or pcap_path.stat().st_size == 0:
                print("  No traffic captured. Moving to next class.")
                break

            print(f"  Captured: {pcap_path.name} ({pcap_path.stat().st_size / 1024:.1f} KB)")

            rows = pcap_to_rows(pcap_path)
            features = rows_to_matrix(rows)
            if len(features) == 0:
                print("  No flows extracted. Moving to next class.")
                break

            print(f"  Extracted {len(features)} flows")

            prediction = predict_attack(model, scaler, features, device)
            true_idx = CLASS_TO_IDX.get(class_id)
            if true_idx is None:
                print(f"  WARNING: {class_id} not in CLASS_TO_IDX; skipping label checks.")
                true_idx = prediction["pred_idx"]

            tactic, mitre_label = mitre_map.get(prediction["pred_class"], ("", ""))
            print(
                f"  Prediction: {prediction['pred_class']} "
                f"(conf={prediction['confidence']:.3f}, attack_prob={prediction['attack_prob']:.3f}) "
                f"-> {'DETECTED' if prediction['detected'] else 'EVADED'}"
            )
            print(f"  MITRE: {tactic} / {mitre_label}")

            correct_class = prediction["pred_class"] == class_id

            if prediction["detected"] and correct_class:
                stats["detected"] += 1
                if evasion_idx >= len(chain):
                    print("  No more evasion tactics. Model wins this round.")
                    break
                evasion = chain[evasion_idx]
                evasion_idx += 1
                print(f"  Trying next evasion: {evasion}")
                continue

            if not prediction["detected"]:
                stats["evaded"] += 1
                reason = "evaded"
            else:
                stats["misclassified"] += 1
                reason = "misclassified"

            save_missed_sample(
                pcap_path, features, class_id, prediction, reason, evasion, round_num, attempt,
            )
            stats["missed_saved"] += 1
            missed_features.append(features)
            missed_labels.append(true_idx)

            if len(missed_features) >= RETRAIN_BATCHES:
                print("  Retraining on missed/evaded samples...")
                retrain_model(model, scaler, missed_features, missed_labels, device)
                missed_features = []
                missed_labels = []
                stats["retrain_cycles"] += 1
            break

    print("\n" + "=" * 70)
    print("Adversarial Training Complete")
    print(f"  Total Attacks:     {stats['total_attacks']}")
    print(f"  Detected (correct):{stats['detected']}")
    print(f"  Evaded:            {stats['evaded']}")
    print(f"  Misclassified:     {stats['misclassified']}")
    print(f"  Missed saved:      {stats['missed_saved']}")
    print(f"  Retrain cycles:    {stats['retrain_cycles']}")
    total = max(stats["total_attacks"], 1)
    print(f"  Detection rate:    {stats['detected'] / total * 100:.1f}%")

    stats_path = SAVE_DIR / "adversarial_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved -> {stats_path}")


if __name__ == "__main__":
    run_adversarial_loop(num_rounds=20)
