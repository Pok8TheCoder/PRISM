"""Adversarial training loop orchestrator.

Architecture:
  - attacker-bot container (172.17.0.3) runs attack scripts against target
  - target-server container (172.17.0.2) runs tcpdump on eth0
  - Host runs world model inference on GPU + retrains on evasion data

Loop:
  1. Start tcpdump on target-server
  2. Run attack via `docker exec attacker-bot python3 attack_script.py`
  3. Stop tcpdump, copy PCAP to host
  4. Extract flow features from PCAP
  5. World model predicts: attack vs benign
  6. If detected -> bot tries next evasion strategy
  7. If evaded -> traffic labeled and added to training dataset
  8. Retrain model on accumulated evasion data
"""

import torch
import numpy as np
import pandas as pd
import subprocess
import time
import json
from pathlib import Path
from collections import defaultdict

from src.adversarial.traffic_capture import TrafficCapture
from src.adversarial.attacker_bot import AttackStrategy, EvasionTactic, STRATEGY_CHAIN
from src.model.world_model import WorldModelTransformer, FEATURE_COLS, NUM_FEATURES, SEQ_LEN

TARGET_IP = "172.17.0.2"
ATTACKER_CONTAINER = "attacker-bot"
TARGET_CONTAINER = "target-server"
EPOCHS_PER_CYCLE = 10
BATCH_SIZE = 64
MAX_EVASION_ATTEMPTS = 4
DETECTION_THRESHOLD = 0.5
SAVE_DIR = Path("data/raw/adversarial")
CHECKPOINT_PATH = Path("models/checkpoints/world_model_poc.pth")

STRATEGY_TO_SCRIPT = {
    AttackStrategy.SSH_BRUTEFORCE: "ssh_bruteforce",
    AttackStrategy.PORT_SCAN_SEQUENTIAL: "port_scan_sequential",
    AttackStrategy.PORT_SCAN_RANDOM: "port_scan_random",
    AttackStrategy.HTTP_FLOOD: "http_flood",
    AttackStrategy.SLOW_LORIS: "slow_loris",
    AttackStrategy.SYN_SCAN_STEALTH: "syn_scan_stealth",
}


def run_attack_in_container(strategy_name: str, evasion_name: str) -> dict:
    """Run attack script inside the attacker-bot Docker container."""
    cmd = [
        "docker", "exec", ATTACKER_CONTAINER,
        "python3", "/tmp/attack_script.py",
        TARGET_IP, strategy_name, evasion_name,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = result.stdout.strip()
        flows = 0
        for line in output.split("\n"):
            if "DONE|flows=" in line:
                flows = int(line.split("flows=")[1])
        return {"flows": flows, "success": True}
    except subprocess.TimeoutExpired:
        return {"flows": 0, "success": False}
    except Exception as e:
        return {"flows": 0, "success": False, "error": str(e)}


def pcap_to_features(pcap_path: Path) -> np.ndarray:
    """Convert PCAP to flow-level feature matrix using Scapy."""
    from scapy.all import rdpcap, TCP, UDP, IP

    packets = rdpcap(str(pcap_path))
    flows = defaultdict(list)

    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "Other")

        if TCP in pkt:
            l4 = pkt[TCP]
            key = (ip.src, l4.sport, ip.dst, l4.dport, proto)
            flags = l4.flags
        elif UDP in pkt:
            l4 = pkt[UDP]
            key = (ip.src, l4.sport, ip.dst, l4.dport, proto)
            flags = 0
        else:
            key = (ip.src, 0, ip.dst, 0, proto)
            flags = 0

        flows[key].append({
            "time": float(pkt.time),
            "len": len(pkt),
            "proto": proto,
            "sport": key[1],
            "dport": key[3],
            "flags": int(flags) if flags else 0,
        })

    feature_rows = []
    for key, pkts in flows.items():
        pkts.sort(key=lambda p: p["time"])
        times = [p["time"] for p in pkts]
        iats = np.diff(times) if len(times) > 1 else [0]
        lengths = [p["len"] for p in pkts]
        flags_list = [p["flags"] for p in pkts]

        dst_port = pkts[0]["dport"]
        protocol = 6 if pkts[0]["proto"] == "TCP" else 17

        syn_cnt = sum(1 for f in flags_list if f & 0x02)
        fin_cnt = sum(1 for f in flags_list if f & 0x01)
        rst_cnt = sum(1 for f in flags_list if f & 0x04)
        psh_cnt = sum(1 for f in flags_list if f & 0x08)
        ack_cnt = sum(1 for f in flags_list if f & 0x10)

        duration = times[-1] - times[0] if len(times) > 1 else 0

        row = {
            "Dst Port": dst_port,
            "Protocol": protocol,
            "Flow Duration": duration * 1e6,
            "Tot Fwd Pkts": len(pkts),
            "Tot Bwd Pkts": 0,
            "Fwd Pkt Len Max": max(lengths),
            "Fwd Pkt Len Min": min(lengths),
            "Fwd Pkt Len Mean": np.mean(lengths),
            "Bwd Pkt Len Max": 0, "Bwd Pkt Len Min": 0, "Bwd Pkt Len Std": 0,
            "Flow Byts/s": sum(lengths) / (duration + 1e-6),
            "Flow Pkts/s": len(pkts) / (duration + 1e-6),
            "Flow IAT Mean": np.mean(iats) if len(iats) > 0 else 0,
            "Flow IAT Std": np.std(iats) if len(iats) > 0 else 0,
            "SYN Flag Cnt": syn_cnt,
            "FIN Flag Cnt": fin_cnt,
            "RST Flag Cnt": rst_cnt,
            "PSH Flag Cnt": psh_cnt,
            "ACK Flag Cnt": ack_cnt,
        }
        feature_rows.append(row)

    if not feature_rows:
        return np.array([]).reshape(0, NUM_FEATURES)

    df = pd.DataFrame(feature_rows)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
    return df.values.astype(np.float32)


def predict_attack(model, features, device):
    """Run world model on captured features. Returns (probability, predicted_label)."""
    if len(features) < SEQ_LEN:
        padding = np.zeros((SEQ_LEN - len(features), NUM_FEATURES), dtype=np.float32)
        features = np.vstack([padding, features])
    if len(features) > SEQ_LEN:
        features = features[-SEQ_LEN:]

    x = torch.from_numpy(features).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
        probs = torch.softmax(logits, dim=1)
        attack_prob = probs[0, 1].item()

    return attack_prob, attack_prob > DETECTION_THRESHOLD


def retrain_model(model, new_features, device, epochs=EPOCHS_PER_CYCLE):
    """Fine-tune the world model on newly captured evasion traffic."""
    if len(new_features) < SEQ_LEN:
        print("  Not enough data for retraining. Skipping.")
        return

    sequences = []
    for i in range(len(new_features) - SEQ_LEN):
        sequences.append(new_features[i:i + SEQ_LEN])

    if not sequences:
        return

    X = np.array(sequences, dtype=np.float32)
    y = np.ones(len(sequences), dtype=np.int64)

    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    n = X_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            _, logits = model(X_t[idx])
            loss = criterion(logits, y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"  Retrain Epoch {epoch+1}/{epochs} | Loss: {total_loss/n:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"  Updated model saved to {CHECKPOINT_PATH}")


def run_adversarial_loop(num_rounds: int = 20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Adversarial Training Loop | Device: {device}")
    print(f"Target: {TARGET_IP} (container: {TARGET_CONTAINER})")
    print(f"Attacker: {ATTACKER_CONTAINER}")
    print("=" * 70)

    model = WorldModelTransformer().to(device)
    if CHECKPOINT_PATH.exists():
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("Loaded pre-trained world model.")
    else:
        print("WARNING: No pre-trained model found. Using random init.")

    capture = TrafficCapture(container_name=TARGET_CONTAINER)

    strategies = [
        AttackStrategy.SSH_BRUTEFORCE,
        AttackStrategy.PORT_SCAN_SEQUENTIAL,
        AttackStrategy.HTTP_FLOOD,
        AttackStrategy.SYN_SCAN_STEALTH,
    ]

    evasion_data = []
    stats = {"total_attacks": 0, "detected": 0, "evaded": 0, "retrain_cycles": 0}

    for round_num in range(1, num_rounds + 1):
        strategy = strategies[(round_num - 1) % len(strategies)]
        evasion = EvasionTactic.NONE
        evasion_idx = 0
        attempt = 0

        print(f"\n--- Round {round_num}/{num_rounds} | Strategy: {strategy.value} ---")

        while attempt <= MAX_EVASION_ATTEMPTS:
            attempt += 1
            stats["total_attacks"] += 1
            pcap_name = f"r{round_num}_a{attempt}_{strategy.value}_{evasion.value}.pcap"

            print(f"  Attempt {attempt} | Evasion: {evasion.value}")

            # Start capture
            capture.start_capture(pcap_name)

            # Run attack inside attacker container
            script_name = STRATEGY_TO_SCRIPT.get(strategy, "port_scan_sequential")
            attack_result = run_attack_in_container(script_name, evasion.value)
            print(f"  Attack result: {attack_result}")

            # Stop capture and get pcap
            time.sleep(1)
            pcap_path = capture.stop_capture()

            if pcap_path is None or not pcap_path.exists() or pcap_path.stat().st_size == 0:
                print("  No traffic captured. Moving to next strategy.")
                break

            print(f"  Captured: {pcap_path.name} ({pcap_path.stat().st_size / 1024:.1f} KB)")

            features = pcap_to_features(pcap_path)
            if len(features) == 0:
                print("  No flows extracted. Moving to next strategy.")
                break

            print(f"  Extracted {len(features)} flows")

            attack_prob, detected = predict_attack(model, features, device)

            print(f"  Model prediction: attack_prob={attack_prob:.4f} -> {'DETECTED' if detected else 'EVADED'}")

            if detected:
                stats["detected"] += 1
                chain = STRATEGY_CHAIN.get(strategy, [])
                if evasion_idx >= len(chain):
                    print("  No more evasion tactics. Model wins this round.")
                    break
                evasion = chain[evasion_idx]
                evasion_idx += 1
                print(f"  Trying next evasion: {evasion.value}")
            else:
                stats["evaded"] += 1
                print("  EVASION SUCCESSFUL! Adding to training data.")
                evasion_data.append(features)

                if len(evasion_data) >= 3:
                    print("  Retraining model on evasion data...")
                    all_evasion = np.vstack(evasion_data)
                    retrain_model(model, all_evasion, device)
                    evasion_data = []
                    stats["retrain_cycles"] += 1
                break

    print("\n" + "=" * 70)
    print("Adversarial Training Complete")
    print(f"  Total Attacks:   {stats['total_attacks']}")
    print(f"  Detected:        {stats['detected']}")
    print(f"  Evaded:          {stats['evaded']}")
    print(f"  Retrain Cycles:  {stats['retrain_cycles']}")
    print(f"  Detection Rate:  {stats['detected']/max(stats['total_attacks'],1)*100:.1f}%")
    print(f"  Evasion Rate:    {stats['evaded']/max(stats['total_attacks'],1)*100:.1f}%")

    stats_path = SAVE_DIR / "adversarial_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to {stats_path}")


if __name__ == "__main__":
    run_adversarial_loop(num_rounds=20)
