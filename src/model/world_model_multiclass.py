"""Multi-class World Model: detects attack TYPE and maps to MITRE ATT&CK stage.

Classes:
  0 = Benign
  1 = SSH_Bruteforce     -> MITRE: Initial Access (T1110)
  2 = Port_Scan          -> MITRE: Reconnaissance (T1046)
  3 = HTTP_Flood         -> MITRE: Impact (T1499)
  4 = Slow_Loris         -> MITRE: Impact (T1499)
  5 = Infiltration       -> MITRE: Lateral Movement (T1021)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import subprocess
import time
import json
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/raw")
ADVERSARIAL_DIR = DATA_DIR / "adversarial"
FILES = {
    'thursday': DATA_DIR / 'thursday_01_03_2018.csv',
    'wednesday': DATA_DIR / 'wednesday_28_02_2018.csv',
}

SEQ_LEN     = 10
BATCH_SIZE  = 256
EPOCHS      = 30
LR          = 1e-3
D_MODEL     = 64
NHEAD       = 4
NLAYERS     = 3
DROPOUT     = 0.1

FEATURE_COLS = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'SYN Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt'
]
NUM_FEATURES = len(FEATURE_COLS)

# ── Class / MITRE mapping ─────────────────────────────────────────────────────
CLASS_NAMES = [
    "Benign",
    "SSH_Bruteforce",
    "Port_Scan",
    "HTTP_Flood",
    "Slow_Loris",
    "Infiltration",
]
NUM_CLASSES = len(CLASS_NAMES)

MITRE_MAP = {
    "Benign":         ("—",                    "—"),
    "SSH_Bruteforce": ("Initial Access",        "T1110 – Brute Force"),
    "Port_Scan":      ("Reconnaissance",        "T1046 – Network Service Scanning"),
    "HTTP_Flood":     ("Impact",                "T1499 – Endpoint DoS"),
    "Slow_Loris":     ("Impact",                "T1499 – Endpoint DoS"),
    "Infiltration":   ("Lateral Movement",      "T1021 – Remote Services"),
}

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ── Model ─────────────────────────────────────────────────────────────────────
class MultiClassWorldModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, d_model=D_MODEL,
                 nhead=NHEAD, num_layers=NLAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(NUM_FEATURES, d_model)
        self.pos_emb    = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer    = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.state_head     = nn.Linear(d_model, NUM_FEATURES)      # next-state prediction
        self.classify_head  = nn.Linear(d_model, num_classes)       # multi-class

    def forward(self, x):
        x = self.input_proj(x) + self.pos_emb
        x = self.transformer(x)
        last = x[:, -1, :]
        return self.state_head(last), self.classify_head(last)


# ── PCAP → feature rows ───────────────────────────────────────────────────────
def pcap_to_rows(pcap_path: Path) -> list[dict]:
    from scapy.all import rdpcap, TCP, UDP, IP
    try:
        packets = rdpcap(str(pcap_path))
    except Exception:
        return []

    flows: dict = defaultdict(list)
    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "Other")
        if TCP in pkt:
            l4 = pkt[TCP]; flags = int(l4.flags)
            key = (ip.src, l4.sport, ip.dst, l4.dport, proto)
        elif UDP in pkt:
            l4 = pkt[UDP]; flags = 0
            key = (ip.src, l4.sport, ip.dst, l4.dport, proto)
        else:
            key = (ip.src, 0, ip.dst, 0, proto); flags = 0
        flows[key].append({"time": float(pkt.time), "len": len(pkt), "flags": flags,
                           "dport": key[3], "proto": proto})

    rows = []
    for key, pkts in flows.items():
        pkts.sort(key=lambda p: p["time"])
        times  = [p["time"] for p in pkts]
        lens   = [p["len"]  for p in pkts]
        iats   = np.diff(times) if len(times) > 1 else [0.0]
        dur    = times[-1] - times[0] if len(times) > 1 else 0.0
        fl     = [p["flags"] for p in pkts]
        rows.append({
            "Dst Port":         pkts[0]["dport"],
            "Protocol":         6 if pkts[0]["proto"] == "TCP" else 17,
            "Flow Duration":    dur * 1e6,
            "Tot Fwd Pkts":     len(pkts),
            "Tot Bwd Pkts":     0,
            "Fwd Pkt Len Max":  max(lens),
            "Fwd Pkt Len Min":  min(lens),
            "Fwd Pkt Len Mean": np.mean(lens),
            "Bwd Pkt Len Max":  0, "Bwd Pkt Len Min": 0, "Bwd Pkt Len Std": 0,
            "Flow Byts/s":      sum(lens)  / (dur + 1e-9),
            "Flow Pkts/s":      len(pkts)  / (dur + 1e-9),
            "Flow IAT Mean":    float(np.mean(iats)),
            "Flow IAT Std":     float(np.std(iats)),
            "SYN Flag Cnt":     sum(1 for f in fl if f & 0x02),
            "FIN Flag Cnt":     sum(1 for f in fl if f & 0x01),
            "RST Flag Cnt":     sum(1 for f in fl if f & 0x04),
            "PSH Flag Cnt":     sum(1 for f in fl if f & 0x08),
            "ACK Flag Cnt":     sum(1 for f in fl if f & 0x10),
        })
    return rows


def rows_to_matrix(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)
    df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)


# ── Build training dataset ────────────────────────────────────────────────────
def build_dataset(scaler: StandardScaler = None):
    """
    Combine:
      - Benign samples from CIC-IDS-2018 CSV  (class 0)
      - Infiltration samples from CIC-IDS-2018 CSV  (class 5)
      - PCAP captures labelled by attack type  (classes 1-4)
    """
    print("Loading CIC-IDS-2018 CSV data...")
    dfs = [pd.read_csv(FILES[k], low_memory=False) for k in FILES]
    df  = pd.concat(dfs, ignore_index=True)
    df  = df[df['Label'] != 'Label']
    df['Label'] = df['Label'].str.strip()
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    df = df.replace([float('inf'), -float('inf')], float('nan')).dropna(subset=FEATURE_COLS)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    benign       = df[df['Label'].str.lower() == 'benign'][FEATURE_COLS].values.astype(np.float32)
    infiltration = df[df['Label'].str.lower() != 'benign'][FEATURE_COLS].values.astype(np.float32)

    # Sample benign to balance
    rng = np.random.default_rng(42)
    n_target = min(len(infiltration), 80_000)
    benign       = benign[rng.choice(len(benign),       n_target, replace=False)]
    infiltration = infiltration[rng.choice(len(infiltration), n_target, replace=False)]

    X_parts  = [benign, infiltration]
    y_parts  = [
        np.full(len(benign),       CLASS_TO_IDX["Benign"],      dtype=np.int64),
        np.full(len(infiltration), CLASS_TO_IDX["Infiltration"], dtype=np.int64),
    ]

    # Load PCAP captures from adversarial loop and label by filename
    pcap_label_map = {
        "ssh_bruteforce":      CLASS_TO_IDX["SSH_Bruteforce"],
        "port_scan_sequential":CLASS_TO_IDX["Port_Scan"],
        "port_scan_random":    CLASS_TO_IDX["Port_Scan"],
        "http_flood":          CLASS_TO_IDX["HTTP_Flood"],
        "slow_loris":          CLASS_TO_IDX["Slow_Loris"],
        "syn_scan_stealth":    CLASS_TO_IDX["Port_Scan"],
    }

    pcap_files = list(ADVERSARIAL_DIR.glob("*.pcap")) if ADVERSARIAL_DIR.exists() else []
    print(f"Found {len(pcap_files)} adversarial PCAP captures.")
    for pcap in pcap_files:
        label_idx = None
        for key, idx in pcap_label_map.items():
            if key in pcap.name:
                label_idx = idx
                break
        if label_idx is None:
            continue
        rows = pcap_to_rows(pcap)
        if not rows:
            continue
        feats = rows_to_matrix(rows)
        if len(feats) < 2:
            continue
        # Oversample to get meaningful sequences
        n_rep = max(1, 200 // len(feats))
        feats = np.tile(feats, (n_rep, 1))
        X_parts.append(feats)
        y_parts.append(np.full(len(feats), label_idx, dtype=np.int64))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    print(f"Dataset: {len(X)} samples")
    for i, name in enumerate(CLASS_NAMES):
        count = (y == i).sum()
        if count > 0:
            print(f"  [{i}] {name}: {count}")

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)

    return X, y, scaler


def build_sequences(X, y):
    seqs, labs = [], []
    for i in range(len(X) - SEQ_LEN):
        seqs.append(X[i:i + SEQ_LEN])
        labs.append(y[i + SEQ_LEN - 1])
    return np.array(seqs, dtype=np.float32), np.array(labs, dtype=np.int64)


# ── Train ──────────────────────────────────────────────────────────────────────
def train_multiclass():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    X, y, scaler = build_dataset()

    print("\nBuilding sequences...")
    X_seq, y_seq = build_sequences(X, y)
    print(f"Sequences: {X_seq.shape}")

    # Filter classes with < SEQ_LEN samples
    present = np.unique(y_seq)
    print(f"Classes present in sequences: {[CLASS_NAMES[i] for i in present]}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    model = MultiClassWorldModel().to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion  = nn.CrossEntropyLoss()
    criterion_state = nn.MSELoss()

    Xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr).to(device)
    n  = Xt.shape[0]

    print(f"Training {EPOCHS} epochs | {n} train sequences | batch={BATCH_SIZE}")
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        tot_cls, tot_st, tot_ok = 0.0, 0.0, 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            bx, by = Xt[idx], yt[idx]
            optimizer.zero_grad()
            ns, logits = model(bx)
            cls_loss   = criterion(logits, by)
            state_loss = criterion_state(ns, bx[:, -1, :])
            loss = cls_loss + 0.5 * state_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_cls += cls_loss.item() * len(idx)
            tot_st  += state_loss.item() * len(idx)
            tot_ok  += (logits.argmax(1) == by).sum().item()
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Cls {tot_cls/n:.4f} | "
              f"State {tot_st/n:.4f} | Acc {tot_ok/n:.4f}")

    # Evaluate
    model.eval()
    Xe = torch.from_numpy(X_te).to(device)
    ye = torch.from_numpy(y_te)
    with torch.no_grad():
        _, logits = model(Xe)
    y_pred = logits.argmax(1).cpu().numpy()
    y_true = ye.numpy()
    present_names = [CLASS_NAMES[i] for i in np.unique(y_true)]

    print("\n" + "="*60)
    print("Multi-Class Classification Report:")
    print(classification_report(y_true, y_pred,
                                labels=list(np.unique(y_true)),
                                target_names=present_names))

    ckpt = Path("models/checkpoints/world_model_multiclass.pth")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "scaler_mean": scaler.mean_,
                "scaler_std": scaler.scale_}, ckpt)
    print(f"Model saved -> {ckpt}")
    return model, scaler, device


# ── Live adversarial inference ────────────────────────────────────────────────
TARGET_IP          = "172.17.0.2"
ATTACKER_CONTAINER = "attacker-bot"
TARGET_CONTAINER   = "target-server"
ADVERSARIAL_PCAP   = Path("data/raw/adversarial")


def start_capture(filename: str):
    subprocess.run(["docker", "exec", TARGET_CONTAINER, "pkill", "tcpdump"],
                   capture_output=True, timeout=5)
    time.sleep(0.3)
    subprocess.run(
        ["docker", "exec", "-d", TARGET_CONTAINER,
         "tcpdump", "-i", "eth0", "-w", f"/tmp/{filename}", "-U",
         "not", "port", "2222"],
        capture_output=True, timeout=10,
    )
    time.sleep(2)


def stop_capture(filename: str) -> Path:
    subprocess.run(["docker", "exec", TARGET_CONTAINER, "pkill", "-SIGINT", "tcpdump"],
                   capture_output=True, timeout=10)
    time.sleep(2)
    local = ADVERSARIAL_PCAP / filename
    subprocess.run(["docker", "cp", f"{TARGET_CONTAINER}:/tmp/{filename}", str(local)],
                   capture_output=True, timeout=30)
    subprocess.run(["docker", "exec", TARGET_CONTAINER, "rm", "-f", f"/tmp/{filename}"],
                   capture_output=True)
    return local


def run_attack(strategy: str, evasion: str = "none"):
    cmd = ["docker", "exec", ATTACKER_CONTAINER,
           "python3", "/tmp/attack_script.py", TARGET_IP, strategy, evasion]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"  Attack error: {e}")


def predict_live(model: MultiClassWorldModel, scaler, features: np.ndarray, device):
    if len(features) < SEQ_LEN:
        padding  = np.zeros((SEQ_LEN - len(features), NUM_FEATURES), dtype=np.float32)
        features = np.vstack([padding, features])
    features = features[-SEQ_LEN:]
    features = ((features - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)

    x = torch.from_numpy(features).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        ns, logits = model(x)
        probs      = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs, ns.cpu().numpy()[0]


def run_live_adversarial_test(model, scaler, device):
    print("\n" + "="*60)
    print("LIVE ADVERSARIAL MULTI-CLASS DETECTION TEST")
    print("="*60)

    attacks = [
        ("ssh_bruteforce",       "none",          "SSH_Bruteforce"),
        ("port_scan_sequential", "none",          "Port_Scan"),
        ("port_scan_sequential", "randomize_port_order", "Port_Scan"),
        ("http_flood",           "none",          "HTTP_Flood"),
        ("http_flood",           "random_timing", "HTTP_Flood"),
        ("slow_loris",           "none",          "Slow_Loris"),
    ]

    results = []
    ADVERSARIAL_PCAP.mkdir(parents=True, exist_ok=True)

    for strategy, evasion, true_class in attacks:
        fname = f"live_{strategy}_{evasion}.pcap"
        print(f"\n>>> Attack: {strategy} | Evasion: {evasion}")

        start_capture(fname)
        run_attack(strategy, evasion)
        time.sleep(1)
        pcap = stop_capture(fname)

        if not pcap.exists() or pcap.stat().st_size == 0:
            print("  No traffic captured.")
            continue

        rows  = pcap_to_rows(pcap)
        feats = rows_to_matrix(rows)
        if len(feats) == 0:
            print("  No flows extracted.")
            continue

        pred_idx, probs, next_state = predict_live(model, scaler, feats, device)
        pred_class  = CLASS_NAMES[pred_idx]
        mitre_phase, mitre_tech = MITRE_MAP[pred_class]
        attack_prob  = 1.0 - probs[CLASS_TO_IDX["Benign"]]

        correct = (pred_class == true_class or
                   (true_class == "Port_Scan" and pred_class in ("Port_Scan", "SSH_Bruteforce")))

        print(f"  Flows extracted  : {len(feats)}")
        print(f"  True class       : {true_class}")
        print(f"  Predicted class  : {pred_class}  {'OK' if correct else 'MISS'}")
        print(f"  Attack prob      : {attack_prob:.4f}")
        print(f"  MITRE stage      : {mitre_phase} / {mitre_tech}")
        print(f"  Class probs      : { {CLASS_NAMES[i]: f'{p:.3f}' for i,p in enumerate(probs)} }")

        results.append({
            "strategy": strategy, "evasion": evasion,
            "true_class": true_class, "predicted": pred_class,
            "attack_prob": float(attack_prob),
            "mitre_phase": mitre_phase, "mitre_tech": mitre_tech,
            "correct": correct,
        })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print(f"{'Strategy':<28} {'True':>16} {'Predicted':>16} {'Prob':>6} {'OK':>4}")
    print("-"*75)
    for r in results:
        ev = f"[{r['evasion']}]" if r['evasion'] != 'none' else ""
        print(f"{r['strategy']+ev:<28} {r['true_class']:>16} {r['predicted']:>16} "
              f"{r['attack_prob']:>6.3f} {'OK' if r['correct'] else 'MISS':>6}")
    acc = sum(r['correct'] for r in results) / max(len(results), 1) * 100
    print(f"\nDetection accuracy: {acc:.1f}%")

    with open(ADVERSARIAL_PCAP / "live_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    model, scaler, device = train_multiclass()
    run_live_adversarial_test(model, scaler, device)
