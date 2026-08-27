"""Multi-class World Model: detects attack TYPE and maps to MITRE ATT&CK stage.

Class definitions are loaded dynamically from data/mitre_network_catalog.json
(generated from configs/attack_catalog.yaml).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import subprocess
import time
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from src.model.attack_catalog import (
    get_class_names,
    get_class_to_idx,
    get_mitre_map,
    get_num_classes,
    get_legacy_strategy_map,
    resolve_pcap_class,
    get_bot_class_ids,
)
from src.pipeline.extract import pcap_to_rows, rows_to_matrix
from src.pipeline.features import FEATURE_COLS, NUM_FEATURES

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

# FEATURE_COLS / NUM_FEATURES imported from src.pipeline.features (flow + packet)

# ── Class / MITRE mapping (from catalog) ──────────────────────────────────────
CLASS_NAMES = get_class_names()
NUM_CLASSES = get_num_classes()
MITRE_MAP = get_mitre_map()
CLASS_TO_IDX = get_class_to_idx()
LEGACY_STRATEGY_MAP = get_legacy_strategy_map()
BOT_CLASS_IDS = get_bot_class_ids()

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


# pcap_to_rows / rows_to_matrix imported from src.pipeline.extract


# ── Build training dataset ────────────────────────────────────────────────────
def build_dataset(scaler: StandardScaler = None):
    """
    Combine:
      - Benign samples from CIC-IDS-2018 CSV  (class 0)
      - Infiltration samples from CIC-IDS-2018 CSV  (class 5)
      - PCAP captures labelled by attack type  (classes 1-4)
    """
    print("Loading CIC-IDS-2018 CSV data...")
    present = [p for p in FILES.values() if p.exists()]
    X_parts: list = []
    y_parts: list = []
    rng = np.random.default_rng(42)

    if present:
        dfs = [pd.read_csv(p, low_memory=False) for p in present]
        df  = pd.concat(dfs, ignore_index=True)
        df  = df[df['Label'] != 'Label']
        df['Label'] = df['Label'].str.strip()
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0.0
        df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
        df = df.replace([float('inf'), -float('inf')], float('nan')).dropna(subset=FEATURE_COLS)
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

        benign       = df[df['Label'].str.lower() == 'benign'][FEATURE_COLS].values.astype(np.float32)
        attack_rows  = df[df['Label'].str.lower() != 'benign'][FEATURE_COLS].values.astype(np.float32)
        n_target = min(len(attack_rows), max(len(benign), 1), 80_000)
        if len(benign) and n_target:
            benign = benign[rng.choice(len(benign), min(n_target, len(benign)), replace=False)]
            X_parts.append(benign)
            y_parts.append(np.full(len(benign), CLASS_TO_IDX["Benign"], dtype=np.int64))
        if len(attack_rows) and n_target:
            attack_rows = attack_rows[rng.choice(len(attack_rows), min(n_target, len(attack_rows)), replace=False)]
            infiltration_idx = CLASS_TO_IDX.get("T1021_remote_services", CLASS_TO_IDX["Benign"])
            X_parts.append(attack_rows)
            y_parts.append(np.full(len(attack_rows), infiltration_idx, dtype=np.int64))
    else:
        print("WARNING: CIC CSVs not found in data/raw. Using PCAP captures only.")

    pcap_files = []
    if ADVERSARIAL_DIR.exists():
        pcap_files = list(ADVERSARIAL_DIR.rglob("*.pcap"))
    print(f"Found {len(pcap_files)} adversarial PCAP captures.")
    for pcap in pcap_files:
        label_name = resolve_pcap_class(pcap.name)
        if label_name is None or label_name not in CLASS_TO_IDX:
            continue
        label_idx = CLASS_TO_IDX[label_name]
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

    X = np.vstack(X_parts) if X_parts else np.zeros((0, NUM_FEATURES), dtype=np.float32)
    y = np.concatenate(y_parts) if y_parts else np.zeros((0,), dtype=np.int64)
    if len(X) == 0:
        raise FileNotFoundError("No CIC CSVs or labeled PCAPs available for training.")

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
TARGET_IP          = "target-server"
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


def run_attack(class_id: str, evasion: str = "none"):
    cmd = ["docker", "exec", ATTACKER_CONTAINER,
           "python3", "-m", "src.adversarial.attack_script", TARGET_IP, class_id, evasion]
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
        ("T1110_ssh_bruteforce", "none", "T1110_ssh_bruteforce"),
        ("T1046_service_scan", "none", "T1046_service_scan"),
        ("T1046_service_scan", "randomize_port_order", "T1046_service_scan"),
        ("T1499_http_flood", "none", "T1499_http_flood"),
        ("T1499_http_flood", "random_timing", "T1499_http_flood"),
        ("T1499_slowloris", "none", "T1499_slowloris"),
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

        correct = pred_class == true_class

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
