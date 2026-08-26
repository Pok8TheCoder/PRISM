"""World Model PoC: Temporal Transformer learning P(S_{t+1} | S_t)."""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("data/raw")
FILES = {
    'thursday': DATA_DIR / 'thursday_01_03_2018.csv',
    'wednesday': DATA_DIR / 'wednesday_28_02_2018.csv'
}

SEQ_LEN = 10
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
D_MODEL = 64
NHEAD = 4
NLAYERS = 3
DROPOUT = 0.1

FEATURE_COLS = [
    'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
    'SYN Flag Cnt', 'FIN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt'
]
NUM_FEATURES = len(FEATURE_COLS)


def load_and_preprocess():
    dfs = [pd.read_csv(FILES[name], low_memory=False) for name in FILES]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Label'] != 'Label']

    benign = df[df['Label'].str.strip().str.lower() == 'benign']
    attack = df[df['Label'].str.strip().str.lower() != 'benign']
    n_attack = len(attack)
    benign = benign.sample(n=n_attack, random_state=42)
    df = pd.concat([benign, attack], ignore_index=True)

    df = df.sort_values('Timestamp').reset_index(drop=True)

    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
    df = df.replace([float('inf'), -float('inf')], float('nan')).dropna(subset=FEATURE_COLS)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    y = (df['Label'].str.strip().str.lower() != 'benign').astype(int).values
    X = df[FEATURE_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    return X, y


def build_sequences(X, y, seq_len=SEQ_LEN):
    sequences, labels = [], []
    for i in range(len(X) - seq_len):
        sequences.append(X[i:i + seq_len])
        labels.append(y[i + seq_len - 1])
    return np.array(sequences), np.array(labels)


class WorldModelTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=NLAYERS, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(NUM_FEATURES, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.state_head = nn.Linear(d_model, NUM_FEATURES)
        self.classify_head = nn.Linear(d_model, 2)
        self.attn_weights = None

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_emb
        x = self.transformer(x)
        last_state = x[:, -1, :]
        next_state_pred = self.state_head(last_state)
        logits = self.classify_head(last_state)
        return next_state_pred, logits


def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading and preprocessing data...")
    X, y = load_and_preprocess()
    print(f"Total samples: {len(X)} | Benign: {(y==0).sum()} | Attack: {(y==1).sum()}")

    print("Building sequences...")
    X_seq, y_seq = build_sequences(X, y)
    print(f"Sequences: {X_seq.shape} | Labels: {y_seq.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_test_t = torch.from_numpy(y_test).long().to(device)

    model = WorldModelTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_state = nn.MSELoss()

    n_samples = X_train_t.shape[0]
    print(f"\nTraining World Model for {EPOCHS} epochs (batch_size={BATCH_SIZE})...\n")

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total_cls_loss, total_state_loss, total_correct = 0.0, 0.0, 0

        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            batch_x = X_train_t[idx]
            batch_y = y_train_t[idx]

            optimizer.zero_grad()
            next_state_pred, logits = model(batch_x)

            cls_loss = criterion_cls(logits, batch_y)
            state_loss = criterion_state(next_state_pred, batch_x[:, -1, :])
            loss = cls_loss + 0.5 * state_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_cls_loss += cls_loss.item() * len(idx)
            total_state_loss += state_loss.item() * len(idx)
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        train_acc = total_correct / n_samples
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Cls Loss: {total_cls_loss/n_samples:.4f} | "
              f"State Loss: {total_state_loss/n_samples:.4f} | "
              f"Train Acc: {train_acc:.4f}")

    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        _, logits = model(X_test_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        y_true = y_test_t.cpu().numpy()

    print("\nWorld Model Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Attack']))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) * 100
    print(f"False Positive Rate: {fpr:.2f}%")
    print(f"Confusion Matrix:\n{cm}")

    ckpt_path = Path("models/checkpoints/world_model_poc.pth")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel saved to {ckpt_path}")

    return model, X_test_t, y_test_t, device


def k_step_rollout(model, X_test, y_test, device, k=5):
    """K-step forward simulation: predict attack probability K steps ahead."""
    model.eval()
    print(f"\n--- K-Step Forward Rollout (K={k}) ---")
    n_samples = min(500, X_test.shape[0])
    sample_idx = torch.randperm(X_test.shape[0], device=device)[:n_samples]
    x = X_test[sample_idx].clone()

    rollout_probs = []
    with torch.no_grad():
        for step in range(k):
            next_state_pred, logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            rollout_probs.append(probs.cpu().numpy())

            next_state_pred_norm = (next_state_pred - next_state_pred.mean(dim=-1, keepdim=True)) / (next_state_pred.std(dim=-1, keepdim=True) + 1e-6)
            x = torch.cat([x[:, 1:, :], next_state_pred_norm.unsqueeze(1)], dim=1)

    rollout_probs = np.array(rollout_probs)
    avg_prob = rollout_probs.mean(axis=1)
    print(f"Avg predicted attack prob across {k} steps: {avg_prob.mean():.4f}")
    print(f"Step-wise avg attack prob: {[f'{p.mean():.4f}' for p in rollout_probs]}")


if __name__ == "__main__":
    model, X_test, y_test, device = train_model()
    k_step_rollout(model, X_test, y_test, device, k=5)
