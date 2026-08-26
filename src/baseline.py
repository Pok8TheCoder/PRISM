"""Baseline: Logistic Regression for flow classification."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path

DATA_DIR = Path("data/raw")
FILES = {
    'thursday': DATA_DIR / 'thursday_01_03_2018.csv',
    'wednesday': DATA_DIR / 'wednesday_28_02_2018.csv'
}

def load_data():
    dfs = [pd.read_csv(FILES[name], low_memory=False) for name in FILES]
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['Label'] != 'Label']  # remove duplicate header rows
    return df

def preprocess(df):
    df['Label'] = df['Label'].str.strip()
    benign = df[df['Label'].str.lower() == 'benign']
    attack = df[df['Label'].str.lower() != 'benign']

    n_attack = len(attack)
    print(f"Benign: {len(benign)} | Attack: {n_attack}")

    benign_sample = benign.sample(n=n_attack, random_state=42)
    df_balanced = pd.concat([benign_sample, attack], ignore_index=True)

    cols = ['Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
            'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Std']

    df_balanced[cols] = df_balanced[cols].apply(pd.to_numeric, errors='coerce')
    df_balanced = df_balanced.replace([float('inf'), -float('inf')], float('nan')).dropna(subset=cols)

    X = df_balanced[cols].astype(float).values
    y = (df_balanced['Label'].str.lower() != 'benign').astype(int).values

    print(f"Balanced dataset -> Benign: {(y==0).sum()} | Attack: {(y==1).sum()}")
    return X, y

def train_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))

if __name__ == "__main__":
    print("Loading Data...")
    data = load_data()
    print("Preprocessing...")
    X, y = preprocess(data)
    print("Training Baseline Model...")
    train_baseline(X, y)
