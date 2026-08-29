"""Is the temporal transformer earning its complexity on this dataset?

Trains a plain Random Forest on the same v2 (YMT.01) windows, same split, same
test set as scripts/train_zmt_ymt.py, and reports the same metrics. RF sees no
sequence structure at all -- it is given the last timestep's 64-d state vector
as one flat row, and the last state already contains everything up to 8 prior
flows via the window aggregation.

This isolates "does the model need to be a temporal transformer" from
"does the feature representation need to be window-aggregated" -- the second
question was already answered by the v1/v2/AMT comparison.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.attack_catalog import get_class_to_idx  # noqa: E402

DATA = ROOT / "data" / "processed" / "zmt_ymt_dataset.npz"
BENIGN_IDX = get_class_to_idx()["Benign"]


def report(name: str, y_true, y_pred, y_prob_attack, source_mask, t_fit: float) -> None:
    attack_true = (y_true != BENIGN_IDX).astype(int)
    attack_pred = (y_pred != BENIGN_IDX).astype(int)
    tp = int(((attack_pred == 1) & (attack_true == 1)).sum())
    fp = int(((attack_pred == 1) & (attack_true == 0)).sum())
    tn = int(((attack_pred == 0) & (attack_true == 0)).sum())
    _, _, bf1, _ = precision_recall_fscore_support(
        attack_true, attack_pred, average="binary", zero_division=0)
    auc = roc_auc_score(attack_true, y_prob_attack)

    acc = (y_pred == y_true).mean()
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    pcap_acc = (y_pred[source_mask] == y_true[source_mask]).mean()

    print(f"\n  {name}")
    print(f"    fit time        {t_fit:.2f}s")
    print(f"    accuracy        {acc:.4f}")
    print(f"    macro F1        {macro:.4f}")
    print(f"    binary F1       {bf1:.4f}")
    print(f"    binary AUC      {auc:.4f}")
    print(f"    FPR             {fp / max(fp + tn, 1):.4f}")
    print(f"    PCAP-only acc   {pcap_acc:.4f}")


def main() -> None:
    d = np.load(DATA, allow_pickle=True)
    X2, y, split, source = d["X2"], d["y"], d["split"], d["source"]
    tr, te = split == "train", split == "test"

    # Last timestep only: one flat 64-d row per sample, no sequence given to the model.
    Xtr_last, Xte_last = X2[tr, -1, :], X2[te, -1, :]
    # Full flattened sequence: 640-d, in case order-blind trees benefit from seeing
    # earlier timesteps explicitly rather than only the final aggregated state.
    Xtr_flat = X2[tr].reshape(tr.sum(), -1)
    Xte_flat = X2[te].reshape(te.sum(), -1)

    print(f"Train {tr.sum()}  Test {te.sum()}  classes {len(np.unique(y))}")

    for name, Xtr, Xte in (
        ("RandomForest, last-timestep-only (64f, non-temporal)", Xtr_last, Xte_last),
        ("RandomForest, full flattened sequence (640f)", Xtr_flat, Xte_flat),
    ):
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced_subsample",
            n_jobs=-1, random_state=0,
        )
        t0 = time.time()
        rf.fit(Xtr, y[tr])
        t_fit = time.time() - t0
        pred = rf.predict(Xte)
        proba = rf.predict_proba(Xte)
        classes = rf.classes_.tolist()
        attack_prob = 1.0 - proba[:, classes.index(BENIGN_IDX)] if BENIGN_IDX in classes else np.zeros(len(Xte))
        report(name, y[te], pred, attack_prob, source[te] == "pcap", t_fit)

    print(f"\n  (for reference) YMT.01 temporal transformer: ~71.5s/seed fit time, "
          f"acc 0.8493, macroF1 0.8818, binF1 0.9172, FPR 0.5307, PCAP-acc 0.9880")


if __name__ == "__main__":
    main()
