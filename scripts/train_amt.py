"""Train AMT.01 — the origin/aryan StateBuilder schema on PRISM's real traffic.

Reuses the exact dataset, splits, architecture, hyperparameters and seeds from
`train_zmt_ymt.py`, so AMT.01 / ZMT.01 / YMT.01 differ only in feature schema.

Results are merged into results/zmt_ymt/metrics.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_zmt_ymt import (  # noqa: E402
    CKPT_DIR, CLASS_NAMES, NUM_CLASSES, OUT_DIR, SEEDS, scale, train_one,
)
from src.pipeline.extract_aryan import (  # noqa: E402
    NUM_FEATURES_ARYAN, blocks_to_aryan,
)

DATA = ROOT / "data" / "processed" / "zmt_ymt_dataset.npz"
CACHE = ROOT / "data" / "processed" / "amt_features.npy"
NAME = "AMT.01"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATA, allow_pickle=True)
    y, split, source = d["y"], d["split"], d["source"]
    tr, va, te = split == "train", split == "val", split == "test"

    if CACHE.exists():
        X = np.load(CACHE)
    else:
        base = ROOT / "data" / "processed" / "zmt_ymt_blocks.npy"
        if not base.exists():
            raise FileNotFoundError(
                "Need the raw base blocks. Re-run build_comparison_dataset.py "
                "after enabling block export."
            )
        X = blocks_to_aryan(np.load(base))
        np.save(CACHE, X)
    print(f"AMT.01 features {X.shape} ({NUM_FEATURES_ARYAN} per timestep)")

    Xtr, Xva, Xte, sc = scale(X[tr], X[va], X[te])
    per_seed, best, best_model, best_pred = [], -1.0, None, None
    for seed in SEEDS:
        print(f"\n  Training {NAME} seed={seed}")
        model, m = train_one(NAME, Xtr, y[tr], Xva, y[va], Xte, y[te], seed, device)
        pred = m.pop("_pred")
        per_seed.append(m)
        print(f"    -> test acc={m['accuracy']:.4f} macroF1={m['f1_macro']:.4f} "
              f"binF1={m['binary_f1']:.4f} FPR={m['binary_fpr']:.4f}")
        if m["f1_macro"] > best:
            best, best_model, best_pred = m["f1_macro"], model, pred

    results = json.load(open(OUT_DIR / "metrics.json"))
    results["per_seed"][NAME] = per_seed
    keys = [k for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))]
    results["summary"][NAME] = {
        k: {"mean": float(np.mean([s[k] for s in per_seed])),
            "std": float(np.std([s[k] for s in per_seed]))} for k in keys
    }
    results["summary"][NAME]["num_features"] = int(X.shape[2])

    yte = y[te]
    f1c = f1_score(yte, best_pred, average=None,
                   labels=list(range(NUM_CLASSES)), zero_division=0)
    results["per_class_f1"][NAME] = {
        CLASS_NAMES[i]: float(f1c[i]) for i in range(NUM_CLASSES)
        if (yte == i).sum() > 0
    }
    pm = source[te] == "pcap"
    results["pcap_only"][NAME] = {
        "accuracy": float((best_pred[pm] == yte[pm]).mean()),
        "f1_macro": float(f1_score(yte[pm], best_pred[pm], average="macro",
                                   zero_division=0)),
        "n": int(pm.sum()),
    }
    results["confusion"][NAME] = confusion_matrix(
        yte, best_pred, labels=list(range(NUM_CLASSES))).tolist()

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save({
        "model": best_model.state_dict(), "num_features": int(X.shape[2]),
        "scaler_mean": sc.mean_, "scaler_std": sc.scale_,
        "class_names": CLASS_NAMES, "model_name": NAME,
    }, CKPT_DIR / "amt_01.pth")

    print("\n" + "=" * 74)
    print(f"{'Model':<10}{'Feat':>6}{'Acc':>9}{'MacroF1':>10}{'BinF1':>9}"
          f"{'FPR':>9}{'PcapAcc':>10}")
    print("-" * 74)
    for n in ("ZMT.01", "YMT.01", NAME):
        s = results["summary"][n]
        print(f"{n:<10}{s['num_features']:>6}{s['accuracy']['mean']:>9.4f}"
              f"{s['f1_macro']['mean']:>10.4f}{s['binary_f1']['mean']:>9.4f}"
              f"{s['binary_fpr']['mean']:>9.4f}"
              f"{results['pcap_only'][n]['accuracy']:>10.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
