"""Leave-one-block-out ablation over the v2 feature schema.

Answers which parts of the 64-column v2 vector actually earn their place, and
in particular whether the gain comes from window aggregation or merely from
having more columns.

Results -> results/zmt_ymt/ablation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_zmt_ymt import evaluate, scale, train_one  # noqa: E402
from src.model.attack_catalog import get_class_names  # noqa: E402
from src.pipeline.features_v2 import FEATURE_BLOCKS_V2, FEATURE_COLS_V2  # noqa: E402

DATA = ROOT / "data" / "processed" / "zmt_ymt_dataset.npz"
OUT = ROOT / "results" / "zmt_ymt" / "ablation.json"
SEEDS = (0, 1)

# Macro-F1 over 33 classes is dominated by classes holding 1-2 test windows,
# where a single flipped sample moves the average by ~0.03. Accuracy over 2728
# windows and mean F1 across the six-class scan/flood family are the stable
# signals, so both are reported alongside it.
CLASS_NAMES = get_class_names()
SCAN_FAMILY = [
    "T1046_service_scan", "T1595_active_scan", "T1498_network_dos",
    "T1499_http_flood", "T1049_connections_discovery", "T1018_remote_discovery",
]
SCAN_IDX = [CLASS_NAMES.index(c) for c in SCAN_FAMILY]

# Blocks A/B/C/E/F are only definable over a window; D is window mean+std of
# per-flow quantities; G is raw current-flow passthrough. "window_only" and
# "flowlike_only" bracket the windowing question from both sides.
VARIANTS = {
    "full_v2": None,
    "minus_A_composition": ["A_composition"],
    "minus_B_protocol": ["B_protocol"],
    "minus_C_portclass": ["C_portclass"],
    "minus_D_dispersion": ["D_dispersion"],
    "minus_E_flags": ["E_flags"],
    "minus_F_packet": ["F_packet"],
    "minus_G_current": ["G_current"],
    "window_only_ABC": ["D_dispersion", "E_flags", "F_packet", "G_current"],
    "flowlike_only_DG": ["A_composition", "B_protocol", "C_portclass",
                         "E_flags", "F_packet"],
}


def cols_for(drop: list[str] | None) -> list[int]:
    dropped: set[str] = set()
    for b in drop or []:
        dropped.update(FEATURE_BLOCKS_V2[b])
    return [i for i, c in enumerate(FEATURE_COLS_V2) if c not in dropped]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATA, allow_pickle=True)
    X2, y, split, source = d["X2"], d["y"], d["split"], d["source"]
    tr, va, te = split == "train", split == "val", split == "test"
    yte, pcap_m = y[te], source[te] == "pcap"

    results = {}
    for name, drop in VARIANTS.items():
        cols = cols_for(drop)
        X = X2[:, :, cols]
        Xtr, Xva, Xte, _ = scale(X[tr], X[va], X[te])
        print(f"\n  {name}  ({len(cols)} features)", flush=True)
        runs = []
        for seed in SEEDS:
            _, m = train_one(name, Xtr, y[tr], Xva, y[va], Xte, yte, seed, device)
            pred = m.pop("_pred")
            m["scan_family_f1"] = float(np.mean(f1_score(
                yte, pred, average=None, labels=SCAN_IDX, zero_division=0)))
            m["pcap_accuracy"] = float((pred[pcap_m] == yte[pcap_m]).mean())
            runs.append(m)
            print(f"    seed{seed} acc={m['accuracy']:.4f} "
                  f"macroF1={m['f1_macro']:.4f} scanF1={m['scan_family_f1']:.4f} "
                  f"pcapAcc={m['pcap_accuracy']:.4f}", flush=True)
        keys = ["accuracy", "f1_macro", "scan_family_f1", "pcap_accuracy",
                "binary_fpr"]
        results[name] = {
            "num_features": len(cols),
            **{k: float(np.mean([r[k] for r in runs])) for k in keys},
            **{f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys},
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    base = results["full_v2"]
    print("\n" + "=" * 92)
    print(f"{'Variant':<24}{'Feat':>5}{'Acc':>8}{'dAcc':>8}{'MacroF1':>9}"
          f"{'ScanF1':>9}{'dScan':>8}{'PcapAcc':>9}{'dPcap':>8}")
    print("-" * 92)
    for name, m in results.items():
        print(f"{name:<24}{m['num_features']:>5}{m['accuracy']:>8.4f}"
              f"{m['accuracy'] - base['accuracy']:>+8.4f}{m['f1_macro']:>9.4f}"
              f"{m['scan_family_f1']:>9.4f}"
              f"{m['scan_family_f1'] - base['scan_family_f1']:>+8.4f}"
              f"{m['pcap_accuracy']:>9.4f}"
              f"{m['pcap_accuracy'] - base['pcap_accuracy']:>+8.4f}")
    print("=" * 92)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
