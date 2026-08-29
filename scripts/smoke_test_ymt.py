"""End-to-end check: load YMT.01 from disk and score real lab captures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_comparison_dataset import windows_from_capture  # noqa: E402
from scripts.train_zmt_ymt import WorldModel  # noqa: E402
from src.model.attack_catalog import resolve_pcap_class  # noqa: E402
from src.pipeline.extract_v2 import base_to_v1, blocks_to_v2, pcap_to_base  # noqa: E402

CKPT = ROOT / "models" / "checkpoints"
ADV = ROOT / "data" / "raw" / "adversarial"


def load(name: str):
    ck = torch.load(CKPT / name, map_location="cpu", weights_only=False)
    model = WorldModel(ck["num_features"])
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["scaler_mean"], ck["scaler_std"], ck["class_names"]


def main() -> None:
    zmt = load("zmt_01.pth")
    ymt = load("ymt_01.pth")
    print(f"ZMT.01 expects {zmt[0].input_proj.in_features} features")
    print(f"YMT.01 expects {ymt[0].input_proj.in_features} features\n")

    rng = np.random.default_rng(7)
    pcaps = sorted(ADV.rglob("*.pcap"))
    picks = [pcaps[i] for i in rng.choice(len(pcaps), 12, replace=False)]

    print(f"{'capture':<40}{'true':<30}{'ZMT.01':<30}{'YMT.01':<30}")
    print("-" * 130)
    hits = {"ZMT.01": 0, "YMT.01": 0}
    n = 0
    for p in picks:
        true = resolve_pcap_class(p.name)
        base = pcap_to_base(p)
        if len(base) == 0 or true is None:
            continue
        blocks = windows_from_capture(base, stride=1)
        preds = {}
        for label, (model, mean, std, names), proj in (
            ("ZMT.01", zmt, base_to_v1), ("YMT.01", ymt, blocks_to_v2),
        ):
            X = proj(blocks)
            X = ((X - mean) / (std + 1e-8)).astype(np.float32)
            with torch.no_grad():
                _, logits = model(torch.from_numpy(X))
            # Majority verdict across the capture's windows.
            votes = np.bincount(logits.argmax(1).numpy(), minlength=len(names))
            preds[label] = names[int(votes.argmax())]
            hits[label] += preds[label] == true
        n += 1
        mark = lambda k: "OK " if preds[k] == true else "MISS"  # noqa: E731
        print(f"{p.name[:38]:<40}{true:<30}"
              f"{mark('ZMT.01')+preds['ZMT.01']:<30}"
              f"{mark('YMT.01')+preds['YMT.01']:<30}")

    print("-" * 130)
    for k in hits:
        print(f"{k}: {hits[k]}/{n} captures correct")


if __name__ == "__main__":
    main()
