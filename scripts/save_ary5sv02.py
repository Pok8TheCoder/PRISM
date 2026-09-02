#!/usr/bin/env python3
"""Save ARY-5sV02 checkpoint metadata."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "checkpoints" / "ary5s_v02.pt"
DST = ROOT / "models" / "checkpoints" / "ary_5sv02.pt"
META = ROOT / "models" / "checkpoints" / "ary_5sv02.json"


def main() -> int:
    if not SRC.exists():
        print(f"Missing {SRC}")
        return 1
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    metrics = dict(ck.get("metrics", {}))
    metrics.update({
        "model_name": "ARY-5sV02",
        "tag": "ARY-5sV02",
        "window_sec": 5,
        "init_from": "ARY-5sV01",
        "recipe": "Fine-tune v01 on CIC 5s + all lab PCAPs (failed PCAPs x20 oversample)",
        "splits_dir": "data/aryan_splits_5s_v02",
    })
    ck["metrics"] = metrics
    ck["model_name"] = "ARY-5sV02"
    DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, DST)
    META.write_text(json.dumps({"checkpoint": str(DST), **metrics}, indent=2), encoding="utf-8")
    print(f"Saved -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
