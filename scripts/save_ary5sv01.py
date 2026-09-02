#!/usr/bin/env python3
"""Save trained 5s checkpoint as canonical ARY-5sV01."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "checkpoints" / "ary5s_v01.pt"
DST = ROOT / "models" / "checkpoints" / "ary_5sv01.pt"
META = ROOT / "models" / "checkpoints" / "ary_5sv01.json"


def main() -> int:
    if not SRC.exists():
        print(f"Missing source checkpoint: {SRC}")
        return 1

    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    metrics = dict(ck.get("metrics", {}))
    metrics.update({
        "model_name": "ARY-5sV01",
        "tag": "ARY-5sV01",
        "window_sec": 5,
        "recipe": "ARY.01 origin/aryan @ 1ec06bc on data/aryan_splits_5s",
        "description": "TemporalTransformerWorldModel trained on CIC-IDS-2018 5s windows",
    })
    ck["metrics"] = metrics
    ck["model_name"] = "ARY-5sV01"

    DST.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, DST)

    sidecar = {
        "model_name": "ARY-5sV01",
        "checkpoint": str(DST.relative_to(ROOT)),
        "alias": str(SRC.relative_to(ROOT)),
        **metrics,
    }
    META.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    # Keep legacy name in sync (copy)
    shutil.copy2(DST, SRC)

    print(json.dumps(sidecar, indent=2))
    print(f"Saved ARY-5sV01 -> {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
