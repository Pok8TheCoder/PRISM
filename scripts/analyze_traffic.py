#!/usr/bin/env python3
"""Offline PS demo: python scripts/analyze_traffic.py path/to/file.pcap"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.adversarial.training_loop import load_model_and_scaler
from src.predict.engine import analyze_file
import torch


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_traffic.py <pcap|csv>")
        return 1
    path = Path(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler = load_model_and_scaler(device)
    result = analyze_file(path, model, scaler, device)
    result.pop("features", None)
    print(json.dumps({k: v for k, v in result.items() if k != "class_probs"}, indent=2, default=str))
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
