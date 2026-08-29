"""
PRISM Model Comparison (src/models/comparison.py)
Direct utility within `src/models/` to evaluate and compare the World Model
against the static baselines (Logistic Regression & Random Forest).

Usage:
    python -m src.models.comparison
    python src/models/comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.model_comparison import run_model_comparison


def main():
    print("[PRISM] Executing Model Comparison from src/models/comparison.py...")
    run_model_comparison()


if __name__ == "__main__":
    main()
