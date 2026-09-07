#!/usr/bin/env python3
"""Offline smoke test for full forecast SHAP report (no live lab)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.explain.forecast_block_explain import explain_hx_block  # noqa: E402
from src.hx.causal import LOOKBACK, SHAUN_DIM, StreamingHXC, load_hxc_bundle  # noqa: E402


def main() -> int:
    bundle = load_hxc_bundle()
    hx = StreamingHXC(bundle, context_skip_steps=0)
    rng = np.random.default_rng(42)
    for _ in range(LOOKBACK):
        hx.buf.append(rng.normal(0, 1, SHAUN_DIM).astype(np.float32))
    report = explain_hx_block(
        hx,
        p_attack=0.91,
        predicted_class=None,
        ground_truth_class="T1046_service_scan",
    )
    required = (
        "waterfall",
        "temporal",
        "class_analysis",
        "top_features",
        "full_shap",
    )
    missing = [k for k in required if k not in report]
    if missing:
        print(f"FAIL missing keys: {missing}", file=sys.stderr)
        return 1
    wf = report["waterfall"]
    if not wf.get("steps"):
        print("FAIL empty waterfall", file=sys.stderr)
        return 1
    if not report["temporal"].get("steps"):
        print("FAIL empty temporal", file=sys.stderr)
        return 1
    print(f"PASS method={report['method']} features={len(report['top_features'])} "
          f"waterfall={len(wf['steps'])} temporal={len(report['temporal']['steps'])}")
    from src.explain.forecast_block_explain import write_shap_artifacts

    paths = write_shap_artifacts(report, window_idx=99)
    print(f"  saved: {paths.get('markdown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
