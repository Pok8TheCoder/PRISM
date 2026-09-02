import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.models.world_model import StateTransformerWorldModel

from src.prediction.ramx import RAMXPredictor, WarmupBaselineCalibrator, EpisodicMemoryBank


def test_warmup_calibrator():
    calibrator = WarmupBaselineCalibrator(warmup_steps=10)
    for _ in range(10):
        calibrator.update(np.random.normal(5.0, 1.0, size=(292,)))
    assert calibrator.calibrated is True
    assert calibrator.baseline_mean is not None

    # Normal state close to baseline
    normal_score = calibrator.get_relative_anomaly_score(np.random.normal(5.0, 1.0, size=(292,)))
    assert normal_score < 0.50

    # Anomalous state with burst
    burst_state = np.random.normal(5.0, 1.0, size=(292,))
    burst_state[8:25] += 50.0  # Massive spike
    burst_score = calibrator.get_relative_anomaly_score(burst_state)
    assert burst_score > 0.85


def test_ramx_predictor():
    model = StateTransformerWorldModel(d_state=292, d_model=256)
    scaler_m = np.zeros((1, 292), dtype=np.float32)
    scaler_s = np.ones((1, 292), dtype=np.float32)

    predictor = RAMXPredictor(model, scaler_m, scaler_s, warmup_steps=10, enable_ttt=False)

    for _ in range(12):
        traj = np.random.randn(20, 292).astype(np.float32)
        res = predictor.predict_state(traj)

    assert res["calibrated"] is True
    assert "p_attack" in res
    assert "mitre_stage" in res


if __name__ == "__main__":
    test_warmup_calibrator()
    test_ramx_predictor()
    print("ALL RAMX TESTS PASSED SUCCESSFULLY!")
