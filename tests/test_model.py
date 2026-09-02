"""Unit tests for StateTransformerWorldModel and RolloutSimulator."""
import torch
import numpy as np
import pytest
from src.models.world_model import StateTransformerWorldModel
from src.prediction.simulator import RolloutSimulator
from src.prediction.scoring import compute_trajectory_risk, generate_alerts


def test_world_model_forward():
    batch_size = 4
    seq_len = 10
    d_state = 110

    model = StateTransformerWorldModel(d_state=d_state, d_model=64, nhead=4, num_layers=2)
    dummy_input = torch.randn(batch_size, seq_len, d_state)

    (
        pred_mean,
        pred_logvar,
        pred_attack_logits,
        pred_mitre_logits,
        pred_fraction,
        attn_weights
    ) = model(dummy_input)

    assert pred_mean.shape == (batch_size, d_state)
    assert pred_logvar.shape == (batch_size, d_state)
    assert pred_attack_logits.shape == (batch_size, 2)
    assert pred_mitre_logits.shape == (batch_size, 7)
    assert pred_fraction.shape == (batch_size, 1)
    assert attn_weights.shape == (batch_size, seq_len, seq_len)


def test_rollout_simulator():
    model = StateTransformerWorldModel(d_state=110, d_model=64, nhead=4, num_layers=2)
    simulator = RolloutSimulator(model, device="cpu")

    dummy_seq = torch.randn(1, 10, 110)
    trajectory = simulator.rollout(dummy_seq, K=5)

    assert len(trajectory) == 5
    assert trajectory[0]["step"] == 1
    assert "infiltration_prob" in trajectory[0]
    assert "mitre_stage" in trajectory[0]

    risk_summary = compute_trajectory_risk(trajectory)
    assert "risk_level" in risk_summary
    assert "max_prob" in risk_summary
