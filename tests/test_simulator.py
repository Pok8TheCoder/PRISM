"""
Unit tests for PRISM Forward Simulator, Attack Mapper, and Scoring Engine.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.models.world_model import StateTransformerWorldModel
from src.prediction.simulator import KStepSimulator
from src.prediction.attack_mapper import AttackStageMapper
from src.prediction.scoring import InfiltrationScorer, StepScore


class TestSimulator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.d_state = 16
        self.lookback = 10
        self.model = StateTransformerWorldModel(
            d_state=self.d_state,
            d_model=32,
            n_layers=1,
            n_heads=2,
            lookback=self.lookback,
        )
        self.state_seq = np.random.randn(self.lookback, self.d_state).astype(np.float32)

    def test_kstep_deterministic_simulation(self):
        """Test K-step forward simulation in deterministic mode."""
        simulator = KStepSimulator(
            model=self.model,
            device="cpu",
            k_steps=5,
            num_rollouts=1,
            deterministic=True,
        )
        result = simulator.simulate(self.state_seq)

        self.assertEqual(len(result.steps), 5)
        self.assertGreaterEqual(result.peak_infiltration_prob, 0.0)
        self.assertLessEqual(result.peak_infiltration_prob, 1.0)
        self.assertIn(result.overall_alert, ["none", "low", "medium", "high", "critical"])

        for step in result.steps:
            self.assertIsInstance(step, StepScore)
            self.assertEqual(step.predicted_state.shape[-1], self.d_state)

    def test_ensemble_simulation(self):
        """Test stochastic ensemble rollout with confidence intervals."""
        simulator = KStepSimulator(
            model=self.model,
            device="cpu",
            k_steps=6,
            num_rollouts=5,
            deterministic=False,
        )
        result = simulator.simulate(self.state_seq)

        self.assertEqual(len(result.steps), 6)
        self.assertIsNotNone(result.ensemble_mean)
        self.assertIsNotNone(result.ensemble_lower)
        self.assertIsNotNone(result.ensemble_upper)
        self.assertEqual(len(result.ensemble_mean), 6)
        self.assertTrue((result.ensemble_upper >= result.ensemble_lower).all())

    def test_attack_mapper(self):
        """Test AttackStageMapper metadata and CAPEC retrieval."""
        mapper = AttackStageMapper()
        meta = mapper.get_tactic_meta("Reconnaissance")
        self.assertEqual(meta["id"], "TA0043")
        self.assertIn("indicators", meta)
        self.assertIn("recommended_actions", meta)

        benign_meta = mapper.get_tactic_meta("Benign")
        self.assertEqual(benign_meta["id"], "BENIGN")

    def test_scoring_engine_alerts(self):
        """Test InfiltrationScorer alert classification and escalation detection."""
        scorer = InfiltrationScorer()
        self.assertEqual(scorer._classify_alert(0.90), "critical")
        self.assertEqual(scorer._classify_alert(0.70), "high")
        self.assertEqual(scorer._classify_alert(0.45), "medium")
        self.assertEqual(scorer._classify_alert(0.25), "low")
        self.assertEqual(scorer._classify_alert(0.05), "none")

        # Test escalating sequence
        steps_escalating = [
            scorer.score_step(step=i + 1, infiltration_prob=0.2 + i * 0.15, mitre_probs=[1.0, 0, 0, 0, 0, 0, 0])
            for i in range(5)
        ]
        res = scorer.score_rollout(steps_escalating)
        self.assertTrue(res.is_escalating)


if __name__ == "__main__":
    unittest.main()
