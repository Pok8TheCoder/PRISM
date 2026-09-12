"""
Unit tests for PRISM Model Architectures & Loss Functions.
Tests StateTransformerWorldModel, LSTMWorldModel, LatentDynamicsWorldModel,
GraphWorldModel, MultiTaskLoss, and Baselines.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.models.world_model import (
    TemporalTransformerWorldModel,
    StateTransformerWorldModel,
    LSTMWorldModel,
)
from src.models.latent_dynamics import LatentDynamicsWorldModel
from src.models.gnn_model import GraphWorldModel
from src.models.components import MultiTaskLoss
from src.models.baseline import LogisticRegressionBaseline, RandomForestBaseline


class TestModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.lookback = 10
        self.d_state = 32
        self.d_model = 64
        self.num_mitre_stages = 7

        self.mock_input = torch.randn(self.batch_size, self.lookback, self.d_state)
        self.mock_target_state = torch.randn(self.batch_size, self.d_state)
        self.mock_y_bin = torch.randint(0, 2, (self.batch_size,))
        self.mock_y_mit = torch.randint(0, self.num_mitre_stages, (self.batch_size,))

    def test_transformer_world_model_forward(self):
        """Test Transformer World Model forward pass shapes and outputs."""
        model = StateTransformerWorldModel(
            d_state=self.d_state,
            d_model=self.d_model,
            n_layers=2,
            n_heads=4,
            lookback=self.lookback,
            num_mitre_stages=self.num_mitre_stages,
        )
        model.eval()
        with torch.no_grad():
            out = model(self.mock_input, return_attention=True)

        self.assertIn("pred_state_mean", out)
        self.assertIn("pred_state_logvar", out)
        self.assertIn("pred_binary", out)
        self.assertIn("pred_mitre", out)

        self.assertEqual(out["pred_state_mean"].shape, (self.batch_size, self.d_state))
        self.assertEqual(out["pred_state_logvar"].shape, (self.batch_size, self.d_state))
        self.assertEqual(out["pred_binary"].shape, (self.batch_size, 2))
        self.assertEqual(out["pred_mitre"].shape, (self.batch_size, self.num_mitre_stages))

    def test_temporal_transformer_features(self):
        """Test Temporal Transformer temporal pooling, conv1d and residual dynamics."""
        model = TemporalTransformerWorldModel(
            d_state=self.d_state,
            d_model=self.d_model,
            n_layers=2,
            n_heads=4,
            lookback=self.lookback,
            num_mitre_stages=self.num_mitre_stages,
            residual_dynamics=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(self.mock_input, return_attention=True)

        self.assertIn("temporal_pool_weights", out)
        self.assertEqual(out["temporal_pool_weights"].shape, (self.batch_size, self.lookback))
        self.assertEqual(out["hidden"].shape, (self.batch_size, self.d_model))
        prob = model.predict_infiltration_prob(self.mock_input)
        self.assertEqual(prob.shape, (self.batch_size,))
        sample = model.sample_next_state(self.mock_input, deterministic=True)
        self.assertEqual(sample.shape, (self.batch_size, self.d_state))

    def test_lstm_world_model_forward(self):
        """Test LSTM World Model forward pass."""
        model = LSTMWorldModel(
            d_state=self.d_state,
            d_model=self.d_model,
            lstm_layers=1,
            num_mitre_stages=self.num_mitre_stages,
        )
        model.eval()
        with torch.no_grad():
            out = model(self.mock_input)

        self.assertEqual(out["pred_state_mean"].shape, (self.batch_size, self.d_state))
        self.assertEqual(out["pred_binary"].shape, (self.batch_size, 2))
        self.assertEqual(out["pred_mitre"].shape, (self.batch_size, self.num_mitre_stages))

    def test_latent_dynamics_world_model(self):
        """Test VAE Latent Dynamics World Model."""
        model = LatentDynamicsWorldModel(
            d_state=self.d_state,
            d_latent=16,
            d_model=32,
            num_mitre_stages=self.num_mitre_stages,
        )
        model.eval()
        with torch.no_grad():
            out = model(self.mock_input)

        self.assertIn("pred_state_mean", out)
        self.assertIn("recon_seq", out)
        self.assertEqual(out["pred_binary"].shape, (self.batch_size, 2))

    def test_graph_world_model_fallback(self):
        """Test GraphWorldModel operates gracefully."""
        model = GraphWorldModel(
            d_node=16,
            d_edge=8,
            d_graph=32,
            d_model=self.d_model,
            num_mitre_stages=self.num_mitre_stages,
        )
        self.assertIsNotNone(model)

    def test_multitask_loss(self):
        """Test MultiTaskLoss backward computation and loss breakdown."""
        loss_fn = MultiTaskLoss(
            lambda_dynamics=1.0,
            lambda_infiltration=0.5,
            lambda_mitre=0.3,
        )
        pred_mu = torch.randn(self.batch_size, self.d_state, requires_grad=True)
        pred_logvar = torch.zeros(self.batch_size, self.d_state, requires_grad=True)
        pred_bin = torch.randn(self.batch_size, 2, requires_grad=True)
        pred_mit = torch.randn(self.batch_size, self.num_mitre_stages, requires_grad=True)

        losses = loss_fn(
            pred_mu,
            pred_logvar,
            self.mock_target_state,
            pred_bin,
            self.mock_y_bin,
            pred_mit,
            self.mock_y_mit,
        )

        self.assertIn("total", losses)
        self.assertIn("dynamics", losses)
        self.assertIn("infiltration", losses)
        self.assertIn("mitre", losses)

        losses["total"].backward()
        self.assertIsNotNone(pred_mu.grad)
        self.assertIsNotNone(pred_bin.grad)

    def test_baselines(self):
        """Test LogisticRegression and RandomForest baselines fitting and inference."""
        X_train = np.random.randn(50, self.d_state)
        y_train = np.random.choice([0, 1], size=50)

        lr = LogisticRegressionBaseline(task="binary", max_iter=100)
        lr.fit(X_train, y_train)
        preds = lr.predict(X_train[:10])
        self.assertEqual(len(preds), 10)

        rf = RandomForestBaseline(task="binary", n_estimators=10)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_train[:10])
        self.assertEqual(len(rf_preds), 10)

    def test_masked_state_embedding_sparsity(self):
        """Test MaskedStateEmbedding handles zero-padded feature blocks gracefully."""
        from src.models.components import MaskedStateEmbedding, HeadAdapter
        emb = MaskedStateEmbedding(d_state=self.d_state, d_model=self.d_model)
        
        # Partially sparse input (simulating CTU-13 with padded columns)
        sparse_input = self.mock_input.clone()
        sparse_input[:, :, 10:] = 0.0
        
        out = emb(sparse_input)
        self.assertEqual(out.shape, (self.batch_size, self.lookback, self.d_model))
        self.assertFalse(torch.isnan(out).any())

        adapter = HeadAdapter(d_model=self.d_model)
        h_adapted = adapter(out[:, -1, :])
        self.assertEqual(h_adapted.shape, (self.batch_size, self.d_model))

    def test_zero_day_anomaly_detection(self):
        """Test compute_zero_day_anomaly returns surprise scores and flags outliers."""
        model = TemporalTransformerWorldModel(
            d_state=self.d_state,
            d_model=self.d_model,
            n_layers=2,
            n_heads=4,
            lookback=self.lookback,
            num_mitre_stages=self.num_mitre_stages,
        )
        model.eval()

        # Normal target vs extreme zero-day spike target
        normal_next_state = torch.randn(self.batch_size, self.d_state) * 0.1
        zero_day_next_state = normal_next_state.clone()
        zero_day_next_state[0, 5:10] = 50.0  # massive anomaly spike

        res = model.compute_zero_day_anomaly(self.mock_input, zero_day_next_state, sigma_threshold=3.0)
        self.assertIn("surprise_score", res)
        self.assertIn("is_zero_day_alert", res)
        self.assertIn("top_anomalous_indices", res)
        # First sample with spike should have higher surprise score
        self.assertGreater(res["surprise_score"][0], res["surprise_score"][1])


if __name__ == "__main__":
    unittest.main()

