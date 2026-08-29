"""
Unit tests for PRISM Explainability Suite: Attention Visualization,
Feature Attribution (Integrated Gradients / SHAP), and Interpretability Reports.
"""

from __future__ import annotations

import unittest
import numpy as np
import torch

from src.models.world_model import StateTransformerWorldModel
from src.explainability.attention_viz import AttentionVisualiser
from src.explainability.feature_importance import FeatureImportanceAnalyser


class TestExplainability(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.d_state = 8
        self.lookback = 6
        self.feature_names = [f"feat_{i}" for i in range(self.d_state)]

        self.model = StateTransformerWorldModel(
            d_state=self.d_state,
            d_model=16,
            n_layers=1,
            n_heads=2,
            lookback=self.lookback,
        )
        self.state_seq = np.random.randn(self.lookback, self.d_state).astype(np.float32)

    def test_attention_visualiser(self):
        """Test extraction of Transformer self-attention weights."""
        viz = AttentionVisualiser(feature_names=self.feature_names)
        tensor_in = torch.tensor(self.state_seq, dtype=torch.float32).unsqueeze(0)
        res = viz.extract_weights(self.model, tensor_in)

        self.assertIn("last_layer_attn", res)
        attn = res["last_layer_attn"]
        self.assertEqual(attn.shape, (self.lookback, self.lookback))
        self.assertTrue((attn >= 0).all())
        self.assertAlmostEqual(float(attn[-1].sum()), 1.0, delta=0.2)

    def test_feature_importance_integrated_gradients(self):
        """Test Integrated Gradients feature attribution calculation."""
        analyser = FeatureImportanceAnalyser(
            model=self.model,
            feature_names=self.feature_names,
            device="cpu",
        )
        ig_result = analyser.integrated_gradients(self.state_seq, n_steps=10)

        self.assertIn("attributions", ig_result)
        self.assertIn("mean_abs", ig_result)
        self.assertEqual(len(ig_result["mean_abs"]), self.d_state)

        # Test ranking top features
        top_k = ig_result["top_features"][:3]
        self.assertGreater(len(top_k), 0)
        self.assertIn("feature", top_k[0])
        self.assertIn("attribution", top_k[0])

    def test_gradient_x_input(self):
        """Test fast Gradient x Input attribution."""
        analyser = FeatureImportanceAnalyser(
            model=self.model,
            feature_names=self.feature_names,
            device="cpu",
        )
        gxi_res = analyser.gradient_x_input(self.state_seq)
        self.assertIn("attributions", gxi_res)
        self.assertEqual(gxi_res["attributions"].shape, (1, self.lookback, self.d_state))


if __name__ == "__main__":
    unittest.main()
