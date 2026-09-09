"""Unit tests for StreamingGen8WorldModel."""

import unittest
import numpy as np
import torch

from src.models.streaming_gen8 import StreamingGen8WorldModel, load_gen8_checkpoint
from src.models.gen8_world_model import Gen8DecoupledMLPWorldModel


class TestStreamingGen8WorldModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Instantiate model directly or via checkpoint
        cls.streamer = StreamingGen8WorldModel()

    def test_single_step(self):
        state = np.random.randn(242).astype(np.float32)
        out = self.streamer.step(state)

        self.assertIn("p_att", out)
        self.assertIn("p_mit", out)
        self.assertIn("pred_state", out)
        self.assertIn("hidden", out)
        self.assertIn("contrastive_z", out)
        self.assertIn("mitre_stage", out)
        self.assertIn("mitre_label", out)

        self.assertIsInstance(out["p_att"], float)
        self.assertGreaterEqual(out["p_att"], 0.0)
        self.assertLessEqual(out["p_att"], 1.0)

        self.assertEqual(out["p_mit"].shape, (7,))
        self.assertAlmostEqual(float(out["p_mit"].sum()), 1.0, places=4)

        self.assertEqual(out["pred_state"].shape, (242,))
        self.assertEqual(out["hidden"].shape, (256,))
        self.assertEqual(out["contrastive_z"].shape, (128,))

    def test_buffer_sliding(self):
        self.streamer.reset()
        for i in range(25):
            s = np.full(242, float(i), dtype=np.float32)
            out = self.streamer.step(s)
            self.assertEqual(out["step"], i + 1)

        self.assertLessEqual(len(self.streamer.buffer), 40)

    def test_forecast(self):
        fc = self.streamer.forecast(horizon=10)
        self.assertIn("states", fc)
        self.assertIn("p_att", fc)
        self.assertIn("p_mit", fc)
        self.assertEqual(fc["states"].shape, (10, 242))
        self.assertEqual(len(fc["p_att"]), 10)
        self.assertEqual(fc["p_mit"].shape, (10, 7))


if __name__ == "__main__":
    unittest.main()
