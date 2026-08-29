"""
Unit tests for PRISM Data Pipeline.
Tests flow extraction, packet parsing heuristics, feature merging,
state building, and dataset temporal loaders.
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import torch

from src.data.flow_extractor import FlowExtractor
from src.data.packet_extractor import PacketExtractor, _is_sequential
from src.data.feature_merger import FeatureMerger
from src.data.state_builder import StateBuilder
from src.data.dataset import StateSequenceDataset, create_dataloaders


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic raw flow DataFrame matching CIC-IDS format
        np.random.seed(42)
        n = 100
        self.mock_flow_df = pd.DataFrame({
            "Timestamp": pd.date_range("2026-01-01 10:00:00", periods=n, freq="1s"),
            "Dst Port": np.random.choice([80, 443, 22, 8080, 21], size=n),
            "Protocol": np.random.choice([6, 17], size=n),
            "Flow Duration": np.random.exponential(1000, size=n),
            "Tot Fwd Pkts": np.random.randint(1, 50, size=n),
            "Tot Bwd Pkts": np.random.randint(0, 50, size=n),
            "TotLen Fwd Pkts": np.random.randint(100, 5000, size=n),
            "TotLen Bwd Pkts": np.random.randint(0, 5000, size=n),
            "Flow Byts/s": np.random.uniform(10, 1000, size=n),
            "Flow Pkts/s": np.random.uniform(1, 100, size=n),
            "Flow IAT Mean": np.random.uniform(0.1, 10, size=n),
            "Flow IAT Std": np.random.uniform(0.01, 5, size=n),
            "Flow IAT Max": np.random.uniform(1, 20, size=n),
            "Flow IAT Min": np.random.uniform(0.01, 1, size=n),
            "Fwd IAT Mean": np.random.uniform(0.1, 10, size=n),
            "Fwd IAT Std": np.random.uniform(0.01, 5, size=n),
            "Fwd IAT Max": np.random.uniform(1, 20, size=n),
            "Bwd IAT Mean": np.random.uniform(0.1, 10, size=n),
            "Bwd IAT Std": np.random.uniform(0.01, 5, size=n),
            "Bwd IAT Max": np.random.uniform(1, 20, size=n),
            "FIN Flag Cnt": np.random.choice([0, 1], size=n),
            "SYN Flag Cnt": np.random.choice([0, 1], size=n),
            "RST Flag Cnt": np.random.choice([0, 1], size=n),
            "PSH Flag Cnt": np.random.choice([0, 1], size=n),
            "ACK Flag Cnt": np.random.choice([0, 1], size=n),
            "URG Flag Cnt": np.zeros(n),
            "Src IP": ["192.168.1.10"] * n,
            "Dst IP": ["10.0.0.1"] * n,
            "Label": np.random.choice(["Benign", "FTP-BruteForce", "Infilteration"], size=n),
        })

    def test_packet_extractor_heuristics(self):
        """Test sequential port scan detection heuristic."""
        seq_ports = [100, 101, 102, 103, 104, 105]
        non_seq_ports = [80, 443, 22, 8080, 53]
        self.assertTrue(_is_sequential(seq_ports, min_length=5))
        self.assertFalse(_is_sequential(non_seq_ports, min_length=5))

    def test_flow_extractor_transformations(self):
        """Test FlowExtractor clean, feature engineering, normalisation, and label mapping."""
        extractor = FlowExtractor(dataset_type="cicids2018")
        df_cleaned = extractor.clean(self.mock_flow_df)
        self.assertFalse(df_cleaned.isnull().values.any())

        df_feat = extractor.engineer_features(df_cleaned)
        self.assertIn("bytes_per_packet", df_feat.columns)
        self.assertIn("fwd_bwd_ratio", df_feat.columns)

        df_log = extractor.log_transform(df_feat)
        self.assertIn("Flow Duration", df_log.columns)

        df_norm = extractor.normalize(df_log, fit=True)
        self.assertTrue(hasattr(extractor, "scaler"))

        df_labeled = extractor.map_labels(df_norm)
        self.assertIn("mitre_stage", df_labeled.columns)
        self.assertIn("mitre_stage_id", df_labeled.columns)

    def test_feature_merger(self):
        """Test FeatureMerger joins flow and packet features with zero filling."""
        merger = FeatureMerger()
        # Merge with None packet df
        merged = merger.merge(self.mock_flow_df, packet_df=None)
        self.assertIn("has_packet_features", merged.columns)
        self.assertEqual(merged["has_packet_features"].sum(), 0)

    def test_state_builder_vector_mode(self):
        """Test StateBuilder converts flow records into temporal state vectors."""
        extractor = FlowExtractor(dataset_type="cicids2018")
        df = extractor.map_labels(self.mock_flow_df)
        builder = StateBuilder(window_size_seconds=10, mode="vector")
        result = builder.build_states(df)

        self.assertIn("states", result)
        self.assertIn("labels_binary", result)
        self.assertIn("labels_mitre", result)
        self.assertIn("feature_names", result)
        self.assertGreater(result["states"].shape[0], 0)
        self.assertGreater(result["states"].shape[1], 10)

    def test_state_sequence_dataset(self):
        """Test StateSequenceDataset yields lookback sequences and targets."""
        T, D = 50, 64
        states = np.random.randn(T, D).astype(np.float32)
        labels_bin = np.random.choice([0, 1], size=T)
        labels_mit = np.random.choice(range(7), size=T)

        lookback = 10
        dataset = StateSequenceDataset(
            states=states,
            labels_binary=labels_bin,
            labels_mitre=labels_mit,
            lookback=lookback,
        )

        self.assertEqual(len(dataset), T - lookback)
        item = dataset[0]
        self.assertEqual(item["state_seq"].shape, (lookback, D))
        self.assertEqual(item["next_state"].shape, (D,))
        self.assertIsInstance(item["label_binary"], torch.Tensor)
        self.assertIsInstance(item["label_mitre"], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
