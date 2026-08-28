"""Build state sequences directly from processed CSV."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.data.state_builder import StateBuilder
from src.data.dataset import create_temporal_splits, save_splits

print("Loading processed CSV...")
df = pd.read_csv("data/processed/cicids2018_processed.csv", low_memory=False)
print(f"Loaded {len(df)} rows. Building state vectors (30s windows)...")

builder = StateBuilder(window_size_seconds=30)
result = builder.build_states(df)
states_shape = result['states'].shape
print(f"Built states shape: {states_shape}")

out_npz = "data/processed/cicids2018_states.npz"
np.savez(out_npz,
         states=result['states'],
         labels_binary=result['labels_binary'],
         labels_mitre=result['labels_mitre'],
         window_ids=result['window_ids'])
print(f"Saved state sequences -> {out_npz}")

splits = create_temporal_splits(result['states'], result['labels_binary'], result['labels_mitre'])
save_splits(splits, "data/splits")
print("Saved train/val/test splits -> data/splits")
print("Done! Ready for GPU training.")
