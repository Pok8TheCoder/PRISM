"""Quick preprocess CIC-IDS-2018 sample for training."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.data.flow_extractor import FlowExtractor
from src.data.state_builder import StateBuilder
from src.data.dataset import create_temporal_splits, save_splits

print('=== Loading CIC-IDS-2018 CSV files (sampling 200k rows each) ===')
extractor = FlowExtractor(dataset_type='cicids2018', raw_dir='data/raw')

dfs = []
files = ['02-14-2018.csv', '02-16-2018.csv', '02-22-2018.csv', '03-01-2018.csv']
for f in files:
    path = f'data/raw/cicids2018/{f}'
    print(f'  Loading {f}...', end=' ')
    chunk = pd.read_csv(path, nrows=200000, low_memory=False)
    chunk.columns = [str(c).strip() for c in chunk.columns]
    print(f'{len(chunk)} rows')
    dfs.append(chunk)

df = pd.concat(dfs, ignore_index=True)
print(f'Total raw rows: {len(df)}')

print('Cleaning...')
df = extractor.clean(df)

print('Engineering features...')
df = extractor.engineer_features(df)

print('Log transform...')
df = extractor.log_transform(df)

print('Normalising...')
df = extractor.normalize(df, fit=True)
extractor.save_scaler('weights/scaler.pkl')

print('Mapping labels...')
df = extractor.map_labels(df)
print(f'Final: {len(df)} rows')
print('Stage distribution:')
print(df['mitre_stage'].value_counts().to_string())

os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/cicids2018_processed.csv', index=False)
print('Saved processed CSV')

print('Building state sequences...')
builder = StateBuilder(window_size_seconds=30)
result = builder.build_states(df)
states_shape = result['states'].shape
print(f'States shape: {states_shape}')

np.savez('data/processed/cicids2018_states.npz',
         states=result['states'],
         labels_binary=result['labels_binary'],
         labels_mitre=result['labels_mitre'],
         window_ids=result['window_ids'])
print('Saved states.npz')

splits = create_temporal_splits(result['states'], result['labels_binary'], result['labels_mitre'])
save_splits(splits, 'data/splits')
print('Done! Pipeline complete.')
