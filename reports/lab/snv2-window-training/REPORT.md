# SNV2 Window Size Retraining (1s / 5s / 15s)

**Date:** 2026-09-03  
**Pipeline:** `PRISM-shaun/scripts/train.py`  
**Data:** CIC-IDS-2018 CSVs from `PRISM/data/raw/`  

## Benchmark results (held-out test)

| Window | Total windows | Attack windows | F1 | Recall | FPR | ROC-AUC | MITRE Acc | Checkpoint |
|--------|----------------:|---------------:|---:|-------:|----:|--------:|----------:|------------|
| **15s** (original) | ~5,540 | — | 0.9035 | 0.8766 | 0.0263 | 0.9883 | 0.9293 | `weights/world_model.pt` |
| **5s** | 28,822 | 5,060 | **0.9626** | **0.9945** | 0.0032 | **0.9997** | **0.9935** | `weights/w5s/world_model.pt` |
| **1s** | 124,572 | 23,266 | 0.9309 | **0.9992** | 0.0109 | **0.9999** | 0.9893 | `weights/w1s/world_model.pt` |

## Conclusions

- **5s window** is the best F1 / ROC-AUC tradeoff for offline CIC benchmark
- **1s window** maximizes recall (0.999) at cost of slightly lower F1 and higher FPR
- **15s baseline** underperforms on recall vs shorter windows

## Training notes

- GPU training with batch 512, AMP on RTX 3060
- Datasets: `PRISM-shaun/data/processed/w1s/`, `w5s/`
- Comparison doc: `PRISM-shaun/results/window_comparison.md`

## Live lab usage

- w5s checkpoint used as default in `live_ips_multi.py` (`--shaun-ckpt`)
- 5s attack capture windows align with w5s training granularity
