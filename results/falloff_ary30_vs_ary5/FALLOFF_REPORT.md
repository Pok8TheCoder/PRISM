# ARY.01 vs ARY-5sV01 — Open-Loop Forecast Falloff

- **ARY.01**: 30s windows, checkpoint `aryan_world_model_best.pt`
- **ARY-5sV01**: 5s windows, checkpoint `ary_5sv01.pt`
- **Scenario**: 10 min context → 20 min open-loop forecast
- **Falloff rule**: first forecast step where abs error > 2.0x median of early steps
- **ARY.01**: 20 ctx + 40 forecast steps @ 30s
- **ARY-5sV01**: 120 ctx + 240 forecast steps @ 5s
- **Features evaluated**: 242

## Summary

| Metric | ARY.01 (30s) | ARY-5sV01 (5s) |
|---|---:|---:|
| Median falloff step | 12.0 | 39.0 |
| Mean falloff step | 12.5 | 51.1 |
| **Median falloff (wall-clock sec)** | **360.0** | **195.0** |
| Mean falloff (wall-clock sec) | 376 | 256 |
| Never falloff (of 242 feats) | 124 | 19 |

## Head-to-head (by forecast step)

- ARY-5s falls off **earlier** (fewer steps): **14** features
- ARY.01 falls off earlier: **94** features
- Same step / both never: **15**

## Head-to-head (by wall-clock seconds)

- ARY-5s falls off earlier in real time: **81** features
- ARY.01 falls off earlier in real time: **29** features

## Interpretation

ARY-5sV01 median falloff occurs at **195s** vs ARY.01 **360s** — shorter windows do **not** buy longer trustworthy forecast horizon in wall-clock time; 5s falls off **sooner in real seconds**.

Note: one forecast step = one window (30s or 5s). More steps at 5s ≠ longer real-time horizon per step.

## Per-feature falloff (earliest 15 where 5s loses in wall-clock time)

| Feature | ARY.01 step (sec) | ARY-5s step (sec) | Δ sec (5−30) |
|---|---:|---:|---:|
| mean_flow_92 | 39 (1170s) | 23 (115s) | -1055 |
| flag_frac_RST | 38 (1140s) | 23 (115s) | -1025 |
| std_flow_47 | 40 (1200s) | 39 (195s) | -1005 |
| mean_flow_102 | 37 (1110s) | 23 (115s) | -995 |
| std_flow_57 | 36 (1080s) | 23 (115s) | -965 |
| std_flow_27 | 30 (900s) | 4 (20s) | -880 |
| std_flow_34 | 31 (930s) | 26 (130s) | -800 |
| std_flow_108 | 21 (630s) | 33 (165s) | -465 |
| std_flow_17 | 22 (660s) | 41 (205s) | -455 |
| std_flow_9 | 16 (480s) | 11 (55s) | -425 |
| mean_flow_61 | 15 (450s) | 8 (40s) | -410 |
| mean_flow_60 | 15 (450s) | 8 (40s) | -410 |
| mean_flow_70 | 14 (420s) | 4 (20s) | -400 |
| std_flow_32 | 16 (480s) | 20 (100s) | -380 |
| std_flow_72 | 14 (420s) | 13 (65s) | -355 |