# ARY.01 vs ARY.02

Two trained checkpoints of the **same** `TemporalTransformerWorldModel`
(d_state=242, d_model=256, 4 layers, 8 heads, 3.8M params). ARY.02 is a
retrain on `origin/aryan` (`de0d27a`), not a new architecture.

| | **ARY.01** | **ARY.02** |
|---|---|---|
| Git | `1ec06bc` | `de0d27a` (2026-08-29 12:13) |
| Local file | `models/checkpoints/aryan_world_model_best.pt` | `models/checkpoints/aryan_world_model_ary02.pt` |
| Best epoch | 13 | 22 |
| Val score | 0.870 | **0.945** (new formula) |
| Val F1 | 0.584 | **0.603** |
| Val loss | **1.745** | 2.104 |
| Test F1 | 0.696 | **0.708** |
| Test precision / FPR | **0.976 / 0.003** | 0.729 / 0.058 |
| Test recall | 0.541 | **0.689** |
| Test MITRE F1 macro | 0.879 | **0.899** |
| Dynamics MSE | ~258 | ~258 |
| Recipe | original multi-task | focal γ=2, 50/50 sampler, higher λ_inf |

ARY.02 catches more attacks (recall) and names MITRE stages slightly better.
It is noisier (more false positives). Next-state dynamics quality is unchanged.

Forecast Player slots: **ARY.01 (frozen)**, **ARY.02 (frozen)**, **RAM-A.01**, **RAM-A.02**.
