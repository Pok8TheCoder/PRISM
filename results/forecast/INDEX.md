# Forecasting results — version index

Each round of forecasting experiments lives in its own numbered folder so old
and new results never get mixed up. **Higher number = more recent.**

> **Note:** Rounds v1–v7 were archived to `archive/2026-09-01-cleanup/results/forecast-v1-v7/`
> during the Sep 2026 workspace cleanup. Only v8+ remain in this folder.

| Folder | What it tested | Models | Script that produced it |
|---|---|---|---|
| `v8_ram_aryan_killchain/` | Same RAM-A.01 protocol applied to **ARY.01** (`world_model_best.pt`, d_state=242, 3.8M params). | ARY.01 frozen, RAM-A.01 | `scripts/ram_aryan_killchain_eval.py` *(archived)* |
| `v9_ary01_vs_ary02/` | **ARY.02** (`origin/aryan` `de0d27a`, epoch 22, focal+balanced) vs **ARY.01** (epoch 13). Same architecture, different training. Player now names MITRE stages, attributes features, and can ingest CSV/PCAP/lab live. | ARY.01, ARY.02, RAM-A.01, RAM-A.02 | dashboard Forecast Player + `docs/ARY01_VS_ARY02.md` |

### Archived rounds (v1–v7)

| Folder | Summary |
|---|---|
| `v1_onestep_ae_rf/` | One-step AE-MLP / RF forecasting |
| `v2_recursive_ae_rf/` | First recursive rollout |
| `v3_recursive_5way/` | + XGBoost, Hybrid, Holt |
| `v4_recursive_temporal/` | + Temporal-GRU |
| `v5_adaptive_memory/` | RAM.01 prototype (ctx=20) |
| `v6_ram01_killchain/` | RAM.01 kill-chain (ctx=60) |
| `v7_transformer_showdown/` | GRU vs Transformer + RAMT.01 |

See `archive/2026-09-01-cleanup/results/forecast-v1-v7/` for plots and JSON.

## Current latest: `v9_ary01_vs_ary02/`

ARY.02 is a retrain, not a new net: +recall / +MITRE F1, worse precision/FPR. See `docs/ARY01_VS_ARY02.md`.

## Previous: `v8_ram_aryan_killchain/`

### Headline result
Loaded Aryan's real trained Transformer World Model checkpoint directly (verified it reproduces his reported val score first) and ran it — frozen vs RAM-A.01 wrapped around it — across two synthetic kill-chain timelines built from his own held-out data. Memory/TTT made **no meaningful difference** either time (2000-step: 469.60 → 469.78, +0.02%; 1000-step: 1.7362 → 1.7529, +1.0%), consistent with `v7`'s finding that RAM helps smaller/weaker backbones more than already-strong ones — his 3.8M-param model is ~100x bigger than TFT.01/Temporal-Y and already fits this small dataset (~2800 windows total) very tightly, leaving little room for a lightweight online-adaptation layer to add value.

Overall MSE swung by ~270x between the 2000-step and 1000-step runs (469.6 vs 1.74) even though the same held-out attack segments were used both times — this is a **benign-traffic-sampling artifact, not a training or eval bug**: the benign filler is a deterministic (non-shuffled) tiling of real benign flows, and the longer timeline simply pulls in more of that pool, increasing the odds of including a rare heavy-tailed burst (e.g. one flow with an extreme byte/packet count). Since MSE is computed in per-feature standardized units (scaled by train-split std), a single such outlier in the *test* pool can dominate the aggregate error even though the model's actual behavior is stable — visible in the 1000-step plot as isolated spikes in `feature[0]`/`feature[3]` that both frozen and RAM-A.01 basically fail to predict (nobody sees a burst 20-40 steps before it happens), while day-to-day benign dynamics track reasonably well.

## Previous: `v7_transformer_showdown/`

### Headline result
Frozen Transformer (TFT.01) beat frozen GRU (Temporal-Y) outright: **6.16 vs 8.52 MSE (-27.7%)**, at ~2.5x the parameter count (83K vs 33K — still far below the 256-dim/16-head classifier config, kept deliberately small for a fair fight). Memory+TTT improved **both** backbones (GRU -4.6%, Transformer -1.9%), and RAMT.01 (Transformer+memory) was the best candidate overall at **6.04 MSE**. So: Transformer > GRU on raw forecasting accuracy on this dataset, and RAM's memory/adaptation layer is genuinely backbone-agnostic — it helps whichever backbone it's bolted onto, just by a smaller *relative* margin on the backbone that was already more accurate. See `showdown_summary.json` for full per-segment numbers and memory-retrieval accuracy (RAMT.01 also edged out RAM.01 on classification-by-retrieval: 76.9% vs 69.2%).

## Key running finding across versions
Every non-recurrent model (RF, XGBoost, plain AE-MLP, flat-window Hybrid) either
freezes into a near-constant line or drifts/oscillates unrealistically under
recursive rollout — see `v2`/`v3` for the mechanism (RF's piecewise-constant
leaf averaging collapses to a 2-state limit cycle; unconstrained MLPs can
explode; bounding only the *step size* doesn't bound cumulative *drift*).
`v4` tests whether giving the model real temporal memory (GRU) instead of
just a flat window of recent states changes this.
