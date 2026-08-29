# Forecasting results — version index

Each round of forecasting experiments lives in its own numbered folder so old
and new results never get mixed up. **Higher number = more recent.**

| Folder | What it tested | Models | Script that produced it |
|---|---|---|---|
| `v1_onestep_ae_rf/` | One-step forecasting: given the **true** previous state, predict the next one. | AE-MLP, Random Forest | `scripts/train_forecast_models.py` (metrics) + `scripts/plot_forecast.py` (plots) |
| `v2_recursive_ae_rf/` | Recursive rollout: feed the model's **own prediction** back in as the next input, all the way through a whole capture, no ground truth after step 0. First look at whether RF/AE-MLP can sustain multi-step forecasts. | AE-MLP, Random Forest | `scripts/test_recursive_rollout.py` |
| `v3_recursive_5way/` | Same recursive rollout protocol, expanded to 5 candidates after testing the "RF flatlines, does XGBoost too?" question and prototyping a bounded-delta hybrid. Warmup changed to first 3 real states. | AE-MLP, Random Forest, **XGBoost**, **Hybrid** (windowed input + bounded delta, but *not* recurrent), **Holt damped-trend** (no training) | `scripts/test_recursive_rollout_v2.py` |
| `v4_recursive_temporal/` | Same protocol again, after replacing the flat-window "Hybrid" with a genuinely **recurrent** (GRU) temporal forecaster — the first candidate here that actually models order/sequence instead of only seeing a concatenated snapshot. | AE-MLP, Random Forest, XGBoost, Hybrid (non-recurrent), **Temporal-GRU** (recurrent + bounded delta), Holt damped-trend | `scripts/test_recursive_rollout_v3.py` |
| `v5_adaptive_memory/` | Prototype of **RAM.01** (Receding-horizon Adaptive Memory): the Temporal-GRU + test-time weight adaptation every horizon-chunk + a surprise-triggered episodic memory bank that recognizes recurring attack-stage shapes across *different* captures. Context=20, horizon=20. Memory seeded from TRAIN captures only, tested cold on held-out TEST captures. | frozen / adaptive / adaptive+memory | `scripts/adaptive_memory_forecaster.py` |
| `v6_ram01_killchain/` | RAM.01 (v2) vs frozen Temporal-Y (v2) vs frozen Temporal-A (amt), all retrained with **context=60 / horizon=40**, run across a synthetic ~1000-step session with 3 real attacks (scan, http flood, DoS) spliced into benign background traffic. | Temporal-Y (frozen), Temporal-A (frozen), **RAM.01** | `scripts/ram01_kill_chain_eval.py` (models trained by `train_temporal_forecaster_ctx60.py`) |
| `v7_transformer_showdown/` | The "GRU vs Transformer" debate settled with real numbers: a genuine self-attention **Transformer forecaster (TFT.01)**, same context=60/horizon=40 protocol and same data as Temporal-Y, plus RAM.01's memory+TTT layer wired onto it unmodified (**RAMT.01**, no new memory/TTT code — `EpisodicMemoryBank`/`OnlineAdaptive` are already model-agnostic). Same synthetic kill-chain timeline as v6, v2 schema only (single-schema comparison, apples-to-apples). | Temporal-Y (frozen GRU), **TFT.01** (frozen Transformer), RAM.01 (GRU+memory), **RAMT.01** (Transformer+memory) | `scripts/transformer_showdown_kill_chain_eval.py` (Transformer trained by `train_transformer_forecaster.py`) |
| `v8_ram_aryan_killchain/` | Same RAM-A.01 protocol applied to **Aryan's actual pushed checkpoint** (`weights/transformer/world_model_best.pt`, d_state=242, 3.8M params, trained on his own CIC-IDS-2018 sample) — his real weights loaded directly and reproduced his reported val metrics, **no retraining**. Kill-chain built from his own held-out val/test splits (Initial Access from val, Lateral Movement from test, Impact from train — noted as seen-during-training). Run at both 2000-step and 1000-step lengths (`--steps` CLI arg) to check sensitivity to timeline length. | Aryan-WM (frozen), **RAM-A.01** (Aryan-WM + TTT + memory) | `scripts/ram_aryan_killchain_eval.py` (run from the `aryan` branch worktree, output saved into main repo) |

## Current latest: `v8_ram_aryan_killchain/`

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
