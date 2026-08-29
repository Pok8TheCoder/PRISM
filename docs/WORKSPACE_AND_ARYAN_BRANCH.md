# PRISM Workspace & `origin/aryan` Branch — Consolidated Reference

> **Generated:** 2026-08-29  
> **Workspace:** `d:\Cursor\PRISM`  
> **Active branch:** `main` (tracking `origin/main`)  
> **Related remote:** `origin/aryan` (4 commits ahead of shared base)

This document ties together the **current main-branch workspace** (including uncommitted research work) and the **Aryan contributor branch** (`origin/aryan`). It is the single entry point for understanding what exists where, how the two lines of development relate, and which numbers are trustworthy.

---

## 1. Executive summary

**PRISM** (*Predictive Risk / Recurrent Infiltration State Model*) is an offline World Model for network attack forecasting: it learns temporal state-transition dynamics from flow and packet telemetry, maps predictions to **MITRE ATT&CK** stages, and supports K-step forward simulation for proactive defense.

Two parallel implementations share a common ancestor but diverged in scope:

| Aspect | **Main (`main`)** | **Aryan (`origin/aryan`)** |
|---|---|---|
| Primary focus | Docker adversarial lab, 33-class MITRE bot catalog, Streamlit SOC dashboard, rigorous schema benchmarks | Full multi-dataset pipeline, GNN/LSTM/Transformer/Latent variants, CAPEC/CVE knowledge layer, packaged training CLI |
| Feature schemas | v1 (27f), v2 (64f), **AMT.01 port of Aryan's StateBuilder (82f)** | Original **StateBuilder** (~110f design intent, 82f in fair comparison) |
| World model | `src/model/world_model_multiclass.py` — multiclass transformer on lab traffic | `src/models/world_model.py` — multi-head transformer (dynamics + binary + MITRE stage) |
| Benchmark rigor | Real traffic: 263 lab PCAPs + CIC-IDS-2018 CSV days; capture-level splits | Committed benchmark used **synthetic demo data** (`demo_states.npz`); real CIC splits exist but were not what produced `benchmark_results.json` |
| Checkpoints | `models/checkpoints/` (ZMT/YMT/AMT classifiers, forecasters) | `weights/` (transformer 3.8M params, LSTM, GNN, latent, LR/RF baselines) |
| Latest research | Forecasting v1–v8, RAM adaptive memory, transformer showdown, Aryan checkpoint kill-chain eval | End-to-end app under `app/`, notebooks, test suite |

**Bottom line:** Aryan's **StateBuilder feature schema** is the strongest classifier input in fair head-to-head tests on real traffic. His **published benchmark JSON should not be cited** — it was trained on Gaussian noise with injected constants. Main has absorbed his schema via `src/pipeline/extract_aryan.py` and validated his real checkpoint in forecasting experiment **v8**.

---

## 2. Git topology

```
455be82  Initial commit
    │
    ├─ … shared history …
    │
27cb629  feat: isolated lab, 32-class catalog, PS-complete inference stack  ← merge-base
    │
    ├── main (current workspace)
    │     └── + large uncommitted research layer (scripts/, results/, docs/, pipeline ports)
    │
    └── origin/aryan (+4 commits)
          6144dc0  full World Model + multi-dataset + CAPEC/CVE
          d5feed3  GPU training all 4 architectures + benchmark plot
          2c2549c  complete pipeline, 200-epoch CIC benchmark, Streamlit app, tests
          1ec06bc  trained checkpoints, baselines, CIC-IDS-2018 splits
```

- **`main` has no commits beyond `27cb629`** that are not also ancestors of `aryan`.
- **All Aryan-specific work lives on `origin/aryan`**, not merged into `main`.
- **Current working tree** on `main` has **5 modified tracked files** and **~60+ untracked files** (research scripts, results, docs, pipeline extensions).

---

## 3. Main branch — current workspace

### 3.1 What shipped on `main`

The committed `main` branch (at `27cb629`) delivers a **lab-driven adversarial prototype**:

| Module | Path | Role |
|---|---|---|
| Isolated Docker lab | `docker/`, `scripts/lab_ctl.py` | Target server + attacker bot on internal network |
| 32+ MITRE attack bots | `src/adversarial/bots/` | One Python bot per technique (T1046 scan, T1499 flood, …) |
| Self-fortifying loop | `src/adversarial/training_loop.py` | Capture → extract → predict → evade → retrain |
| Feature pipeline v1 | `src/pipeline/features.py`, `extract.py` | 27-column per-flow vector (20 flow + 7 packet) |
| Multiclass world model | `src/model/world_model_multiclass.py` | Temporal transformer, 33 MITRE-mapped classes |
| K-step engine | `src/predict/engine.py` | Forward rollout simulation |
| Explainability | `src/explain/attribution.py` | SHAP / attention hooks |
| SOC dashboard | `src/ui/app.py`, `run_dashboard.py` | Streamlit tabs: PS Demo, Live Monitor, Runbook, Retrain, MITRE Catalog |
| MITRE KB | `data/mitre_attack.json` | 222 Enterprise techniques |
| Baseline | `src/baseline.py` | Logistic regression (54% F1 → WM 85% F1 on early benchmark) |

### 3.2 Uncommitted research layer (local only)

The workspace contains substantial work **not yet committed** to `main`. This is the active research surface.

#### Feature schema & classifier benchmarks

| Artifact | Description |
|---|---|
| `src/pipeline/features_v2.py` | v2 window schema (64 features) |
| `src/pipeline/extract_v2.py` | Trailing 8-flow window aggregation |
| `src/pipeline/extract_aryan.py` | **Faithful port** of Aryan's `StateBuilder::_aggregate_window_vector` (82 features) |
| `scripts/build_comparison_dataset.py` | Shared capture-split dataset for fair comparison |
| `scripts/train_zmt_ymt.py` | Trains ZMT.01 + YMT.01 (3 seeds each) |
| `scripts/train_amt.py` | Trains AMT.01 (Aryan schema on same traffic) |
| `scripts/analyze_zmt_ymt.py`, `ablate_ymt_blocks.py`, `plot_zmt_ymt.py` | Analysis, ablation, plots |
| `docs/MODEL_COMPARISON.md`, `docs/FEATURE_SCHEMA_V2.md` | Full write-ups |
| `results/zmt_ymt/` | Metrics, ablation, comparison chart |

#### Forecasting experiment lineage (v1 → v8)

Documented in `results/forecast/INDEX.md`:

| Version | Question answered |
|---|---|
| **v1** | One-step next-state prediction (AE-MLP, RF) |
| **v2** | Recursive rollout — RF/AE collapse under feedback |
| **v3** | + XGBoost, hybrid, Holt baseline |
| **v4** | + Temporal-GRU (first recurrent forecaster) |
| **v5** | RAM.01 prototype (TTT + episodic memory) |
| **v6** | RAM vs frozen GRU on synthetic kill-chain (ctx=60) |
| **v7** | Transformer (TFT.01) vs GRU; RAMT.01 best at 6.04 MSE |
| **v8** | **Aryan's real 3.8M-param checkpoint** — RAM-A.01 adds ~0% |

#### Dashboard & lab enhancements (modified, uncommitted)

| File | Changes |
|---|---|
| `src/ui/app.py` | Live wire banner, verdict cards, packet feed integration |
| `src/ui/live_feed.py` | **New** — Scapy packet visualizer, JSONL live stream |
| `src/ui/lab_controller.py` | Background loop, train-on-all-dashboard-attacks |
| `src/adversarial/training_loop.py` | Expanded loop telemetry for live feed |
| `run_dashboard.py` | Entry-point tweaks |

---

## 4. Aryan branch (`origin/aryan`)

### 4.1 Philosophy & packaging

Aryan's branch reframes PRISM as a **production-shaped research platform**:

- **Multi-dataset registry** — CIC-IDS-2018, CTU-13, UNSW-NB15, CICIoT2023, LANL, DARPA
- **Unified extractors** — `src/data/flow_extractor.py`, `packet_extractor.py`, per-dataset adapters
- **StateBuilder** — `src/data/state_builder.py` converts merged features into **30-second wall-clock windows**
- **Knowledge bases** — MITRE stages, CAPEC patterns, CVE/NVD offline lookups
- **Four model families** — Transformer, LSTM, GNN (GraphSAGE), Latent dynamics
- **Full eval stack** — `src/evaluation/benchmark.py`, ablation, metrics, SHAP/attention viz
- **Multi-page Streamlit app** — `app/streamlit_app.py` + pages (timeline, explainability, attack map, upload)
- **Tests & notebooks** — `tests/`, `notebooks/01–04_*.ipynb`
- **Shipped weights** — `weights/transformer/world_model_best.pt` (3.8M params), plus LSTM/GNN/latent/baselines

### 4.2 StateBuilder schema (the part that held up)

The core aggregation in `_aggregate_window_vector` produces, per window:

1. **Composition** — `num_flows`, unique src/dst IPs, unique dst ports, port entropy  
2. **Dispersion** — mean **and** std of all numeric columns (58 of 82 features in the ported version)  
3. **Flag profile** — SYN/ACK/FIN/RST/PSH/URG fractions (by flow count)  
4. **Protocol mix** — TCP/UDP/ICMP fractions  
5. **Top-port counts** — hits on ports 21, 22, 23, 25, 53, 80, 110, 135, 139, 143  

On main, this is reimplemented as **AMT.01** in `extract_aryan.py`, using an **8-flow trailing window** (not 30-second windows) so comparisons isolate *schema* from *windowing policy*.

### 4.3 World model architecture (Aryan)

From `origin/aryan:src/models/world_model.py`:

```
Input: (B, L, D_state)  — L=lookback windows, D_state≈110 (config) / 242 (trained checkpoint)
  → StateEmbedding (Linear + LayerNorm)
  → Causal TemporalConv1D
  → Positional encoding (learnable or sinusoidal)
  → Causal Transformer encoder (4 layers, 8 heads, d_model=256)
  → Temporal attention pooling
  → Residual dynamics head: S_{t+1} = S_t + ΔS_t
  → Binary infiltration head
  → MITRE stage classification head
```

Trained checkpoint used in v8 eval:

| Property | Value |
|---|---|
| Path (on aryan branch) | `weights/transformer/world_model_best.pt` |
| Parameters | **3,800,941** |
| Best epoch | 13 |
| Val score (reproduced) | 0.870 |
| Val F1 | 0.584 |
| Val loss | 1.745 |

### 4.4 What did **not** hold up — the committed benchmark

`origin/aryan:results/benchmark/benchmark_results.json` was generated from **`demo_states.npz`**:

```python
# scripts/download_data.py::generate_demo_data (on aryan branch)
T, D = 500, 50
states[:] = np.random.randn(T, D) * 0.3
# attacks injected as fixed +2..+4 offsets on hand-picked dimensions
```

Evidence it is synthetic:

- Exactly 200 benign / 300 attack windows, 50 per MITRE stage  
- `per_feature_mse` has exactly **50** entries while `configs/model.yaml` sets `d_state: 110`  
- **LR and RF baselines score F1 = 1.000** — linearly separable by construction  
- World Model *loses* to LR on this data → benchmark argues **against** the transformer on fake data

**Do not cite those numbers.** Use AMT.01 results on real traffic (§5) or Aryan's reproduced val metrics on his real CIC sample (§6.3).

### 4.5 The 374-feature `FEATURES.md` list

Aryan's branch includes `docs/FEATURES.md` documenting a **374-column union** across six datasets. Important caveats (detailed in `MODEL_COMPARISON.md` §7):

- Never extracted from real traffic as a unified matrix  
- Contains label columns, identifiers, host-only telemetry, ~40% duplicated port one-hots  
- **Not consumed by StateBuilder** — separate aspirational catalog  
- Main's root `FEATURES.md` mirrors this list; it describes design intent, not the live pipeline  

---

## 5. Feature schema head-to-head (real traffic)

All three schemas project from the **same 29-column base record** and **same 8-flow trailing window**. Only the feature engineering differs.

| Model | Schema | Columns | Checkpoint |
|---|---|---:|---|
| **ZMT.01** | v1 per-flow | 27 | `models/checkpoints/zmt_01.pth` |
| **YMT.01** | v2 window (designed on main) | 64 | `models/checkpoints/ymt_01.pth` |
| **AMT.01** | Aryan StateBuilder port | 82 | `models/checkpoints/amt_01.pth` |

### 5.1 Classifier results (held-out test, 3-seed mean)

| Metric | ZMT.01 | YMT.01 | AMT.01 |
|---|---:|---:|---:|
| Accuracy (33-class) | 0.809 | 0.849 | **0.857** |
| Macro F1 | 0.872 | 0.882 | **0.891** |
| Binary F1 | 0.921 | 0.917 | **0.928** |
| FPR (all attacks) | 0.544 | 0.531 | **0.435** |
| **FPR excluding Infiltration** | 0.123 | 0.109 | **0.000** |
| Lab PCAP accuracy | 0.917 | **0.988** | 0.966 |
| Scan/flood family F1 | 0.845 | **0.984** | 0.952 |

**Interpretation:**

- **Windowing beats per-flow** by ~4–5 accuracy points — the main structural lesson.  
- **AMT.01 wins benign discrimination** — zero spurious alarms into real attack classes (488 benign test windows). Best default for a dashboard that must not cry wolf.  
- **YMT.01 wins attack typing** in the synthetic lab — scan/flood family and lab-capture accuracy.  
- Schemas are **complementary**; recommended merge: AMT mean/std breadth + YMT log-compression and port-sequence ratios (see `MODEL_COMPARISON.md` §8).

### 5.2 Ablation insight (YMT blocks)

Removing any single v2 block changes accuracy by ≤ ~0.01. The gain comes from **computing statistics over neighbouring flows**, especially **std** (regularity/beacon detection). AMT.01's 58 mean/std columns are the same mechanism applied more broadly — explaining its edge.

---

## 6. Forecasting experiments

### 6.1 Main-branch forecaster stack

| Component | Script | Architecture |
|---|---|---|
| One-step baselines | `train_forecast_models.py` | AE-MLP, Random Forest |
| Recursive tests | `test_recursive_rollout*.py` | v2–v4 protocols |
| Temporal GRU | `train_temporal_forecaster_ctx60.py` | Recurrent + bounded delta |
| Transformer forecaster | `train_transformer_forecaster.py` | TFT.01 (self-attention) |
| RAM adaptive memory | `adaptive_memory_forecaster.py`, `ram01_kill_chain_eval.py` | TTT + episodic memory bank |
| Transformer showdown | `transformer_showdown_kill_chain_eval.py` | GRU vs Transformer vs RAM variants |

**Key finding:** Non-recurrent models (RF, XGB, flat MLP) **flatline or oscillate** under recursive rollout. GRU helps; Transformer helps more on v7; RAM helps weaker backbones most.

### 6.2 v7 — GRU vs Transformer (main traffic, v2 schema)

| Model | Overall MSE |
|---|---:|
| Temporal-Y (GRU, frozen) | 8.52 |
| TFT.01 (Transformer, frozen) | **6.16** (−27.7%) |
| RAM.01 (GRU + memory) | 8.13 |
| **RAMT.01 (Transformer + memory)** | **6.04** (best) |

### 6.3 v8 — Aryan's checkpoint kill-chain eval

Uses Aryan's **real** `world_model_best.pt` (no retraining). Synthetic timeline: benign filler + Initial Access (val) + Lateral Movement (test) + Impact (train — sanity check only).

| Run | Steps | Aryan-WM MSE | RAM-A.01 MSE | Δ |
|---|---:|---:|---:|---|
| Full | 2000 | 469.60 | 469.78 | +0.02% |
| Short | 1000 | 1.736 | 1.753 | +1.0% |

RAM-A.01 (RAM wrapper on Aryan's model) made **no meaningful difference**. Consistent with v7: a 3.8M-param model already overfits a ~2800-window CIC sample tightly; lightweight online adaptation has no room to help.

The **270× MSE swing** between 2000-step and 1000-step runs is a **benign-sampling artifact** (deterministic tiling of heavy-tailed flows), not a training bug — see `results/forecast/INDEX.md`.

Script: `scripts/ram_aryan_killchain_eval.py` (executed from an `aryan` worktree; outputs saved to main's `results/forecast/v8_ram_aryan_killchain/`).

---

## 7. Side-by-side architecture map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SHARED CONCEPT                                  │
│   Telemetry → State vectors S_t → Sequence model → P(S_{t+1}|S_t) → MITRE  │
└─────────────────────────────────────────────────────────────────────────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────────────┐          ┌─────────────────────────┐
│   MAIN (lab-centric)    │          │  ARYAN (dataset-centric)  │
├─────────────────────────┤          ├─────────────────────────┤
│ PCAP via Scapy bots     │          │ 6 dataset extractors      │
│ 27f / 64f / 82f schemas │          │ StateBuilder 30s windows  │
│ 33-class MITRE bots     │          │ 7 MITRE stage heads       │
│ Docker self-healing     │          │ GNN + LSTM + Latent       │
│ Streamlit SOC (5 tabs)  │          │ Multi-page app/           │
│ RAM forecasting research│          │ CAPEC/CVE knowledge       │
└─────────────────────────┘          └─────────────────────────┘
         │                                        │
         └──────── extract_aryan.py ──────────────┘
                    (schema port)
```

---

## 8. Directory reference

### Main branch (committed + local)

```
PRISM/
├── configs/attack_catalog.yaml     # 32-class bot → MITRE mapping
├── docker/                         # Isolated lab compose stack
├── data/
│   ├── mitre_attack.json           # 222 Enterprise techniques
│   ├── mitre_network_catalog.json
│   └── raw/adversarial/            # Captured PCAPs from lab loop
├── docs/
│   ├── FEATURE_SCHEMA_V2.md        # v1 vs v2 schema spec
│   ├── MODEL_COMPARISON.md         # ZMT/YMT/AMT head-to-head
│   ├── JUDGE_GUIDE.md
│   └── WORKSPACE_AND_ARYAN_BRANCH.md  ← this file
├── models/checkpoints/             # Trained weights (classifiers, forecasters)
├── results/
│   ├── zmt_ymt/                    # Schema comparison outputs
│   └── forecast/v1…v8/             # Versioned forecasting experiments
├── scripts/                        # Training, eval, dataset builders
├── src/
│   ├── adversarial/                # Lab, bots, training loop
│   ├── model/                      # world_model, world_model_multiclass
│   ├── pipeline/                   # features v1/v2, extract_aryan
│   ├── predict/                    # K-step engine
│   ├── explain/                    # Attribution
│   └── ui/                         # Streamlit dashboard + live feed
├── run_dashboard.py
├── README.md                       # Shipped prototype overview
└── FEATURES.md                     # 374-feature catalog (design doc)
```

### Aryan branch only (on `origin/aryan`)

```
PRISM/  (aryan layout)
├── app/                            # Multi-page Streamlit app
├── configs/data.yaml, model.yaml, train.yaml
├── data/processed/, data/splits/   # CIC-IDS-2018 NPZ splits
├── src/data/                       # Extractors, StateBuilder, registry
├── src/models/                     # Transformer, LSTM, GNN, latent
├── src/evaluation/                 # Benchmark, ablation, metrics
├── src/explainability/             # SHAP, attention, feature importance
├── src/knowledge/                  # CAPEC, CVE
├── src/prediction/                 # Simulator, attack mapper, scoring
├── weights/                        # All trained checkpoints + scaler
├── tests/                          # pytest suite
└── notebooks/                      # EDA → training → eval pipeline
```

---

## 9. How to work with both lines

### Stay on main (recommended for lab + benchmarks)

```powershell
cd d:\Cursor\PRISM
.\venv\Scripts\activate

# Lab
python scripts/lab_ctl.py up
python -m src.adversarial.training_loop

# Schema comparison (CUDA)
python scripts/build_comparison_dataset.py
python scripts/train_zmt_ymt.py
python scripts/train_amt.py
python scripts/analyze_zmt_ymt.py

# Dashboard
python run_dashboard.py
```

### Inspect or run Aryan's branch

```powershell
git fetch origin
git worktree add ..\PRISM-aryan origin/aryan   # isolated checkout
cd ..\PRISM-aryan
pip install -r requirements.txt

python scripts/download_data.py --list-datasets
python scripts/train.py --states data/splits/train.npz --epochs 30
streamlit run app/streamlit_app.py
```

To load Aryan's transformer checkpoint into v8-style eval, use the aryan worktree paths referenced in `results/forecast/INDEX.md`.

---

## 10. Recommendations

| Decision | Recommendation | Rationale |
|---|---|---|
| Default classifier schema for dashboard | **AMT.01** | Zero false alarms into real attack classes on benign traffic |
| Attack typing in adversarial lab | **YMT.01** | Best scan/flood family + lab PCAP accuracy |
| Cite Aryan's benchmark JSON | **Never** | Synthetic demo data; LR beats transformer by construction |
| Cite Aryan's StateBuilder | **Yes** | Wins fair head-to-head on real traffic; ported faithfully |
| Forecasting backbone | **Transformer + RAM** (RAMT.01 on main traffic) | v7 best MSE; RAM is backbone-agnostic |
| Wrap RAM around Aryan's 3.8M model | **Low priority** | v8 shows +0–1% — model already saturated on small CIC sample |
| Merge branches | **Selective** | Port StateBuilder integration + knowledge layers; keep main's lab and fair benchmark harness |

### Suggested next steps

1. **Commit** the research layer on `main` (or branch `research/forecast-v8`) — scripts, results index, pipeline ports, docs.  
2. **Cherry-pick or port** Aryan's `src/knowledge/` (CAPEC/CVE) and evaluation metrics without pulling synthetic benchmark paths.  
3. **Build merged schema** (AMT mean/std + YMT log/port-seq) and re-run `train_zmt_ymt.py` pattern.  
4. **Add IP columns** to base records where available so AMT's dead `num_unique_src/dst_ips` features activate.  
5. **Replace** root `FEATURES.md` cross-reference with a clear "design catalog vs live pipeline" banner to avoid confusion.

---

## 11. Related documents

| Document | Contents |
|---|---|
| [`README.md`](../README.md) | Shipped main prototype — lab, dashboard, quickstart |
| [`docs/MODEL_COMPARISON.md`](MODEL_COMPARISON.md) | Full ZMT/YMT/AMT analysis, ablation, Aryan benchmark autopsy |
| [`docs/FEATURE_SCHEMA_V2.md`](FEATURE_SCHEMA_V2.md) | v1 vs v2 column definitions and design rules |
| [`results/forecast/INDEX.md`](../results/forecast/INDEX.md) | Forecasting experiment changelog v1–v8 |
| [`FEATURES.md`](../FEATURES.md) | 374-feature multi-dataset catalog (not live pipeline) |
| [`idea.txt`](../idea.txt) | Original NCIIPC challenge specification |

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **ZMT.01** | Z-generation Model Transformer on v1 (27f per-flow) schema |
| **YMT.01** | Y-generation Model Transformer on v2 (64f window) schema |
| **AMT.01** | Aryan-ported Model Transformer on StateBuilder (82f) schema |
| **RAM.01** | Receding-horizon Adaptive Memory — TTT + episodic memory on forecaster |
| **TFT.01** | Small Transformer forecaster (main branch, ctx=60) |
| **StateBuilder** | Aryan's window aggregation class (`src/data/state_builder.py`) |
| **Kill-chain eval** | Synthetic timeline splicing benign + attack segments for rollout MSE |

---

*This document subsumes the workspace state as of 2026-08-29. For numeric results, prefer JSON artifacts in `results/` over prose here — they are the source of truth when this doc and code diverge.*
