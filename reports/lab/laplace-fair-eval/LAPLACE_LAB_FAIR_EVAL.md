# Laplace (Gen10) Fair Lab Evaluation

**Date:** 2026-09-12  
**Model:** PRISM Laplace Model (`StreamingLaplaceModel` / Gen10 world model + Tier-2 flow context)  
**Checkpoint:** `PRISM-aryan-branch/weights/universal_gen10/world_model_best.pt`

## Are we using the streaming path?

**Yes — for the fair lab PCAP test.**

| Test | Path | Fair? |
|------|------|-------|
| **Fair lab PCAP eval** | `PCAP → gen10_pcap_ingest (raw 249-d) → StreamingLaplaceModel.step()` with per-window `flow_context` | ✅ Correct |
| **Live IPS** (`live_ips_multi.py`, backend `laplace`) | Same: `pcap_to_gen10_windows` + `StreamingLaplaceModel` | ✅ Correct |
| **Aryan batch benchmark** (`scratch/evaluate_gen10_testset.py`) | NPZ `DataLoader` batch inference on `splits_universal_gen10_5s` | Batch only — not streaming |
| **Early unfair compare** (`eval_prism_universal_compare.py`) | Padded Gen8 242-d NPZ → Laplace batch/stream | ❌ Wrong feature space |

Aryan’s published ~75.8% MITRE number comes from **batch eval on his universal Gen10 NPZ split** (not yet in repo). Production intent is **streaming**: `StreamingGen10WorldModel` / `StreamingLaplaceModel` in `src/models/streaming_gen10.py` and `streaming_laplace.py`, wired through `scripts/live_ips_multi.py`.

Our fair harness (`Automode/train/eval_laplace_lab_fair.py`) calls the same streaming class and Tier-2 API as live IPS:

```
lab PCAP → StateBuilder(5s) → raw 249-d state + flow_context → StreamingLaplaceModel.step(raw, flow_context=...)
```

Continuous lookback=20 is carried across PCAPs (no per-PCAP buffer reset).

## Results on our realistic lab PCAPs (148 files, 272 scored windows)

| Metric | Laplace | Gen8 | Gen8+RXI v2_staged |
|--------|---------|------|---------------------|
| MITRE accuracy | **14.8%** | 4.0% | 2.6% |
| Binary accuracy | 54.8% | 77.6% | 18.4% |
| Weak-stage recall (Init/Lateral/Exfil mean) | 13.6% | 33.3%* | 33.3%* |

\*Gen8 variants only recall **Initial Access** (7 windows); other weak stages stay at 0%.

### Laplace per-stage recall

| Stage | Recall |
|-------|--------|
| Benign | 16.0% |
| Reconnaissance | 9.7% |
| Initial Access | 0% |
| Lateral Movement | **40.7%** |
| Command & Control | 0% |
| Exfiltration | 0% |
| Impact | **54.5%** |

Artifacts: `Automode/baselines/laplace_lab_fair.json`, `Automode/baselines/lab_fair_compare.json`

## Why Laplace is lacking on our lab data

### 1. Distribution shift (trained universal, tested lab)

Weights and scaler were fit on CIC / universal bulk traffic (dense 5s windows, different topology and volume). Our adversarial lab PCAPs are **short (~30s)**, **sparse** (often 1–7 windows per capture), and use RFC5737/documentation address ranges that differ from training corpora.

### 2. Lookback starvation

`lookback=20` at 5s needs **~100s of prior context** before the first scored window. Many lab PCAPs contribute only a handful of windows; continuous streaming helps but total scored windows remain low (272 across 148 PCAPs).

### 3. Stage confusion, not total failure

Confusion matrix shows heavy **Benign ↔ Recon** and **Recon → Impact** mixing. Lateral and Impact are partially detected — the model is not uniformly blind, but MITRE stage boundaries do not align with our capture taxonomy.

### 4. Weak stages still miss

**Initial Access, C2, and Exfiltration** are 0% recall on lab PCAPs. These are the same weak surfaces Gen8 struggles with on universal data; Laplace does not fix them under lab shift.

### 5. Binary head is mediocre here

54.8% binary accuracy vs Gen8’s 77.6% on the same PCAPs — Tier-1 attack probability is not well calibrated for sparse lab features even when Tier-2 flow context is present.

### 6. Unfair tests were misleading (now ruled out)

Feeding **pre-scaled Gen8 242-d NPZ** into Laplace (padding to 249-d) produced ~39% batch / ~3% stream — that path **double-wrong** (feature space + no flow context). Fair PCAP streaming is the only valid offline comparison.

## Comparison to Aryan’s numbers

| Benchmark | MITRE acc | Notes |
|-----------|-----------|-------|
| Aryan Gen10 universal test (batch, his NPZ) | ~75.8% | Not our lab distribution; splits not in repo |
| Aryan stress test (`hardest_lab_stress_test.py`) | 10/10 pass | Synthetic states + flow_context, not real PCAPs |
| **Our fair lab PCAP stream** | **14.8%** | Honest production path on realistic captures |

Laplace is **not proven worse than Gen8 on lab traffic** in absolute terms — it has the **best MITRE stage spread** in our 3-way fair compare — but **14.8% is not deployment-ready** on our realistic lab corpus without retraining, denser captures, or calibration on lab benign warmup.

## Reproduce

```bash
# Fair Laplace only
python Automode/train/eval_laplace_lab_fair.py

# Laplace vs Gen8 vs Gen8+RXI v2_staged
python Automode/train/eval_lab_fair_compare.py
```

Requires: `PRISM-aryan-branch` (Laplace weights + `gen10_pcap_ingest.py`), `Automode/train/data/bulk_5s_cache/index.json`, lab PCAPs under `PRISM/data/raw/`.
