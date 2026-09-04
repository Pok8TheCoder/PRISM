# Shaun Window Checkpoint — Lab PCAP Comparison

Base + RAMX v2/v3 for each checkpoint (15s / w5s / w1s). Best RAMX picked by
**mean F1 on attack windows only** per checkpoint.

## Why was mean F1 so low before?

The old **`mean_f1_attack`** metric mixed **warmup + attack windows**:

- 10 lab-benign warmup steps precede each PCAP replay
- Base models often fire during warmup (high P(attack) on OOD benign) → counted as FP
- Example: 15s base had **harm=10** on every PCAP but **0% benign FP** on the test slice
  because `detected` only checks attack-labeled windows
- **Detection rate** can be 86% while **F1 stays ~0.2** — one TP in attack phase, many warmup FPs

Use **`f1_attack_only`** (attack windows after warmup) for fair model quality comparison.

## Protocol

- Ingest window matches training (15s / 5s / 1s); lab benign warmup; no oracle
- Threshold P(attack) ≥ 0.5

## Base vs best RAMX companion (full lab set)

| Checkpoint | Window | System | Attack det | Benign FP | F1 (all win) | **F1 (attack only)** | Warmup harm |
|------------|--------|--------|------------|-----------|--------------|----------------------|-------------|
| base_15s | 15.0s | base | 86.5% | 0.0% | 0.195 | **0.621** | 10.0 |
| base_15s | 15.0s | **ramx_v2** | 86.5% | 0.0% | 0.197 | **0.624** | 10.0 |
| w5s | 5.0s | base | 25.0% | 0.0% | 0.204 | **0.204** | 0.0 |
| w5s | 5.0s | **ramx_v2** | 66.7% | 0.0% | 0.276 | **0.276** | 0.0 |
| w1s | 1.0s | base | 0.0% | 0.0% | 0.000 | **0.000** | 0.0 |
| w1s | 1.0s | **ramx_v2** | 33.3% | 0.0% | 0.043 | **0.043** | 0.0 |

## All RAMX variants (attack-only F1)

| Checkpoint | base | ramx_v2 | ramx_v3 | best |
|------------|------|---------|---------|------|
| base_15s | 0.621 | 0.624 | 0.621 | **ramx_v2** |
| w5s | 0.204 | 0.276 | 0.241 | **ramx_v2** |
| w1s | 0.000 | 0.043 | 0.043 | **ramx_v2** |

## Same-PCAP intersection (9 PCAPs) — base vs best RAMX

| Checkpoint | base det | base F1atk | best RAMX | best det | best F1atk |
|------------|----------|------------|-----------|----------|------------|
**9 PCAPs** scorable at 15s, 5s, and 1s.

| Checkpoint | base det | base F1atk | best RAMX | best det | best F1atk |
|------------|----------|------------|-----------|----------|------------|
| base_15s | 100.0% | 0.565 | ramx_v2 | 100.0% | 0.587 |
| w5s | 33.3% | 0.262 | ramx_v2 | 83.3% | 0.340 |
| w1s | 0.0% | 0.000 | ramx_v2 | 33.3% | 0.043 |

## Offline CIC (training benchmark)

| Checkpoint | CIC F1 |
|------------|--------|
| base_15s | 0.9035 |
| w5s | **0.9626** |
| w1s | 0.9309 |

Raw JSON: `Automode/baselines/shaun_window_compare_results.json`