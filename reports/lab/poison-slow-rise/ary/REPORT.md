# Slow-Rising Poison Test — `ary5_ramx` (5s windows)

**Date:** 2026-09-03  
**Checkpoint:** `models/checkpoints/ary_5sv01.pt`  
**Raw data:** `results/ips_redteam/poison_slow_rise/ary/`

## Setup

- All phases use **5s windows**
- Warmup 60s (12 windows) → ramp 300s (58 windows) → strike 60s (12 windows)
- Ramp scored with `true_bin=0` → attack traffic ingested into **RAMX memory as benign**
- ML detection only (threshold 0.5)

## Comparison

| Metric | Control | Poison |
|--------|---------|--------|
| **Detected at strike** | **Yes** | **No** |
| **Max P(attack)** | **0.584** | **0.439** |
| Delta max P | — | **−0.145** |
| Poison suppressed detection | — | **Yes** |
| TTD | ~42s | n/a |

## Ramp phase (poison run)

- 58 windows, max P = 0.291 (below threshold)
- Mean P = 0.038
- No alerts during ramp (`any_alert_during_ramp: false`)

## Strike phase (poison run)

First windows: P = 0.142, 0.141, 0.014 — never crossed 0.5.

## Verdict

**ARY RAMX v01 is vulnerable to slow-rising memory poisoning** in this test. Label-gated k-NN memory accepted ramp traffic as benign reference; full strike was misclassified (max P 0.439 < 0.5). This confirms the tiered-memory dilution failure mode documented in `streaming_variants.py`.

## Mitigation ideas

- Never write unconfirmed traffic to baseline tier
- Suspicious-traffic gate on memory ingest regardless of label
- Rolling baseline refresh with drift detection
