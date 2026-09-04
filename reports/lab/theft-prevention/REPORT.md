# Live IPS Cred-Theft Prevention — All Models

**Date:** 2026-09-03  
**Script:** `scripts/live_ips_multi.py`  
**Policy:** `--event-block-stages recon` + background event watcher  
**Attack windows:** 5s | **Scale:** 10× / replicate 5× | **Speed:** 25×  

## Primary outcome (theft prevented?)

| Model | Checkpoint | Block @ | Outcome | Creds stolen | Benign harm |
|-------|------------|---------|---------|--------------|-------------|
| ARY RAMX v01 | ary_5sv01 | 17.3s | **prevented** | No | 0 |
| SN2RX v2 | w5s | 16.4s | **prevented** | No | 0 |
| SN2RX v3 | w5s | 12.8s | **prevented** | No | 0 |
| SN2RX v2 | w1s | 13.5s | prevented* | Race | 0 |
| SN2RX v3 | w1s | 17.3s | **prevented** | No | 0 |
| SN2RX v2 | w15s | 17.7s | **prevented** | No | 0 |
| SN2RX v3 | w15s | 17.3s | **prevented** | No | 0 |

\* w1s sn2rx: block at 13.5s but post_exploit event logged at 14.2s (in-flight race); scored as prevented by timestamp.

## ML detection at strike (secondary)

| Model | F1 | ML detected | Notes |
|-------|-----|-------------|-------|
| ARY | 0.00 | No | Event block, not ML |
| SN2RX v3 w5s | 0.00 | No | Event block |
| SN2RX v2 w15s | 0.80 | Yes | ML also fired |
| SN2RX v3 w15s | 1.00 | Yes | ML also fired |

## What changed vs earlier failure

Before fix: ML blocked at **window end** (~20s); theft at ~16.5s → `too_late`.

After fix:
- Background event watcher (50ms poll, in-container tail)
- Block on recon stage before post_exploit
- Poll during capture, ingest, and scoring

## Raw data

- `results/ips_redteam/theft_prevention_all/`
- `results/ips_redteam/theft_prevention/sn2rx3/`

## Related

- [Poison slow-rise tests](../poison-slow-rise/SUMMARY.md) — separate ML-only poisoning study
