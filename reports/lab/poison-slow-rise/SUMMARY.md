# Slow-Rising Threat / RAMX Poisoning — Summary

**Test script:** `scripts/test_ramx_poison_live.py`  
**Date:** 2026-09-03  
**Window size:** 5s (all phases)  
**Traffic:** 10× scale, 5× replicate, scaled benign client  

## Design

Two conditions per model:

| Condition | Phases |
|-----------|--------|
| **Control** | 60s warmup (benign) → 60s full-speed cred theft |
| **Poison** | 60s warmup → **300s slow-rising ramp** (scored as benign) → 60s strike |

Slow agent phases: noise → recon → mild SQLi → full UNION theft (`docker/redteam/redteam_agent_slow.py`).

No IPS blocking — ML detection only (threshold 0.5).

## Results

| Model | Control detected | Control max P | Poison detected | Poison max P | Poison suppressed? |
|-------|------------------|---------------|-----------------|--------------|-------------------|
| **ARY RAMX v01** (`ary5_ramx`) | **Yes** | **0.584** | **No** | **0.439** | **Yes** |
| **SN2RX v3** (`sn2rx3` w5s) | Yes | 0.617 | Yes | 0.603 | No (−0.014) |
| **SNv2 base** (`sn_base` w5s) | No | 0.040 | No | 0.044 | N/A (never detected) |

## Interpretation

- **ARY RAMX is vulnerable** to slow-rising label poisoning: ramp traffic written to memory as benign (`true_bin=0`) diluted k-NN baseline; strike max P dropped below 0.5.
- **SN2RX v3** calibrators did **not** suppress detection at 10× scale; relative-anomaly fusion kept P above threshold during ramp and strike.
- **SNv2 base** (classifier only, no RAMX) did not detect cred theft at these settings in either condition — RAMX/adaptation layer required for this lab traffic profile.

## Per-model reports

- [sn2rx3/REPORT.md](sn2rx3/REPORT.md)
- [ary/REPORT.md](ary/REPORT.md)
- [sn_base/REPORT.md](sn_base/REPORT.md)

## Raw data

`results/ips_redteam/poison_slow_rise/{backend}/`
