# IPS Compare: ARY RAMX v01 vs v02

**Date:** 2026-09-02  
**Raw data:** `results/ips_redteam/v01/`, `v02/`, `COMPARE.md`  
**Duration:** ~20 min wall | Warmup 30s | 5s windows  

## Results

| Model | Outcome | Creds stolen | Stolen at | IPS blocked | Blocked at | Prevented | F1 | TTD | Benign harm |
|-------|---------|--------------|-----------|-------------|------------|-----------|-----|-----|-------------|
| **v01** | undetected_theft | Yes | 35.7s | No | n/a | No | 0.000 | n/a | 0 |
| **v02** | prevented | No | n/a | Yes | 5.4s | Yes | 0.788 | 0.0s | 7 |

## Notes

- v01 checkpoint: `models/checkpoints/ary_5sv01.pt`
- v02 uses tiered RAMX memory improvements
- v02 prevented theft but had 7 benign-harm FPs during warmup
- v01 failed to detect or block on this red-team SQLi objective

## Context

Pre-dates recon-stage event blocking. v02's early block at 5.4s was ML/IPS timing, not event diary.
