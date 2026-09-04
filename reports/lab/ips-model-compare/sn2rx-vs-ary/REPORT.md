# IPS Compare: SN2RX v2 vs ARY-5s+RAMX (Scaled Live Lab)

**Date:** 2026-09-02  
**Raw data:** `results/ips_redteam/sn2rx_compare/`  
**Traffic:** 10× scale + 5× replicate; benign 10 workers @ 0.5s  

## IPS blocking (primary)

| System | Outcome | Prevented | F1 | TTD | Benign harm | Blocked at |
|--------|---------|-----------|-----|-----|-------------|------------|
| ARY-5s+RAMX | undetected_theft | No | 0.000 | n/a | 0 | n/a |
| **SN2RX v2** | too_late | No | 0.800 | 0.0s | 3 | ~20.2s |

## Detection (same trace)

| System | F1 | Detected | Max P | TTD |
|--------|-----|----------|-------|-----|
| sn_base | 0.800 | Yes | 0.985 | 0.0s |
| sn2rx | 0.800 | Yes | 0.985 | 0.0s |

## Key finding (pre-fix)

SN2RX **detected** the attack (recall=1, TTD=0) but IPS blocked **after** cred theft (~16.5s stolen, ~20s block) → `too_late`.

ARY missed entirely on this live fair-ingest run (F1=0).

This run motivated the event-watcher + recon-stage block fix documented in [theft-prevention](../../theft-prevention/REPORT.md).
