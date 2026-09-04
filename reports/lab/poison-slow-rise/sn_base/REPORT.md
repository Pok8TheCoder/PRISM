# Slow-Rising Poison Test — `sn_base` (Shaun V2, w5s, 5s windows)

**Date:** 2026-09-03  
**Checkpoint:** `PRISM-shaun/weights/w5s/world_model.pt`  
**System:** `StreamingShaunBase` (classifier only, no RAMX)  
**Raw data:** `results/ips_redteam/poison_slow_rise/sn_base/`

## Setup

- 5s windows throughout
- Same control vs poison protocol as other backends
- w5s world model, 292-d Shaun states

## Comparison

| Metric | Control | Poison |
|--------|---------|--------|
| Detected at strike | No | No |
| Max P(attack) | 0.040 | 0.044 |
| Delta max P | — | +0.005 |
| Poison suppressed detection | — | No (N/A) |

## Ramp phase (poison)

- 57 windows, max P = 0.050
- Mean P = 0.043

## Verdict

**SNv2 base classifier did not detect cred theft** in either control or poison runs at 10× scaled lab traffic. Max P stayed ~0.04 (well below 0.5). Poisoning is moot — the base model already misses this zero-day scenario without RAMX fusion.

Compare to **sn2rx3** on the same traffic profile (max P ~0.60 with RAMX) or **sn2rx v2** in earlier IPS runs (F1 0.80 on scaled trace).
