# Slow-Rising Poison Test — `sn2rx3` (w5s, 5s windows)

**Date:** 2026-09-03  
**Checkpoint:** `PRISM-shaun/weights/w5s/world_model.pt`  
**Raw data:** `results/ips_redteam/poison_slow_rise/poison_slow_rise/` (legacy path) + `comparison.json`

## Setup

- Warmup 60s → ramp 300s → strike 60s @ speed 10×
- Warmup windows: 15s; attack windows: 5s (first run used mixed windows)
- ML only, no IPS block

## Comparison

| Metric | Control | Poison |
|--------|---------|--------|
| Detected at strike | Yes | Yes |
| Max P(attack) | 0.617 | 0.603 |
| Delta max P | — | −0.014 |
| TTD at strike | ~37s | ~5s |
| Raw offset drift (ramp) | — | +0.033 |

## Ramp phase

- 57 windows, max P = **0.602** (alert fired during ramp)
- Mean P = 0.184
- Calibrator offset drift negligible (+0.033)

## Verdict

**Poison did not suppress sn2rx3 detection.** Strike remained above threshold. Ramp may have *helped* early detection (lower TTD). SN2RX v3 calibrators + relative-anomaly fusion resist this poisoning scenario at lab scale.
