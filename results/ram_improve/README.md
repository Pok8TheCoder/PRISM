# RAM improvement experiments

Structured results from `scripts/ram_improve_eval.py`, `ram_1000step_eval.py`,
`timesfm_ram_compare.py`, and `ram_sensitivity_sweep.py`.

| Path | What |
|---|---|
| `eval.json` | V0–V11 classify-blend sweep (binary F1 on test set) |
| `1000step_*.png/json` | P(attack) timeline: base vs orig RAM vs V8 vs V10 |
| `ctx100_hor200/` | TimesFM vs ARY feature forecast (ctx=100, hor=200) |
| `ctx100_hor200_attackgate/` | Attack-gated RAM variant |
| `sensitivity/` | Gate-threshold + horizon sweeps |

Superseded flat TimesFM PNGs → `archive/2026-09-01-cleanup/results/ram_improve-legacy/`

| Path | What |
|---|---|
| `falloff_500/` | 500-step context → 500-step open-loop forecast falloff test |
