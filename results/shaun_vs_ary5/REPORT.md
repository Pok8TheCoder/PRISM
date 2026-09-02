# Shaun PRISM V2 vs ARY-5sV01 (base + RAMX)

Shaun: `D:\Cursor\PRISM-shaun\weights\world_model.pt` (`origin/shaun`, 292D @ 15s, L=30)
ARY-5s: `D:\Cursor\PRISM\models\checkpoints\ary_5sv01.pt` (242D @ 5s, RAMX uses oracle labels)

Shaun lab ingest: **fair path** — PCAP → CIC-style rows → `SchemaAligner` → `StateBuilder` (15s).
Shaun RAMX: **v2 context-gated** — baseline from context buffer, fusion alerts only on lab PCAP (fixes warmup FP).
ARY RAMX: hidden-key memory bank (`RAMX_V.01`, oracle labels).

## Lab PCAP aggregate

| Model | Mean F1 | Det rate | Mean attack max P | Mean warmup P |
|---|---|---|---|---|
| ARY-5s base | 0.000 | 0.0% | 0.010 | 0.008 |
| ARY-5s + RAMX | 0.931 | 98.6% | 0.565 | 0.008 |
| Shaun V2 base | 0.059 | 6.5% | 0.197 | 0.156 |
| Shaun V2 + RAMX v2 | **1.000** | **100%** | **0.679** | 0.156 |

## Live Docker rounds

### cred_theft/none
- windows: 24 @ 15.0s
- **ary5_base**: F1=0.000 det=False TTD=None warmup_P=0.039 attack_max_P=0.054
- **ary5_ramx**: F1=1.000 det=True TTD=0 warmup_P=0.039 attack_max_P=0.581

### key_theft/none
- windows: 24 @ 15.0s
- **ary5_base**: F1=0.000 det=False TTD=None warmup_P=0.044 attack_max_P=0.048
- **ary5_ramx**: F1=1.000 det=True TTD=0 warmup_P=0.044 attack_max_P=0.587

### defacement/none
- windows: 24 @ 15.0s
- **ary5_base**: F1=0.000 det=False TTD=None warmup_P=0.045 attack_max_P=0.055
- **ary5_ramx**: F1=1.000 det=True TTD=0 warmup_P=0.045 attack_max_P=0.583
