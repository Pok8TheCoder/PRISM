# Scaled Lab Bench: ARY-5sV01 vs Shaun V2 (base + RAMX)

Traffic scaling: **10×** rate/count features, **5×** flow replication
(targets CIC-like density; quiet lab median ~20 flows → ~100 flows/capture)

ARY-5s: `D:\Cursor\PRISM\models\checkpoints\ary_5sv01.pt` @ 5s windows | ARY RAMX = hidden-key bank (oracle labels)
Shaun V2: `D:\Cursor\PRISM-shaun\weights\world_model.pt` @ 15s SchemaAligner ingest | Shaun RAMX v2 = context-gated calibrator

## Aggregate (attack PCAPs)

| Model | Mean F1 | Det rate | Mean attack max P | Mean warmup P |
|---|---|---|---|---|
| ARY-5s base | 0.000 | 0.0% | 0.009 | 0.008 |
| ARY-5s + RAMX | 0.937 | 99.3% | 0.569 | 0.008 |
| Shaun V2 base | 0.539 | 55.8% | 0.528 | 0.156 |
| Shaun V2 + RAMX v2 | **1.000** | **100%** | **0.811** | 0.156 |
