# IPS Red-Team Compare: ARY-5sV01+RAMX vs ARY-5sV02+RAMX

Generated: 2026-09-02T17:04:30.103786+00:00

| Model | Outcome | Creds stolen | Stolen at | IPS blocked | Blocked at | Prevented theft | F1 | TTD | Benign harm |
|---|---|---|---|---|---|---|---|---|---|
| v01 | undetected_theft | yes | 35.7s | no | n/a | no | 0.000 | n/a | 0 |
| v02 | prevented | no | n/a | yes | 5.4s | yes | 0.788 | 0.0s | 7 |

## Notes

### v01
- Round: `a84e4a360602`
- Warmup: 30.0s (7 windows)
- Attack windows: 20 @ 5.0s state / 0.5s wall
- Checkpoint: `D:\Cursor\PRISM\models\checkpoints\ary_5sv01.pt`

### v02
- Round: `5e297ede413b`
- Warmup: 30.0s (7 windows)
- Attack windows: 20 @ 5.0s state / 0.5s wall
- Checkpoint: `D:\Cursor\PRISM\models\checkpoints\ary_5sv02.pt`
