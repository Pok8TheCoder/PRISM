# IPS Quick Compare: ARY-5s+RAMX vs SN2RXv2 (scaled live lab)

Generated: 2026-09-02T18:56:35.350323+00:00

Traffic: PCAP row scale 10x + 5x replicate; benign-client 10 workers @ 0.5s.
Attack: live SQLi cred theft (zero-day vs CIC).

## IPS blocking (primary)

| System | Outcome | Prevented theft | F1 | TTD | Benign harm | Blocked at |
|---|---|---|---|---|---|---|
| ARY-5s+RAMX | undetected_theft | no | 0.000 | n/a | 0 | n/a |
| **SN2RXv2** | too_late | no | 0.800 | 0.000 | 3 | 20.191 |

## Zero-day side test (same SN2RX run, shadow scores)

| System | F1 | Detected | Max P(attack) | TTD | Benign harm |
|---|---|---|---|---|---|
| Shaun V2 base (sn_base) | 0.800 | yes | 0.985 | 0.000 | 3 |
| **SN2RXv2** | 0.800 | yes | 0.985 | 0.000 | 3 |

## Notes

- `sn_base` vs `sn2rx` on the **same** scaled live trace tests zero-day detection without extra Docker time.
- ARY RAMX uses oracle labels for memory; Shaun RAMX v2 does not.
