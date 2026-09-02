# Zero-Day Live Objective Sweep

Generated: 2026-09-02T19:08:18.191556+00:00

HTTP lab objectives (never in CIC-IDS-2018 training):
- `T1555_sqli_cred_theft`
- `T1552_key_theft`
- `T1491_web_defacement`

## Live rescore (all 4 systems, same scaled traffic)

Traffic: PCAP 10× + 5× replicate, benign 10 workers @ 0.5s, 15s windows.

| Objective | System | F1 | Det | Max P (attack) | TTD | Warmup harm |
|---|---|---:|---|---:|---:|---:|
| cred_theft | ary5_base | 0.000 | no | 0.032 | n/a | 0 |
| cred_theft | ary5_ramx | 0.000 | no | 0.032 | n/a | 0 |
| cred_theft | sn_base | 0.444 | yes | 0.976 | 0.0 | 5 |
| cred_theft | sn2rx | 0.444 | yes | 0.976 | 0.0 | 5 |
| key_theft | ary5_base | 0.000 | no | 0.027 | n/a | 0 |
| key_theft | ary5_ramx | 0.000 | no | 0.027 | n/a | 0 |
| key_theft | sn_base | 0.250 | yes | 0.985 | 0.0 | 6 |
| key_theft | sn2rx | 0.250 | yes | 0.985 | 0.0 | 6 |
| defacement | ary5_base | 0.000 | no | 0.036 | n/a | 0 |
| defacement | ary5_ramx | 0.000 | no | 0.036 | n/a | 0 |
| defacement | sn_base | 0.400 | yes | 0.976 | 0.0 | 6 |
| defacement | sn2rx | 0.400 | yes | 0.976 | 0.0 | 6 |

## Stored live_lab replay (ARY-5s only, historical captures)

| Objective | System | F1 | Det | Max P | Warmup mean P |
|---|---|---:|---|---:|---:|
| cred_theft | ary5_base | 0.000 | no | 0.054 | 0.039 |
| cred_theft | ary5_ramx | 1.000 | yes | 0.581 | 0.039 |
| key_theft | ary5_base | 0.000 | no | 0.048 | 0.044 |
| key_theft | ary5_ramx | 1.000 | yes | 0.587 | 0.044 |
| defacement | ary5_base | 0.000 | no | 0.055 | 0.045 |
| defacement | ary5_ramx | 1.000 | yes | 0.583 | 0.045 |

## Notes

- Stored `live_lab` rounds embed 242-d ARY states only — Shaun requires fair PCAP ingest (live rescore).
- Zero-day = lab objective class IDs not in CIC training set.
