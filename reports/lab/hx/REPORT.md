# PRISM-HX fair bench

Fair protocol: no oracle labels, 5s Shaun 292-d ingest, detect @ 0.5, same 13 held-out internet-benign PCAPs + 17 scorable lab attacks as the Shaun w5s + RAMX v2 champion.

| Model | Attack det | Benign FP (protocol) | n scored | notes |
|-------|------------|----------------------|----------|-------|
| Shaun w5s + RAMX v2 (champion) | 47.1% (8/17) | 0.0% | 30 | production `sn2rx` |
| HX Attempt 1 | 23.5% (4/17) | 0.0% | 30 | 15s confirm capped several true attacks at 0.49 |
| HX-C Attempt 2 | **70.6% (12/17)** | 0.0% | 30 | new causal 33-way net; technique names unused |

**Champion bar:** det ≥ 47.1% and **0% protocol FP**.
**HX Attempt 1 LOSES.**
**HX-C Attempt 2 WINS the detection bar.** Technique-on-alert precision is 0/4 (gate ≥ 0.8 missed) — emit **binary + MITRE stage only**; do not trust catalog class yet.

Raw: `D:\Cursor\Automode\baselines\hx_bench_results.json`

## Attempt 1 — why it lost

HX wrapped frozen/finetuned Shaun w5s and required a 15s confirm before `p_att` could pass 0.5. Several attacks the champion already catches (T1568, some DNS tunnels) peaked at **0.49** (`suspect` only). Holdout train FP never reached 0 (best 0.1%). Dual-window confirm kept protocol FP at 0 and **cut recall in half**.

## Attempt 2 — HX-C

New ARY-style causal Pre-LN transformer on 292-d Shaun states. Primary head is 33-way softmax (catalog + Benign). Binary = `1 - p(Benign)`. RAMX-HX context gate / cooldown / self-memory, **no 15s confirm**. Trained from random init on the rebuilt mix (internet + matched recaptures + lab PCAPs + weak CIC family labels). Best epoch 9: holdout FP 0.9%, train recall 91% — RAMX gate is what zeros protocol FP on the PCAP bench.

### Hits vs champion (same 17 scorable attacks)

HX-C keeps the champion’s floods / T1568 / some DNS, and **adds**:

- `r15` and `r79` DNS tunnel
- sequential port scan (`T1046`)
- active scan (`T1595`)

Still miss: **all T1095** and **slow SSH brute** (max P ≪ 0.5). Quiet 1× classes stay below the ML floor.

### Benign scores (honesty check)

The streaming protocol counts a benign PCAP as FP only if a true-attack label is present, so protocol FP is structurally 0. Checking `max_score ≥ 0.5` on the 13 test benign PCAPs:

| Model | benign PCAPs with max ≥ 0.5 | mean max |
|-------|-----------------------------|----------|
| Shaun RAMX v2 | 4 / 13 | 0.236 |
| HX-C | 4 / 13 | 0.234 |

Same four internet windows trip both stacks. HX-C is **not** a firehose relative to the champion.

### Technique head

Four confirmed alerts emitted a class (`T1046` on HTTP floods, `T1205` on T1568). All four names are wrong. Abstain unless this head is retrained; binary detection does not depend on it.

## Live wiring

`--backend hx_c` is available in `scripts/live_ips_multi.py` (5s Shaun ingest). Default live backend is unchanged. Production IPS stays **`sn2rx`** until you choose to A/B `hx_c` — the 33-class names are not ready even though binary det beats the champion.

Weights (gitignored): `Automode/train/checkpoints/hx_c_w5s.pt`
