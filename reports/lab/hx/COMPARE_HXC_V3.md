# HX-C vs Shaun RAMX v3

Fair no-oracle protocol: 5s Shaun 292-d ingest, detect @ 0.5, 10 internet-benign warmup windows, 13 held-out benign + 17 scorable attacks. Shaun v3 from `internet_corpus_results.json`; HX-C from `hx_bench_results.json` (same 30 PCAPs).

## RAMX in HX-C

**Yes.** `StreamingHXC` scores through **RAMX-HX**, which is the v3 gate (cap 0.49) + raw-score median offset + relative-anomaly fusion. Extra vs Shaun v3: **self-write episodic memory** (no oracle) and **post-alert cooldown**. Shaun v3’s memory bank is never written without labels.

## Headline

| | Shaun RAMX v3 (`sn2rx3`) | HX-C |
|--|--|--|
| Attack det | 47.1% (8/17) | **70.6% (12/17)** |
| Protocol FP | 0% | 0% |
| Benign PCAPs with max P ≥ 0.5 | 4/13 (w009–w012) | **same 4/13** (slightly lower peaks) |
| Technique names | none | 4 alerts, **0 correct** |

v2 and v3 have **identical detection decisions** on this set; max P can differ by up to ~0.16.

## HX-C-only hits

- r15 / r79 DNS tunnel
- sequential port scan (T1046)
- T1595 active scan

## Missed by both

All T1095 PCAPs and slow SSH brute (max P ≪ 0.5).
