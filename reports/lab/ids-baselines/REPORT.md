# IDS Baseline Validation Report

Generated: 2026-09-03 (full validation run)

## What we tested

Four gaps from the prior baseline research were closed:

1. **Benign PCAP corpus** — 10 live captures at 1× traffic (`data/raw/adversarial/benign/`)
2. **Fair PCAP benchmark** — same 5s window / 1× scale protocol for all models
3. **Suricata** — ET Open rules, offline PCAP replay via Docker (`jasonish/suricata:7.0.3`)
4. **Live cred theft** — ML-only (`--event-block-stages none`), 1× benign, 5s attack windows

Scripts: `Automode/baselines/{capture_benign_pcaps,bench_fair_ids,suricata_pcap_bench,run_ids_validation}.py`

Raw JSON: `Automode/baselines/{fair_ids_results,suricata_results,suricata_benign_results}.json`

---

## 1. PCAP benchmark (fair protocol)

| Model | Attack det | Benign FP | Scored (atk/ben) | Notes |
|-------|------------|-----------|------------------|-------|
| **KitNET-style AE** | **100%** | **100%** | 23 (13/10) | Flags everything — useless |
| **Isolation Forest** | **100%** | **100%** | 23 (13/10) | Same — flags everything |
| Random Forest | 0% | 0% | 23 (13/10) | Never crosses 0.5 threshold |
| ARY-5s base | 0% | 0% | 148 (138/10) | No RAMX memory |
| **ARY-5s + RAMX** | **98.6%** | **0%** | 148 (138/10) | Mean F1 0.93 on attacks |
| Shaun base w5s | 17.6% | 0% | 27 (17/10) | Only 17 attack PCAPs had enough 5s windows |
| **Shaun + RAMX v2 w5s** | **100%** | **0%** | 27 (17/10) | Mean F1 1.0 on scorable attacks |

**Key takeaway:** KitNET/IF looked like they "beat" PRISM on the old 13-PCAP attack-only run. With benign PCAPs included, they **alert on 100% of benign traffic too** — they are not viable IDS baselines on this lab domain. On a fair bench, **Shaun+RAMX is best on detection with zero benign FP**, followed by ARY+RAMX.

Sklearn models only score PCAPs with ≥5 ingest windows (23/151 total). ARY streaming scores more PCAPs (148) because CIC warmup padding handles short captures.

---

## 2. Suricata (ET Open, offline PCAP)

| Corpus | PCAPs | Detection | FP rate |
|--------|-------|-----------|---------|
| Attack (first 35) | 35 | **0%** | n/a |
| Benign (live captures) | 10 | n/a | **0%** |

Suricata produced **zero alerts** on all 45 PCAPs tested. The Docker lab traffic is internal, low-volume HTTP — it does not match ET Open signature patterns (no obvious SQLi strings in PCAP bytes at this capture granularity, quiet flows, no external C2).

**Suricata does not outperform PRISM models here** — it does not detect anything at all on this lab corpus without rule tuning or inline mirroring during live attacks.

---

## 3. Live zero-day cred theft (ML-only, 1× scale, 5s windows)

Objective: `T1555_sqli_cred_theft` — never in CIC training. Event blocking disabled.

| System | F1 | Detected | Max P(attack) | Outcome | Benign harm |
|--------|-----|----------|---------------|---------|-------------|
| **ARY-5s + RAMX** | 0.0 | No | ~0.05 | undetected_theft | 0 |
| **SN2RX3 w5s + RAMX** | 0.0 | No | ~0.02 | undetected_theft | 0 |
| SN base w5s (shadow) | 0.0 | No | ~0.02 | — | 0 |

Creds stolen at ~21–24s wall time. At **1× traffic**, attack windows contain too few flows for any ML model to cross the 0.5 threshold — even Shaun.

**Prior result at 10× scale** (`results/ips_redteam/sn2rx_compare/`):

| System | F1 | Detected |
|--------|-----|----------|
| SN2RX v2 + RAMX | **0.80** | Yes |
| ARY-5s + RAMX | 0.0 | No |

Live detection requires either **scaled traffic** (10×) or **event-based blocking** (recon stage watcher — the theft-prevention fix). No external baseline was tested live; KitNET/IF are offline-only in our harness.

---

## 4. Does anything outperform PRISM?

| Test | Winner | Beats PRISM? |
|------|--------|--------------|
| PCAP bench (fair, with benign FP audit) | **Shaun+RAMX v2 w5s** (100% det, 0% FP) | No — PRISM wins |
| PCAP bench | KitNET/IF (100% det) | **Yes on detection, but 100% benign FP** — not usable |
| Suricata ET Open | Nobody (0% det) | No |
| Live cred theft @ 1× | Nobody (all ML fail) | Tie at zero |
| Live cred theft @ 10× | **Shaun SN2RX** (F1 0.8) | No — PRISM wins |
| Theft prevention (event block) | Event watcher | N/A — not ML |

**Bottom line:** No fairly benchmarked external model outperforms PRISM. KitNET/IF only "win" by flagging everything. Suricata is blind on quiet Docker lab PCAPs. On the tests we trust, **Shaun+RAMX is the strongest detector**; **ARY+RAMX is close on PCAP replay** but weaker on live zero-day cred theft at scale.

---

## 5. Recommendations

1. **Do not trust KitNET/IF 100% numbers** without benign FP audit — confirmed 100% FP here.
2. **Suricata needs inline mirroring + tuned rules** for SQLi on the lab webapp; offline PCAP replay is insufficient.
3. **Live cred theft @ 1× is below all ML models' sensitivity floor** — use event blocking or scale traffic for ML evaluation.
4. **True KitNET-py on raw bytes** remains untested; our baseline uses PRISM 242-d states (same handicap as before, but now with benign corpus).

---

## Artifacts

| Path | Content |
|------|---------|
| `data/raw/adversarial/benign/benign_w*.pcap` | 10 benign captures |
| `Automode/baselines/fair_ids_results.json` | Full PCAP bench |
| `Automode/baselines/suricata_results.json` | Suricata on 35 attack PCAPs |
| `Automode/baselines/suricata_benign_results.json` | Suricata on 10 benign PCAPs |
| `results/ips_redteam/ids_validation/ary_ml_only/` | Live ARY @ 1× ML-only |
| `results/ips_redteam/ids_validation/sn2rx3_ml_only/` | Live Shaun @ 1× ML-only |
