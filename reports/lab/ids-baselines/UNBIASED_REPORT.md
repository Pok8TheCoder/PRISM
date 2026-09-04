# Unbiased IDS Baseline Report

Generated: 2026-09-04

This report replaces the biased comparison in `REPORT.md`. Read this for conclusions.

## What was wrong before

| Bias | Old behavior | Fix |
|------|--------------|-----|
| KitNET/IF threshold | CIC p95 on OOD lab traffic → 100% det **and** 100% FP | Calibrate on **held-out lab benign** (7 PCAPs) |
| PRISM warmup | CIC validation windows | **Lab benign PCAPs only** |
| ARY RAMX | Oracle labels fed to memory (`true_bin` from filename) | Primary row: **`true_bin=None`** |
| “KitNET wins” | Misleading headline | Separate **lab-trained** baselines + FP on 3 test benign |

## Fairness protocol

- **Threshold:** p95 of window scores from 7 calibration benign PCAPs
- **FP test:** 3 held-out benign PCAPs (never used for threshold)
- **Attack set:** 141 PCAPs (only 13–17 have ≥5 scorable windows depending on ingest path)
- **PRISM RAMX:** no oracle in primary rows; oracle row labeled upper-bound only
- **Suricata:** `suricata_lab.yaml` with `172.16.0.0/12` in HOME_NET (pending Docker rerun)

Scripts: `Automode/baselines/bench_unbiased_ids.py`, `run_unbiased_validation.py`

---

## Primary results (no oracle, lab-calibrated thresholds)

| Model | Attack det | Benign FP | Scored (atk/ben) |
|-------|------------|-----------|------------------|
| KitNET lab-train + lab-cal | **100%** | **33.3%** | 16 (13/3) |
| Isolation Forest lab-train + lab-cal | **100%** | **33.3%** | 16 (13/3) |
| KitNET CIC-train + lab-cal | 92.3% | 0% | 16 (13/3) |
| Isolation Forest CIC-train + lab-cal | 0% | 33.3% | 16 (13/3) |
| Random Forest (CIC supervised) | 0% | 0% | 16 (13/3) |
| ARY-5s base (no oracle) | 7.7% | 0% | 16 (13/3) |
| ARY-5s + RAMX (no oracle) | 7.7% | 0% | 16 (13/3) |
| **Shaun base w5s (no oracle)** | 17.6% | 0% | 20 (17/3) |
| **Shaun + RAMX v2 w5s (no oracle)** | **47.1%** | **0%** | 20 (17/3) |
| **Shaun + RAMX v3 w5s (no oracle)** | **47.1%** | **0%** | 20 (17/3) |

### Oracle upper bound (NOT fair — shows prior bias source)

| Model | Attack det | Benign FP |
|-------|------------|-----------|
| ARY-5s + RAMX **with oracle labels** | 53.8% | 0% |

The old biased bench showed ARY RAMX at **98.6%** because oracle labels + CIC warmup inflated memory. Without oracle it drops to **7.7%** — same as base.

---

## Calibrated thresholds (lab benign p95)

| Model | Threshold |
|-------|-----------|
| KitNET CIC-train | 179,190 |
| KitNET lab-train | 0.577 |
| IF CIC-train | 0.109 |
| IF lab-train | 0.000 |

Lab-trained AE threshold is sensible (~0.58). CIC-trained AE threshold is orders of magnitude wrong for lab traffic unless recalibrated.

---

## Honest conclusions

### Do external models beat PRISM fairly?

**Partially yes, with tradeoffs:**

- **Lab-trained KitNET-style AE and IF** detect **100%** of scorable attacks vs Shaun RAMX **47%**
- But they incur **33% benign FP** on held-out lab benign (1/3 PCAPs)
- **Shaun + RAMX v2/v3** is the **best PRISM system without cheating**: 47% det, **0% FP** (tie on this bench)
- **ARY + RAMX without oracle is not useful** on this lab (7.7% = base model)

### Was the old “PRISM wins” conclusion wrong?

**Mostly yes for ARY RAMX.** The 98.6% figure depended on oracle labels. Fair ARY RAMX ≈ base ≈ useless here.

**Shaun RAMX still leads PRISM** without oracle, but does **not** beat lab-adapted unsupervised baselines on detection rate.

### What about commercial IDS (Suricata)?

Not re-run (Docker offline). Prior run with default config: 0% on quiet lab PCAPs. With `HOME_NET` fix pending — do not conclude commercial IDS is bad from our lab alone.

---

## Remaining limitations (still not perfect)

1. **Not real KitNET-py** — still MLP AE on PRISM 242-d states, not Kitsune bytes
2. **Small scorable set** — 13–17 attack PCAPs pass MIN_WINDOWS=5
3. **Small benign test** — only 3 held-out PCAPs for FP
4. **Suricata** — offline replay only; inline sidecar not run yet
5. **RF/ARY** — trained on CIC, evaluated on lab (OOD); no lab fine-tune for them

---

## Artifacts

| File | Content |
|------|---------|
| `Automode/baselines/unbiased_ids_results.json` | Full per-PCAP results |
| `Automode/baselines/bench_unbiased_ids.py` | Unbiased benchmark script |
| `Automode/baselines/suricata_lab.yaml` | HOME_NET-aware Suricata config |
| `reports/lab/ids-baselines/REPORT.md` | Original (biased) run — keep for history |

Re-run:

```powershell
D:\Cursor\PRISM\venv\Scripts\python.exe D:\Cursor\Automode\baselines\bench_unbiased_ids.py
# With Docker up:
D:\Cursor\PRISM\venv\Scripts\python.exe D:\Cursor\Automode\baselines\run_unbiased_validation.py
```
