# PRISM Lab & Docker Reports

Central index for all adversarial-lab, Docker IPS, and training experiment reports.

**Convention:** When a lab or Docker test completes, add a markdown report under the appropriate subdirectory and update this index.

| Category | Path | Description |
|----------|------|-------------|
| Lab — theft prevention | [lab/theft-prevention/](lab/theft-prevention/REPORT.md) | Live IPS cred-theft block tests |
| Lab — slow-rising poison | [lab/poison-slow-rise/](lab/poison-slow-rise/SUMMARY.md) | RAMX calibrator/memory poisoning |
| Lab — SNV2 windows | [lab/snv2-window-training/](lab/snv2-window-training/REPORT.md) | CIC offline + [lab PCAP compare](lab/snv2-window-training/LAB_COMPARE.md) |
| Lab — IPS model compare | [lab/ips-model-compare/](lab/ips-model-compare/SUMMARY.md) | ARY vs Shaun head-to-head |
| Lab — IDS baselines | [lab/ids-baselines/](lab/ids-baselines/REPORT.md) | KitNET/IF/Suricata vs PRISM (biased — see UNBIASED) |
| Lab — IDS unbiased | [lab/ids-baselines/UNBIASED_REPORT.md](lab/ids-baselines/UNBIASED_REPORT.md) | Fair bench: lab-cal thresholds, no oracle |
| Lab — ARY RAMX versions | [lab/ips-model-compare/ary-v01-v02/](lab/ips-model-compare/ary-v01-v02/REPORT.md) | v01 vs v02 IPS |
| Docker — infrastructure | [docker/infrastructure/](docker/infrastructure/REPORT.md) | Compose lab setup |

## Report status

| Report | Date | Status |
|--------|------|--------|
| Theft prevention (all models) | 2026-09-03 | Done |
| Poison slow-rise — sn2rx3 w5s | 2026-09-03 | Done |
| Poison slow-rise — ary5_ramx 5s | 2026-09-03 | Done |
| Poison slow-rise — sn_base w5s | 2026-09-03 | Done |
| SNV2 window training | 2026-09-03 | Done |
| IPS sn2rx vs ARY compare | 2026-09-02 | Done |
| ARY RAMX v01 vs v02 | 2026-09-02 | Done |
| Docker lab infrastructure | 2026-09-03 | Done |
| IDS baseline validation (fair bench) | 2026-09-03 | Superseded — see unbiased |
| IDS unbiased validation | 2026-09-04 | Done (incl. Shaun RAMX v3) |

## Raw data locations

- IPS live rounds: `results/ips_redteam/`
- Poison experiments: `results/ips_redteam/poison_slow_rise/`
- SNV2 training: `PRISM-shaun/results/`, `PRISM-shaun/weights/`
