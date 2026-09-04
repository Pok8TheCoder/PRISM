# Internet-realistic corpus — all models (same PCAPs)

Benign corpus: multi-persona internet lab captures (`internet_benign/captured/`).
Calibration: first 7 benign PCAPs (p95 thresholds). FP test: held-out benign.
Attack set: existing lab adversarial PCAPs.

## Primary comparison

| Category | Model | Attack det | Benign FP | n scored |
|----------|-------|------------|-----------|----------|
| commercial | KitNET (lab train + live cal) | 100.0% | 100.0% | 26 |
| commercial | KitNET (CIC train + live cal) | 84.6% | 46.2% | 26 |
| commercial | Isolation Forest (lab train) | 100.0% | 84.6% | 26 |
| commercial | Isolation Forest (CIC train) | 0.0% | 38.5% | 26 |
| commercial | Random Forest (CIC supervised) | 0.0% | 0.0% | 26 |
| commercial | Suricata ET Open | 2.8% | 0.0% | 154 |
| prism_base | ARY-5s base | 0.0% | 0.0% | 26 |
| prism_base | Shaun w5s base | 17.6% | 0.0% | 30 |
| prism_ramx | ARY-5s + RAMX | 0.0% | 0.0% | 26 |
| prism_ramx | Shaun w5s + RAMX v2 (best) | 47.1% | 0.0% | 30 |

## All ML models (incl. RAMX v3, oracle upper bound)

| Model | Attack det | Benign FP | n scored |
|-------|------------|-----------|----------|
| kitnet_cic_train | 84.6% | 46.2% | 26 |
| kitnet_lab_train | 100.0% | 100.0% | 26 |
| iforest_cic_train | 0.0% | 38.5% | 26 |
| iforest_lab_train | 100.0% | 84.6% | 26 |
| random_forest_cic | 0.0% | 0.0% | 26 |
| ary5_base_no_oracle | 0.0% | 0.0% | 26 |
| ary5_ramx_no_oracle | 0.0% | 0.0% | 26 |
| ary5_ramx_oracle_upper_bound | 53.8% | 0.0% | 26 |
| shaun_base_w5s_no_oracle | 17.6% | 0.0% | 30 |
| shaun_ramx_v2_w5s_no_oracle | 47.1% | 0.0% | 30 |
| shaun_ramx_v3_w5s_no_oracle | 47.1% | 0.0% | 30 |

ML raw: `Automode/baselines/internet_corpus_results.json`
Suricata raw: `Automode/baselines/internet_corpus_suricata.json`