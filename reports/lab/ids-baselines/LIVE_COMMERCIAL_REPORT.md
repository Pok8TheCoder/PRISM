# Live Commercial IDS Benchmark (Docker lab, not PCAP replay)

## Protocol

- **Live** 5s window captures from `target-server` @ 1× benign traffic
- **Calib:** warmup benign windows → lab p95 thresholds
- **FP test:** post-warmup benign-only windows
- **Attack:** zero-day cred theft red-team agent
- Models: KitNET-style AE, IF, RF (CIC + lab train variants), Suricata ET Open per window

## Results

| Model | Attack det | Benign FP |
|-------|------------|-----------|
| kitnet_cic_train | 16.7% | 0.0% |
| kitnet_lab_train | 100.0% | 100.0% |
| iforest_cic_train | 16.7% | 0.0% |
| iforest_lab_train | 0.0% | 0.0% |
| random_forest_cic | 0.0% | 0.0% |
| Suricata (ET Open, per-window PCAP) | 0.0% | 0.0% |

Thresholds: `{
  "kitnet_cic_live_p95": 108707.646875,
  "kitnet_lab_live_p95": 0.026384481228888035,
  "iforest_cic_live_p95": 0.09951318267138265,
  "iforest_lab_live_p95": -1.214306433183765e-17
}`

Raw: `Automode/baselines/live_commercial_ids_results.json`