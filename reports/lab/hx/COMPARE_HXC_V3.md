# HX-C vs Shaun RAMX v3 — full metrics

Fair no-oracle protocol: 5s Shaun 292-d ingest, detect @ 0.5, 10 internet-benign warmup windows, 13 held-out benign + 17 scorable attacks. Both models scored in one pass (`bench_hxc_v3_detailed.py`).

## RAMX

**HX-C uses RAMX-HX** (v3 gate, raw-score offset, relative-anomaly fusion) plus self-write memory and post-alert cooldown. Shaun v3 never writes episodic memory without oracle labels and has no cooldown.

## Headline

| Metric | Shaun RAMX v3 | HX-C |
|--------|---------------|------|
| Attack detection (PCAP) | 47.1% (8/17) | **70.6% (12/17)** |
| Protocol benign FP (detected flag) | 0.0% | 0.0% |
| Honest benign PCAP FP (any test window ≥ 0.5) | 30.8% | 30.8% |
| Honest benign window FPR | 4.2% | 4.2% |
| Mean F1 (attack PCAPs, mixed warmup+test) | 0.170 | 0.168 |
| Mean F1 attack-only | 0.170 | **0.168** |
| Mean precision attack-only | 0.471 | 0.706 |
| Mean recall attack-only | 0.119 | 0.096 |
| Pooled window F1 (attack PCAPs, test slice) | 0.292 | 0.207 |
| Pooled window precision | 1.000 | 1.000 |
| Pooled window recall | 0.171 | 0.116 |
| Pooled TP / FN windows | 34 / 165 | 23 / 176 |
| Attack-window alert rate | 17.1% | 11.6% |
| Mean warmup harm (FP windows) | 0.00 | 0.00 |
| Mean / median TTD (windows, detected only) | 8.50 / 7.50 | 5.92 / 4.00 |
| Mean TTD (seconds, detected only) | 42.5 | 29.6 |
| Mean max P on attacks | 0.380 | 0.616 |
| Mean max P on benign | 0.236 | 0.234 |
| Mean P on attack windows | 0.146 | 0.303 |
| Mean P on benign windows | 0.073 | 0.058 |
| Mean raw P (attack / benign) | 0.139 / 0.043 | 0.410 / 0.026 |
| Mean relative anomaly (attack / benign) | 0.086 / 0.069 | 0.033 / 0.067 |
| Technique alerts / precision | 0 / 0.0% | 4 / 0.0% |

Protocol FP is structurally 0 on benign PCAPs because `detected` requires a true-attack label. **Honest FP** counts any test-slice window with P ≥ 0.5 on a benign PCAP.

### How to read the F1 / recall split

HX-C **wins PCAP-level detection and TTD** (fires sooner on more attacks). Shaun v3 **wins pooled window recall and F1** (34 TP windows vs 23). HX-C’s post-alert **cooldown** caps follow-on windows at 0.49, so floods that stay above threshold for many windows (v3 F1 ~0.63–0.74) drop to ~0.30 F1 even though they still count as detected. Macro precision-on-attack-PCAPs equals detection rate here because the test slice of attack PCAPs has **zero FPs** (pooled precision 1.0) and misses contribute precision 0.

## Per class (attack PCAPs)

| Class | n | v3 det | HX-C det |
|-------|---|--------|----------|
| T1046_service_scan | 1 | 0.0% | 100.0% |
| T1071_dns_tunnel | 4 | 50.0% | 100.0% |
| T1095_non_app_protocol | 4 | 0.0% | 0.0% |
| T1110_ssh_bruteforce | 1 | 0.0% | 0.0% |
| T1499_http_flood | 3 | 100.0% | 100.0% |
| T1568_dynamic_resolution | 3 | 100.0% | 100.0% |
| T1595_active_scan | 1 | 0.0% | 100.0% |

## Per PCAP

| PCAP | Class | v3 maxP | v3 det | v3 F1-atk | v3 TTD | HX-C maxP | HX-C det | HX-C F1-atk | HX-C TTD | HX-C tech |
|------|-------|---------|--------|-----------|--------|-----------|----------|-------------|----------|-----------|
| live_http_flood_random_timing.pcap | T1499_http_flood | 0.833 | 1 | 0.692 | 7 | 1.000 | 1 | 0.300 | 1 | T1046_service_scan |
| r15_a1_T1071_dns_tunnel_none.pcap | T1071_dns_tunnel | 0.099 | 0 | 0.000 | — | 0.622 | 1 | 0.200 | 8 | — |
| r19_a1_T1095_non_app_protocol_none.pcap | T1095_non_app_protocol | 0.083 | 0 | 0.000 | — | 0.118 | 0 | 0.000 | — | — |
| r1_a3_ssh_bruteforce_slow_timing.pcap | T1110_ssh_bruteforce | 0.061 | 0 | 0.000 | — | 0.051 | 0 | 0.000 | — | — |
| r2_a2_T1595_active_scan_random_timing.pcap | T1595_active_scan | 0.084 | 0 | 0.000 | — | 1.000 | 1 | 0.154 | 21 | — |
| r2_a2_port_scan_sequential_random_timing.pcap | T1046_service_scan | 0.219 | 0 | 0.000 | — | 0.943 | 1 | 0.333 | 1 | — |
| r30_a1_T1568_dynamic_resolution_none.pcap | T1568_dynamic_resolution | 0.600 | 1 | 0.143 | 12 | 0.944 | 1 | 0.143 | 12 | T1205_traffic_signaling |
| r3_a2_http_flood_random_timing.pcap | T1499_http_flood | 0.827 | 1 | 0.741 | 6 | 1.000 | 1 | 0.300 | 1 | T1046_service_scan |
| r3_a3_http_flood_slow_timing.pcap | T1499_http_flood | 0.870 | 1 | 0.625 | 5 | 1.000 | 1 | 0.308 | 1 | T1046_service_scan |
| r47_a1_T1071_dns_tunnel_none.pcap | T1071_dns_tunnel | 0.607 | 1 | 0.200 | 6 | 0.507 | 1 | 0.200 | 4 | — |
| r51_a1_T1095_non_app_protocol_none.pcap | T1095_non_app_protocol | 0.086 | 0 | 0.000 | — | 0.255 | 0 | 0.000 | — | — |
| r62_a1_T1568_dynamic_resolution_none.pcap | T1568_dynamic_resolution | 0.600 | 1 | 0.143 | 12 | 0.589 | 1 | 0.267 | 4 | — |
| r79_a1_T1071_dns_tunnel_none.pcap | T1071_dns_tunnel | 0.141 | 0 | 0.000 | — | 0.614 | 1 | 0.182 | 8 | — |
| r83_a1_T1095_non_app_protocol_none.pcap | T1095_non_app_protocol | 0.078 | 0 | 0.000 | — | 0.314 | 0 | 0.000 | — | — |
| ui_r15_a1_T1071_dns_tunnel_none.pcap | T1071_dns_tunnel | 0.606 | 1 | 0.200 | 8 | 0.641 | 1 | 0.200 | 6 | — |
| ui_r19_a1_T1095_non_app_protocol_none.pcap | T1095_non_app_protocol | 0.069 | 0 | 0.000 | — | 0.273 | 0 | 0.000 | — | — |
| ui_r30_a1_T1568_dynamic_resolution_none.pcap | T1568_dynamic_resolution | 0.600 | 1 | 0.143 | 12 | 0.603 | 1 | 0.267 | 4 | — |
| internet_benign_w008_30s.pcap | Benign | 0.072 | 0 | 0.000 | — | 0.063 | 0 | 0.000 | — | — |
| internet_benign_w009_30s.pcap | Benign | 0.618 | 0 | 0.000 | — | 0.602 | 0 | 0.000 | — | — |
| internet_benign_w010_30s.pcap | Benign | 0.616 | 0 | 0.000 | — | 0.602 | 0 | 0.000 | — | — |
| internet_benign_w011_30s.pcap | Benign | 0.618 | 0 | 0.000 | — | 0.602 | 0 | 0.000 | — | — |
| internet_benign_w012_30s.pcap | Benign | 0.618 | 0 | 0.000 | — | 0.602 | 0 | 0.000 | — | — |
| internet_benign_w013_30s.pcap | Benign | 0.056 | 0 | 0.000 | — | 0.078 | 0 | 0.000 | — | — |
| internet_benign_w014_30s.pcap | Benign | 0.058 | 0 | 0.000 | — | 0.064 | 0 | 0.000 | — | — |
| internet_benign_w015_30s.pcap | Benign | 0.055 | 0 | 0.000 | — | 0.084 | 0 | 0.000 | — | — |
| internet_benign_w016_30s.pcap | Benign | 0.056 | 0 | 0.000 | — | 0.083 | 0 | 0.000 | — | — |
| internet_benign_w017_30s.pcap | Benign | 0.071 | 0 | 0.000 | — | 0.056 | 0 | 0.000 | — | — |
| internet_benign_w018_30s.pcap | Benign | 0.062 | 0 | 0.000 | — | 0.070 | 0 | 0.000 | — | — |
| internet_benign_w019_30s.pcap | Benign | 0.061 | 0 | 0.000 | — | 0.053 | 0 | 0.000 | — | — |
| internet_benign_w020_30s.pcap | Benign | 0.103 | 0 | 0.000 | — | 0.088 | 0 | 0.000 | — | — |

Raw JSON: `D:\Cursor\Automode\baselines\hxc_v3_detailed.json`