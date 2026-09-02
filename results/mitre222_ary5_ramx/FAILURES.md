# ARY-5sV01 + RAMX — MITRE ATT&CK (222 techniques) benchmark

**Model:** ARY-5sV01 + RAMX (oracle eval, 5s windows, 138 lab PCAPs)

## Summary

| Metric | Count |
|---|---:|
| Total MITRE techniques | 222 |
| Host-only (not network evaluable) | 161 |
| Network-observable | 61 |
| Tested (lab PCAP exists) | 61 |
| **Pass (all PCAPs detect)** | **52** |
| **Fail (any PCAP missed)** | **9** |
| Partial misses (some PCAPs) | 2 classes |
| Network-observable, no PCAP yet | 0 |

## Detection failures (tested, missed)

### `T1046_service_scan`
- MITRE techniques (8): T1046, T1590, T1592, T1593, T1594, T1596, T1597, T1598
- Lab PCAPs: 9 — all missed
- Mean F1: 0.873
  - `live_port_scan_sequential_none.pcap`

### `T1135_share_discovery`
- MITRE techniques (1): T1135
- Lab PCAPs: 4 — all missed
- Mean F1: 0.750
  - `r44_a1_T1135_share_discovery_none.pcap`


## Partial failures (class detected on some PCAPs, missed on others)

### `T1046_service_scan` — 89% det rate (8/9 PCAPs)
- MISS `live_port_scan_sequential_none.pcap` max_p=0.0079

### `T1135_share_discovery` — 75% det rate (3/4 PCAPs)
- MISS `r44_a1_T1135_share_discovery_none.pcap` max_p=0.0081

## Full technique failure list

| Technique | Name | Mapped class | Status |
|---|---|---|---|
| T1046 | Network Service Discovery | T1046_service_scan | FAIL |
| T1135 | Network Share Discovery | T1135_share_discovery | FAIL |
| T1590 | Gather Victim Network Information | T1046_service_scan | FAIL |
| T1592 | Gather Victim Host Information | T1046_service_scan | FAIL |
| T1593 | Search Open Websites/Domains | T1046_service_scan | FAIL |
| T1594 | Search Victim-Owned Websites | T1046_service_scan | FAIL |
| T1596 | Search Open Technical Databases | T1046_service_scan | FAIL |
| T1597 | Search Closed Sources | T1046_service_scan | FAIL |
| T1598 | Phishing for Information | T1046_service_scan | FAIL |

## Host-only techniques (161) — not network-evaluable

These require host telemetry (process, registry, etc.); flow-based IDS cannot score them.

- T1003: OS Credential Dumping
- T1005: Data from Local System
- T1006: Direct Volume Access
- T1011: Exfiltration Over Other Network Medium
- T1012: Query Registry
- T1014: Rootkit
- T1036: Masquerading
- T1037: Boot or Logon Initialization Scripts
- T1039: Data from Network Shared Drive
- T1047: Windows Management Instrumentation
- T1052: Exfiltration Over Physical Medium
- T1053: Scheduled Task/Job
- T1055: Process Injection
- T1056: Input Capture
- T1057: Process Discovery
- T1059: Command and Scripting Interpreter
- T1068: Exploitation for Privilege Escalation
- T1069: Permission Groups Discovery
- T1070: Indicator Removal
- T1072: Software Deployment Tools
- ... and 141 more (see JSON)