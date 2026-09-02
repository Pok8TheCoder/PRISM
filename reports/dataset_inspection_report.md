# CIC-IDS2017 Dataset Inspection Report (MachineLearningCVE)

## Executive Summary

- **Total Files**: 8
- **Total Records**: 2,830,743
- **Total Features**: 79 numerical/categorical features + 1 label
- **Schema Consistency**: YES (All 8 CSVs share identical column names and order)
- **Timestamp Column in MachineLearningCVE**: NONE (Not present in MachineLearningCVE folder)
  *(Note: Timestamps are present in `data/raw/TrafficLabelling/` under column name `Timestamp`)*

---

## 1 & 2. CSV Files and Sizes

| File Name | Size (MB) | Total Rows | Duplicate Rows | Missing Values | Inf Values |
| --- | --- | --- | --- | --- | --- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 73.55 MB | 225,745 | 2,633 | 4 | 64 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 73.34 MB | 286,467 | 72,353 | 15 | 727 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 55.62 MB | 191,033 | 6,888 | 28 | 216 |
| `Monday-WorkingHours.pcap_ISCX.csv` | 168.73 MB | 529,918 | 26,935 | 64 | 810 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 79.25 MB | 288,602 | 35,630 | 18 | 396 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 49.61 MB | 170,366 | 6,066 | 20 | 250 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 128.82 MB | 445,909 | 24,065 | 201 | 327 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 214.74 MB | 692,703 | 81,909 | 1,008 | 1,586 |

---

## 3. Schema & Feature Details

All files contain **79** columns. Header names contain leading/trailing whitespaces (e.g. `' Label'`). Column names after stripping whitespace:

```text
Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets, Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean, Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size, Avg Bwd Segment Size, Fwd Header Length.1, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd, min_seg_size_forward, Active Mean, Active Std, Active Max, Active Min, Idle Mean, Idle Std, Idle Max, Idle Min, Label
```


---

## 4 & 5. Unique Labels & Class Distribution

| Label | Record Count | Percentage |
| --- | --- | --- |
| `BENIGN` | 2,273,097 | 80.30% |
| `DoS Hulk` | 231,073 | 8.16% |
| `PortScan` | 158,930 | 5.61% |
| `DDoS` | 128,027 | 4.52% |
| `DoS GoldenEye` | 10,293 | 0.36% |
| `FTP-Patator` | 7,938 | 0.28% |
| `SSH-Patator` | 5,897 | 0.21% |
| `DoS slowloris` | 5,796 | 0.20% |
| `DoS Slowhttptest` | 5,499 | 0.19% |
| `Bot` | 1,966 | 0.07% |
| `Web Attack � Brute Force` | 1,507 | 0.05% |
| `Web Attack � XSS` | 652 | 0.02% |
| `Infiltration` | 36 | 0.00% |
| `Web Attack � Sql Injection` | 21 | 0.00% |
| `Heartbleed` | 11 | 0.00% |

### Per-File Class Breakdown

#### `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `DDoS` | 128,027 |
| `BENIGN` | 97,718 |

#### `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `PortScan` | 158,930 |
| `BENIGN` | 127,537 |

#### `Friday-WorkingHours-Morning.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `BENIGN` | 189,067 |
| `Bot` | 1,966 |

#### `Monday-WorkingHours.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `BENIGN` | 529,918 |

#### `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `BENIGN` | 288,566 |
| `Infiltration` | 36 |

#### `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `BENIGN` | 168,186 |
| `Web Attack � Brute Force` | 1,507 |
| `Web Attack � XSS` | 652 |
| `Web Attack � Sql Injection` | 21 |

#### `Tuesday-WorkingHours.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `BENIGN` | 432,074 |
| `FTP-Patator` | 7,938 |
| `SSH-Patator` | 5,897 |

#### `Wednesday-workingHours.pcap_ISCX.csv`
| Label | Count |
| --- | --- |
| `BENIGN` | 440,031 |
| `DoS Hulk` | 231,073 |
| `DoS GoldenEye` | 10,293 |
| `DoS slowloris` | 5,796 |
| `DoS Slowhttptest` | 5,499 |
| `Heartbleed` | 11 |

---

## 6. Timestamp / Date Column Status

❌ **No timestamp or date column exists in `data/raw/MachineLearningCVE/` CSV files.**
The MachineLearningCVE dataset consists of pre-extracted 78 network flow statistical features calculated per flow over windows by ISCXFlowMeter, but timestamp strings were stripped in this specific folder.
💡 *Crucial Note for Time-Series / Predictive Windowing*: The `data/raw/TrafficLabelling/` folder contains the raw labeled CSV files **WITH `Timestamp` column** (e.g., `Timestamp` formatted as `dd/MM/yyyy HH:mm:ss` or `d/M/yyyy H:mm`).

---

## 7. Missing Values & Duplicate Rows Analysis

- **Total Duplicate Rows across files**: 256,479 (9.06%)
- **Total Missing (Null/NaN) Values**: 1,358
- **Total Infinite (`Inf`/`Infinity`) Values**: 4,376

> [!NOTE]
> Missing and infinite values occur primarily in rate/flow metrics like `Flow Bytes/s` and `Flow Packets/s` when flow duration is zero. These must be cleaned during preprocessing.

---

## 8. Schema Consistency Check

✅ **All 8 CSV files share 100% consistent column names and structural schemas.**


---

## 9. File Mapping by Traffic / Attack Category

- **BENIGN Traffic**: Present in **ALL 8 files** (Total: 2,273,097 records)
- **DoS / DDoS**:
  - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` -> **DDoS** (128,027)
  - `Wednesday-workingHours.pcap_ISCX.csv` -> **DoS Hulk** (231,073), **DoS GoldenEye** (10,293), **DoS slowloris** (5,796), **DoS Slowhttptest** (5,499)
- **Brute Force (Patator)**:
  - `Tuesday-WorkingHours.pcap_ISCX.csv` -> **FTP-Patator** (7,938), **SSH-Patator** (5,897)
- **PortScan**:
  - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` -> **PortScan** (158,930)
- **Other Attacks**:
  - `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` -> Web Attacks (Brute Force: 1,507, XSS: 652, Sql Injection: 21)
  - `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` -> Infiltration (36)
  - `Friday-WorkingHours-Morning.pcap_ISCX.csv` -> Bot (1,966)

---

## 10. Record Estimate for Selected 4 Categories

If we select only **BENIGN, DoS/DDoS, Brute Force, and PortScan**:

| Category | Included Labels | Total Count | Percentage of 4-Group Total |
| --- | --- | --- | --- |
| `BENIGN` | 2,273,097 | 80.42% |
| `DoS/DDoS` | 380,688 | 13.47% |
| `Brute Force (Patator)` | 13,835 | 0.49% |
| `PortScan` | 158,930 | 5.62% |

**Total Selected Category Records**: **2,826,550** out of 2,830,743 total dataset records (99.85% of full dataset).


---

## 💡 Recommendation for First Prototype

To keep the initial prototype fast, lightweight, and balanced while covering diverse attack behaviors, we recommend:

### Recommended Files (3 CSV files):

1. `Tuesday-WorkingHours.pcap_ISCX.csv` (~135 MB) -> Contains **BENIGN** + **Brute Force (FTP-Patator, SSH-Patator)**
2. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (~77 MB) -> Contains **BENIGN** + **PortScan**
3. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (~77 MB) -> Contains **BENIGN** + **DDoS**

### Combined Recommended Subset Profile:

- **Total Disk footprint**: ~289 MB (vs 824 MB full dataset)
- **Total Records in subset**: ~717,143 records
- **Class Distribution in Subset**:
  - `BENIGN`: ~424,248 (59.2%)
  - `PortScan`: 158,930 (22.2%)
  - `DDoS`: 128,027 (17.9%)
  - `FTP-Patator`: 7,938 (1.1%)
  - `SSH-Patator`: 5,897 (0.8%)

**Why this subset?**
- Gives clean representation of Reconnaissance (PortScan), Password Attacks (Brute Force), and Volumetric Attacks (DDoS).
- Avoids the huge 225 MB Wednesday file while maintaining full coverage of essential attack types.
- Can further sub-sample BENIGN to 100k-200k records for near-instant iteration during initial model/pipeline development.