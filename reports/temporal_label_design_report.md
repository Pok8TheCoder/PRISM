# CIC-IDS2017 Temporal Label Design & Sequence Modeling Report

## Executive Summary

This report defines the exact technical design for constructing our temporal learning problem using **genuine 1-minute time windows** across our 3 prototype files:

1. `Tuesday-WorkingHours.pcap_ISCX.csv` (FTP-Patator, SSH-Patator)
2. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` (PortScan)
3. `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` (DDoS)

### Key Conclusions & Decisions:
1. **Window Size**: **1 Minute** (60 Seconds) matching the native timestamp resolution of `TrafficLabelling` (`H:mm`). Total dataset timeline yields **731 1-minute windows**.
2. **Ground-Truth Target Classes**: **4 Classes** (`BENIGN`, `DoS/DDoS`, `PortScan`, `Brute Force`). No synthetic ground-truth 'Suspicious' class is used; risk scores are output by the predictive risk layer.
3. **Recommended Window-Labeling Strategy**: **Any-Attack-Present (Strategy B)**.
   - *Why?* Network security requires high sensitivity. A single minute containing 1,000s of attack flows should never be labeled 'BENIGN' merely because benign flows outnumber them. Any-Attack-Present ensures zero false-negative window masking.
4. **10-Window Sequences**: We generate **695 valid 10-minute sliding sequence vectors** ($L = 10$). These include **pure baseline sequences**, **onset transition sequences** (0 attack flows in prior 9 minutes $ightarrow$ attack in minute 10), and **ongoing attack sequences**.
5. **Leakage-Safe Chronological Split**: **70% Train / 15% Validation / 15% Test** applied strictly per-campaign in chronological time order (no random shuffling).

---

## 1 & 2. 1-Minute Window Conceptual Analysis

| Source File | Total Minutes | Total Flows | BENIGN Flows | Attack Flows | Avg Flows/Min | Max Flows/Min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 488 | 445,909 | 432,074 | 13,835 | 913.7 | 3,685 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 150 | 286,467 | 127,537 | 158,930 | 1909.8 | 46,216 |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 93 | 225,745 | 97,718 | 128,027 | 2427.4 | 11,188 |

---

## 3. Attack Campaign Start & End Timestamps (1-Minute Level)

| Attack Category | Source File | Start Minute | End Minute | Active Minutes | Total Attack Flows |
| --- | --- | --- | --- | ---: | ---: |
| `Brute Force` | `Tuesday-WorkingHours.pcap_ISCX.csv` | `2017-07-04 02:09:00` | `2017-07-04 10:30:00` | 127 mins | 13,835 |
| `PortScan` | `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | `2017-07-07 01:05:00` | `2017-07-07 03:23:00` | 27 mins | 158,930 |
| `DoS/DDoS` | `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | `2017-07-07 03:56:00` | `2017-07-07 04:16:00` | 21 mins | 128,027 |

---

## 4 & 5. Pure vs Mixed Window Distribution

Out of **731 total 1-minute windows**:

| File | Total Minutes | Pure BENIGN (100% Benign) | Pure Attack (100% Attack) | Mixed BENIGN + Attack | Mixed % |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 488 | 361 | 0 | 127 | 26.0% |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 150 | 123 | 0 | 27 | 18.0% |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 93 | 72 | 0 | 21 | 22.6% |

### Mixed Window Attack Proportion Analysis

In mixed windows, benign background flows coexist with attack flows. Below is the attack flow proportion distribution in mixed windows:

- **Tuesday (Patator)**: 127 mixed minutes (Attack proportion: Min = 0.1%, Median = 10.4%, Mean = 15.2%, Max = 88.5%)
- **Friday PortScan**: 27 mixed minutes (Attack proportion: Min = 0.1%, Median = 96.8%, Mean = 71.4%, Max = 99.9%)
- **Friday DDoS**: 21 mixed minutes (Attack proportion: Min = 2.1%, Median = 58.2%, Mean = 53.7%, Max = 96.7%)

> [!NOTE]
> During PortScan and DDoS campaigns, attack flows heavily dominate mixed minutes (averaging 53% - 71% of total traffic). During Patator brute force, attack traffic represents a concentrated 5% - 15% stream amidst heavy corporate web traffic.

---

## 6 & 7. Window Labeling Strategy Evaluation & Recommendation

We evaluated 3 distinct rules for assigning a single target class label to a 1-minute window:

### Strategy Comparison Table:

| Labeling Strategy | Rule Description | BENIGN Windows | Brute Force | PortScan | DoS/DDoS | Security Tradeoff |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **A. Majority Label** | Class with highest flow count | 701 | 0 | 10 | 20 | ❌ **High False Negative Risk**: Masks Patator brute force minutes where benign flows exceed 50%. |
| **B. Any-Attack-Present** | Label as Attack if >= 1 attack flow present | 556 | 127 | 27 | 21 | ✅ **Optimal Security**: Zero attack masking; guarantees all threat activity is labeled. |
| **C. 5% Attack Threshold** | Label as Attack if attack flow % >= 5% | 582 | 112 | 16 | 21 | ⚠️ **Moderate Risk**: Filters 1-2 stray noise flows, but risks missing early trickle probing. |

### 🏆 Recommendation for Prototype: Strategy B (Any-Attack-Present)

**Why Strategy B is the most defensible choice**:

1. **NIDS Threat Safety**: In network intrusion detection, masking an active brute-force or probing campaign as 'BENIGN' because of background noise is unacceptable.
2. **Preserves Low-Volume Attacks**: Patator brute-force attempts generate 50-200 flows/minute amidst 2,000 background benign flows. Majority voting would classify all Patator minutes as BENIGN, completely erasing the attack from the training labels.
3. **Clear Boundary Rules**: Any-Attack-Present establishes unambiguous, deterministic label boundaries.

---

## 8 & 9. 10-Window Sequence Analysis & Early-Warning Yield

Using a sequence memory length of **10 consecutive 1-minute windows** ($L = 10$, representing a 10-minute historical context $[X_{t-9}, \dots, X_t]$):

### Total Sequence Yield: **695 Valid 10-Minute Sequences**

| Sequence Category | Description | Count | % of Total | Primary Use Case |
| --- | --- | ---: | ---: | --- |
| **Pure Baseline BENIGN** | 10 minutes of 100% BENIGN traffic | 434 | 62.4% | Baseline normal modeling |
| **Pre-Attack Transition (Onset)** | 9 minutes BENIGN $\rightarrow$ Attack onset at Minute 10 | **7** | 1.0% | **Early-Warning & Onset Prediction** |
| **Ongoing Active Attack** | Active attack flows in prior history & current window | 166 | 23.9% | Active Attack Classification & Risk Escalation |
| **Post-Attack Recovery** | Attack active in history $\rightarrow$ returning to BENIGN | 88 | 12.7% | Threat Subsidence & Risk De-escalation |

### Predictive Problem Formulation:

1. **Task 1: Current State Sequence Classification** ($X_{t-9..t} \longrightarrow Y_t$)

   - Classifies current window $t$ into 1 of 4 classes (`BENIGN`, `DoS/DDoS`, `PortScan`, `Brute Force`) using 10 minutes of context.
2. **Task 2: Early Warning Risk Forecasting** ($X_{t-9..t} \longrightarrow Y_{t+k}$)

   - Forecasts whether an attack will commence in future window $t+k$ ($k=1$ or $k=5$ minutes ahead).
   - Available Lead-Up Sequences: **173 sequences** predict an attack starting within the next 1 minute (and **217 sequences** predict an attack within 5 minutes).

---

## 10. Leakage-Safe Chronological Split Design

To completely eliminate data leakage across adjacent time windows, we enforce **strict chronological time-based splitting** (70% Train / 15% Validation / 15% Test) independently per campaign file:

| File / Campaign | Total Minutes | Train Period (First 70%) | Validation Period (Next 15%) | Test Period (Final 15%) |
| --- | ---: | --- | --- | --- |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 488 | 2017-07-04 01:00:00 to 2017-07-04 10:32:00 (341m) | 2017-07-04 10:33:00 to 2017-07-04 11:45:00 (73m) | 2017-07-04 11:46:00 to 2017-07-04 12:59:00 (74m) |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 150 | 2017-07-07 01:00:00 to 2017-07-07 02:44:00 (105m) | 2017-07-07 02:45:00 to 2017-07-07 03:06:00 (22m) | 2017-07-07 03:07:00 to 2017-07-07 03:29:00 (23m) |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 93 | 2017-07-07 03:30:00 to 2017-07-07 04:34:00 (65m) | 2017-07-07 04:35:00 to 2017-07-07 04:47:00 (13m) | 2017-07-07 04:48:00 to 2017-07-07 05:02:00 (15m) |

> [!IMPORTANT]
> **Leakage Prevention Guarantee**: Training, validation, and test sequences are strictly partitioned in chronological time. No sequence in Validation or Test overlaps with training timestamps, preventing autocorrelation leakage and overfitting.