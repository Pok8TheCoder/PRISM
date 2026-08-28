# PRISM: Predictive Recurrent Infiltration State Model

> **A World Model for Network Infiltration Trajectory Forecasting & MITRE ATT&CK Mapping**

---

### Reference & Contact Information
- **Agency Reference:** National Critical Information Infrastructure Protection Centre (NCIIPC), India
- **Official Website:** [nciipc.gov.in](https://nciipc.gov.in)
- **Helpdesk Contact:** [helpdesk1@nciipc.gov.in](mailto:helpdesk1@nciipc.gov.in)

---

## 📌 Overview

**PRISM** is an open-source, fully offline World Model AI architecture designed for Critical Information Infrastructure (CII) defense. Unlike static classifiers that operate on isolated alerts or single packet flows, PRISM models the **state-transition dynamics** $P(S_{t+1} \mid S_t, a_t)$ of an enterprise network environment.

By combining temporal sequence models (Transformers/LSTMs/GNNs) with multi-task prediction heads, PRISM performs **$K$-step autoregressive forward simulation**, enabling defenders to forecast multi-stage attack progression (e.g. Reconnaissance $\rightarrow$ Initial Access $\rightarrow$ Lateral Movement $\rightarrow$ C2 $\rightarrow$ Exfiltration/Impact) up to 10–30 minutes before critical impact occurs.

---

## 📚 Supported Public Datasets & Knowledge Bases

PRISM natively supports ingestion, feature normalization, and temporal windowing for major open cybersecurity datasets and threat intelligence frameworks:

### Public Datasets
1. **CSE-CIC-IDS2018:** Canadian Institute for Cybersecurity intrusion dataset (AWS S3 mirror support).
2. **CTU-13:** Stratosphere IPS botnet traffic dataset.
3. **UNSW-NB15:** UNSW Canberra Cyber Range network intrusion dataset (49 features, 9 attack categories).
4. **CICIoT2023:** Real-time IoT security dataset (86 features, 33 attack vectors).
5. **LANL Authentication Dataset:** Los Alamos National Laboratory host and network log events.
6. **DARPA Intrusion Detection Datasets:** DARPA 1998/1999/2000 network traffic evaluations.

### Open Knowledge Bases
- **MITRE ATT&CK Framework:** 7-stage attack kill chain mapping (TA0043, TA0001, TA0008, TA0011, TA0010, TA0040).
- **CAPEC (Common Attack Pattern Enumeration and Classification):** Integrated attack pattern lookups (e.g. CAPEC-287 TCP Port Scan, CAPEC-66 SQLi, CAPEC-640 Pass the Hash).
- **CVE / NVD (National Vulnerability Database):** Pre-compiled offline mapping to critical vulnerabilities (Log4Shell CVE-2021-44228, EternalBlue CVE-2017-0144, ProxyShell, Spring4Shell, Zerologon).

---

## ⚙️ Key System Architecture

1. **Flow & Packet Extraction Pipeline:** Standardises raw netflow CSVs and Scapy PCAP features into fixed-size temporal state vectors $S_t \in \mathbb{R}^{D}$.
2. **State Transformer World Model:** Causal multi-head self-attention network trained with multi-task loss:
   $$\mathcal{L} = \lambda_{\text{dyn}} \mathcal{L}_{\text{NLL}}(S_{t+1}, \hat{\mu}, \hat{\sigma}^2) + \lambda_{\text{inf}} \mathcal{L}_{\text{BCE}}(y_{\text{inf}}, \hat{y}) + \lambda_{\text{mitre}} \mathcal{L}_{\text{CE}}(y_{\text{stage}}, \hat{m})$$
3. **K-Step Autoregressive Simulator:** Predicts future state trajectories $S_{t+1}, \dots, S_{t+K}$ with Monte Carlo ensemble uncertainty estimation.
4. **Explainability Suite:** Integrated Gradients, SHAP DeepExplainer/KernelExplainer, and temporal attention rollouts.
5. **Streamlit Interactive Command Center:** Real-time SOC dashboard for trajectory visualization, alert escalation, and playbook guidance.

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/PRISM-WorldModel/PRISM.git
cd PRISM
pip install -r requirements.txt
```

### 2. Generate Synthetic Demo Data
To test the pipeline out of the box without downloading massive PCAPs:
```bash
python scripts/download_data.py --demo
```

### 3. List & Preprocess Supported Datasets
List registered datasets:
```bash
python scripts/download_data.py --list-datasets
```
Preprocess any supported dataset (e.g., UNSW-NB15):
```bash
python scripts/download_data.py --dataset unsw-nb15 --preprocess
```

### 4. Train the World Model
```bash
python scripts/train.py --states data/raw/demo_states.npz --epochs 30
```

### 5. Run Offline Inference & $K$-Step Rollout
```bash
python scripts/infer.py --states data/raw/demo_states.npz --k-steps 10 --shap
```

### 6. Run Benchmark Evaluation (World Model vs Static Baselines)
```bash
python scripts/evaluate.py --states data/raw/demo_states.npz --train-baselines --plot
```

### 7. Launch Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 📄 License & Compliance

PRISM is released under the **MIT License**. All dataset parsers use open-source public schemas and operate 100% offline without external network or API dependencies.
