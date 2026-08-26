# PRISM: Predictive Risk Intelligence for Security Monitoring

**PRISM** is an AI-powered network attack forecasting platform leveraging **World Models** to move cyber defense from reactive intrusion detection to proactive, forward-looking attack prediction.

Rather than classifying isolated network flows in hindsight, PRISM learns the temporal state-transition dynamics of computer networks $P(S_{t+1} \mid S_t)$, simulates potential future attack trajectories $K$-steps ahead, maps predicted behaviors to recognized **MITRE ATT&CK** stages, and provides interpretable decision support for defenders in Enterprise and Critical Information Infrastructure (CII) environments.

---

## 🚀 Current Implementation Status

PRISM is currently implemented as a functional prototype with the following core modules:

### 1. Telemetry Processing & Feature Pipeline
- **Dual-Level Feature Extraction**: Ingests flow-level attributes (NetFlow/IPFIX format: IP/ports, TCP bitmasks, byte/packet counts, IAT statistics) and packet-level details (TTL variance, payload distributions, port access patterns).
- **Benchmark & Live Telemetry Support**: Pre-processes benchmark datasets (CSE-CIC-IDS-2018) and parses raw `.pcap` network traffic in real time via `Scapy`.

### 2. Temporal Transformer World Model
- **Dynamics Learning**: Trained to predict the probability distribution over future network states $S_{t+1}$ given sequence history $S_t, S_{t-1}, \dots, S_{t-N}$.
- **Dual-Head Architecture**:
  - **State Prediction Head**: Predicts full future feature state vector $S_{t+1}$ via MSE loss.
  - **Multi-Class Attack Classifier**: Classifies network state across multiple threat types (`Benign`, `SSH_Bruteforce`, `Port_Scan`, `HTTP_Flood`, `Slow_Loris`, `Infiltration`).
- **GPU-Accelerated**: Optimized for NVIDIA GPUs using PyTorch and PyTorch Lightning.

### 3. MITRE ATT&CK Mapping & Knowledge Base
- **Automatic Stage Tagging**: Maps predicted future states directly to MITRE ATT&CK tactics and techniques (e.g., `T1110 Brute Force` $\rightarrow$ Initial Access, `T1046 Network Service Scanning` $\rightarrow$ Reconnaissance, `T1499 Endpoint DoS` $\rightarrow$ Impact).
- **Embedded CTI Database**: Integrated dataset (`data/mitre_attack.json`) containing **222 official Enterprise MITRE ATT&CK techniques**, descriptions, and real-world threat group examples (e.g., APT28, Sandworm, Gamaredon).

### 4. Self-Fortifying Adversarial Training Loop
- **Dockerized Environment**: Deploys an isolated Debian target server (`target-server`) running SSH and Apache, paired with a containerized attacker bot (`attacker-bot`).
- **Live Traffic Capture**: Runs `tcpdump` inside target containers, transfers `.pcap` files to host GPU for real-time feature extraction and prediction.
- **Adaptive Evasion & Self-Healing**: When the model detects an attack, the bot automatically escalates evasion tactics (timing randomization, port order scrambling, source port manipulation). Successful evasions are automatically appended to training data to trigger GPU retraining cycles, continuously hardening the model.

### 5. Benchmark Validation
- **Logistic Regression Baseline**: Built non-temporal baseline classifier.
- **Performance**: Temporal Transformer World Model achieves an **85% F1 Score** (vs. 54% baseline) and cuts the False Positive Rate from ~41% down to **16.7%**, demonstrating a +31% measurable improvement from temporal dynamics learning.

---

## 🎯 Target Vision & Future Roadmap

The ultimate goal of PRISM is an enterprise-ready, fully autonomous predictive defense system.

```
                                  PRISM Target Vision
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│   Network Telemetry       Temporal & Spatial       K-Step Rollout        Proactive       │
│   (PCAP / NetFlow)  ───►    Graph World     ───►  Future Infiltration ──► Defensive     │
│                             Model (GNN)              Forecaster         Mitigation       │
│                                  │                       │            (eBPF / Firewall)  │
│                                  ▼                       ▼                               │
│                         Interpretable SHAP &   MITRE ATT&CK Stage                        │
│                         Attention Heatmaps     Timeline Dashboard                        │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Planned Features & Enhancements
1. **Hybrid Temporal-Graph Neural Network (TGNN)**: Incorporate Graph Neural Networks (PyTorch Geometric) to model network topology natively—treating hosts as nodes and active flows as edges to capture complex lateral movement across subnets.
2. **Uncertainty-Aware Autoregressive Rollout**: Extend $K$-step forward simulation to output confidence intervals and probability decay bounds as predictions project further into the future.
3. **Full SHAP & Attention Interpretability Engine**: Expose exact feature attribution heatmaps (showing which specific TCP flag, inter-arrival time, or payload anomaly triggered a prediction).
4. **Proactive Active Defense (eBPF / iptables Integration)**: Automatically generate and push defensive firewall rules or eBPF filters to sever predicted C2 channels or lateral movement paths *before* compromise is completed.

---

## 🖥️ Dashboard Architecture & Visual Design Specification

The PRISM visual interface is designed as an offline, high-density SOC (Security Operations Center) decision support web app built with **Streamlit** and **Plotly**.

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  PRISM // Predictive Risk Intelligence for Security Monitoring             [SYSTEM: ONLINE] │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  [ METRIC: Ingestion Rate ]   [ METRIC: Threat Level ]   [ METRIC: Model Accuracy ]      │
│  12.4 MB/s (1,420 flows/s)    ELEVATED (Prob 0.84)       85.2% F1 (GPU Active)           │
│                                                                                          │
├──────────────────────────────────────────────────────┬───────────────────────────────────┤
│                                                      │                                   │
│  📈 Live & Forecasted Infiltration Probability       │  🛡️ MITRE ATT&CK Kill-Chain Stage  │
│  ┌────────────────────────────────────────────────┐  │  ┌─────────────────────────────┐  │
│  │ 1.0 ┤                      /-- Forecast       │  │  │ Reconnaissance    [98%] ✓  │  │
│  │ 0.5 ┤         /----\      /                   │  │  │ Initial Access    [84%] ⚡ │  │
│  │ 0.0 └────────/──────\────/─────────────────── │  │  │ Lateral Movement  [42%] ⏳ │  │
│  │     t-10  t-5    t    t+1  t+2  t+3  t+4  t+5 │  │  │ Command & Control [12%] ░  │  │
│  └────────────────────────────────────────────────┘  │  └─────────────────────────────┘  │
│                                                      │                                   │
├──────────────────────────────────────────────────────┴───────────────────────────────────┤
│                                                                                          │
│  🔍 Feature Attribution (SHAP / Attention Heatmap)                                      │
│  Flow IAT Variance    ██████████████████████ 42%                                         │
│  SYN Flag Ratio       ██████████████ 28%                                                 │
│  Dst Port Anomaly     ████████ 16%                                                       │
│                                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  ⚡ Live Adversarial Lab Control & Self-Healing Telemetry                                │
│  [ Run Adversarial Bot ]   [ Trigger Retrain ]   [ Download Forensic Report (JSON) ]     │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Components
1. **Header & System Telemetry Bar**:
   - Live network ingestion rate (MB/s, flows/sec).
   - Global network status (`NORMAL`, `ELEVATED`, `CRITICAL`).
   - GPU VRAM consumption & inference latency metrics.
2. **Infiltration Timeline (Interactive Line Chart)**:
   - Historical observed traffic (solid line) transitioning seamlessly into $K$-step forecasted probability (dashed line with shaded confidence interval).
   - Interactive slider to scrub through past time windows or simulate future projections.
3. **MITRE ATT&CK Stage Progression Panel**:
   - Visual kill-chain pipeline highlighting active and forecasted attack stages.
   - Expandable technique cards detailing matching technique IDs (e.g., `T1046`), description snippets, and recommended defender remediation steps.
4. **Interpretable Feature Attribution Panel**:
   - Ranked horizontal bar chart of SHAP values / attention weights explaining *why* the model predicted an attack (e.g., flagging unusual inter-arrival times or SYN flag ratios).
5. **Adversarial Red-Team Control Dashboard**:
   - Interactive buttons to initiate containerized attacks (`SSH Brute Force`, `Port Scan`, `HTTP Flood`).
   - Live feedback terminal displaying bot evasion escalation, traffic capture logs, and model retraining triggers.

---

## 📁 Project Structure

```
.
├── configs/                  # Model & pipeline YAML configuration files
│   └── default.yaml
├── data/
│   ├── raw/                  # Raw PCAP captures & CSV flow datasets
│   │   └── adversarial/      # Live captured pcap files from Docker loop
│   ├── processed/            # Timestamped normalized feature matrices (Parquet)
│   └── mitre_attack.json     # 222 Enterprise MITRE ATT&CK techniques database
├── models/
│   └── checkpoints/          # PyTorch model weights (.pth)
├── notebooks/
│   └── data_exploration.ipynb
├── scripts/
│   ├── download_data.py      # Automated dataset downloader
│   └── fetch_mitre_attack.py # Script fetching official MITRE ATT&CK STIX 2.1 data
├── src/
│   ├── adversarial/          # Dockerized target server, bot, and self-healing loop
│   │   ├── attack_script.py
│   │   ├── attacker_bot.py
│   │   ├── traffic_capture.py
│   │   └── training_loop.py
│   ├── model/                # World Model PyTorch architectures
│   │   ├── world_model.py            # Binary temporal transformer
│   │   └── world_model_multiclass.py # Multi-class MITRE transformer
│   ├── baseline.py           # Logistic Regression static baseline
│   ├── pipeline/             # NetFlow & PCAP feature extraction engine
│   ├── predict/              # K-step rollout simulation
│   ├── explain/              # SHAP & attention interpretability
│   └── ui/                   # Streamlit web dashboard
├── requirements.txt
├── idea.txt                  # Original project specification
└── README.md
```

---

## ⚡ Quickstart

### 1. Environment Setup
```powershell
# Create & activate virtual environment (inherits system CUDA PyTorch)
python -m venv venv --system-site-packages
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Baseline vs. World Model Benchmark
```powershell
# Train & evaluate static logistic regression baseline
python src/baseline.py

# Train & evaluate multi-class Temporal Transformer World Model
python src/model/world_model_multiclass.py
```

### 3. Run Adversarial Self-Fortification Loop
```powershell
# Requires Docker Desktop running
python -m src.adversarial.training_loop
```

### 4. Launch Web Dashboard
```powershell
streamlit run src/ui/app.py
```
