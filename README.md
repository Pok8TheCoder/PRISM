# PRISM: Predictive Risk Intelligence for Security Monitoring

AI-Based Network Attack Forecasting platform using World Models to learn network state-transition dynamics, forecast attack trajectories before compromise, and map predictions to MITRE ATT&CK stages.

## Project Structure

```
.
├── configs/          # YAML configuration files
├── data/             # Telemetry data & MITRE ATT&CK JSON
├── models/           # Trained checkpoints
├── notebooks/        # Data exploration notebooks
├── scripts/          # Execution entry points & data fetch scripts
├── src/
│   ├── adversarial/  # Attacker bot & live capture loop
│   ├── pipeline/     # PCAP & NetFlow feature extraction
│   ├── model/        # Multi-class World Model architectures (Transformer)
│   ├── predict/      # Forward simulation & MITRE stage classifier
│   ├── explain/      # SHAP & attention feature attribution
│   └── ui/           # Streamlit dashboard
├── requirements.txt
└── idea.txt
```

## Quickstart

1. **Activate Environment**:
   `.\venv\Scripts\activate`

2. **Run Interface**:
   `streamlit run src/ui/app.py`
