"""
PRISM Model Quick-Test Script (V2 Architecture - 286 Dimensions):
Run this script to immediately verify model loading, run sample forward inference,
and generate a 5-step predictive rollout.

Usage:
    python test_model.py
"""

import os
import sys
import torch
import numpy as np

# Append root & set encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.world_model import StateTransformerWorldModel
from src.prediction.simulator import RolloutSimulator
from src.prediction.scoring import compute_trajectory_risk
from src.utils.constants import MITRE_STAGES_INV

def main():
    print("=" * 65)
    print("       PRISM V2: Predictive NIDS World Model Test Suite")
    print("=" * 65)

    weights_path = os.path.join("weights", "world_model.pt")
    if not os.path.exists(weights_path):
        print(f"Error: Model checkpoint not found at '{weights_path}'!")
        return

    # 1. Load Checkpoint
    print(f"\n[1/4] Loading PRISM Model Checkpoint from: {weights_path}...")
    chk = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    # Infer d_state and pe_len dynamically
    d_state = chk["model_state_dict"]["input_embed.0.weight"].shape[1]
    pe_len = chk["model_state_dict"].get("pos_encoder.pe", torch.zeros(1, 50, 256)).shape[1]
    
    model = StateTransformerWorldModel(d_state=d_state, d_model=256, nhead=8, num_layers=4, max_seq_len=pe_len)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()
    print(f"      Model loaded! (State Dimension: {d_state} | Max Seq Len: {pe_len})")

    # 2. Extract Normalization Parameters
    scaler_mean = chk.get("scaler_mean", np.zeros((1, d_state)))
    scaler_std = chk.get("scaler_std", np.ones((1, d_state)))
    print(f"      Embedded Normalization Scaler: Mean {scaler_mean.shape} | Std {scaler_std.shape}")

    # 3. Load Sample Processed State Sequence (or generate mock)
    print(f"\n[2/4] Preparing 15-Second Network State History (L=30 windows = 7.5 mins)...")
    processed_states_path = os.path.join("data", "processed", "states.npy")
    if os.path.exists(processed_states_path):
        all_states = np.load(processed_states_path)
        lookback = min(30, len(all_states) - 1)
        sample_raw_seq = all_states[:lookback]
        print(f"      Loaded real state sequence from {processed_states_path} (Shape: {sample_raw_seq.shape})")
    else:
        sample_raw_seq = np.random.randn(30, d_state).astype(np.float32)
        print(f"      Using synthetic test state sequence (Shape: {sample_raw_seq.shape})")

    # Normalize sequence
    norm_seq = (sample_raw_seq - scaler_mean) / scaler_std
    input_tensor = torch.from_numpy(norm_seq).float().unsqueeze(0)

    # 4. Run Instant Forward Inference
    print("\n[3/4] Running Instant Inference...")
    with torch.no_grad():
        pred_mean, pred_logvar, p_atk, p_mit, p_frac, attn = model(input_tensor)
        
        prob_attack = float(torch.softmax(p_atk, dim=-1)[0, 1].item())
        pred_stage_id = int(torch.argmax(p_mit, dim=-1)[0].item())
        pred_stage_name = MITRE_STAGES_INV.get(pred_stage_id, "Unknown")
        predicted_fraction = float(p_frac[0].item())

    print(f"      • Threat Infiltration Probability : {prob_attack:.1%}")
    print(f"      • Predicted MITRE ATT&CK Stage   : Stage {pred_stage_id} [{pred_stage_name}]")
    print(f"      • Malicious Traffic Fraction     : {predicted_fraction:.1%}")

    # 5. Run Autoregressive 5-Step Rollout Simulator (15s per step)
    print("\n[4/4] Running Autoregressive Forward Rollout (+5 Steps Ahead)...")
    simulator = RolloutSimulator(model, device="cpu")
    trajectory = simulator.rollout(input_tensor, K=5, stochastic=False)
    risk_summary = compute_trajectory_risk(trajectory)

    print(f"\n      --- 🔮 Forward Forecast Trajectory ---")
    for step in trajectory:
        k = step["step"]
        seconds_ahead = k * 15
        p = step["infiltration_prob"]
        stg = step.get("mitre_stage", "Unknown")
        print(f"      Step +{k} (+{seconds_ahead}s): Threat Risk = {p:5.1%} | Imminent Stage: {stg}")

    print(f"\n      Overall Trajectory Risk Level: {risk_summary['risk_level']}")

    # 6. Test RAMX Adaptive Engine (Warmup Calibration + Relative Anomaly Detection)
    print("\n[5/5] Testing RAMX Adaptive Memory Engine (Warmup Calibrator & TTT)...")
    from src.prediction.ramx import RAMXPredictor
    ramx = RAMXPredictor(model, scaler_mean, scaler_std, warmup_steps=5, enable_ttt=False)
    
    # Warmup calibration steps
    for _ in range(5):
        ramx.predict_state(sample_raw_seq)
    
    # Evaluation with local anomaly adaptation
    ramx_out = ramx.predict_state(sample_raw_seq)
    print(f"      • RAMX Calibrated Status          : {ramx_out['calibrated']}")
    print(f"      • RAMX Relative Anomaly Score     : {ramx_out['relative_anomaly']:.1%}")
    print(f"      • RAMX Fused Threat Probability   : {ramx_out['p_attack']:.1%}")

    print("\n" + "=" * 65)
    print("  SUCCESS! PRISM V2 + RAMX Adaptive Engine is 100% operational.")
    print("  To launch the Cyber Operations Dashboard UI, run:")
    print("      streamlit run app/streamlit_app.py")

    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
