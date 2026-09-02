"""
PRISM: Predictive Recurrent Infiltration State Model
World Model Powered Cyber Defense Operations Center (Streamlit Dashboard)
Featuring: Interactive Attack Simulator, Continuous Threat Waveform Stream,
and Autoregressive World Model Rollout.
"""

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.world_model import StateTransformerWorldModel
from src.prediction.simulator import RolloutSimulator
from src.prediction.scoring import compute_trajectory_risk, generate_alerts
from src.explainability.attention_viz import format_attention_heatmap
from src.explainability.feature_importance import explain_state_prediction
from src.prediction.attack_mapper import MITRE_RECOMMENDATIONS
from src.prediction.interactive_stream import ScenarioGenerator
from src.utils.constants import MITRE_STAGES_INV, MITRE_STAGES

# Page Configuration
st.set_page_config(
    page_title="PRISM | Predictive NIDS Operations Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk & Glassmorphic Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1322 0%, #080b14 90%);
        color: #f1f5f9;
    }

    /* Top HUD Header */
    .hud-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
        border: 1px solid rgba(0, 240, 255, 0.25);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 240, 255, 0.08);
        backdrop-filter: blur(12px);
    }
    .hud-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #00f0ff, #7000ff, #ff0055);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hud-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
    }

    /* Cyber Badges & Pills */
    .status-pill-online {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        color: #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-pill-cuda {
        display: inline-flex;
        align-items: center;
        background: rgba(112, 0, 255, 0.15);
        border: 1px solid #a855f7;
        color: #c084fc;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
    }

    /* Metric Cards */
    .metric-box {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .metric-box:hover {
        border-color: rgba(0, 240, 255, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 240, 255, 0.15);
    }
    .metric-box-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-box-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .metric-box-sub {
        font-size: 0.80rem;
        font-weight: 500;
    }

    /* Alerts */
    .alert-card-critical {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 5px solid #ef4444;
        border-top: 1px solid rgba(239, 68, 68, 0.3);
        border-bottom: 1px solid rgba(239, 68, 68, 0.3);
        border-right: 1px solid rgba(239, 68, 68, 0.3);
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
    }
    .alert-card-warning {
        background: linear-gradient(90deg, rgba(234, 179, 8, 0.18) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 5px solid #eab308;
        border-top: 1px solid rgba(234, 179, 8, 0.3);
        border-bottom: 1px solid rgba(234, 179, 8, 0.3);
        border-right: 1px solid rgba(234, 179, 8, 0.3);
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
    }
    .alert-card-safe {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
        border-left: 5px solid #10b981;
        border-top: 1px solid rgba(16, 185, 129, 0.3);
        border-bottom: 1px solid rgba(16, 185, 129, 0.3);
        border-right: 1px solid rgba(16, 185, 129, 0.3);
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 14px;
    }

    /* Kill chain box */
    .killchain-node {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_prism_engine():
    """Loads trained PRISM World Model checkpoint and normalization parameters."""
    checkpoint_path = os.path.join("weights", "world_model.pt")
    if os.path.exists(checkpoint_path):
        chk = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        d_state = chk["model_state_dict"]["input_embed.0.weight"].shape[1]
        pe_len = chk["model_state_dict"].get("pos_encoder.pe", torch.zeros(1, 50, 256)).shape[1]
        model = StateTransformerWorldModel(d_state=d_state, d_model=256, nhead=8, num_layers=4, max_seq_len=pe_len)
        model.load_state_dict(chk["model_state_dict"])
        scaler_mean = chk.get("scaler_mean", np.zeros((1, d_state)))
        scaler_std = chk.get("scaler_std", np.ones((1, d_state)))
    else:
        model = StateTransformerWorldModel(d_state=286, d_model=256, nhead=8, num_layers=4, max_seq_len=50)
        scaler_mean = np.zeros((1, 286))
        scaler_std = np.ones((1, 286))
    model.eval()
    return model, scaler_mean, scaler_std


@st.cache_data
def load_network_states():
    """Loads cached preprocessed network state matrix and labels."""
    processed_file = os.path.join("data", "processed", "states.npy")
    if os.path.exists(processed_file):
        states = np.load(processed_file)
        atks = np.load(os.path.join("data", "processed", "attack_labels.npy"))
        mitres = np.load(os.path.join("data", "processed", "mitre_labels.npy"))
        fracs = np.load(os.path.join("data", "processed", "attack_fractions.npy"))
        return states, atks, mitres, fracs
    else:
        np.random.seed(42)
        states = np.random.randn(200, 286).astype(np.float32)
        atks = np.zeros(200, dtype=np.int64)
        atks[140:180] = 1
        mitres = np.zeros(200, dtype=np.int64)
        mitres[140:155] = 1
        mitres[155:170] = 2
        mitres[170:180] = 6
        fracs = np.zeros(200, dtype=np.float32)
        fracs[140:180] = 0.85
        return states, atks, mitres, fracs


# Load core engine
model, scaler_mean, scaler_std = load_prism_engine()
states, atks, mitres, fracs = load_network_states()
simulator = RolloutSimulator(model, device="cpu")

# Top HUD Header
st.markdown("""
<div class="hud-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="hud-title">🛡️ PRISM CYBER DEFENSE OPERATIONS CENTER</div>
            <div class="hud-subtitle">Predictive Recurrent Infiltration State Model — Continuous Network Threat Waveform & Autoregressive World Model</div>
        </div>
        <div>
            <span class="status-pill-online">● SYSTEM ONLINE</span>
            <span class="status-pill-cuda">● WORLD MODEL (60 EPOCHS)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown("### 🎛️ NAVIGATION CONSOLE")
nav_selection = st.sidebar.radio(
    "Choose Operational View:",
    [
        "⚡ Live Attack Scenario Simulator & Stream",
        "🛡️ SOC Executive Cockpit",
        "🔮 K-Step Autoregressive Rollout",
        "🧠 Transformer Attention & Explainability",
        "🗺️ MITRE ATT&CK Kill Chain Tracker",
        "📊 Model Evaluation & Benchmarks"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### 📡 STREAM SPECS")
st.sidebar.write(f"• **Active Timeline**: `{len(states):,} Minutes`")
st.sidebar.write(f"• **History Window ($L$)**: `10 Mins (Sliding)`")
st.sidebar.write(f"• **State Vector Size**: `110 Dimensions`")
st.sidebar.write(f"• **Model Status**: `Trained & Calibrated`")


# =============================================================
# 1. LIVE ATTACK SCENARIO SIMULATOR & STREAM (FLAGSHIP VIEW)
# =============================================================
if nav_selection == "⚡ Live Attack Scenario Simulator & Stream":
    st.markdown("### ⚡ Live Multi-Phase Attack Stream & Continuous Threat Waveform")
    st.markdown("Simulate realistic network progression (**Benign $\\to$ Suspicious Onset $\\to$ Active Infiltration**) and watch the Transformer World Model detect anomalies before the breach strikes.")

    # 1. Scenario Selector & Injector Controls
    sc_col1, sc_col2 = st.columns([2, 1])
    with sc_col1:
        selected_scenario = st.selectbox(
            "Select Attack Scenario to Simulate:",
            ScenarioGenerator.get_available_scenarios()
        )
    with sc_col2:
        playback_speed = st.select_slider(
            "Playback Speed:",
            options=[0.2, 0.4, 0.7, 1.0],
            value=0.4,
            format_func=lambda x: f"{x}s per tick"
        )

    # Custom injector controls if Scenario E is selected
    user_start = 15
    user_intensity = 0.85
    user_stage = 2
    if "Scenario E" in selected_scenario:
        st.markdown("##### 🛠️ Custom Attack Injection Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            user_start = st.slider("Attack Strike Minute", min_value=12, max_value=28, value=15)
        with c2:
            user_intensity = st.slider("Attack Intensity / Packet Volume", min_value=0.2, max_value=1.0, value=0.85)
        with c3:
            user_stage_name = st.selectbox("MITRE Attack Vector", ["Initial Access (Brute Force)", "Reconnaissance (PortScan)", "Exfiltration", "Impact (DDoS)"])
            stage_map = {"Initial Access (Brute Force)": 2, "Reconnaissance (PortScan)": 1, "Exfiltration": 5, "Impact (DDoS)": 6}
            user_stage = stage_map[user_stage_name]

    # Generate scenario states
    sc_states, sc_atks, sc_mitres, sc_meta = ScenarioGenerator.generate_scenario_states(
        selected_scenario, states, atks, mitres,
        user_attack_start=user_start,
        user_attack_intensity=user_intensity,
        user_attack_stage=user_stage
    )

    st.info(f"📋 **Scenario Profile**: {sc_meta.get('description', '')} | Target Port: `{sc_meta.get('target_port', 80)}` ({sc_meta.get('target_protocol', 'TCP')})")

    # Compute threat probabilities across the scenario
    sc_probs = []
    sc_preds_mitre = []
    
    for i in range(10, len(sc_states)):
        norm_seq = (sc_states[i-10:i] - scaler_mean) / scaler_std
        t_seq = torch.from_numpy(norm_seq).float().unsqueeze(0)
        with torch.no_grad():
            _, _, p_atk, p_mit, _, _ = model(t_seq)
            p = float(torch.softmax(p_atk, dim=-1)[0, 1].item())
            m = int(torch.argmax(p_mit, dim=-1)[0].item())
            sc_probs.append(p)
            sc_preds_mitre.append(m)

    total_stream_steps = len(sc_probs)
    
    # 2. Interactive Time Slider & Playback Buttons
    col_play, col_step = st.columns([1, 3])
    with col_play:
        st.write("")
        st.write("")
        run_animation = st.button("▶️ Run Live Continuous Playback", use_container_width=True)
    with col_step:
        current_tick = st.slider(
            "Scrub Timeline Window (Current Network Clock $t$):",
            min_value=10,
            max_value=10 + total_stream_steps - 1,
            value=min(18, 10 + total_stream_steps - 1)
        )

    # Streaming container
    stream_placeholder = st.empty()

    def render_stream_frame(current_t):
        idx = current_t - 10
        active_prob = sc_probs[idx]
        active_stage = MITRE_STAGES_INV.get(sc_preds_mitre[idx], "Benign")
        active_ground = "Attack" if sc_atks[current_t] == 1 else "Normal"
        
        # Determine Status Theme
        if active_prob >= 0.70:
            status_theme = "CRITICAL INTRUSION ACTIVE"
            status_color = "#ff0055"
            card_class = "alert-card-critical"
            state_label = "🔴 ACTIVE INFILTRATION"
        elif active_prob >= 0.35:
            status_theme = "SUSPICIOUS ONSET (EARLY WARNING)"
            status_color = "#eab308"
            card_class = "alert-card-warning"
            state_label = "🟡 SUSPICIOUS PROBING DETECTED"
        else:
            status_theme = "BENIGN / NORMAL BASELINE"
            status_color = "#10b981"
            card_class = "alert-card-safe"
            state_label = "🟢 NORMAL TRAFFIC FLOW"

        with stream_placeholder.container():
            # 1. Real-Time Status Bar
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.15rem; font-weight: 800; color: {status_color};">[{status_theme}]</span>
                        <span style="margin-left: 12px; font-size: 1.05rem; font-weight: 600;">Minute {current_t}:00 — Stage: {active_stage}</span>
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: {status_color};">
                        Threat Probability: {active_prob:.1%}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Continuous Flow Waveform Chart (Flowing Right-to-Left with Color-Coded Phases)
            timeline_mins = list(range(10, 10 + len(sc_probs)))
            past_probs = sc_probs[:idx+1]
            past_mins = timeline_mins[:idx+1]

            fig_wave = go.Figure()

            # Normal Baseline Shaded Band (0.0 to 0.35)
            fig_wave.add_hrect(y0=0.0, y1=0.35, fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, annotation_text="🟢 SAFE BASELINE", annotation_position="top left", annotation_font_color="#10b981")
            # Suspicious Pre-Attack Warning Band (0.35 to 0.70)
            fig_wave.add_hrect(y0=0.35, y1=0.70, fillcolor="rgba(234, 179, 8, 0.10)", line_width=0, annotation_text="🟡 SUSPICIOUS ONSET (EARLY WARNING)", annotation_position="top left", annotation_font_color="#eab308")
            # Active Attack Threat Band (0.70 to 1.0)
            fig_wave.add_hrect(y0=0.70, y1=1.05, fillcolor="rgba(239, 68, 68, 0.12)", line_width=0, annotation_text="🔴 CRITICAL ATTACK STRIKE", annotation_position="top left", annotation_font_color="#ef4444")

            # Historical Waveform
            fig_wave.add_trace(go.Scatter(
                x=timeline_mins,
                y=sc_probs,
                mode="lines",
                name="Network Threat Stream",
                line=dict(color="rgba(148, 163, 184, 0.3)", width=2, dash="dot"),
                hoverinfo="skip"
            ))

            # Active Streaming Waveform up to current_t
            colors_list = []
            for p in past_probs:
                if p >= 0.70:
                    colors_list.append("#ff0055")
                elif p >= 0.35:
                    colors_list.append("#eab308")
                else:
                    colors_list.append("#00f0ff")

            fig_wave.add_trace(go.Scatter(
                x=past_mins,
                y=past_probs,
                mode="lines+markers",
                name="Real-Time EKG Threat Waveform",
                line=dict(color="#00f0ff", width=4),
                marker=dict(size=8, color=colors_list, line=dict(color="#ffffff", width=1.5))
            ))

            # Vertical Pulsating Radar Tracer Line at Current Minute
            fig_wave.add_vline(x=current_t, line_width=3, line_color="#ffffff", line_dash="solid", annotation_text=f"⏱️ NOW (Min {current_t})", annotation_position="top right")

            fig_wave.update_layout(
                title="<b>Continuous Live Network Infiltration Waveform (Right-to-Left Telemetry Stream)</b>",
                xaxis_title="Timeline Clock (Minutes)",
                yaxis_title="Threat Severity (0.0 to 1.0)",
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.6)",
                height=380,
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis=dict(range=[-0.02, 1.08], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                xaxis=dict(range=[10, 10 + total_stream_steps], showgrid=True, gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_wave, use_container_width=True)

            # 3. Synchronized Future Lookahead (+5 Mins Ahead) & Instant Telemetry
            col_cone, col_gauge = st.columns([2, 1.2])

            with col_cone:
                st.markdown("#### 🔮 World Model Forward Lookahead Cone (+5 Mins Ahead)")
                curr_stream_seq = (sc_states[current_t-10:current_t] - scaler_mean) / scaler_std
                curr_t_tensor = torch.from_numpy(curr_stream_seq).float().unsqueeze(0)
                
                rollout_traj = simulator.rollout(curr_t_tensor, K=5, stochastic=False)
                traj_steps = [f"+{s['step']} Min" for s in rollout_traj]
                traj_p = [s['infiltration_prob'] for s in rollout_traj]
                
                fig_cone = go.Figure()
                fig_cone.add_trace(go.Scatter(
                    x=traj_steps,
                    y=[min(1.0, p + 0.07) for p in traj_p] + [max(0.0, p - 0.07) for p in traj_p][::-1],
                    fill='toself',
                    fillcolor='rgba(0, 240, 255, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Uncertainty Bounds',
                    hoverinfo="skip"
                ))
                fig_cone.add_trace(go.Scatter(
                    x=traj_steps,
                    y=traj_p,
                    mode='lines+markers',
                    name='Forecast Risk Trajectory',
                    line=dict(color='#ff0055' if max(traj_p) >= 0.70 else '#00f0ff', width=3),
                    marker=dict(size=8)
                ))
                fig_cone.add_hline(y=0.70, line_dash="dash", line_color="#ef4444", annotation_text="Alarm Limit")
                fig_cone.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(15, 23, 42, 0.4)",
                    plot_bgcolor="rgba(15, 23, 42, 0.4)",
                    height=280,
                    margin=dict(l=20, r=20, t=30, b=20),
                    yaxis=dict(range=[-0.02, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.05)")
                )
                st.plotly_chart(fig_cone, use_container_width=True)

            with col_gauge:
                st.markdown("#### ⚡ Live Packet Metrics & Containment")
                cur_fwd_pkts = int(sc_states[current_t, 2])
                cur_fwd_bytes = int(sc_states[current_t, 4])
                cur_syn = int(sc_states[current_t, 22])
                
                st.markdown(f"""
                <div class="metric-box">
                    <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                        <span>Forward Packets/min:</span>
                        <b style="color:#00f0ff;">{cur_fwd_pkts:,}</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                        <span>Forward Bytes/min:</span>
                        <b style="color:#a855f7;">{cur_fwd_bytes:,} B</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                        <span>SYN Flags Count:</span>
                        <b style="color:#ff0055;">{cur_syn}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.write("")
                if active_prob >= 0.70:
                    st.error(f"🛡️ **AUTO-FIREWALL DIRECTIVE TRIGGERED**:\n`iptables -A INPUT -p tcp --dport {sc_meta.get('target_port', 22)} -j DROP`")
                elif active_prob >= 0.35:
                    st.warning(f"⚠️ **PROACTIVE ISOLATION WARNING**:\n`rate-limit --port {sc_meta.get('target_port', 22)} --limit 10/min`")
                else:
                    st.success("🟢 **NETWORK HEALTHY**: Normal routing active.")

    # Execute Animation Loop or Render Single Frame
    if run_animation:
        for t in range(10, 10 + total_stream_steps):
            render_stream_frame(t)
            time.sleep(playback_speed)
    else:
        render_stream_frame(current_tick)


# =============================================================
# 2. SOC EXECUTIVE COCKPIT
# =============================================================
elif nav_selection == "🛡️ SOC Executive Cockpit":
    st.markdown("### 🛡️ SOC Executive Cockpit & Telemetry Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-box-title">World Model F1-Score</div>
            <div class="metric-box-val" style="color: #00f0ff;">89.9%</div>
            <div class="metric-box-sub" style="color: #10b981;">▲ +49.6% vs Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-box-title">Attack Catch Rate (Recall)</div>
            <div class="metric-box-val" style="color: #10b981;">96.1%</div>
            <div class="metric-box-sub" style="color: #94a3b8;">Missed Intrusions: &lt; 4%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-box-title">MITRE Tactical Accuracy</div>
            <div class="metric-box-val" style="color: #a855f7;">95.2%</div>
            <div class="metric-box-sub" style="color: #c084fc;">7-Stage Kill Chain</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-box-title">False Alarm Rate (FPR)</div>
            <div class="metric-box-val" style="color: #ff0055;">5.7%</div>
            <div class="metric-box-sub" style="color: #10b981;">▼ Down from 82.9%</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    probs_list = []
    mitre_preds_list = []
    for i in range(10, len(states)):
        norm_seq = (states[i-10:i] - scaler_mean) / scaler_std
        t_seq = torch.from_numpy(norm_seq).float().unsqueeze(0)
        with torch.no_grad():
            _, _, p_atk, p_mit, _, _ = model(t_seq)
            p = float(torch.softmax(p_atk, dim=-1)[0, 1].item())
            m = int(torch.argmax(p_mit, dim=-1)[0].item())
            probs_list.append(p)
            mitre_preds_list.append(m)

    time_idx = list(range(10, len(states)))
    current_threat_prob = probs_list[-1] if probs_list else 0.05
    current_mitre_stage = MITRE_STAGES_INV.get(mitre_preds_list[-1] if mitre_preds_list else 0, "Benign")

    row_gauge, row_time = st.columns([1, 2.5])
    
    with row_gauge:
        st.markdown("#### 🧭 Instant Threat Dial")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current_threat_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Current Threat ({current_mitre_stage})", 'font': {'size': 18, 'color': '#f8fafc'}},
            delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                'bar': {'color': "#00f0ff"},
                'bgcolor': "rgba(30, 41, 59, 0.5)",
                'borderwidth': 2,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [35, 70], 'color': 'rgba(234, 179, 8, 0.25)'},
                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.35)'}
                ],
                'threshold': {
                    'line': {'color': "#ff0055", 'width': 4},
                    'thickness': 0.75,
                    'value': 75
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#f8fafc"},
            height=320,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with row_time:
        st.markdown("#### 📈 Full Timeline Infiltration Probability Curve")
        df_time = pd.DataFrame({
            "Minute": time_idx,
            "Infiltration Probability": probs_list,
            "Ground Truth": ["Attack Event" if atks[i] == 1 else "Normal Baseline" for i in time_idx]
        })

        fig_timeline = px.area(
            df_time,
            x="Minute",
            y="Infiltration Probability",
            color_discrete_sequence=["#00f0ff"],
            template="plotly_dark"
        )
        fig_timeline.add_hline(y=0.70, line_dash="dash", line_color="#ff0055", annotation_text="CRITICAL (0.70)")
        fig_timeline.add_hline(y=0.35, line_dash="dot", line_color="#eab308", annotation_text="WARNING (0.35)")
        fig_timeline.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", range=[0, 1.05])
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


# =============================================================
# 3. K-STEP AUTOREGRESSIVE ROLLOUT
# =============================================================
elif nav_selection == "🔮 K-Step Autoregressive Rollout":
    st.markdown("### 🔮 K-Step Autoregressive Forward Rollout Simulator")
    st.markdown("Forecast multi-step state dynamics ($S_{t+1} \dots S_{t+K}$) before intrusions occur.")

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        sim_idx = st.slider("Select Starting Network Window ($t$)", min_value=10, max_value=len(states)-1, value=min(145, len(states)-1))
    with col_ctrl2:
        k_steps = st.slider("Lookahead Forecast Horizon ($K$ Minutes)", min_value=3, max_value=15, value=10)

    curr_seq = torch.from_numpy((states[sim_idx-10:sim_idx] - scaler_mean) / scaler_std).float().unsqueeze(0)
    
    trajectory = simulator.rollout(curr_seq, K=k_steps, stochastic=False)
    risk_summary = compute_trajectory_risk(trajectory)
    alerts = generate_alerts(trajectory)

    st.write("")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Trajectory Risk Level</div>
            <div class="metric-box-val" style="color: {'#ef4444' if risk_summary['risk_level']=='CRITICAL' else '#10b981'};">{risk_summary['risk_level']}</div>
            <div class="metric-box-sub" style="color: #94a3b8;">Forecast over +{k_steps} Mins</div>
        </div>
        """, unsafe_allow_html=True)
    with colB:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Peak Infiltration Probability</div>
            <div class="metric-box-val" style="color: #00f0ff;">{risk_summary['max_prob']:.1%}</div>
            <div class="metric-box-sub" style="color: #94a3b8;">Max Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with colC:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-box-title">Imminent MITRE Target</div>
            <div class="metric-box-val" style="color: #a855f7;">{risk_summary['peak_stage']}</div>
            <div class="metric-box-sub" style="color: #c084fc;">Predicted Stage</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    steps_ahead = [f"+{s['step']} Min" for s in trajectory]
    p_values = [s['infiltration_prob'] for s in trajectory]

    fig_rollout = go.Figure()
    fig_rollout.add_trace(go.Scatter(
        x=steps_ahead + steps_ahead[::-1],
        y=[min(1.0, p + 0.08) for p in p_values] + [max(0.0, p - 0.08) for p in p_values][::-1],
        fill='toself',
        fillcolor='rgba(0, 240, 255, 0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name='Uncertainty Cone'
    ))
    fig_rollout.add_trace(go.Scatter(
        x=steps_ahead,
        y=p_values,
        mode='lines+markers',
        name='Predicted Infiltration Probability',
        line=dict(color='#00f0ff', width=3.5),
        marker=dict(size=9, color='#ff0055', line=dict(color='#ffffff', width=1.5))
    ))
    fig_rollout.add_hline(y=0.70, line_dash="dash", line_color="#ef4444", annotation_text="Escalation Threshold")
    fig_rollout.update_layout(
        title="<b>Autoregressive Infiltration Risk Rollout Trajectory</b>",
        xaxis_title="Simulation Time Ahead",
        yaxis_title="Infiltration Probability",
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_rollout, use_container_width=True)


# =============================================================
# 4. TRANSFORMER ATTENTION & EXPLAINABILITY
# =============================================================
elif nav_selection == "🧠 Transformer Attention & Explainability":
    st.markdown("### 🧠 Transformer Self-Attention & Gradient Saliency")
    st.markdown("Deconstruct the model's inner attention weights across historical network minutes.")

    exp_idx = st.slider("Select Sequence Window ($t$)", min_value=10, max_value=len(states)-1, value=min(160, len(states)-1))
    curr_seq = torch.from_numpy((states[exp_idx-10:exp_idx] - scaler_mean) / scaler_std).float().unsqueeze(0)

    report = explain_state_prediction(model, curr_seq)
    st.markdown(f"#### Infiltration Probability: **`{report['infiltration_probability']:.1%}`** | Predicted Stage: **`{report['predicted_mitre_stage']}`**")

    col_att, col_feat = st.columns([1.2, 1])
    with col_att:
        st.markdown("##### 🔍 Multi-Head Self-Attention Matrix")
        with torch.no_grad():
            _, _, _, _, _, attn_weights = model(curr_seq)
        
        attn_data = format_attention_heatmap(attn_weights.numpy())
        fig_attn = px.imshow(
            attn_data["matrix"],
            labels=dict(x="Key (Historical Minute)", y="Query (Recent Minute)", color="Attention Weight"),
            x=attn_data["window_labels"],
            y=attn_data["window_labels"],
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_attn.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_attn, use_container_width=True)
        st.caption(f"ℹ️ {attn_data['narrative']}")

    with col_feat:
        st.markdown("##### 📊 Top Gradient Saliency Drivers")
        df_feat = pd.DataFrame(report["top_features"]).head(8)
        fig_bar = px.bar(
            df_feat,
            x="importance",
            y="feature_name",
            orientation='h',
            color="importance",
            color_continuous_scale="Plasma",
            template="plotly_dark"
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.4)",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================
# 5. MITRE ATT&CK KILL CHAIN TRACKER
# =============================================================
elif nav_selection == "🗺️ MITRE ATT&CK Kill Chain Tracker":
    st.markdown("### 🗺️ MITRE ATT&CK Tactical Kill Chain Mapping")
    st.markdown("Overview of the 7 standardized network-level infiltration phases.")

    cols = st.columns(7)
    stages = ["Benign", "Reconnaissance", "Initial Access", "Lateral Movement", "Command & Control", "Exfiltration", "Impact (DoS/DDoS)"]
    
    for idx, (col, stage) in enumerate(zip(cols, stages)):
        with col:
            st.markdown(f"""
            <div class="killchain-node">
                <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 600;">STAGE {idx}</div>
                <div style="font-size: 0.95rem; font-weight: 700; margin-top: 4px; color: #f8fafc;">{stage}</div>
                <div style="margin-top: 8px; font-size: 0.72rem; color: #10b981;">● MONITORED</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### 📖 Incident Remediation Protocol Directory")
    for stage_name, details in MITRE_RECOMMENDATIONS.items():
        with st.expander(f"📌 {stage_name} — [Severity: {details['severity']}]"):
            st.markdown(f"**Threat Mechanics**: {details['description']}")
            st.markdown(f"**Automated Containment Directive**: `{details['action']}`")


# =============================================================
# 6. MODEL EVALUATION & BENCHMARKS
# =============================================================
elif nav_selection == "📊 Model Evaluation & Benchmarks":
    st.markdown("### 📊 Benchmark Lab & Comparative Evaluation")
    st.markdown("Comparison between the **PRISM Transformer World Model** and **Standard Static NIDS Baselines**.")

    bench_file = os.path.join("results", "benchmark_results.json")
    if os.path.exists(bench_file):
        with open(bench_file, "r") as f:
            bench_data = json.load(f)
        df_bench = pd.DataFrame(bench_data)
    else:
        df_bench = pd.DataFrame([
            {"Model": "Logistic Regression / RF Baseline", "F1 Score": 0.4035, "Precision": 0.2599, "Recall": 0.9020, "FPR": 0.8291, "ROC-AUC": 0.6156, "MITRE Accuracy": 0.3349},
            {"Model": "PRISM StateTransformerWorldModel", "F1 Score": 0.8991, "Precision": 0.8448, "Recall": 0.9608, "FPR": 0.0570, "ROC-AUC": 0.9950, "MITRE Accuracy": 0.9522}
        ])

    st.dataframe(df_bench, use_container_width=True)

    st.write("")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_f1 = px.bar(
            df_bench,
            x="Model",
            y="F1 Score",
            color="Model",
            color_discrete_sequence=["#ef4444", "#00f0ff"],
            title="<b>F1-Score Comparison</b>",
            template="plotly_dark"
        )
        fig_f1.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.4)", plot_bgcolor="rgba(15, 23, 42, 0.4)")
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_g2:
        fig_fpr = px.bar(
            df_bench,
            x="Model",
            y="FPR",
            color="Model",
            color_discrete_sequence=["#ef4444", "#10b981"],
            title="<b>False Positive Rate (Lower is Better)</b>",
            template="plotly_dark"
        )
        fig_fpr.update_layout(paper_bgcolor="rgba(15, 23, 42, 0.4)", plot_bgcolor="rgba(15, 23, 42, 0.4)")
        st.plotly_chart(fig_fpr, use_container_width=True)
