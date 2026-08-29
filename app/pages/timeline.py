"""
PRISM Infiltration Timeline & Simulation Studio (Timeline Page)
Performs K-step forward simulation, Monte Carlo uncertainty bounds,
and alert escalation tracking.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from src.models.world_model import build_world_model, load_checkpoint
from src.prediction.simulator import KStepSimulator
from src.utils.config import load_config
from src.utils.constants import MITRE_STAGE_COLORS

st.set_page_config(
    page_title="PRISM — Forecast Timeline",
    page_icon="📈",
    layout="wide",
)

css_path = ROOT_DIR / "app" / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="prism-title">📈 Infiltration Probability Forecast Timeline</h1>', unsafe_allow_html=True)
st.markdown('<p class="prism-subtitle">Autoregressively roll out $K$-step future states to forecast threat trajectory before compromise completes.</p>', unsafe_allow_html=True)

# Ensure telemetry is available
if "states" not in st.session_state:
    # Auto-load test set if available
    default_test = ROOT_DIR / "data" / "splits" / "test.npz"
    if default_test.exists():
        data = np.load(default_test)
        st.session_state["states"] = data["states"]
        st.session_state["labels_binary"] = data["labels_binary"]
        st.session_state["labels_mitre"] = data["labels_mitre"]
        st.session_state["feature_names"] = [f"feat_{i}" for i in range(data["states"].shape[1])]
    else:
        st.warning("Please upload or load telemetry in the Ingestion page first.")
        st.stop()

states = st.session_state["states"]
feature_names = st.session_state.get("feature_names", [f"feat_{i}" for i in range(states.shape[1])])
labels_bin = st.session_state.get("labels_binary", None)
lookback = 20

if len(states) <= lookback:
    st.error(f"Need at least {lookback + 1} states; current telemetry has {len(states)}.")
    st.stop()

# Sidebar simulation configuration
with st.sidebar:
    st.markdown("### Simulation Parameters")
    k_steps = st.slider("Forecast Horizon ($K$ steps)", min_value=5, max_value=30, value=12)
    n_rollouts = st.slider("Monte Carlo Ensemble Rollouts", min_value=1, max_value=20, value=10)
    deterministic = st.checkbox("Deterministic Mean Rollout", value=False)
    
    st.markdown("---")
    st.markdown("### Model Selection")
    ckpt_path = st.text_input("Checkpoint Path", value="weights/world_model_best.pt")

col_time, col_btn = st.columns([4, 1])
with col_time:
    window_idx = st.slider(
        "Current Time Window ($t$)",
        min_value=lookback,
        max_value=len(states) - 1,
        value=min(lookback + 20, len(states) - 1),
    )
with col_btn:
    st.write("")
    st.write("")
    run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

# Context sequence: [t-L, ..., t]
state_context = states[window_idx - lookback : window_idx]

@st.cache_resource
def get_cached_model(path_str: str, d_state: int):
    cfg = load_config(ROOT_DIR / "configs" / "model.yaml")
    cfg.model.d_state = d_state
    model = build_world_model(cfg.model)
    full_path = ROOT_DIR / path_str
    if full_path.exists():
        load_checkpoint(model, str(full_path), device="cpu")
    model.eval()
    return model

if run_sim or "rollout_result" not in st.session_state:
    with st.spinner(f"Simulating {k_steps} future steps with {n_rollouts} rollouts..."):
        try:
            model = get_cached_model(ckpt_path, d_state=states.shape[1])
            simulator = KStepSimulator(
                model=model,
                device="cpu",
                k_steps=k_steps,
                num_rollouts=n_rollouts,
                deterministic=deterministic,
            )
            res = simulator.simulate(state_context, feature_names=feature_names)
            st.session_state["rollout_result"] = res
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.stop()

result = st.session_state["rollout_result"]

# Alert & KPI summary
alert_class = f"alert-pill alert-pill-{result.overall_alert}"
st.markdown(
    f'<div style="margin-bottom: 1.5rem; display: flex; align-items: center; gap: 1rem;">'
    f'<span class="{alert_class}">OVERALL STATUS: {result.overall_alert.upper()}</span>'
    f'<span>Peak Threat Probability: <strong>{result.peak_infiltration_prob:.1%}</strong> at Step +{result.peak_step}</span>'
    f'<span>Escalation: <strong>{"DETECTED" if result.is_escalating else "STABLE"}</strong></span>'
    f'</div>',
    unsafe_allow_html=True,
)

# Plotly Timeline Chart
steps = [s.step for s in result.steps]
probs = [s.infiltration_prob for s in result.steps]
stages = [s.mitre_stage for s in result.steps]
marker_colors = [MITRE_STAGE_COLORS.get(s, "#3b82f6") for s in stages]

fig = go.Figure()

# Shaded uncertainty band if available
if result.ensemble_upper is not None and result.ensemble_lower is not None:
    fig.add_trace(go.Scatter(
        x=steps + steps[::-1],
        y=result.ensemble_upper.tolist() + result.ensemble_lower.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% Confidence Interval",
        hoverinfo="skip",
    ))

# Mean trajectory line
fig.add_trace(go.Scatter(
    x=steps,
    y=probs,
    mode="lines+markers",
    line=dict(color="#ef4444", width=3),
    marker=dict(size=10, color=marker_colors, line=dict(color="#ffffff", width=1.5)),
    name="P(Infiltration)",
    text=stages,
    hovertemplate="<b>Step +%{x}</b><br>P(Infiltration): %{y:.3f}<br>MITRE Stage: %{text}<extra></extra>",
))

# Threshold reference lines
fig.add_hline(y=0.85, line_dash="dash", line_color="#ef4444", annotation_text="Critical (0.85)")
fig.add_hline(y=0.65, line_dash="dash", line_color="#f97316", annotation_text="High (0.65)")
fig.add_hline(y=0.40, line_dash="dash", line_color="#eab308", annotation_text="Medium (0.40)")

fig.update_layout(
    title="Forward Rollout Trajectory ($t+1$ to $t+K$)",
    xaxis_title="Forecast Window Offset (Steps into future)",
    yaxis_title="Infiltration Probability",
    yaxis=dict(range=[0, 1.05]),
    template="plotly_dark",
    plot_bgcolor="#0b0f19",
    paper_bgcolor="#0b0f19",
    height=450,
    margin=dict(l=40, r=40, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# Step-by-Step Table
st.subheader("Forecast Steps Breakdown")
step_data = []
for s in result.steps:
    step_data.append({
        "Step": f"+{s.step} ({s.step * 30}s)",
        "P(Infiltration)": f"{s.infiltration_prob:.3f}",
        "Alert Level": s.alert_level.upper(),
        "Predicted MITRE Stage": s.mitre_stage,
        "Confidence": f"{s.confidence:.2f}",
        "Top Driving Feature": s.top_features[0]["feature"] if s.top_features else "N/A",
    })

st.dataframe(pd.DataFrame(step_data), use_container_width=True, hide_index=True)
