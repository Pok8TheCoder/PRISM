"""
PRISM Explainability & Incident Investigation Studio (Explainability Page)
Extracts Transformer temporal attention maps, SHAP / Integrated Gradients
feature attribution, and formats SOC incident response reports.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch

from src.models.world_model import build_world_model, load_checkpoint
from src.explainability.attention_viz import AttentionVisualiser
from src.explainability.feature_importance import FeatureImportanceAnalyser
from src.utils.config import load_config

st.set_page_config(
    page_title="PRISM — Explainability & Attribution",
    page_icon="🔍",
    layout="wide",
)

css_path = ROOT_DIR / "app" / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="prism-title">🔍 Explainable AI (XAI) & Attribution Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="prism-subtitle">Unpack the model\'s internal causal mechanisms: temporal attention heatmaps and feature attribution.</p>', unsafe_allow_html=True)

if "states" not in st.session_state:
    default_test = ROOT_DIR / "data" / "splits" / "test.npz"
    if default_test.exists():
        data = np.load(default_test)
        st.session_state["states"] = data["states"]
        st.session_state["feature_names"] = [f"feat_{i}" for i in range(data["states"].shape[1])]
    else:
        st.warning("Please upload or load telemetry in the Ingestion page first.")
        st.stop()

states = st.session_state["states"]
feature_names = st.session_state.get("feature_names", [f"feat_{i}" for i in range(states.shape[1])])
lookback = 20

col_sel, col_method = st.columns([3, 1])
with col_sel:
    step_idx = st.slider(
        "Select Time Window to Explain ($t$)",
        min_value=lookback,
        max_value=len(states) - 1,
        value=min(lookback + 10, len(states) - 1),
    )
with col_method:
    xai_method = st.selectbox("Attribution Method", ["Integrated Gradients", "Gradient × Input", "Temporal Attention"])

seq_to_explain = states[step_idx - lookback : step_idx]

@st.cache_resource
def get_explainer_model(d_state: int):
    cfg = load_config(ROOT_DIR / "configs" / "model.yaml")
    cfg.model.d_state = d_state
    model = build_world_model(cfg.model)
    ckpt = ROOT_DIR / "weights" / "world_model_best.pt"
    if ckpt.exists():
        load_checkpoint(model, str(ckpt), device="cpu")
    model.eval()
    return model

model = get_explainer_model(states.shape[1])

tab1, tab2 = st.tabs(["📊 Visual Explanations", "📋 Structured SOC Incident Report"])

with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Temporal Attention Heatmap")
        st.caption("Which historical time windows ($t-L$ to $t$) the causal multi-head attention heads focused on:")
        
        tensor_in = torch.tensor(seq_to_explain, dtype=torch.float32).unsqueeze(0)
        viz = AttentionVisualiser(feature_names=feature_names)
        attn_res = viz.extract_weights(model, tensor_in)
        attn_matrix = attn_res.get("last_layer_attn")

        if attn_matrix is not None:
            time_labels = [f"t-{lookback - 1 - i}" for i in range(lookback)]
            fig_attn = go.Figure(data=go.Heatmap(
                z=attn_matrix,
                x=time_labels,
                y=time_labels,
                colorscale="Viridis",
                hoverongaps=False,
            ))
            fig_attn.update_layout(
                title="Causal Self-Attention Weights Matrix (Last Layer)",
                xaxis_title="Key Time Step",
                yaxis_title="Query Time Step",
                template="plotly_dark",
                plot_bgcolor="#0b0f19",
                paper_bgcolor="#0b0f19",
                height=400,
            )
            st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.info("Attention weights extraction not available for this model architecture.")

    with col_right:
        st.markdown("### Top Contributing Features")
        st.caption(f"Calculated via {xai_method}:")
        
        analyser = FeatureImportanceAnalyser(model=model, feature_names=feature_names, device="cpu")
        if xai_method == "Gradient × Input":
            attr_res = analyser.gradient_x_input(seq_to_explain)
        else:
            attr_res = analyser.integrated_gradients(seq_to_explain, n_steps=25)

        top_feats = attr_res.get("top_features", [])[:10]
        if top_feats:
            f_names = [f["feature"] for f in top_feats][::-1]
            f_attrs = [f["attribution"] for f in top_feats][::-1]
            bar_colors = ["#ef4444" if a > 0 else "#3b82f6" for a in f_attrs]

            fig_bar = go.Figure(go.Bar(
                x=f_attrs,
                y=f_names,
                orientation="h",
                marker_color=bar_colors,
                hovertemplate="<b>%{y}</b><br>Attribution: %{x:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                title="Feature Impact on Infiltration Score (Red = Threat Driver)",
                xaxis_title="Attribution Magnitude",
                template="plotly_dark",
                plot_bgcolor="#0b0f19",
                paper_bgcolor="#0b0f19",
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No feature attributions returned.")

with tab2:
    st.markdown("### Structured Security Interpretability Report (Phase 4.3)")
    
    # Generate structured JSON report
    report_data = {
        "timestamp_window_id": int(step_idx),
        "lookback_horizon_seconds": lookback * 30,
        "xai_attribution_method": xai_method,
        "driving_features": [
            {
                "feature": f.get("feature", "unknown"),
                "attribution_weight": float(f.get("attribution", 0.0)),
                "observed_value": float(f.get("value", 0.0)),
                "security_context": (
                    "High port entropy indicating port scan or sweep" if "entropy" in f.get("feature", "").lower()
                    else "Spike in SYN packet flags without completion" if "syn" in f.get("feature", "").lower()
                    else "Abnormal inter-arrival time standard deviation" if "iat" in f.get("feature", "").lower()
                    else "Elevated traffic volume characteristic of lateral movement / exfil"
                )
            }
            for f in top_feats[:5]
        ],
        "soc_recommended_action": (
            "Review firewall ingress logs for high-rate SYN packets; isolate suspected source IP subnet if lateral movement continues."
        )
    }

    st.json(report_data)
    st.download_button(
        label="Download Incident Report JSON",
        data=json.dumps(report_data, indent=2),
        file_name=f"prism_xai_report_window_{step_idx}.json",
        mime="application/json",
    )
