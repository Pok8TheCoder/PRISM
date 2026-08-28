"""
PRISM - Streamlit Demo Application
Predictive Recurrent Infiltration State Model
Fully offline, no cloud API dependencies.

Run with: streamlit run app/streamlit_app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import streamlit as st

# -----------------------------------------------------------------------
# Page config (must be FIRST Streamlit call)
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="PRISM — Predictive Cyber Defence",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.utils.constants import (
    MITRE_STAGES_INV, MITRE_STAGE_COLORS, ALERT_THRESHOLDS
)
from src.prediction.attack_mapper import MITRE_TACTIC_META

# -----------------------------------------------------------------------
# CSS styling
# -----------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #e74c3c, #3498db);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .alert-critical { background: #c0392b22; border-left: 4px solid #c0392b;
                      padding: 10px; border-radius: 4px; margin: 5px 0; }
    .alert-high     { background: #e74c3c22; border-left: 4px solid #e74c3c;
                      padding: 10px; border-radius: 4px; margin: 5px 0; }
    .alert-medium   { background: #f39c1222; border-left: 4px solid #f39c12;
                      padding: 10px; border-radius: 4px; margin: 5px 0; }
    .alert-low      { background: #3498db22; border-left: 4px solid #3498db;
                      padding: 10px; border-radius: 4px; margin: 5px 0; }
    .alert-none     { background: #2ecc7122; border-left: 4px solid #2ecc71;
                      padding: 10px; border-radius: 4px; margin: 5px 0; }
    .metric-box     { background: #1a1a2e; border-radius: 8px;
                      padding: 15px; text-align: center; }
    .feature-row-pos { color: #e74c3c; font-weight: 600; }
    .feature-row-neg { color: #3498db; font-weight: 600; }
    .stage-badge    { display: inline-block; padding: 4px 12px;
                      border-radius: 12px; font-size: 0.85rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("## PRISM Settings")
        st.markdown("---")

        model_path = st.text_input(
            "Model checkpoint", value="weights/world_model_best.pt"
        )
        k_steps = st.slider("Forecast horizon (K steps)", 5, 30, 10)
        n_rollouts = st.slider("Ensemble rollouts", 1, 20, 10)
        deterministic = st.checkbox("Deterministic rollout", value=False)

        st.markdown("---")
        st.markdown("### Alert Thresholds")
        thresh_critical = st.slider("Critical", 0.5, 1.0, 0.85)
        thresh_high = st.slider("High", 0.3, 0.9, 0.65)
        thresh_medium = st.slider("Medium", 0.1, 0.7, 0.40)

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "**PRISM** is a World Model AI system for predictive cyber defence. "
            "It learns network state transition dynamics and forecasts attack "
            "progression before compromise is completed."
        )

    return {
        "model_path": model_path,
        "k_steps": k_steps,
        "n_rollouts": n_rollouts,
        "deterministic": deterministic,
        "thresholds": {
            "critical": thresh_critical,
            "high": thresh_high,
            "medium": thresh_medium,
            "low": 0.20,
        },
    }


# -----------------------------------------------------------------------
# Model loading (cached)
# -----------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str, d_state: int = 50):
    """Load or create a demo world model."""
    import torch
    from src.utils.config import load_config
    from src.models.world_model import build_world_model

    cfg = load_config("configs/model.yaml")
    cfg.model.d_state = d_state

    model = build_world_model(cfg.model)

    if os.path.exists(model_path):
        from src.models.world_model import load_checkpoint
        load_checkpoint(model, model_path, device="cpu")
        st.sidebar.success(f"Model loaded: {model_path}")
    else:
        st.sidebar.warning("No checkpoint found. Using untrained model for demo.")

    model.eval()
    return model


# -----------------------------------------------------------------------
# Data loading helpers
# -----------------------------------------------------------------------
def load_uploaded_file(uploaded_file, cfg_settings: dict):
    """Process uploaded CSV or PCAP file -> state sequence."""
    import tempfile

    filename = uploaded_file.name
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if suffix == ".csv":
            return _load_csv(tmp_path)
        elif suffix in (".pcap", ".pcapng"):
            return _load_pcap(tmp_path)
        else:
            st.error(f"Unsupported file type: {suffix}")
            return None, [], None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _load_csv(csv_path: str):
    from src.data.flow_extractor import FlowExtractor
    from src.data.feature_merger import FeatureMerger
    from src.data.state_builder import StateBuilder

    extractor = FlowExtractor(dataset_type="cicids2018")
    df = extractor.extract(file_path=csv_path, fit_scaler=True)
    merger = FeatureMerger()
    merged = merger.merge(df)
    builder = StateBuilder(window_size_seconds=30)
    result = builder.build_states(merged)

    states = result["states"]
    feature_names = result.get("feature_names", [])
    labels = result.get("labels_binary", np.zeros(len(states)))
    return states, feature_names, labels


def _load_pcap(pcap_path: str):
    from src.data.packet_extractor import PacketExtractor
    from src.data.feature_merger import FeatureMerger
    from src.data.state_builder import StateBuilder

    pe = PacketExtractor(window_size_seconds=30)
    pkt_df = pe.extract_from_pcap(pcap_path)
    merger = FeatureMerger()
    merged = merger.merge(None, pkt_df)
    builder = StateBuilder()
    result = builder.build_states(merged)

    states = result["states"]
    feature_names = result.get("feature_names", [])
    labels = result.get("labels_binary", np.zeros(len(states)))
    return states, feature_names, labels


def load_demo_data():
    """Load synthetic demo data."""
    demo_path = "data/raw/demo_states.npz"
    if not os.path.exists(demo_path):
        with st.spinner("Generating demo data..."):
            from scripts.download_data import generate_demo_data
            generate_demo_data()

    data = np.load(demo_path)
    return (
        data["states"],
        [f"feature_{i}" for i in range(data["states"].shape[1])],
        data["labels_binary"],
    )


# -----------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------
def run_simulation(model, state_seq, feature_names, settings):
    """Run K-step simulation and return RolloutResult."""
    import torch
    from src.prediction.simulator import KStepSimulator

    simulator = KStepSimulator(
        model=model,
        device="cpu",
        k_steps=settings["k_steps"],
        num_rollouts=settings["n_rollouts"],
        deterministic=settings["deterministic"],
        thresholds=settings["thresholds"],
    )
    return simulator.simulate(state_seq, feature_names=feature_names or None)


# -----------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------
def render_alert_banner(result):
    """Show top-level alert banner."""
    level = result.overall_alert
    prob = result.peak_infiltration_prob

    alert_msgs = {
        "critical": f"CRITICAL THREAT DETECTED — Peak P(attack)={prob:.1%} | {result.trajectory_summary}",
        "high": f"HIGH THREAT — Peak P(attack)={prob:.1%} | {result.trajectory_summary}",
        "medium": f"MEDIUM THREAT — Peak P(attack)={prob:.1%} | {result.trajectory_summary}",
        "low": f"LOW ANOMALY — Peak P(attack)={prob:.1%} | {result.trajectory_summary}",
        "none": f"NETWORK APPEARS BENIGN — Max P(attack)={prob:.1%}",
    }

    st.markdown(
        f'<div class="alert-{level}">'
        f'<strong>{alert_msgs.get(level, "")}</strong>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_kpi_row(result):
    """Key metrics row."""
    cols = st.columns(5)
    with cols[0]:
        st.metric("Peak P(attack)", f"{result.peak_infiltration_prob:.1%}")
    with cols[1]:
        st.metric("Alert Level", result.overall_alert.upper())
    with cols[2]:
        peak_step = result.steps[result.peak_step - 1] if result.steps else None
        st.metric("Peak Stage", peak_step.mitre_stage if peak_step else "—")
    with cols[3]:
        st.metric("Early Warning", f"{result.early_warning_steps} steps")
    with cols[4]:
        st.metric("Escalating", "YES" if result.is_escalating else "No")


def render_timeline_chart(result):
    """Infiltration probability timeline using Plotly."""
    import plotly.graph_objects as go

    steps = [s.step for s in result.steps]
    probs = [s.infiltration_prob for s in result.steps]
    stages = [s.mitre_stage for s in result.steps]
    colors = [MITRE_STAGE_COLORS.get(s, "#888") for s in stages]

    fig = go.Figure()

    # Ensemble band (if available)
    if result.ensemble_mean is not None and result.ensemble_lower is not None:
        fig.add_trace(go.Scatter(
            x=steps + steps[::-1],
            y=result.ensemble_upper.tolist() + result.ensemble_lower.tolist()[::-1],
            fill="toself",
            fillcolor="rgba(52,152,219,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% CI",
        ))

    # Main probability line
    fig.add_trace(go.Scatter(
        x=steps, y=probs,
        mode="lines+markers",
        line=dict(color="#e74c3c", width=2.5),
        marker=dict(color=colors, size=10, line=dict(color="white", width=1)),
        name="P(infiltration)",
        hovertemplate="Step %{x}: P=%{y:.3f}<br>Stage: %{text}",
        text=stages,
    ))

    # Threshold lines
    for name, val, colour in [
        ("Critical", result.steps[0].mitre_probs[0] if False else 0.85, "#c0392b"),
        ("High", 0.65, "#e74c3c"),
        ("Medium", 0.40, "#f39c12"),
    ]:
        fig.add_hline(y=val, line_dash="dash", line_color=colour,
                      annotation_text=name, annotation_position="right")

    fig.update_layout(
        title="Infiltration Probability Forecast Timeline",
        xaxis_title="Forecast Step",
        yaxis_title="P(Infiltration)",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.15),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_mitre_stage_view(result):
    """MITRE ATT&CK stage progression chart."""
    import plotly.graph_objects as go

    stages = [s.mitre_stage for s in result.steps]
    stage_order = list(MITRE_STAGES_INV.values())
    stage_ids = [stage_order.index(s) if s in stage_order else 0 for s in stages]
    cols = [MITRE_STAGE_COLORS.get(s, "#888") for s in stages]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[s.step for s in result.steps],
        y=stage_ids,
        mode="lines+markers",
        marker=dict(color=cols, size=12),
        line=dict(color="#95a5a6", width=1.5, dash="dot"),
        hovertemplate="Step %{x}: %{text}",
        text=stages,
    ))

    fig.update_layout(
        title="Predicted MITRE ATT&CK Stage Progression",
        xaxis_title="Forecast Step",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(stage_order))),
            ticktext=stage_order,
            gridcolor="#2c3e50",
        ),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_attribution(step):
    """Feature attribution table for a single step."""
    if not step.top_features:
        st.info("No feature attribution available for this step.")
        return

    import plotly.graph_objects as go

    names = [f["feature"] for f in step.top_features]
    vals = [f["attribution"] for f in step.top_features]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Feature Attribution — Step {step.step} (Gradient × Input)",
        xaxis_title="Attribution (impact on P(attack))",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_attack_details(step):
    """Show MITRE tactic metadata for the predicted stage."""
    meta = MITRE_TACTIC_META.get(step.mitre_stage, MITRE_TACTIC_META["Benign"])
    color = meta.get("color", "#888")

    st.markdown(
        f'<span class="stage-badge" style="background:{color}33; color:{color}; '
        f'border: 1px solid {color};">'
        f'MITRE {meta["id"]} — {meta["tactic"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'**Description:** {meta["description"]}')

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Indicators:**")
        for ind in meta.get("indicators", []):
            st.markdown(f"- {ind}")
    with col2:
        st.markdown("**Recommended Actions:**")
        for action in meta.get("recommended_actions", []):
            st.markdown(f"- {action}")


# -----------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------
def main():
    # Header
    st.markdown(
        '<h1 class="main-header">PRISM — Predictive Recurrent Infiltration State Model</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "*World Model AI for Proactive Cyber Defence | "
        "Predicts attack progression before compromise is complete*"
    )
    st.markdown("---")

    # Sidebar
    settings = render_sidebar()

    # ---------------------------------------------------------------
    # Input section
    # ---------------------------------------------------------------
    st.subheader("Input Data")
    input_mode = st.radio(
        "Select input source",
        ["Demo Data (synthetic)", "Upload CSV", "Upload PCAP", "Load States NPZ"],
        horizontal=True,
    )

    states = feature_names = labels = None

    if input_mode == "Demo Data (synthetic)":
        if st.button("Load Demo Data", type="primary"):
            with st.spinner("Loading demo scenario..."):
                states, feature_names, labels = load_demo_data()
            st.session_state["states"] = states
            st.session_state["feature_names"] = feature_names
            st.session_state["labels"] = labels
            st.success(f"Demo loaded: {len(states)} time windows, {states.shape[1]} features")

    elif input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CIC-IDS-2018 CSV", type=["csv"])
        if uploaded:
            with st.spinner("Processing CSV..."):
                states, feature_names, labels = load_uploaded_file(uploaded, settings)
            if states is not None:
                st.session_state["states"] = states
                st.session_state["feature_names"] = feature_names
                st.success(f"CSV processed: {len(states)} windows, {states.shape[1]} features")

    elif input_mode == "Upload PCAP":
        uploaded = st.file_uploader("Upload PCAP file", type=["pcap", "pcapng"])
        if uploaded:
            with st.spinner("Parsing PCAP..."):
                states, feature_names, labels = load_uploaded_file(uploaded, settings)
            if states is not None:
                st.session_state["states"] = states
                st.session_state["feature_names"] = feature_names
                st.success(f"PCAP parsed: {len(states)} windows, {states.shape[1]} features")

    elif input_mode == "Load States NPZ":
        npz_path = st.text_input("Path to states.npz", value="data/splits/test.npz")
        if st.button("Load NPZ"):
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                states = data["states"]
                feature_names = [f"feature_{i}" for i in range(states.shape[1])]
                labels = data.get("labels_binary", np.zeros(len(states)))
                st.session_state["states"] = states
                st.session_state["feature_names"] = feature_names
                st.session_state["labels"] = labels
                st.success(f"Loaded: {len(states)} windows")
            else:
                st.error(f"File not found: {npz_path}")

    # Pull from session state
    if "states" in st.session_state:
        states = st.session_state["states"]
        feature_names = st.session_state.get("feature_names", [])
        labels = st.session_state.get("labels", None)

    if states is None:
        st.info("Select an input source above to begin analysis.")
        st.stop()

    # ---------------------------------------------------------------
    # Window selector
    # ---------------------------------------------------------------
    st.markdown("---")
    st.subheader("Simulation Window")
    lookback = 20
    max_window = len(states) - 1

    if max_window < lookback:
        st.warning(f"Need at least {lookback} windows. Data has only {len(states)}.")
        st.stop()

    col_slider, col_info = st.columns([3, 1])
    with col_slider:
        start_idx = st.slider(
            "Start window (lookback context ends here)",
            min_value=lookback,
            max_value=max_window,
            value=min(lookback + 5, max_window),
        )
    with col_info:
        if labels is not None and start_idx < len(labels):
            true_label = "Attack" if labels[start_idx] else "Benign"
            from src.utils.constants import MITRE_STAGES_INV
            true_stage = MITRE_STAGES_INV.get(
                int(st.session_state.get("labels_mitre", np.zeros(len(states)))[start_idx]) if "labels_mitre" in st.session_state else 0,
                "Unknown",
            )
            st.metric("Ground Truth", true_label)

    state_seq = states[start_idx - lookback : start_idx]  # (L, D)

    # ---------------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------------
    st.markdown("---")
    if st.button("Run PRISM Forecast", type="primary", use_container_width=True):
        d_state = states.shape[1]
        model = load_model(settings["model_path"], d_state=d_state)

        with st.spinner(f"Running {settings['k_steps']}-step forward simulation..."):
            try:
                result = run_simulation(model, state_seq, feature_names, settings)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)
                st.stop()

    if "result" not in st.session_state:
        st.info("Click 'Run PRISM Forecast' to begin simulation.")
        st.stop()

    result = st.session_state["result"]

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------
    st.markdown("---")
    render_alert_banner(result)
    st.markdown("")
    render_kpi_row(result)

    # Timeline
    st.markdown("---")
    st.subheader("Infiltration Probability Timeline")
    render_timeline_chart(result)

    # MITRE stages
    st.subheader("Attack Stage Progression")
    render_mitre_stage_view(result)

    # Drill-down
    st.markdown("---")
    st.subheader("Step Detail")
    step_select = st.selectbox(
        "Select forecast step for detail view",
        options=list(range(1, len(result.steps) + 1)),
        format_func=lambda x: (
            f"Step {x} — P={result.steps[x-1].infiltration_prob:.3f} "
            f"| {result.steps[x-1].mitre_stage} "
            f"[{result.steps[x-1].alert_level.upper()}]"
        ),
    )
    step = result.steps[step_select - 1]

    col_feat, col_attack = st.columns([1, 1])
    with col_feat:
        st.markdown("##### Feature Attribution")
        render_feature_attribution(step)
    with col_attack:
        st.markdown("##### MITRE ATT&CK Detail")
        render_attack_details(step)

    # Step table
    st.markdown("---")
    st.subheader("Full Forecast Table")
    table_data = [
        {
            "Step": s.step,
            "P(attack)": f"{s.infiltration_prob:.4f}",
            "Alert": s.alert_level.upper(),
            "MITRE Stage": s.mitre_stage,
            "Confidence": f"{s.confidence:.3f}",
            "Top Feature": s.top_features[0]["feature"] if s.top_features else "—",
        }
        for s in result.steps
    ]
    import pandas as pd
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # Download
    st.download_button(
        label="Download Forecast JSON",
        data=__import__("json").dumps(
            {
                "overall_alert": result.overall_alert,
                "peak_prob": result.peak_infiltration_prob,
                "is_escalating": result.is_escalating,
                "steps": [
                    {
                        "step": s.step,
                        "prob": s.infiltration_prob,
                        "stage": s.mitre_stage,
                        "alert": s.alert_level,
                    }
                    for s in result.steps
                ],
            },
            indent=2,
        ),
        file_name="prism_forecast.json",
        mime="application/json",
    )

    st.markdown("---")
    st.caption(
        "PRISM — Predictive Recurrent Infiltration State Model | "
        "World Model AI for Cyber Defence | Running fully offline"
    )


if __name__ == "__main__":
    main()
