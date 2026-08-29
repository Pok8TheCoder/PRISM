"""PRISM SOC dashboard — live adversarial lab control and visualization."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.adversarial.lab_config import DETECTION_THRESHOLD
from src.adversarial.training_loop import load_model_and_scaler
from src.model.attack_catalog import get_bot_class_ids, get_evasion_chains, get_mitre_map, load_catalog
from src.model.world_model_multiclass import CLASS_NAMES, FEATURE_COLS
from src.pipeline.features import FLOW_FEATURE_COLS, NUM_FEATURES, PACKET_FEATURE_COLS
from src.predict.engine import analyze_file
from src.ui.lab_controller import (
    K_ROLLOUT,
    MITRE_TACTICS,
    aggregate_tactic_probs,
    build_forensic_report,
    compute_feature_attribution,
    forecast_attack_probability,
    get_catalog_summary,
    get_device,
    get_flow_preview_table,
    get_lab_snapshot,
    list_missed_samples,
    retrain_from_missed,
    run_attack_session,
    run_evasion_chain,
    save_result_to_missed,
    start_lab,
    stop_lab,
    threat_level,
)
import torch
from src.prediction.simulator import KStepSimulator
from src.utils.constants import MITRE_STAGES_INV

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PRISM // SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

SOC_CSS = """
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    [data-testid="stMetricValue"] { color: #58a6ff; font-size: 1.4rem; }
    [data-testid="stMetricLabel"] { color: #8b949e; }
    .prism-header {
        font-family: 'Consolas', monospace;
        font-size: 1.1rem;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-online { color: #3fb950; font-weight: bold; }
    .status-offline { color: #f85149; font-weight: bold; }
    .suspect-box {
        background: #161b22;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    div[data-testid="stExpander"] { background: #161b22; border: 1px solid #30363d; }
</style>
"""
st.markdown(SOC_CSS, unsafe_allow_html=True)


def _init_session():
    defaults = {
        "timeline": [],
        "alerts": [],
        "event_log": [],
        "runbook_step": 1,
        "pending_suspect": None,
        "last_result": None,
        "round_counter": 0,
        "auto_chain": False,
        "ps_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    st.session_state.event_log.insert(0, f"[{ts}] {msg}")
    st.session_state.event_log = st.session_state.event_log[:100]


@st.cache_resource(show_spinner="Loading world model...")
def _load_model_bundle():
    device = get_device()
    model, scaler = load_model_and_scaler(device)
    return model, scaler, device


def _append_timeline(result, prediction: dict):
    st.session_state.timeline.append({
        "time": datetime.now(timezone.utc).isoformat(),
        "class_id": result.class_id,
        "evasion": result.evasion,
        "flows": result.flow_count,
        "attack_prob": prediction.get("attack_prob", 0),
        "pred_class": prediction.get("pred_class", "Benign"),
        "confidence": prediction.get("confidence", 0),
    })
    st.session_state.timeline = st.session_state.timeline[-50:]


@st.cache_resource(show_spinner="Loading PRISM World Model...")
def _load_prism_model(model_name: str, device: str = "cpu"):
    from src.models.world_model import StateTransformerWorldModel, LSTMWorldModel, load_checkpoint
    from src.models.gnn_model import GraphWorldModel
    from src.models.latent_dynamics import LatentDynamicsWorldModel

    d_state = 242
    if "GNN" in model_name or "Graph" in model_name:
        m = GraphWorldModel(d_node=d_state, d_graph=d_state, d_model=256, lookback=20)
        load_checkpoint(m, "weights/gnn/world_model_best.pt", device=device)
    elif "LSTM" in model_name:
        m = LSTMWorldModel(d_state=d_state, d_model=256, lstm_layers=2, lookback=20)
        load_checkpoint(m, "weights/lstm/world_model_best.pt", device=device)
    elif "Latent" in model_name:
        m = LatentDynamicsWorldModel(d_state=d_state, d_latent=64, d_model=256, n_layers=2, lookback=20)
        load_checkpoint(m, "weights/latent/world_model_best.pt", device=device)
    else:
        m = StateTransformerWorldModel(d_state=d_state, d_model=256, n_layers=4, n_heads=8, lookback=20)
        load_checkpoint(m, "weights/world_model_best.pt", device=device)
    m.eval()
    return m


@st.cache_data(show_spinner="Loading CIC-IDS-2018 test telemetry...")
def _load_test_telemetry():
    from src.data.dataset import load_states_from_npz
    npz_path = ROOT / "data" / "splits" / "test.npz"
    if npz_path.exists():
        d = load_states_from_npz(str(npz_path))
        return d["states"], d.get("labels_binary", None), d.get("labels_mitre", None)
    return None, None, None


def _render_header(lab, active_prism_model: str = "Transformer World Model (200ep GPU)"):
    online = lab.all_running
    status_cls = "status-online" if online else "status-offline"
    status_txt = "ONLINE" if online else "STANDALONE"
    st.markdown(
        f'<div class="prism-header">'
        f'PRISM // Predictive Recurrent Infiltration State Model &nbsp; '
        f'<span class="{status_cls}">[{status_txt}]</span> &nbsp; '
        f'Active: <span style="color: #58a6ff; font-weight: bold;">{active_prism_model}</span> &nbsp; '
        f'Device: {lab.device}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_metrics(lab, last_pred: dict | None):
    attack_prob = last_pred.get("attack_prob", 0) if last_pred else 0
    flows = st.session_state.last_result.flow_count if st.session_state.last_result else 0
    elapsed = st.session_state.last_result.elapsed_sec if st.session_state.last_result else 0
    fps = flows / elapsed if elapsed > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Threat Level", threat_level(attack_prob))
    c2.metric("Attack Probability", f"{attack_prob:.1%}")
    c3.metric("Last Flows", flows)
    c4.metric("Flow Rate", f"{fps:.1f}/s")
    c5.metric("Missed Samples", lab.missed_count)


def _plot_timeline(timeline: list[dict], forecast: list[float] | None):
    if not timeline and not forecast:
        st.info("Run an attack to populate the live timeline.")
        return

    observed_x, observed_y = [], []
    for i, pt in enumerate(timeline):
        observed_x.append(i - len(timeline))
        observed_y.append(pt["attack_prob"])

    fig = go.Figure()
    if observed_x:
        fig.add_trace(go.Scatter(
            x=observed_x, y=observed_y,
            mode="lines+markers", name="Observed",
            line=dict(color="#58a6ff", width=2),
        ))

    if forecast:
        fx = list(range(0, len(forecast)))
        fig.add_trace(go.Scatter(
            x=fx, y=forecast,
            mode="lines+markers", name="Forecast",
            line=dict(color="#d29922", dash="dash", width=2),
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Live & Forecasted Attack Probability",
        xaxis_title="Time window (t=now at 0)",
        yaxis_title="Attack probability",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_tactics(probs: np.ndarray):
    tactic_scores = aggregate_tactic_probs(probs)
    labels = [t for t in MITRE_TACTICS if tactic_scores.get(t, 0) > 0.01]
    if not labels:
        labels = MITRE_TACTICS[:6]
    values = [tactic_scores.get(t, 0) for t in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=values, colorscale="Reds", cmin=0, cmax=1),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
        title="MITRE ATT&CK Tactic Scores",
        xaxis=dict(range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_attribution(scores: list[tuple[str, float]]):
    if not scores:
        st.caption("Feature attribution appears after a scored attack.")
        return
    names = [s[0] for s in scores][::-1]
    vals = [s[1] for s in scores][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color="#58a6ff"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        title="Feature Attribution (z-score proxy)",
        xaxis_title="Relative contribution",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_suspect_review():
    suspect = st.session_state.pending_suspect
    if not suspect:
        return

    pred = suspect["prediction"]
    result = suspect["result"]
    mitre_map = get_mitre_map()
    tactic, tech = mitre_map.get(pred["pred_class"], ("", ""))

    st.markdown('<div class="suspect-box">', unsafe_allow_html=True)
    st.warning("SUSPECT ACTIVITY — review before alerting")
    c1, c2 = st.columns(2)
    c1.write(f"**True attack:** `{result.class_id}`")
    c1.write(f"**Predicted:** `{pred['pred_class']}` ({pred['confidence']:.1%})")
    c1.write(f"**Attack prob:** {pred['attack_prob']:.1%}")
    c2.write(f"**MITRE:** {tactic} / {tech}")
    c2.write(f"**Flows:** {result.flow_count} | **Evasion:** {result.evasion}")

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("Confirm Alert", type="primary"):
        st.session_state.alerts.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "result": result.to_dict(),
            "prediction": {k: v for k, v in pred.items() if k != "probs"},
            "status": "confirmed",
        })
        _log(f"ALERT confirmed: {pred['pred_class']} ({result.class_id})")
        st.session_state.pending_suspect = None
        st.rerun()

    if a2.button("Dismiss"):
        _log(f"Dismissed suspect: {pred['pred_class']}")
        st.session_state.pending_suspect = None
        st.rerun()

    if a3.button("Save to missed dataset"):
        path = save_result_to_missed(result, reason="operator_save")
        _log(f"Saved missed sample -> {path}")
        st.session_state.pending_suspect = None
        st.cache_resource.clear()
        st.rerun()

    if a4.button("Save + Dismiss"):
        save_result_to_missed(result, reason="operator_save")
        st.session_state.pending_suspect = None
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_ps_demo(model, scaler, device):
    st.subheader("Problem-statement demo — upload traffic, forecast, explain")
    st.caption(
        f"Schema: **{NUM_FEATURES} features** "
        f"({len(FLOW_FEATURE_COLS)} flow-level + {len(PACKET_FEATURE_COLS)} packet-level). "
        "CSV rows missing packet columns are filled with 0. Fully offline."
    )
    uploaded = st.file_uploader(
        "Upload a PCAP or CIC/PRISM CSV",
        type=["pcap", "pcapng", "csv"],
        key="ps_upload",
    )
    if not uploaded:
        st.info("Drop a `.pcap` / `.csv` file to run K-step forecast, MITRE mapping, and SHAP/attention.")
        return

    if st.button("Run inference", type="primary", key="ps_run"):
        out_dir = ROOT / "data" / "processed"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        with st.spinner("Extracting features and running world model..."):
            result = analyze_file(dest, model, scaler, device)
        st.session_state.ps_result = result
        _log(f"PS demo: {uploaded.name} -> {result.get('pred_class', 'fail')}")

    result = st.session_state.get("ps_result")
    if not result:
        return
    if not result.get("ok"):
        st.error(result.get("message", "Inference failed"))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows", result["flow_count"])
    c2.metric("Predicted class", result["pred_class"])
    c3.metric("Attack probability", f"{result['attack_prob']:.1%}")
    c4.metric("MITRE tactic", result["mitre_tactic"] or "—")
    st.write(f"**Technique:** {result['mitre_technique']}")
    st.write(f"**Explainability:** {result['explain_method']}")

    observed = [{"attack_prob": p} for p in result.get("observed_curve", [])]
    _plot_timeline(observed, result.get("forecast"))
    _plot_attribution(result.get("driving_features") or [])

    with st.expander("Attention / gradient / SHAP detail"):
        st.write("Attention", result.get("attention_features"))
        st.write("Gradient", result.get("gradient_features"))
        st.write("SHAP", result.get("shap_features"))
    st.write("Top classes", result.get("top_classes"))


def _render_prism_simulation(prism_model, prism_model_name: str, device: str, test_states, test_labels_binary, test_labels_mitre):
    st.subheader(f"🔮 PRISM World Model Simulation & Infiltration Forecast")
    st.caption(
        f"Active Architecture: **{prism_model_name}** | Target: CSE-CIC-IDS2018 Temporal States ($S_t \\in \\mathbb{{R}}^{{242}}$) | "
        f"Forecasting $P(S_{{t+k}} \\mid S_t)$ over lookahead horizon $K$."
    )

    if test_states is None:
        st.error("CIC-IDS-2018 test telemetry not found at `data/splits/test.npz`.")
        return

    c_scen, c_k, c_roll = st.columns([2, 1, 1])
    with c_scen:
        scen = st.selectbox(
            "Telemetry Scenario",
            [
                "🚨 Scenario A: Attack Escalation (Window 255 -> Rapid Critical Breach)",
                "🔍 Scenario B: Infiltration Reconnaissance (Window 60 -> Early Probe)",
                "🛡️ Scenario C: Clean Benign Baseline (Window 25 -> Normal Operations)",
                "🎛️ Custom Window Slider",
            ],
            index=0,
        )
        if "255" in scen:
            default_window = 255
        elif "60" in scen:
            default_window = 60
        elif "25" in scen:
            default_window = 25
        else:
            default_window = 100

        window_idx = st.slider("Current Time Window (t)", 20, len(test_states) - 1, default_window, step=1)

    with c_k:
        k_steps = st.slider("Lookahead Horizon (K steps)", 5, 25, 10, help="Future steps to simulate")
    with c_roll:
        n_rollouts = st.slider("Monte Carlo Rollouts", 1, 15, 5, help="Stochastic rollouts for 95% CI")

    lookback_seq = test_states[window_idx - 20 : window_idx]
    true_label = int(test_labels_binary[window_idx]) if test_labels_binary is not None else -1
    true_mitre = MITRE_STAGES_INV.get(int(test_labels_mitre[window_idx]), "Unknown") if test_labels_mitre is not None else "N/A"

    sim = KStepSimulator(model=prism_model, device=device, k_steps=k_steps, num_rollouts=n_rollouts)
    result = sim.simulate(lookback_seq)

    current_prob = result.steps[0].infiltration_prob
    peak_prob = result.peak_infiltration_prob
    alert = result.overall_alert.upper()

    alert_colors = {
        "NONE": "#3fb950",
        "LOW": "#58a6ff",
        "MEDIUM": "#d29922",
        "HIGH": "#db6d28",
        "CRITICAL": "#f85149",
    }
    color = alert_colors.get(alert, "#58a6ff")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Current Attack Prob", f"{current_prob:.1%}")
    m2.metric("Peak Forecast Risk", f"{peak_prob:.1%}")
    m3.markdown(
        f"**Threat Assessment**<br><span style='color:{color}; font-size:1.3rem; font-weight:bold;'>{alert}</span>",
        unsafe_allow_html=True,
    )
    m4.metric("Escalation Detected", "YES ⚠️" if result.is_escalating else "NO ✅")
    m5.metric("Ground Truth Label", "ATTACK 🚨" if true_label == 1 else "BENIGN 🛡️")

    st.write(f"**Trajectory Assessment**: *{result.trajectory_summary}*")

    # Estimate historical probabilities for the lookback window
    step_indices = [s.step for s in result.steps]
    step_probs = [s.infiltration_prob for s in result.steps]

    fig = go.Figure()
    if result.ensemble_upper is not None and result.ensemble_lower is not None:
        fig.add_trace(go.Scatter(
            x=step_indices + step_indices[::-1],
            y=list(result.ensemble_upper) + list(result.ensemble_lower)[::-1],
            fill="toself",
            fillcolor="rgba(210, 153, 34, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% Confidence Band",
        ))

    all_x = [0] + step_indices
    all_y = [current_prob] + step_probs
    fig.add_trace(go.Scatter(
        x=all_x, y=all_y,
        mode="lines+markers",
        name=f"Rollout ({prism_model_name.split()[0]})",
        line=dict(color="#d29922", width=3),
        marker=dict(size=8, color="#f0883e"),
    ))

    fig.add_hline(y=0.50, line_dash="dash", line_color="#d29922", annotation_text="Warning Threshold (0.50)")
    fig.add_hline(y=0.75, line_dash="dash", line_color="#f85149", annotation_text="Critical Threshold (0.75)")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        title=f"Autoregressive Forecast Horizon (t=0 to t+{k_steps})",
        xaxis_title="Simulation Step Ahead (k)",
        yaxis_title="Infiltration Probability P(Attack | S_{t+k})",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    c_mitre, c_explain = st.columns([1, 1])
    with c_mitre:
        st.subheader("MITRE ATT&CK Kill-Chain Rollout")
        peak_s = result.steps[min(result.peak_step, len(result.steps) - 1)]
        mitre_names = [MITRE_STAGES_INV.get(i, f"Stage {i}") for i in range(len(peak_s.mitre_probs))]
        mitre_vals = [float(p) for p in peak_s.mitre_probs]
        fig_m = go.Figure(go.Bar(
            x=mitre_vals, y=mitre_names, orientation="h",
            marker=dict(color=mitre_vals, colorscale="Turbo", cmin=0, cmax=1),
        ))
        fig_m.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig_m, use_container_width=True)
        st.info(f"Most Likely Kill-Chain Phase at Step +{peak_s.step}: **{peak_s.mitre_stage}** (True Stage: **{true_mitre}**)")

    with c_explain:
        st.subheader("Feature Deviation & Drivers")
        curr_state = lookback_seq[-1]
        top_feat_idx = np.argsort(np.abs(curr_state))[-10:][::-1]
        feat_names = [f"Feature [{i}]" for i in top_feat_idx]
        feat_mags = [float(curr_state[i]) for i in top_feat_idx]
        fig_f = go.Figure(go.Bar(x=feat_mags, y=feat_names, orientation="h", marker_color="#58a6ff"))
        fig_f.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            height=300,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_f, use_container_width=True)

    # 200-Epoch Benchmark Summary Expander
    with st.expander("📊 View Full 200-Epoch Multi-Model Benchmark Comparison Table"):
        st.markdown("""
        | Model Architecture | Model Family | Binary F1 | Precision | Recall | False Positive Rate | ROC AUC | MITRE F1 | Dynamics MSE | Can Forecast Ahead? |
        |:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        | **Transformer World Model** | Causal Attention | **0.7552** | 0.7826 | **0.7297** | 0.0460 | 0.9151 | 0.4506 | 268.49 | **Yes (Autoregressive)** |
        | **Graph World Model (GNN)** | Graph Topology | **0.7107** | **0.9149** | 0.5811 | **0.0123** | 0.8806 | 0.4329 | **164.91** | **Yes (Autoregressive)** |
        | **LSTM World Model** | Recurrent Sequence | 0.6875 | 0.8148 | 0.5946 | 0.0307 | **0.9335** | **0.8586** | 259.13 | **Yes (Autoregressive)** |
        | **Latent Dynamics (VAE)** | Stochastic VAE | 0.3122 | 0.1850 | **1.0000** | 1.0000 | 0.5159 | 0.4616 | 291.89 | **Yes (Autoregressive)** |
        | **Logistic Regression** | Static Linear | 0.6240 | 0.7800 | 0.5200 | 0.0319 | 0.9130 | 0.2316 | N/A | No (Static only) |
        | **Random Forest** | Static Ensemble | 0.5000 | **1.0000** | 0.3333 | **0.0000** | **0.9729** | 0.2305 | N/A | No (Static only) |
        """)


def main():
    _init_session()
    lab = get_lab_snapshot()
    model, scaler, device = _load_model_bundle()

    # ── Sidebar: PRISM World Model Selector ──────────────────────────────────
    with st.sidebar:
        st.subheader("PRISM Model Selector")
        prism_model_names = [
            "Transformer World Model (Primary 200ep GPU)",
            "Graph World Model (GNN 200ep)",
            "LSTM World Model (200ep)",
            "Latent Dynamics World Model (200ep)",
        ]
        active_model_name = st.selectbox(
            "Active World Model",
            prism_model_names,
            index=0,
            help="Select which deep world model checkpoint to evaluate and simulate.",
        )
        prism_model = _load_prism_model(active_model_name, device=str(device))
        test_states, test_labels_binary, test_labels_mitre = _load_test_telemetry()

        st.divider()
        st.subheader("Lab Control")
        if st.button("Start Lab", use_container_width=True):
            ok, msg = start_lab()
            _log(msg)
            st.cache_resource.clear()
            st.rerun()
        if st.button("Stop Lab", use_container_width=True):
            ok, msg = stop_lab()
            _log(msg)
            st.rerun()
        if st.button("Refresh Status", use_container_width=True):
            st.rerun()

        for name, running in lab.containers.items():
            icon = "🟢" if running else "🔴"
            st.write(f"{icon} `{name}`")

        st.divider()
        st.subheader("Lab Model")
        if not lab.checkpoint_compatible:
            st.info("Docker lab model: 33-class catalog.")
        if st.button("Retrain on missed samples", use_container_width=True):
            ok, msg, n = retrain_from_missed(model, scaler, device)
            _log(msg)
            if ok:
                st.cache_resource.clear()
            st.success(msg) if ok else st.error(msg)

        st.divider()
        summary = get_catalog_summary()
        st.caption(
            f"Catalog: {summary.get('trainable_classes', '?')} classes | "
            f"{summary.get('network_distinct_bots', '?')} bots"
        )

    _render_header(lab, active_prism_model=active_model_name)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_prism, tab_demo, tab_live, tab_runbook, tab_missed, tab_mitre = st.tabs([
        "🔮 PRISM World Model", "PS Demo", "Live Monitor", "Attack Runbook", "Missed & Retrain", "MITRE Catalog",
    ])

    last_pred = None
    if st.session_state.last_result and st.session_state.last_result.prediction:
        last_pred = st.session_state.last_result.prediction

    with tab_prism:
        _render_prism_simulation(prism_model, active_model_name, str(device), test_states, test_labels_binary, test_labels_mitre)

    with tab_demo:
        _render_ps_demo(model, scaler, device)

    with tab_live:
        _render_metrics(lab, last_pred)
        _render_suspect_review()

        col_chart, col_tactic = st.columns([3, 2])
        forecast = None
        if st.session_state.last_result and st.session_state.last_result.features is not None:
            forecast = forecast_attack_probability(
                model, scaler, st.session_state.last_result.features, device, K_ROLLOUT,
            )
        with col_chart:
            _plot_timeline(st.session_state.timeline, forecast)
        with col_tactic:
            if last_pred and "probs" in last_pred:
                _plot_tactics(last_pred["probs"])
            else:
                st.caption("Tactic panel fills after inference.")

        st.subheader("Live Flow Stream (model input)")
        if st.session_state.last_result and st.session_state.last_result.features is not None:
            preview = get_flow_preview_table(st.session_state.last_result.features)
            st.dataframe(preview, use_container_width=True, hide_index=True)
            scores = compute_feature_attribution(
                st.session_state.last_result.features, scaler,
            )
            _plot_attribution(scores)
        else:
            st.caption("Flow table shows the latest captured window.")

        st.subheader("Event Log")
        for line in st.session_state.event_log[:15]:
            st.code(line, language=None)

    with tab_runbook:
        st.subheader("Step-wise Attack Procedure")
        step = st.session_state.runbook_step
        st.progress(step / 5, text=f"Step {step} of 5")

        bot_classes = [c for c in get_bot_class_ids()]
        evasion_chains = get_evasion_chains()

        if step == 1:
            st.write("**Step 1 — Lab health**")
            if lab.all_running:
                st.success("All lab containers running on isolated network.")
            else:
                st.error("Lab offline. Start lab from sidebar.")
            if st.button("Next: Configure attack"):
                st.session_state.runbook_step = 2
                st.rerun()

        elif step == 2:
            st.write("**Step 2 — Select attack class & evasion**")
            class_id = st.selectbox("Attack class", bot_classes, index=0)
            chain = ["none"] + evasion_chains.get(class_id, [])
            evasion = st.selectbox("Evasion tactic", chain)
            auto_chain = st.checkbox("Auto evasion chain on detection", value=False)
            st.session_state._run_class = class_id
            st.session_state._run_evasion = evasion
            st.session_state._run_auto = auto_chain
            c1, c2 = st.columns(2)
            if c1.button("Back"):
                st.session_state.runbook_step = 1
                st.rerun()
            if c2.button("Next: Execute"):
                st.session_state.runbook_step = 3
                st.rerun()

        elif step == 3:
            st.write("**Step 3 — Execute attack & capture**")
            class_id = st.session_state.get("_run_class", bot_classes[0])
            evasion = st.session_state.get("_run_evasion", "none")
            auto = st.session_state.get("_run_auto", False)

            st.info(f"Target: `target-server` | Class: `{class_id}` | Evasion: `{evasion}`")

            if st.button("Run Attack Now", type="primary"):
                if not lab.all_running:
                    st.error("Start the lab first.")
                else:
                    st.session_state.round_counter += 1
                    if auto:
                        results = run_evasion_chain(class_id, model, scaler, device)
                        result = results[-1] if results else None
                        for r in results:
                            _log(f"Chain: {r.class_id}/{r.evasion} -> {r.message}")
                    else:
                        result = run_attack_session(
                            class_id, evasion,
                            round_num=st.session_state.round_counter,
                            model=model, scaler=scaler, device=device,
                        )
                        _log(f"Attack {class_id}/{evasion}: {result.message}")

                    if result and result.success and result.prediction:
                        st.session_state.last_result = result
                        _append_timeline(result, result.prediction)
                        if (
                            result.prediction["attack_prob"] >= DETECTION_THRESHOLD
                            or result.prediction["detected"]
                        ):
                            st.session_state.pending_suspect = {
                                "result": result,
                                "prediction": result.prediction,
                            }
                        st.session_state.runbook_step = 4
                    elif result:
                        st.session_state.last_result = result
                        st.error(result.message)
                    st.rerun()

            if st.button("Back"):
                st.session_state.runbook_step = 2
                st.rerun()

        elif step == 4:
            st.write("**Step 4 — Review inference**")
            result = st.session_state.last_result
            if result and result.prediction:
                pred = result.prediction
                st.json({
                    "true_class": result.class_id,
                    "predicted": pred["pred_class"],
                    "confidence": pred["confidence"],
                    "attack_prob": pred["attack_prob"],
                    "detected": pred["detected"],
                    "flows": result.flow_count,
                })
                top_probs = sorted(
                    [(CLASS_NAMES[i], float(pred["probs"][i])) for i in range(len(CLASS_NAMES))],
                    key=lambda x: x[1], reverse=True,
                )[:6]
                st.write("Top class probabilities:", top_probs)
            else:
                st.warning("No scored result yet.")

            _render_suspect_review()

            c1, c2 = st.columns(2)
            if c1.button("Back"):
                st.session_state.runbook_step = 3
                st.rerun()
            if c2.button("Next: Actions"):
                st.session_state.runbook_step = 5
                st.rerun()

        elif step == 5:
            st.write("**Step 5 — Save, retrain, export**")
            result = st.session_state.last_result

            c1, c2, c3 = st.columns(3)
            if c1.button("Save last as missed") and result:
                path = save_result_to_missed(result, reason="runbook_save")
                _log(f"Saved -> {path}")
                st.cache_resource.clear()
                st.success("Saved to missed dataset.")

            if c2.button("Retrain on all missed"):
                ok, msg, _ = retrain_from_missed(model, scaler, device)
                _log(msg)
                st.cache_resource.clear()
                st.success(msg) if ok else st.error(msg)

            report = build_forensic_report(
                st.session_state.timeline,
                st.session_state.alerts,
                lab,
            )
            if c3.download_button(
                "Download forensic JSON",
                data=json.dumps(report, indent=2, default=str),
                file_name="prism_forensic_report.json",
                mime="application/json",
            ):
                _log("Forensic report downloaded.")

            if st.button("Run full loop (6 classes)"):
                from src.adversarial.training_loop import run_adversarial_loop
                with st.spinner("Running adversarial loop..."):
                    run_adversarial_loop(num_rounds=6)
                _log("Full adversarial loop complete.")
                st.cache_resource.clear()
                st.rerun()

            if st.button("Reset runbook"):
                st.session_state.runbook_step = 1
                st.rerun()

    with tab_missed:
        st.subheader("Missed / evaded samples")
        samples = list_missed_samples()
        st.write(f"**{len(samples)}** samples in `{Path('data/raw/adversarial/missed')}`")
        if samples:
            st.dataframe(
                [{
                    "time": s.get("timestamp"),
                    "true": s.get("true_class_id"),
                    "predicted": s.get("predicted_class"),
                    "reason": s.get("reason"),
                    "confidence": s.get("confidence"),
                } for s in samples[:30]],
                use_container_width=True,
                hide_index=True,
            )
        if st.button("Retrain model on all missed", key="retrain_missed_tab"):
            ok, msg, n = retrain_from_missed(model, scaler, device)
            st.cache_resource.clear()
            st.success(msg) if ok else st.error(msg)

    with tab_mitre:
        st.subheader("MITRE ATT&CK catalog")
        cat = load_catalog()
        s = cat.get("summary", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total techniques", s.get("total_mitre_techniques", 0))
        m2.metric("Network bots", s.get("network_distinct_bots", 0))
        m3.metric("Family mapped", s.get("network_family", 0))
        m4.metric("Host-only", s.get("host_only", 0))

        bucket = st.selectbox("Filter", ["all", "network_distinct", "network_family", "host_only"])
        query = st.text_input("Search technique ID or name")
        rows = []
        for tech in cat.get("techniques", []):
            if bucket != "all" and tech.get("bucket") != bucket:
                continue
            if query and query.lower() not in tech["id"].lower() and query.lower() not in tech["name"].lower():
                continue
            rows.append({
                "id": tech["id"],
                "name": tech["name"],
                "bucket": tech["bucket"],
                "class": tech.get("mapped_class_id") or "—",
                "detectable": tech.get("detectable_from_network"),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
