"""
PRISM Telemetry Ingestion Studio (Upload Page)
Handles PCAP, CSV (CIC-IDS-2018, CTU-13, UNSW-NB15), and NPZ state ingestion,
feature extraction inspection, and normalization stats.
"""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="PRISM — Telemetry Ingestion",
    page_icon="📥",
    layout="wide",
)

# Custom CSS
css_path = ROOT_DIR / "app" / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="prism-title">📥 Telemetry Ingestion & Preprocessing</h1>', unsafe_allow_html=True)
st.markdown('<p class="prism-subtitle">Ingest raw PCAPs, NetFlow CSV records, or pre-built temporal state matrices for PRISM World Model analysis.</p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    source_type = st.radio(
        "Select Telemetry Source",
        ["Real CIC-IDS-2018 Splits", "Upload PCAP / PCAPNG", "Upload Flow CSV", "Load Custom NPZ"],
        horizontal=True,
    )

    if source_type == "Real CIC-IDS-2018 Splits":
        split_choice = st.selectbox("Select Partition", ["test.npz (Unseen Evaluation)", "val.npz (Validation)", "train.npz (Training Set)"])
        split_path = ROOT_DIR / "data" / "splits" / split_choice.split(" ")[0]
        
        if st.button("Load Selected Split", type="primary"):
            if split_path.exists():
                data = np.load(split_path)
                st.session_state["states"] = data["states"]
                st.session_state["labels_binary"] = data["labels_binary"]
                st.session_state["labels_mitre"] = data["labels_mitre"]
                st.session_state["feature_names"] = [f"feat_{i}" for i in range(data["states"].shape[1])]
                st.success(f"Loaded {len(data['states'])} state windows ({data['states'].shape[1]} features) from {split_choice.split(' ')[0]}")
            else:
                st.error(f"Split file not found: {split_path}")

    elif source_type == "Upload Flow CSV":
        uploaded_csv = st.file_uploader("Upload NetFlow CSV (CIC-IDS-2018 / CTU-13 / UNSW-NB15)", type=["csv"])
        if uploaded_csv is not None:
            if st.button("Process & Extract States", type="primary"):
                with st.spinner("Running FlowExtractor and StateBuilder..."):
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(uploaded_csv.read())
                        tmp_path = tmp.name
                    try:
                        from src.data.flow_extractor import FlowExtractor
                        from src.data.state_builder import StateBuilder
                        from src.data.feature_merger import FeatureMerger
                        
                        extractor = FlowExtractor(dataset_type="cicids2018")
                        df = extractor.extract(file_path=tmp_path, fit_scaler=True)
                        merger = FeatureMerger()
                        merged = merger.merge(df)
                        builder = StateBuilder(window_size_seconds=30)
                        res = builder.build_states(merged)
                        
                        st.session_state["states"] = res["states"]
                        st.session_state["labels_binary"] = res["labels_binary"]
                        st.session_state["labels_mitre"] = res["labels_mitre"]
                        st.session_state["feature_names"] = res["feature_names"]
                        st.success(f"Extracted {len(res['states'])} time windows across {res['states'].shape[1]} state dimensions!")
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

    elif source_type == "Upload PCAP / PCAPNG":
        uploaded_pcap = st.file_uploader("Upload Network Packet Capture (.pcap, .pcapng)", type=["pcap", "pcapng"])
        if uploaded_pcap is not None:
            if st.button("Parse Packets & Build States", type="primary"):
                with st.spinner("Extracting packet-level features with Scapy..."):
                    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
                        tmp.write(uploaded_pcap.read())
                        tmp_path = tmp.name
                    try:
                        from src.data.packet_extractor import PacketExtractor
                        from src.data.state_builder import StateBuilder
                        from src.data.feature_merger import FeatureMerger
                        
                        pe = PacketExtractor(window_size_seconds=30)
                        pkt_df = pe.extract_from_pcap(tmp_path)
                        merger = FeatureMerger()
                        merged = merger.merge(None, pkt_df)
                        builder = StateBuilder()
                        res = builder.build_states(merged)
                        
                        st.session_state["states"] = res["states"]
                        st.session_state["labels_binary"] = res["labels_binary"]
                        st.session_state["labels_mitre"] = res["labels_mitre"]
                        st.session_state["feature_names"] = res["feature_names"]
                        st.success(f"Parsed {len(res['states'])} time windows from PCAP!")
                    except Exception as e:
                        st.error(f"PCAP parsing failed: {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

    elif source_type == "Load Custom NPZ":
        npz_input = st.text_input("Filesystem Path to states .npz", value="data/splits/test.npz")
        if st.button("Load NPZ", type="primary"):
            p = Path(npz_input)
            if p.exists():
                data = np.load(p)
                st.session_state["states"] = data["states"]
                st.session_state["labels_binary"] = data["labels_binary"]
                st.session_state["labels_mitre"] = data["labels_mitre"]
                st.session_state["feature_names"] = [f"feat_{i}" for i in range(data["states"].shape[1])]
                st.success(f"Loaded {len(data['states'])} state windows from {npz_input}")
            else:
                st.error("File does not exist.")

with col2:
    st.markdown("### Telemetry Status")
    if "states" in st.session_state:
        states = st.session_state["states"]
        labels_bin = st.session_state.get("labels_binary", np.zeros(len(states)))
        attack_count = int(np.sum(labels_bin))
        
        st.metric("Total Windows (W=30s)", len(states))
        st.metric("State Vector Dim ($D$)", states.shape[1])
        st.metric("Attack Windows", f"{attack_count} ({100 * attack_count / max(len(states), 1):.1f}%)")
        st.success("Telemetry active in session. Ready for forecasting!")
    else:
        st.info("No active telemetry. Load a dataset split or upload traffic above.")

st.markdown("---")
if "states" in st.session_state:
    st.subheader("State Matrix Sample Preview")
    preview_df = pd.DataFrame(
        st.session_state["states"][:15, :12],
        columns=[f"S_{i}" for i in range(12)]
    )
    st.dataframe(preview_df, use_container_width=True)
