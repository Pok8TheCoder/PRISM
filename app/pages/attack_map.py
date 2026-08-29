"""
PRISM MITRE ATT&CK & Defense Playbook Studio (Attack Map Page)
Maps predicted progression to the MITRE ATT&CK kill-chain, links CAPEC patterns
and CVE vulnerabilities, and provides automated SOC playbooks.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.prediction.attack_mapper import AttackStageMapper, MITRE_TACTIC_META
from src.utils.constants import MITRE_STAGES_INV, MITRE_STAGE_COLORS

st.set_page_config(
    page_title="PRISM — MITRE ATT&CK Matrix",
    page_icon="🗺️",
    layout="wide",
)

css_path = ROOT_DIR / "app" / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="prism-title">🗺️ MITRE ATT&CK Kill-Chain & Playbook Studio</h1>', unsafe_allow_html=True)
st.markdown('<p class="prism-subtitle">Correlate network state transitions against the MITRE ATT&CK framework, CAPEC patterns, and actionable SOC playbooks.</p>', unsafe_allow_html=True)

mapper = AttackStageMapper()

# MITRE Stage Navigation Tabs
stages = [s for s in MITRE_STAGES_INV.values() if s != "Benign"]
cols = st.columns(len(stages))

for i, stage_name in enumerate(stages):
    color = MITRE_STAGE_COLORS.get(stage_name, "#3b82f6")
    meta = mapper.get_tactic_meta(stage_name)
    with cols[i]:
        st.markdown(
            f'<div style="border-top: 4px solid {color}; background: #111827; border-radius: 8px; padding: 12px; text-align: center; height: 100%;">'
            f'<div style="font-size: 0.75rem; color: #94a3b8; font-weight: 600;">{meta.get("id", "TAXXXX")}</div>'
            f'<div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-top: 4px;">{stage_name}</div>'
            f'<div style="font-size: 0.75rem; color: {color}; margin-top: 6px;">Severity {meta.get("severity", 1)}/6</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# Detailed Tactic Inspector
selected_stage = st.selectbox("Select MITRE Stage for Deep-Dive Investigation", stages, index=2)
tactic_info = mapper.get_tactic_meta(selected_stage)

col_details, col_playbook = st.columns([1, 1])

with col_details:
    st.subheader(f"Tactic Overview: {selected_stage} ({tactic_info.get('id', '')})")
    st.markdown(f"**Description:** {tactic_info.get('description', 'N/A')}")
    
    st.markdown("#### Primary Network Indicators")
    for ind in tactic_info.get("indicators", []):
        st.markdown(f"- 🚩 **{ind}**")
        
    st.markdown("#### Related CAPEC Attack Patterns")
    capec_patterns = tactic_info.get("capec_patterns", [])
    if capec_patterns:
        for cap in capec_patterns:
            if isinstance(cap, dict):
                st.markdown(f"- 🏷️ **{cap.get('id', 'CAPEC')}**: {cap.get('name', '')} ({cap.get('likelihood', 'Medium')} likelihood)")
            else:
                st.markdown(f"- 🏷️ {cap}")
    else:
        st.info("No explicit CAPEC mappings recorded for this stage.")

    st.markdown("#### Known Vulnerability Correlations (CVE)")
    cves = tactic_info.get("cve_vulnerabilities", [])
    if cves:
        for c in cves:
            if isinstance(c, dict):
                st.markdown(f"- 🛡️ **{c.get('cve', 'CVE')}**: {c.get('description', '')} (CVSS: `{c.get('cvss', 'N/A')}`)")
            else:
                st.markdown(f"- 🛡️ {c}")
    else:
        st.info("No direct CVEs registered.")

with col_playbook:
    st.subheader("SOC Incident Response Containment Playbook")
    st.markdown(
        f'<div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 12px; border-radius: 6px; margin-bottom: 15px;">'
        f'<strong>Action Priority:</strong> Level {tactic_info.get("severity", 1)} Response Protocol Activated'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    st.markdown("#### Immediate Containment Checklist")
    for action in tactic_info.get("recommended_actions", []):
        st.checkbox(f"{action}", key=f"chk_{selected_stage}_{action[:15]}")

    st.markdown("#### NCIIPC / Enterprise Escalation Guidance")
    st.info(
        "1. Isolate the affected subnet or host immediately if Lateral Movement or C2 is confirmed.\n"
        "2. Capture full PCAP telemetry at the perimeter router for forensic analysis.\n"
        "3. Notify the internal Computer Security Incident Response Team (CSIRT).\n"
        "4. Contact NCIIPC Helpdesk (helpdesk1@nciipc.gov.in) if Critical Information Infrastructure (CII) assets are targeted."
    )
