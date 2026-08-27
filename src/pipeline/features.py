"""Canonical feature schema used by training, live capture, and the PS demo.

Flow-level columns match CIC-IDS-2018 / CICFlowMeter names.
Packet-level columns are derived from PCAP (zeros when ingesting flow-only CSV).
"""

from __future__ import annotations

FLOW_FEATURE_COLS = [
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "SYN Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
]

PACKET_FEATURE_COLS = [
    "TTL Mean",
    "TTL Std",
    "TCP Win Mean",
    "IP Frag Cnt",
    "Payload Len Mean",
    "Payload Len Std",
    "Retrans Cnt",
]

FEATURE_COLS = FLOW_FEATURE_COLS + PACKET_FEATURE_COLS
NUM_FEATURES = len(FEATURE_COLS)

CIC_LABEL_COL = "Label"
