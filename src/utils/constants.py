"""
PRISM Constants
Central definitions for MITRE mappings, feature names, and model defaults.
"""

# ---------------------------------------------------------------------------
# NCIIPC Metadata & Attribution
# ---------------------------------------------------------------------------
NCIIPC_INFO = {
    "agency": "National Critical Information Infrastructure Protection Centre (NCIIPC)",
    "website": "nciipc.gov.in",
    "helpdesk": "helpdesk1@nciipc.gov.in",
    "country": "India",
    "role": "National Nodal Agency for Critical Information Infrastructure Protection",
}

# ---------------------------------------------------------------------------
# MITRE ATT&CK Stage Definitions
# ---------------------------------------------------------------------------
MITRE_STAGES = {
    "Benign": 0,
    "Reconnaissance": 1,
    "Initial Access": 2,
    "Lateral Movement": 3,
    "Command & Control": 4,
    "Exfiltration": 5,
    "Impact": 6,
}

MITRE_STAGES_INV = {v: k for k, v in MITRE_STAGES.items()}

NUM_MITRE_STAGES = len(MITRE_STAGES)

MITRE_STAGE_COLORS = {
    "Benign": "#2ecc71",
    "Reconnaissance": "#3498db",
    "Initial Access": "#f39c12",
    "Lateral Movement": "#e67e22",
    "Command & Control": "#e74c3c",
    "Exfiltration": "#9b59b6",
    "Impact": "#c0392b",
}

# ---------------------------------------------------------------------------
# CIC-IDS-2018 Label -> MITRE Stage Mapping
# ---------------------------------------------------------------------------
CICIDS_LABEL_TO_MITRE = {
    "Benign": "Benign",
    "FTP-BruteForce": "Initial Access",
    "SSH-Bruteforce": "Initial Access",
    "Bot": "Command & Control",
    "Infilteration": "Lateral Movement",
    "DoS attacks-Hulk": "Impact",
    "DoS attacks-SlowHTTPTest": "Impact",
    "DoS attacks-Slowloris": "Impact",
    "DoS attacks-GoldenEye": "Impact",
    "DDoS attacks-LOIC-HTTP": "Impact",
    "DDOS attack-HOIC": "Impact",
    "DDOS attack-LOIC-UDP": "Impact",
    "Brute Force -Web": "Initial Access",
    "Brute Force -XSS": "Initial Access",
    "SQL Injection": "Initial Access",
}

# CTU-13 Label Mapping
CTU13_LABEL_TO_MITRE = {
    "Normal": "Benign",
    "Botnet": "Command & Control",
    "Background": "Benign",
}

# UNSW-NB15 Label Mapping
UNSWNB15_LABEL_TO_MITRE = {
    "Normal": "Benign",
    "normal": "Benign",
    "Generic": "Impact",
    "generic": "Impact",
    "Exploits": "Initial Access",
    "exploits": "Initial Access",
    "Fuzzers": "Reconnaissance",
    "fuzzers": "Reconnaissance",
    "DoS": "Impact",
    "dos": "Impact",
    "Reconnaissance": "Reconnaissance",
    "reconnaissance": "Reconnaissance",
    "Backdoor": "Initial Access",
    "backdoor": "Initial Access",
    "Backdoors": "Initial Access",
    "Analysis": "Reconnaissance",
    "analysis": "Reconnaissance",
    "Worms": "Impact",
    "worms": "Impact",
    "Shellcode": "Initial Access",
    "shellcode": "Initial Access",
}

# CICIoT2023 Label Mapping
CICIOT2023_LABEL_TO_MITRE = {
    "Benign": "Benign",
    "BenignTraffic": "Benign",
    "DDoS-ICMP_Flood": "Impact",
    "DDoS-UDP_Flood": "Impact",
    "DDoS-TCP_Flood": "Impact",
    "DDoS-PSHACK_Flood": "Impact",
    "DDoS-SYN_Flood": "Impact",
    "DDoS-RSTFINFlood": "Impact",
    "DDoS-SynonymIP-Flood": "Impact",
    "DDoS-SynonymousIP_Flood": "Impact",
    "DDoS-SlowLoris": "Impact",
    "DDoS-HTTP_Flood": "Impact",
    "DDoS-ICMP_Fragmentation": "Impact",
    "DDoS-ACK_Fragmentation": "Impact",
    "DDoS-UDP_Fragmentation": "Impact",
    "DoS-UDP_Flood": "Impact",
    "DoS-TCP_Flood": "Impact",
    "DoS-SYN_Flood": "Impact",
    "DoS-HTTP_Flood": "Impact",
    "Mirai-greeth_flood": "Command & Control",
    "Mirai-greip_flood": "Command & Control",
    "Mirai-udpplain": "Command & Control",
    "Recon-PingSweep": "Reconnaissance",
    "Recon-OSScan": "Reconnaissance",
    "Recon-PortScan": "Reconnaissance",
    "VulnerabilityScan": "Reconnaissance",
    "Recon-HostDiscovery": "Reconnaissance",
    "DNS_Spoofing": "Initial Access",
    "MITM-ArpSpoofing": "Lateral Movement",
    "DictionaryBruteForce": "Initial Access",
    "BrowserExtraction": "Exfiltration",
    "BrowserHijacking": "Exfiltration",
    "CommandInjection": "Initial Access",
    "XSS": "Initial Access",
    "SqlInjection": "Initial Access",
    "Uploading_Attack": "Initial Access",
    "Backdoor_Malware": "Command & Control",
}

# LANL Authentication Dataset Label Mapping
LANL_LABEL_TO_MITRE = {
    "Normal": "Benign",
    "RedTeam": "Lateral Movement",
    "AuthenticationFailure": "Initial Access",
    "PrivilegeEscalation": "Lateral Movement",
}

# DARPA Intrusion Detection Dataset Label Mapping
DARPA_LABEL_TO_MITRE = {
    "normal": "Benign",
    "probe": "Reconnaissance",
    "dos": "Impact",
    "u2r": "Lateral Movement",
    "r2l": "Initial Access",
}

SUPPORTED_DATASETS = ["cicids2018", "ctu13", "unsw-nb15", "ciciot2023", "lanl", "darpa"]

# ---------------------------------------------------------------------------
# TCP Flag Definitions
# ---------------------------------------------------------------------------
TCP_FLAGS = ["SYN", "ACK", "FIN", "RST", "PSH", "URG"]
NUM_TCP_FLAGS = len(TCP_FLAGS)

TCP_FLAG_BITS = {
    "FIN": 0x01,
    "SYN": 0x02,
    "RST": 0x04,
    "PSH": 0x08,
    "ACK": 0x10,
    "URG": 0x20,
}

# ---------------------------------------------------------------------------
# Protocol Mapping
# ---------------------------------------------------------------------------
PROTOCOLS = {"TCP": 0, "UDP": 1, "ICMP": 2, "Other": 3}
NUM_PROTOCOLS = len(PROTOCOLS)

# ---------------------------------------------------------------------------
# Port Ranges and Notable Ports
# ---------------------------------------------------------------------------
PORT_BINS = {
    "well_known": (0, 1023),
    "registered": (1024, 49151),
    "ephemeral": (49152, 65535),
}

TOP_ATTACKED_PORTS = [
    21, 22, 23, 25, 53, 80, 110, 135, 139, 143,
    443, 445, 993, 995, 1433, 1723, 3306, 3389, 5900, 8080,
]

# ---------------------------------------------------------------------------
# Flow-Level Feature Names (CIC-IDS-2018 CSV columns)
# ---------------------------------------------------------------------------
FLOW_FEATURES_RAW = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg",
    "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts",
    "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label",
]

# Numeric features to log-transform (highly skewed)
LOG_TRANSFORM_FEATURES = [
    "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Flow Byts/s", "Flow Pkts/s",
    "Fwd Header Len", "Bwd Header Len",
]

# ---------------------------------------------------------------------------
# Packet-Level Feature Names (PCAP-derived)
# ---------------------------------------------------------------------------
PACKET_FEATURES = [
    "ttl_mean", "ttl_std",
    "tcp_window_mean", "tcp_window_std",
    "ip_frag_flag",
    "payload_size_mean", "payload_size_std",
    "port_scan_sequential", "port_scan_random",
    "retransmission_count",
    "syn_ack_ratio", "rst_rate",
]

# ---------------------------------------------------------------------------
# State Vector Feature Names (aggregated per time window)
# ---------------------------------------------------------------------------
STATE_AGGREGATE_FEATURES = [
    "num_flows",
    "num_unique_src_ips",
    "num_unique_dst_ips",
    "num_unique_dst_ports",
    "port_entropy",
]

# ---------------------------------------------------------------------------
# Model Default Hyperparameters
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = {
    "d_state": 110,
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 8,
    "lookback": 20,
    "dropout": 0.1,
    "head_dropout": 0.3,
}

# ---------------------------------------------------------------------------
# Training Defaults
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 50,
    "patience": 10,
    "grad_clip": 1.0,
    "lambda_dynamics": 1.0,
    "lambda_infiltration": 0.5,
    "lambda_mitre": 0.3,
}

# ---------------------------------------------------------------------------
# Time Window Settings
# ---------------------------------------------------------------------------
WINDOW_SIZE_SECONDS = 30
K_STEP_DEFAULT = 10

# ---------------------------------------------------------------------------
# Alert Thresholds
# ---------------------------------------------------------------------------
ALERT_THRESHOLDS = {
    "critical": 0.85,
    "high": 0.65,
    "medium": 0.40,
    "low": 0.20,
}
