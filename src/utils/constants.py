"""
PRISM Constants, Feature Names, and MITRE ATT&CK Mappings (V2 Architecture - 286 Dimensions).
"""

# MITRE ATT&CK Stage Encodings (Network-Level Tactics)
MITRE_STAGES = {
    "Benign": 0,
    "Reconnaissance": 1,
    "Initial Access": 2,       # Includes Credential Access / Brute Force / Web Exploits
    "Lateral Movement": 3,     # Includes Infiltration / ARP Spoofing / Worms
    "Command & Control": 4,    # Includes Botnets / Mirai / Backdoors / C2 Channels
    "Exfiltration": 5,         # Includes Data Theft / Covert Exfil
    "Impact": 6                # Includes DoS / DDoS / Packet Floods
}

MITRE_STAGES_INV = {v: k for k, v in MITRE_STAGES.items()}
NUM_MITRE_STAGES = len(MITRE_STAGES)

# Mapping from heterogeneous dataset labels across CICIOT23, CIC-IDS2018, CIC-IDS2017, UNSW-NB15 to MITRE
LABEL_TO_MITRE = {
    # Benign traffic
    "benign": "Benign",
    "normal": "Benign",
    "benigntraffic": "Benign",
    
    # Reconnaissance / Probing (Stage 1)
    "portscan": "Reconnaissance",
    "portsweep": "Reconnaissance",
    "ipsweep": "Reconnaissance",
    "nmap": "Reconnaissance",
    "reconnaissance": "Reconnaissance",
    "analysis": "Reconnaissance",
    "probe": "Reconnaissance",
    "recon-hostdiscovery": "Reconnaissance",
    "recon-osscan": "Reconnaissance",
    "recon-portscan": "Reconnaissance",
    "recon-pingweep": "Reconnaissance",
    "vulnerabilityscan": "Reconnaissance",
    "dns_spoofing": "Reconnaissance",
    
    # Initial Access & Credential Access / Web Exploits (Stage 2)
    "ftp-patator": "Initial Access",
    "ssh-patator": "Initial Access",
    "ftp-bruteforce": "Initial Access",
    "ssh-bruteforce": "Initial Access",
    "brute force -web": "Initial Access",
    "brute force -xss": "Initial Access",
    "sql injection": "Initial Access",
    "sqlinjection": "Initial Access",
    "commandinjection": "Initial Access",
    "browserhijacking": "Initial Access",
    "dictionarybruteforce": "Initial Access",
    "xss": "Initial Access",
    "uploading_attack": "Initial Access",
    "web attack  brute force": "Initial Access",
    "web attack  xss": "Initial Access",
    "web attack  sql injection": "Initial Access",
    "web attack - brute force": "Initial Access",
    "web attack - xss": "Initial Access",
    "web attack - sql injection": "Initial Access",
    "exploits": "Initial Access",
    "fuzzers": "Initial Access",
    "heartbleed": "Initial Access",
    
    # Lateral Movement (Stage 3)
    "infilteration": "Lateral Movement",
    "infiltration": "Lateral Movement",
    "mitm-arpspoofing": "Lateral Movement",
    "arpspoofing": "Lateral Movement",
    "shellcode": "Lateral Movement",
    "worm": "Lateral Movement",
    "worms": "Lateral Movement",
    
    # Command & Control / Botnets (Stage 4)
    "bot": "Command & Control",
    "botnet": "Command & Control",
    "c&c": "Command & Control",
    "c2": "Command & Control",
    "backdoor": "Command & Control",
    "backdoors": "Command & Control",
    "backdoor_malware": "Command & Control",
    "mirai-greeth_flood": "Command & Control",
    "mirai-udpplain": "Command & Control",
    "mirai-greip_flood": "Command & Control",
    "neris": "Command & Control",
    "rbot": "Command & Control",
    "virut": "Command & Control",
    "menti": "Command & Control",
    "sogou": "Command & Control",
    "murlo": "Command & Control",
    "nsis": "Command & Control",
    
    # Exfiltration (Stage 5)
    "exfiltration": "Exfiltration",
    "generic": "Exfiltration",
    "theft": "Exfiltration",
    
    # Impact (DoS / DDoS) (Stage 6)
    "dos": "Impact",
    "ddos": "Impact",
    "ddos-icmp_flood": "Impact",
    "ddos-udp_flood": "Impact",
    "ddos-tcp_flood": "Impact",
    "ddos-pshack_flood": "Impact",
    "ddos-syn_flood": "Impact",
    "ddos-rstfinflood": "Impact",
    "ddos-synonymousip_flood": "Impact",
    "ddos-icmp_fragmentation": "Impact",
    "ddos-udp_fragmentation": "Impact",
    "ddos-ack_fragmentation": "Impact",
    "dos-udp_flood": "Impact",
    "dos-tcp_flood": "Impact",
    "dos-syn_flood": "Impact",
    "dos hulk": "Impact",
    "dos goldeneye": "Impact",
    "dos slowloris": "Impact",
    "dos slowhttptest": "Impact",
    "dos attacks-hulk": "Impact",
    "dos attacks-slowhttptest": "Impact",
    "dos attacks-slowloris": "Impact",
    "dos attacks-goldeneye": "Impact",
    "ddos attacks-loic-http": "Impact",
    "ddos attack-hoic": "Impact",
    "ddos attack-loic-udp": "Impact",
    "smurf": "Impact",
    "neptune": "Impact",
    "pod": "Impact",
    "teardrop": "Impact",
    "land": "Impact",
    "back": "Impact"
}

# 64 Unified Flow Features across CICIOT23, CIC-IDS2018, CIC-IDS2017, UNSW-NB15
UNIFIED_FLOW_FEATURES = [
    # 1. Flow Timing & Duration (8)
    "flow_duration",
    "flow_byts_s",
    "flow_pkts_s",
    "fwd_pkts_s",
    "bwd_pkts_s",
    "rate",
    "srate",
    "drate",
    
    # 2. Packet & Byte Volumes (18)
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "tot_fwd_bytes",
    "tot_bwd_bytes",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "fwd_pkt_len_min",
    "fwd_pkt_len_max",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "bwd_pkt_len_min",
    "bwd_pkt_len_max",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    "pkt_size_avg",
    
    # 3. Inter-Arrival Times (IAT) (14)
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_min",
    "flow_iat_max",
    "fwd_iat_tot",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_min",
    "fwd_iat_max",
    "bwd_iat_tot",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_min",
    "bwd_iat_max",
    
    # 4. TCP Header & Control Flags (12)
    "fin_flag_cnt",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "psh_flag_cnt",
    "ack_flag_cnt",
    "urg_flag_cnt",
    "ece_flag_cnt",
    "cwr_flag_cnt",
    "fwd_header_len",
    "bwd_header_len",
    "init_win_bytes_fwd",
    "init_win_bytes_bwd",
    
    # 5. Activity, Idle & Subflows (9)
    "down_up_ratio",
    "active_mean",
    "active_std",
    "active_max",
    "active_min",
    "idle_mean",
    "idle_std",
    "idle_max",
    "idle_min",
    
    # 6. Protocols & Application Signatures (3)
    "dst_port_binned",
    "protocol_type",       # 1: TCP, 2: UDP, 3: ICMP, 0: Other
    "app_proto_flag"       # Encoded application protocol flag (HTTP, DNS, SSH, etc.)
]

NUM_FLOW_FEATURES = len(UNIFIED_FLOW_FEATURES)  # 64

# Key continuous features selected for median quantile computation (20)
MEDIAN_FEATURE_INDICES = [
    0, 1, 2, 5, 8, 9, 10, 11, 12, 16, 22, 23, 25, 26, 31, 36, 48, 50, 52, 62
]

STATE_DIM = 292  # 64 Means + 64 Stds + 64 Maxs + 64 Mins + 20 Medians + 16 Macro & Graph Descriptors

# 16 Macro & Graph Topological Descriptors:
# [0..9]: num_flows_log, tcp_frac, udp_frac, icmp_frac, other_proto_frac, dst_port_entropy, syn_ack_ratio, down_up_mean, burst_factor, log_syn_volume
# [10..15]: max_src_out_degree, max_dst_in_degree, src_ip_entropy, dst_ip_entropy, graph_density_ratio, one_way_edge_ratio

# High-risk well-known port mapping for destination port binning
COMMON_ATTACK_PORTS = {
    21: 1,    # FTP
    22: 2,    # SSH
    23: 3,    # Telnet
    25: 4,    # SMTP
    53: 5,    # DNS
    80: 6,    # HTTP
    110: 7,   # POP3
    143: 8,   # IMAP
    443: 9,   # HTTPS
    445: 10,  # SMB
    1433: 11, # MSSQL
    3306: 12, # MySQL
    3389: 13, # RDP
    8080: 14, # HTTP-Alt
    8443: 15  # HTTPS-Alt
}

