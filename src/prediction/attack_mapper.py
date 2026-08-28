"""
PRISM - MITRE ATT&CK Stage Mapper
Maps predicted model outputs to MITRE ATT&CK stages with
tactic descriptions and recommended defender actions.
"""

from src.utils.constants import MITRE_STAGES, MITRE_STAGES_INV, MITRE_STAGE_COLORS
from src.knowledge.capec import CAPECKnowledgeBase
from src.knowledge.cve import CVEKnowledgeBase


# ---------------------------------------------------------------------------
# MITRE ATT&CK Tactic Metadata
# ---------------------------------------------------------------------------
MITRE_TACTIC_META = {
    "Benign": {
        "id": "BENIGN",
        "tactic": "Normal Traffic",
        "description": "No malicious activity detected. Network behaviour is within normal parameters.",
        "indicators": [],
        "recommended_actions": ["Continue monitoring", "Update baseline models"],
        "severity": 0,
        "color": "#2ecc71",
    },
    "Reconnaissance": {
        "id": "TA0043",
        "tactic": "Reconnaissance",
        "description": (
            "Adversary is actively gathering information about the target network. "
            "Common techniques include port scanning, service enumeration, and DNS probing."
        ),
        "indicators": [
            "High unique destination port count",
            "Sequential or random port scanning patterns",
            "High port entropy",
            "Low bytes-per-flow (scan probes)",
            "Multiple SYN without ACK (half-open scans)",
        ],
        "recommended_actions": [
            "Enable port scan detection alerts",
            "Block source IPs exceeding port-scan thresholds",
            "Review firewall rules for exposed services",
            "Capture full PCAP for forensic analysis",
        ],
        "severity": 1,
        "color": "#3498db",
    },
    "Initial Access": {
        "id": "TA0001",
        "tactic": "Initial Access",
        "description": (
            "Adversary is attempting to enter the network via exploitation, "
            "brute-force authentication, or phishing. "
            "Common techniques: SSH/FTP brute force, web application exploitation."
        ),
        "indicators": [
            "High rate of authentication failures",
            "Credential stuffing patterns (many src IPs -> same dst port)",
            "SQL injection payloads in HTTP flows",
            "Repeated connections to port 22, 21, 3389, 80, 443",
            "RST flag spikes after SYN (blocked probe responses)",
        ],
        "recommended_actions": [
            "Immediately review authentication logs",
            "Enforce MFA on all remote access services",
            "Block brute-force source IPs at perimeter",
            "Isolate targeted services behind VPN",
            "Alert SOC for manual triage",
        ],
        "severity": 2,
        "color": "#f39c12",
    },
    "Lateral Movement": {
        "id": "TA0008",
        "tactic": "Lateral Movement",
        "description": (
            "Adversary has gained initial foothold and is moving through the network "
            "to reach higher-value targets. "
            "Common techniques: SMB enumeration, WMI execution, pass-the-hash."
        ),
        "indicators": [
            "Internal east-west traffic spikes",
            "New host-to-host communication pairs",
            "Port 445 (SMB) / 135 (RPC) / 5985 (WinRM) internal flows",
            "Abnormal LDAP queries",
            "Credential relay attempts",
        ],
        "recommended_actions": [
            "CRITICAL: Segment the network immediately",
            "Isolate compromised host(s) from internal network",
            "Disable SMBv1 and enforce SMB signing",
            "Review Active Directory event logs (4624, 4625, 4648)",
            "Initiate incident response procedure",
        ],
        "severity": 4,
        "color": "#e67e22",
    },
    "Command & Control": {
        "id": "TA0011",
        "tactic": "Command & Control",
        "description": (
            "Adversary has established a persistent channel to control compromised systems. "
            "Traffic may be encrypted, beaconing, or tunnelled through legitimate protocols."
        ),
        "indicators": [
            "Regular beaconing intervals (low IAT variance)",
            "Connections to unusual external IPs on uncommon ports",
            "DNS queries to algorithmically generated domains (DGA)",
            "Encrypted non-standard port traffic",
            "Low-and-slow data exfiltration precursor flows",
        ],
        "recommended_actions": [
            "CRITICAL: Block C2 IP/domain at perimeter firewall immediately",
            "Enable DNS sinkholing for suspicious domains",
            "Review egress firewall rules — whitelist-only outbound",
            "Inspect all encrypted tunnels (SSL inspection)",
            "Escalate to senior incident response team",
        ],
        "severity": 5,
        "color": "#e74c3c",
    },
    "Exfiltration": {
        "id": "TA0010",
        "tactic": "Exfiltration",
        "description": (
            "Adversary is actively transferring sensitive data out of the network. "
            "Data may be staged, compressed, encrypted, and transferred via "
            "cloud services, DNS tunnelling, or direct connections."
        ),
        "indicators": [
            "Abnormally large outbound byte transfers",
            "High bytes-per-flow ratio with unusual external destinations",
            "DNS TXT record queries with large payloads (DNS tunnelling)",
            "ICMP with large payloads (ICMP tunnelling)",
            "Traffic to cloud storage endpoints (S3, Dropbox) outside business hours",
        ],
        "recommended_actions": [
            "CRITICAL: Block all outbound traffic to suspected exfil destinations NOW",
            "Preserve forensic evidence — snapshot affected systems",
            "Notify legal and compliance teams (data breach protocol)",
            "Invoke business continuity plan",
            "Engage external incident response retainer",
        ],
        "severity": 6,
        "color": "#9b59b6",
    },
    "Impact": {
        "id": "TA0040",
        "tactic": "Impact",
        "description": (
            "Adversary is actively disrupting, degrading, or destroying systems. "
            "Includes ransomware, DDoS attacks, data destruction, and service disruption."
        ),
        "indicators": [
            "Massive volumetric traffic from multiple sources (DDoS)",
            "SYN flood signatures",
            "Service availability degradation",
            "Disk encryption activity (ransomware precursor)",
            "Mass file modification events",
        ],
        "recommended_actions": [
            "CRITICAL: Activate DDoS mitigation / scrubbing centre",
            "Enable null-routing for attack traffic sources",
            "Activate BCP / DR failover immediately",
            "Preserve unencrypted backups — isolate from network",
            "Contact upstream ISP for traffic filtering",
        ],
        "severity": 6,
        "color": "#c0392b",
    },
}


class AttackStageMapper:
    """
    Maps model predictions to MITRE ATT&CK stages with full metadata,
    CAPEC attack patterns, CVE vulnerability references, and defender guidance.
    """

    def __init__(self) -> None:
        self.capec_kb = CAPECKnowledgeBase()
        self.cve_kb = CVEKnowledgeBase()

    def map_stage_id(self, stage_id: int) -> dict:
        """
        Given a numeric stage ID, return full MITRE metadata enriched with
        CAPEC attack patterns and related CVE vulnerabilities.

        Parameters
        ----------
        stage_id : int  — from MITRE_STAGES encoding

        Returns
        -------
        dict with tactic, description, indicators, recommended_actions,
             severity, color, stage_name, capec_patterns, cve_vulnerabilities
        """
        stage_name = MITRE_STAGES_INV.get(stage_id, "Benign")
        meta = dict(MITRE_TACTIC_META.get(stage_name, MITRE_TACTIC_META["Benign"]))
        
        # Enrich with CAPEC attack patterns and CVE vulnerabilities
        meta["capec_patterns"] = self.capec_kb.get_capec_for_stage(stage_name)
        meta["cve_vulnerabilities"] = self.cve_kb.get_cves_for_stage(stage_name)

        return {"stage_name": stage_name, "stage_id": stage_id, **meta}

    def map_from_probs(self, mitre_probs: list[float]) -> dict:
        """
        Map from a probability distribution over stages.

        Parameters
        ----------
        mitre_probs : list of float, length = NUM_MITRE_STAGES
                      softmax probabilities for each stage

        Returns
        -------
        dict with predicted stage metadata + full probability breakdown
        """
        predicted_id = int(max(range(len(mitre_probs)), key=lambda i: mitre_probs[i]))
        meta = self.map_stage_id(predicted_id)

        # Build probability breakdown for all stages
        prob_breakdown = []
        for stage_id, prob in enumerate(mitre_probs):
            name = MITRE_STAGES_INV.get(stage_id, "Unknown")
            prob_breakdown.append({
                "stage_id": stage_id,
                "stage_name": name,
                "probability": float(prob),
                "color": MITRE_TACTIC_META.get(name, {}).get("color", "#888888"),
            })
        prob_breakdown.sort(key=lambda x: x["probability"], reverse=True)

        meta["prob_breakdown"] = prob_breakdown
        meta["prediction_confidence"] = float(mitre_probs[predicted_id])
        return meta

    def infer_stage_from_features(self, top_features: list[dict]) -> str:
        """
        Heuristic stage inference from top contributing features
        (used as fallback / cross-check against model output).

        Parameters
        ----------
        top_features : list of dicts with 'feature' and 'shap_value' keys

        Returns
        -------
        Inferred MITRE stage name
        """
        feature_names = {f["feature"].lower() for f in top_features}

        if any("scan" in f or "port_entropy" in f for f in feature_names):
            return "Reconnaissance"
        if any("syn_flag" in f or "rst_rate" in f or "brute" in f for f in feature_names):
            return "Initial Access"
        if any("smb" in f or "445" in f or "internal" in f for f in feature_names):
            return "Lateral Movement"
        if any("beacon" in f or "iat_std" in f or "c2" in f for f in feature_names):
            return "Command & Control"
        if any("bytes" in f and "outbound" in f for f in feature_names):
            return "Exfiltration"
        if any("syn_flood" in f or "ddos" in f for f in feature_names):
            return "Impact"
        return "Benign"

    def get_kill_chain_position(self, stage_name: str) -> dict:
        """
        Return position in the Cyber Kill Chain for a given MITRE stage.
        """
        kill_chain = {
            "Benign": {"position": 0, "phase": "None", "urgency": "low"},
            "Reconnaissance": {"position": 1, "phase": "Reconnaissance", "urgency": "low"},
            "Initial Access": {"position": 2, "phase": "Weaponisation / Delivery", "urgency": "medium"},
            "Lateral Movement": {"position": 3, "phase": "Exploitation / Installation", "urgency": "high"},
            "Command & Control": {"position": 4, "phase": "Command & Control", "urgency": "critical"},
            "Exfiltration": {"position": 5, "phase": "Actions on Objectives", "urgency": "critical"},
            "Impact": {"position": 5, "phase": "Actions on Objectives", "urgency": "critical"},
        }
        return kill_chain.get(stage_name, kill_chain["Benign"])
