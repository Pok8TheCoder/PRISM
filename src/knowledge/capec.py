"""
PRISM — CAPEC Knowledge Base Module
Provides CAPEC (Common Attack Pattern Enumeration and Classification) mappings
and lookups to enrich MITRE ATT&CK stage predictions with attack patterns.
Fully functional offline with pre-compiled pattern dictionary.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("prism.knowledge.capec")

# Pre-compiled CAPEC catalog mapped to MITRE ATT&CK tactics & stages
CAPEC_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "Reconnaissance": [
        {
            "id": "CAPEC-287",
            "name": "TCP Port Scan",
            "abstraction": "Standard",
            "description": "An adversary sends TCP SYN or ACK packets to target ports to identify open services.",
            "prerequisites": "Network connectivity to target system.",
            "mitre_technique": "T1046",
        },
        {
            "id": "CAPEC-310",
            "name": "Scanning for Vulnerable Software",
            "abstraction": "Standard",
            "description": "An adversary probes network services using automated scanners to detect unpatched software.",
            "prerequisites": "Target service version banners exposed.",
            "mitre_technique": "T1595.002",
        },
        {
            "id": "CAPEC-112",
            "name": "Brute Force",
            "abstraction": "Standard",
            "description": "Adversary probes service authentication endpoints to discover valid user credentials.",
            "prerequisites": "Exposed login endpoint (SSH, RDP, FTP, HTTP).",
            "mitre_technique": "T1110",
        },
    ],
    "Initial Access": [
        {
            "id": "CAPEC-16",
            "name": "Dictionary Password Attack",
            "abstraction": "Standard",
            "description": "Adversary uses a list of common passwords to attempt authentication against network services.",
            "prerequisites": "Target account username list or default accounts.",
            "mitre_technique": "T1110.001",
        },
        {
            "id": "CAPEC-66",
            "name": "SQL Injection",
            "abstraction": "Detailed",
            "description": "Adversary injects malicious SQL statements into user input fields to bypass authentication.",
            "prerequisites": "Unsanitized database queries in web application.",
            "mitre_technique": "T1190",
        },
        {
            "id": "CAPEC-242",
            "name": "Code Injection",
            "abstraction": "Meta",
            "description": "Adversary executes arbitrary code by introducing attacker-controlled data into application logic.",
            "prerequisites": "Vulnerable application parser or command interpreter.",
            "mitre_technique": "T1059",
        },
    ],
    "Lateral Movement": [
        {
            "id": "CAPEC-555",
            "name": "Remote Services Login",
            "abstraction": "Standard",
            "description": "Adversary uses stolen credentials to log into adjacent internal hosts via SSH/RDP/SMB.",
            "prerequisites": "Valid domain or local administrator credentials.",
            "mitre_technique": "T1021",
        },
        {
            "id": "CAPEC-640",
            "name": "Pass the Hash",
            "abstraction": "Detailed",
            "description": "Adversary uses NTLM or Kerberos ticket hash to authenticate to remote systems without cracking password.",
            "prerequisites": "Extracted NTLM hash or Kerberos ticket from compromised host memory.",
            "mitre_technique": "T1550.002",
        },
    ],
    "Command & Control": [
        {
            "id": "CAPEC-584",
            "name": "Custom Command and Control Protocol",
            "abstraction": "Standard",
            "description": "Adversary establishes encrypted bi-directional command channel over non-standard protocols.",
            "prerequisites": "Compromised internal host running C2 agent (Botnet/Trojan).",
            "mitre_technique": "T1095",
        },
        {
            "id": "CAPEC-269",
            "name": "DNS Tunneling",
            "abstraction": "Detailed",
            "description": "Adversary encodes C2 traffic and commands inside DNS query requests and TXT responses.",
            "prerequisites": "Authoritative DNS server under attacker control.",
            "mitre_technique": "T1071.004",
        },
    ],
    "Exfiltration": [
        {
            "id": "CAPEC-118",
            "name": "Data Exfiltration over Network Protocol",
            "abstraction": "Meta",
            "description": "Adversary steals sensitive data by transmitting it over authorized outbound network channels (HTTPS, DNS, FTP).",
            "prerequisites": "Access to target sensitive data files and outbound internet access.",
            "mitre_technique": "T1041",
        },
    ],
    "Impact": [
        {
            "id": "CAPEC-125",
            "name": "Flooding",
            "abstraction": "Meta",
            "description": "Adversary exhausts network bandwidth or server processing capacity using high-volume traffic floods.",
            "prerequisites": "Distributed botnet or high-bandwidth reflection sources.",
            "mitre_technique": "T1498",
        },
        {
            "id": "CAPEC-488",
            "name": "HTTP Flood Denial of Service",
            "abstraction": "Standard",
            "description": "Adversary sends high volumes of HTTP GET/POST requests to overwhelm web server application pool.",
            "prerequisites": "Accessible HTTP/HTTPS service.",
            "mitre_technique": "T1499.002",
        },
    ],
}


class CAPECKnowledgeBase:
    """CAPEC Query and Mapping Engine."""

    def __init__(self) -> None:
        self.catalog = CAPEC_CATALOG

    def get_capec_for_stage(self, stage_name: str) -> List[Dict[str, Any]]:
        """Return matching CAPEC patterns for a given MITRE ATT&CK stage."""
        stage_name = stage_name.strip()
        return self.catalog.get(stage_name, [])

    def get_pattern_by_id(self, capec_id: str) -> Optional[Dict[str, Any]]:
        """Find specific CAPEC entry by ID (e.g. CAPEC-287)."""
        capec_id_upper = capec_id.upper().strip()
        for patterns in self.catalog.values():
            for p in patterns:
                if p["id"].upper() == capec_id_upper:
                    return p
        return None

    def search_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Search CAPEC entries by keyword query."""
        q = query.lower()
        results = []
        for patterns in self.catalog.values():
            for p in patterns:
                if q in p["name"].lower() or q in p["description"].lower() or q in p.get("mitre_technique", "").lower():
                    results.append(p)
        return results
