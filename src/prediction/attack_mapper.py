"""
MITRE ATT&CK Mapping and Defensive Action Recommendations.
"""

from typing import Dict, Any

MITRE_RECOMMENDATIONS = {
    "Benign": {
        "severity": "LOW",
        "description": "Normal baseline network traffic observed.",
        "action": "No remediation required. Continuing routine monitoring."
    },
    "Reconnaissance": {
        "severity": "MEDIUM",
        "description": "Active port scanning or network probing detected.",
        "action": "Inspect external firewall ACLs, rate-limit aggressive sweepers, and verify closed ports."
    },
    "Initial Access": {
        "severity": "HIGH",
        "description": "Brute-force authentication or web application exploitation attempt detected.",
        "action": "Enforce MFA, temporarily block repeated failing source IPs, and patch targeted public-facing services."
    },
    "Lateral Movement": {
        "severity": "CRITICAL",
        "description": "Internal host-to-host exploitation (e.g. SMB/RDP probing) detected.",
        "action": "Isolate affected subnet immediately, revoke active session tokens, and analyze domain controller logs."
    },
    "Command & Control": {
        "severity": "CRITICAL",
        "description": "Botnet beaconing or malicious outbound protocol tunneling detected.",
        "action": "Sinkhole C2 domain/IP at the DNS firewall, isolate infected endpoints, and run endpoint remediation."
    },
    "Exfiltration": {
        "severity": "CRITICAL",
        "description": "Anomalous large outbound data transfer or protocol abuse detected.",
        "action": "Sever outbound data connection immediately, freeze affected accounts, and begin data loss incident response."
    },
    "Impact": {
        "severity": "HIGH",
        "description": "High-volume or low-and-slow Denial of Service (DoS/DDoS) attack observed.",
        "action": "Activate DDoS mitigation scrubbing filters, scale bandwidth limits, and drop malformed packet streams."
    }
}


def get_stage_guidance(stage_name: str) -> Dict[str, str]:
    """Returns security context and remediation recommendations for a MITRE stage."""
    return MITRE_RECOMMENDATIONS.get(stage_name, {
        "severity": "MEDIUM",
        "description": f"Unclassified security event associated with {stage_name}.",
        "action": "Conduct manual packet capture investigation."
    })
