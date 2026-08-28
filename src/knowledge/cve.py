"""
PRISM — CVE / NVD Knowledge Base Module
Provides Common Vulnerabilities and Exposures (CVE) and NVD metadata lookups
to map predicted infiltration states to known infrastructure vulnerabilities.
Works offline with pre-populated vulnerability database.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("prism.knowledge.cve")

# Curated catalog of critical infrastructure CVEs grouped by MITRE ATT&CK stage
CVE_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "Initial Access": [
        {
            "cve_id": "CVE-2021-44228",
            "title": "Apache Log4j RCE (Log4Shell)",
            "cvss_score": 10.0,
            "severity": "CRITICAL",
            "cwe": "CWE-502",
            "description": "Apache Log4j2 JNDI features used in configuration, log messages, and parameters do not protect against attacker controlled LDAP and other JNDI endpoints.",
            "mitre_technique": "T1190",
            "affected_component": "Apache Log4j 2.0-beta9 to 2.15.0",
        },
        {
            "cve_id": "CVE-2021-34473",
            "title": "Microsoft Exchange Server RCE (ProxyShell)",
            "cvss_score": 9.8,
            "severity": "CRITICAL",
            "cwe": "CWE-918",
            "description": "Microsoft Exchange Server Remote Code Execution Vulnerability allowing unauthenticated attacker to execute code as SYSTEM.",
            "mitre_technique": "T1190",
            "affected_component": "Microsoft Exchange Server 2013/2016/2019",
        },
        {
            "cve_id": "CVE-2023-3519",
            "title": "Citrix ADC and Gateway Unauthenticated RCE",
            "cvss_score": 9.8,
            "severity": "CRITICAL",
            "cwe": "CWE-94",
            "description": "Unauthenticated Remote Code Execution in Citrix ADC and Citrix Gateway when configured as Gateway or AAA virtual server.",
            "mitre_technique": "T1190",
            "affected_component": "Citrix ADC / NetScaler Gateway",
        },
    ],
    "Lateral Movement": [
        {
            "cve_id": "CVE-2017-0144",
            "title": "SMBv1 Remote Code Execution (EternalBlue)",
            "cvss_score": 9.8,
            "severity": "CRITICAL",
            "cwe": "CWE-20",
            "description": "Microsoft Server Message Block 1.0 (SMBv1) protocol handling allows remote code execution via crafted packets.",
            "mitre_technique": "T1210",
            "affected_component": "Microsoft Windows SMBv1",
        },
        {
            "cve_id": "CVE-2020-1472",
            "title": "Netlogon Privilege Escalation (Zerologon)",
            "cvss_score": 10.0,
            "severity": "CRITICAL",
            "cwe": "CWE-330",
            "description": "An elevation of privilege vulnerability exists when an attacker establishes a vulnerable Netlogon secure channel connection to a domain controller.",
            "mitre_technique": "T1210",
            "affected_component": "Microsoft Windows Netlogon / Active Directory Domain Controllers",
        },
    ],
    "Command & Control": [
        {
            "cve_id": "CVE-2022-22965",
            "title": "Spring Framework RCE (Spring4Shell)",
            "cvss_score": 9.8,
            "severity": "CRITICAL",
            "cwe": "CWE-94",
            "description": "Spring Framework Data Binding ClassLoader Exposure allows unauthenticated attacker to drop web shell for C2 communications.",
            "mitre_technique": "T1505.003",
            "affected_component": "Spring Framework 5.3.0 to 5.3.17",
        },
    ],
    "Impact": [
        {
            "cve_id": "CVE-2023-44487",
            "title": "HTTP/2 Rapid Reset DDoS Vulnerability",
            "cvss_score": 7.5,
            "severity": "HIGH",
            "cwe": "CWE-400",
            "description": "The HTTP/2 protocol allows a request cancellation attack (Rapid Reset) leading to server resource exhaustion and Denial of Service.",
            "mitre_technique": "T1499",
            "affected_component": "HTTP/2 Web Servers and Load Balancers",
        },
    ],
}


class CVEKnowledgeBase:
    """CVE / NVD Query and Risk Assessment Engine."""

    def __init__(self) -> None:
        self.catalog = CVE_CATALOG

    def get_cves_for_stage(self, stage_name: str) -> List[Dict[str, Any]]:
        """Retrieve related CVEs for a predicted MITRE stage."""
        stage_name = stage_name.strip()
        return self.catalog.get(stage_name, [])

    def lookup_cve(self, cve_id: str) -> Optional[Dict[str, Any]]:
        """Lookup specific CVE entry by ID."""
        cve_id_upper = cve_id.upper().strip()
        for cves in self.catalog.values():
            for entry in cves:
                if entry["cve_id"].upper() == cve_id_upper:
                    return entry
        return None

    def assess_severity(self, cvss_score: float) -> str:
        """Categorize CVSS v3 score into standard severity bucket."""
        if cvss_score >= 9.0:
            return "CRITICAL"
        elif cvss_score >= 7.0:
            return "HIGH"
        elif cvss_score >= 4.0:
            return "MEDIUM"
        else:
            return "LOW"
