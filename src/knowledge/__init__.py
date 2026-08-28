"""
PRISM Knowledge Base Package.
Exposes CAPEC and CVE knowledge engines for threat intelligence enrichment.
"""

from src.knowledge.capec import CAPECKnowledgeBase
from src.knowledge.cve import CVEKnowledgeBase

__all__ = ["CAPECKnowledgeBase", "CVEKnowledgeBase"]
