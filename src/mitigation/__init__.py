"""
PRISM Preemptive Intrusion Prevention Subsystem (P-IPS).
Provides safety guards, cross-platform firewall drivers, TTL lease management,
and entity attribution for autonomous and co-pilot network mitigation.
"""

from src.mitigation.guard import AllowlistGuard
from src.mitigation.drivers import (
    BaseFirewallDriver,
    ActionType,
    WindowsFirewallDriver,
    LinuxFirewallDriver,
    SandboxFirewallDriver,
    get_system_driver
)
from src.mitigation.lease_manager import LeaseManager, RuleLease
from src.mitigation.attributor import EntityAttributor, CulpritEntity
from src.mitigation.audit_logger import MitigationAuditLogger
from src.mitigation.engine import MitigationEngine, MitigationMode, MitigationAction

__all__ = [
    "AllowlistGuard",
    "BaseFirewallDriver",
    "ActionType",
    "WindowsFirewallDriver",
    "LinuxFirewallDriver",
    "SandboxFirewallDriver",
    "get_system_driver",
    "LeaseManager",
    "RuleLease",
    "EntityAttributor",
    "CulpritEntity",
    "MitigationAuditLogger",
    "MitigationEngine",
    "MitigationMode",
    "MitigationAction",
]
