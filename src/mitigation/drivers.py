"""
Firewall & Kernel Enforcement Drivers for PRISM P-IPS.
Provides cross-platform enforcement (Windows Filtering Platform, Linux iptables/nftables,
and a safe in-memory Sandbox driver for test environments).
"""

import abc
import platform
import subprocess
from enum import Enum
from typing import Dict, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("FirewallDriver")


class ActionType(str, Enum):
    """Graduated response levels based on threat confidence and attack class."""
    DROP = "DROP"              # Null-route / drop packet immediately
    RATE_LIMIT = "RATE_LIMIT"  # Token-bucket clamp (e.g. 5 pkts/sec)
    TCP_RST = "TCP_RST"        # Immediate TCP connection termination
    TARPIT = "TARPIT"          # Delay ACKs to freeze hostile scanners


class BaseFirewallDriver(abc.ABC):
    """Abstract interface for OS-level firewall and packet-filtering drivers."""

    @abc.abstractmethod
    def block_ip(
        self,
        ip: str,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        reason: str = "",
        action_type: ActionType = ActionType.DROP
    ) -> bool:
        """Applies a preemptive defense rule for the given target IP."""
        pass

    @abc.abstractmethod
    def unblock_ip(self, ip: str) -> bool:
        """Removes the drop rule for the given target IP."""
        pass

    @abc.abstractmethod
    def is_blocked(self, ip: str) -> bool:
        """Checks if the given target IP currently has an active block rule."""
        pass

    @abc.abstractmethod
    def get_active_rules(self) -> List[Dict[str, any]]:
        """Returns list of all active firewall rules managed by PRISM."""
        pass


class SandboxFirewallDriver(BaseFirewallDriver):
    """
    In-memory mock firewall driver.
    Provides realistic rule tracking, simulation logging, and audit history
    without requiring root or Windows Administrator privileges.
    """

    def __init__(self):
        self.rules: Dict[str, Dict[str, any]] = {}
        logger.info("Initialized SandboxFirewallDriver (Safe non-privileged mode)")

    def block_ip(
        self,
        ip: str,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        reason: str = "",
        action_type: ActionType = ActionType.DROP
    ) -> bool:
        self.rules[ip] = {
            "ip": ip,
            "port": port or "ALL",
            "protocol": protocol or "ALL",
            "reason": reason,
            "action_type": action_type.value,
            "driver": "SANDBOX",
            "status": "ACTIVE"
        }
        logger.info(f"[SANDBOX FIREWALL] Applied PREEMPTIVE {action_type.value} rule for IP {ip} | Reason: {reason}")
        return True

    def unblock_ip(self, ip: str) -> bool:
        if ip in self.rules:
            del self.rules[ip]
            logger.info(f"[SANDBOX FIREWALL] Removed DROP rule for IP {ip} (TTL Expired / Manual Override)")
            return True
        return False

    def is_blocked(self, ip: str) -> bool:
        return ip in self.rules

    def get_active_rules(self) -> List[Dict[str, any]]:
        return list(self.rules.values())


class WindowsFirewallDriver(BaseFirewallDriver):
    """
    Windows Filtering Platform driver using netsh advfirewall.
    Requires running with Administrator privileges for hardware rule injection.
    """

    RULE_PREFIX = "PRISM_PREEMPT_"

    def __init__(self):
        self._check_privileges()
        self.tracked_rules: Dict[str, Dict[str, any]] = {}

    def _check_privileges(self) -> bool:
        try:
            # Query firewall status to test admin permission
            res = subprocess.run(
                ["netsh", "advfirewall", "show", "currentprofile"],
                capture_output=True,
                text=True,
                check=False
            )
            return res.returncode == 0
        except Exception:
            return False

    def block_ip(
        self,
        ip: str,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        reason: str = "",
        action_type: ActionType = ActionType.DROP
    ) -> bool:
        port_tag = f"_PORT{port}" if port else ""
        rule_name = f"{self.RULE_PREFIX}{action_type.value}_{ip.replace(':', '_').replace('/', '_')}{port_tag}"
        cmd = [
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={rule_name}",
            "dir=in",
            "action=block",
            f"remoteip={ip}"
        ]
        # Surgical Port Blocking: If specific service port is attacked, block ONLY that port
        if port:
            proto_str = protocol.upper() if protocol and protocol.upper() in ["TCP", "UDP"] else "TCP"
            cmd.extend([f"protocol={proto_str}", f"localport={port}"])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                self.tracked_rules[ip] = {
                    "ip": ip,
                    "port": port or "ALL",
                    "protocol": protocol or "ALL",
                    "rule_name": rule_name,
                    "reason": reason,
                    "action_type": action_type.value,
                    "driver": "WINDOWS_NETSH"
                }
                logger.info(f"[WINDOWS NETSH] Injected SURGICAL {action_type.value} rule '{rule_name}' for IP {ip} on port {port or 'ALL'}")
                return True
            else:
                logger.error(f"[WINDOWS NETSH] Failed to add rule: {res.stderr.strip()}")
                return False
        except Exception as e:
            logger.error(f"[WINDOWS NETSH] Execution error: {e}")
            return False

    def unblock_ip(self, ip: str) -> bool:
        rule_name = f"{self.RULE_PREFIX}*{ip.replace(':', '_').replace('/', '_')}*"
        cmd = [
            "netsh", "advfirewall", "firewall", "delete", "rule",
            f"name={rule_name}"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if ip in self.tracked_rules:
                del self.tracked_rules[ip]
            logger.info(f"[WINDOWS NETSH] Deleted rule for IP {ip}")
            return res.returncode == 0
        except Exception as e:
            logger.error(f"[WINDOWS NETSH] Failed to delete rule: {e}")
            return False

    def is_blocked(self, ip: str) -> bool:
        return ip in self.tracked_rules

    def get_active_rules(self) -> List[Dict[str, any]]:
        return list(self.tracked_rules.values())


class LinuxFirewallDriver(BaseFirewallDriver):
    """
    Linux iptables/nftables driver with surgical port targeting and graduated response.
    Requires root/sudo capabilities.
    """

    def __init__(self):
        self.tracked_rules: Dict[str, Dict[str, any]] = {}

    def block_ip(
        self,
        ip: str,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        reason: str = "",
        action_type: ActionType = ActionType.DROP
    ) -> bool:
        proto = (protocol or "tcp").lower()
        port_args = ["-p", proto, "--dport", str(port)] if port else []

        if action_type == ActionType.RATE_LIMIT:
            cmd = ["iptables", "-I", "INPUT", "-s", ip] + port_args + ["-m", "limit", "--limit", "5/s", "-j", "ACCEPT"]
        elif action_type == ActionType.TCP_RST:
            cmd = ["iptables", "-I", "INPUT", "-s", ip, "-p", "tcp"] + (["--dport", str(port)] if port else []) + ["-j", "REJECT", "--reject-with", "tcp-reset"]
        elif action_type == ActionType.TARPIT:
            cmd = ["iptables", "-I", "INPUT", "-s", ip, "-p", "tcp"] + (["--dport", str(port)] if port else []) + ["-j", "TARPIT"]
        else:
            cmd = ["iptables", "-I", "INPUT", "-s", ip] + port_args + ["-j", "DROP"]

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                self.tracked_rules[ip] = {
                    "ip": ip,
                    "port": port or "ALL",
                    "protocol": protocol or "ALL",
                    "reason": reason,
                    "action_type": action_type.value,
                    "driver": "LINUX_IPTABLES"
                }
                logger.info(f"[LINUX IPTABLES] Injected SURGICAL {action_type.value} for IP {ip} on port {port or 'ALL'}")
                return True
            return False
        except Exception as e:
            logger.error(f"[LINUX IPTABLES] Failed to add rule: {e}")
            return False

    def unblock_ip(self, ip: str) -> bool:
        cmd = ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if ip in self.tracked_rules:
                del self.tracked_rules[ip]
            logger.info(f"[LINUX IPTABLES] Removed DROP rule for IP {ip}")
            return res.returncode == 0
        except Exception as e:
            logger.error(f"[LINUX IPTABLES] Failed to remove rule: {e}")
            return False

    def is_blocked(self, ip: str) -> bool:
        return ip in self.tracked_rules

    def get_active_rules(self) -> List[Dict[str, any]]:
        return list(self.tracked_rules.values())


def get_system_driver(force_sandbox: bool = False) -> BaseFirewallDriver:
    """
    Factory function: returns the best available firewall driver.
    If run without administrator/root privileges or force_sandbox=True,
    gracefully uses SandboxFirewallDriver to guarantee zero permission crashes.
    """
    if force_sandbox:
        return SandboxFirewallDriver()

    sys_name = platform.system().lower()
    if sys_name == "windows":
        driver = WindowsFirewallDriver()
        if driver._check_privileges():
            logger.info("Elevated privileges detected. Using WindowsFirewallDriver.")
            return driver
        else:
            logger.warning("No Windows Administrator rights. Falling back to SandboxFirewallDriver.")
            return SandboxFirewallDriver()
    elif sys_name == "linux":
        try:
            check = subprocess.run(["iptables", "-L", "-n"], capture_output=True)
            if check.returncode == 0:
                logger.info("Root privileges detected. Using LinuxFirewallDriver.")
                return LinuxFirewallDriver()
        except Exception:
            pass
        return SandboxFirewallDriver()
    
    return SandboxFirewallDriver()
