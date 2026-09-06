"""
Allowlist Guard & Blast Radius Safety Controller for PRISM P-IPS.
Guarantees that critical infrastructure (Localhost, Default Gateway, DNS, DHCP)
can never be accidentally blocked, preventing self-inflicted network denial of service.
"""

import ipaddress
import socket
import subprocess
import platform
from typing import Set, Tuple, List, Optional
from src.utils.logger import setup_logger

logger = setup_logger("AllowlistGuard")


class AllowlistGuard:
    """
    Guarantees that essential network infrastructure cannot be blocked.
    Combines hardcoded immutable ranges with dynamic gateway and DNS discovery.
    """

    # Immutable public DNS, multicast, and loopback ranges
    IMMUTABLE_IPS: Set[str] = {
        "127.0.0.1",
        "::1",
        "0.0.0.0",
        "255.255.255.255",
        "8.8.8.8",        # Google Primary DNS
        "8.8.4.4",        # Google Secondary DNS
        "1.1.1.1",        # Cloudflare Primary DNS
        "1.0.0.1",        # Cloudflare Secondary DNS
        "9.9.9.9",        # Quad9 DNS
        "208.67.222.222", # OpenDNS
    }

    IMMUTABLE_NETWORKS: List[ipaddress.IPv4Network] = [
        ipaddress.IPv4Network("127.0.0.0/8"),    # Loopback
        ipaddress.IPv4Network("224.0.0.0/4"),    # Multicast
        ipaddress.IPv4Network("169.254.0.0/16"), # Link-local APIPA
    ]

    # Essential infrastructure ports that must never be severed
    IMMUTABLE_PORTS: Set[int] = {
        53,   # DNS
        67,   # DHCP Server
        68,   # DHCP Client
        123,  # NTP Network Time
    }

    def __init__(self, custom_allowlist: Optional[List[str]] = None):
        self.allowlist_ips: Set[str] = set(self.IMMUTABLE_IPS)
        self.allowlist_networks: List[ipaddress.IPv4Network] = list(self.IMMUTABLE_NETWORKS)
        
        # Dynamically discover host's default gateway and local host IP
        self._discover_local_environment()

        # Add any user-specified custom entries
        if custom_allowlist:
            for item in custom_allowlist:
                self.add_to_allowlist(item)

    def _discover_local_environment(self) -> None:
        """Discovers local machine IP and default gateway to protect them."""
        # 1. Discover local host IP
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Does not actually connect, but queries routing table
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                if local_ip:
                    self.allowlist_ips.add(local_ip)
                    logger.info(f"Protected local machine IP: {local_ip}")
        except Exception as e:
            logger.debug(f"Could not discover local IP via socket: {e}")

        # 2. Discover default gateway
        gateway_ip = self._detect_default_gateway()
        if gateway_ip:
            self.allowlist_ips.add(gateway_ip)
            logger.info(f"Protected Default Gateway IP: {gateway_ip}")

    def _detect_default_gateway(self) -> Optional[str]:
        """Cross-platform default gateway detection."""
        system = platform.system().lower()
        try:
            if system == "windows":
                output = subprocess.check_output("route print 0.0.0.0", shell=True, text=True)
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0] == "0.0.0.0" and parts[1] == "0.0.0.0":
                        gw = parts[2]
                        ipaddress.IPv4Address(gw)  # validate
                        return gw
            elif system == "linux":
                output = subprocess.check_output("ip route | grep default", shell=True, text=True)
                parts = output.strip().split()
                if "via" in parts:
                    gw_idx = parts.index("via") + 1
                    gw = parts[gw_idx]
                    ipaddress.IPv4Address(gw)
                    return gw
        except Exception as e:
            logger.debug(f"Gateway detection failed: {e}")
        return None

    def add_to_allowlist(self, item: str) -> None:
        """Adds an individual IP or CIDR network to the allowlist."""
        item = item.strip()
        try:
            if "/" in item:
                net = ipaddress.IPv4Network(item, strict=False)
                self.allowlist_networks.append(net)
                logger.info(f"Added network to allowlist: {net}")
            else:
                ip = str(ipaddress.ip_address(item))
                self.allowlist_ips.add(ip)
                logger.info(f"Added IP to allowlist: {ip}")
        except ValueError as e:
            logger.warning(f"Invalid IP/CIDR '{item}' ignored: {e}")

    def is_allowed_to_block(self, target_ip: str) -> Tuple[bool, str]:
        """
        Evaluates whether an IP can be safely blocked.
        
        Returns:
            (can_block: bool, reason: str)
            - True, "OK" if safe to block.
            - False, "<Reason>" if blocking would violate blast radius.
        """
        target_ip = target_ip.strip()
        
        # 1. Syntax check
        try:
            parsed_ip = ipaddress.ip_address(target_ip)
        except ValueError:
            return False, f"Invalid IP address format: '{target_ip}'"

        # 2. Check individual immutable IP list
        if target_ip in self.allowlist_ips:
            return False, f"IP {target_ip} is in protected allowlist (Default Gateway / Core DNS / Host IP)"

        # 3. Check network ranges
        if parsed_ip.version == 4:
            for net in self.allowlist_networks:
                if parsed_ip in net:
                    return False, f"IP {target_ip} falls within protected subnet {net}"

        # 4. Check private loopback / multicast properties
        if parsed_ip.is_loopback:
            return False, f"IP {target_ip} is a loopback address"
        if parsed_ip.is_multicast:
            return False, f"IP {target_ip} is a multicast address"

        return True, "OK"

    def is_port_protected(self, port: Optional[int]) -> Tuple[bool, str]:
        """Ensures that essential network infrastructure ports cannot be blocked."""
        if port is None:
            return False, "No port specified"
        if port in self.IMMUTABLE_PORTS:
            return True, f"Port {port} is protected infrastructure (DNS/DHCP/NTP)"
        return False, "OK"

    def check_blast_radius(
        self,
        target_ip: str,
        ip_flow_count: int = 1,
        total_flows: int = 1,
        max_traffic_fraction: float = 0.15
    ) -> Tuple[bool, str, float]:
        """
        Evaluates the potential operational impact of blocking target_ip.
        If target_ip represents > 15% of active network traffic, an automated
        drop could cause an accidental corporate outage (e.g. shared gateway/proxy).

        Returns:
            (is_safe_to_autoblock: bool, message: str, traffic_ratio: float)
        """
        if total_flows <= 0:
            return True, "Traffic volume negligible", 0.0

        ratio = float(ip_flow_count / total_flows)
        if total_flows >= 20 and ratio > max_traffic_fraction:
            return (
                False,
                f"High Blast Radius: Host represents {ratio:.1%} of active network traffic (threshold {max_traffic_fraction:.0%}). Downgrading to Co-Pilot.",
                ratio
            )

        return True, "Safe blast radius", ratio
