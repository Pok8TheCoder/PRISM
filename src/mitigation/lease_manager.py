"""
Lease Manager for PRISM P-IPS.
Enforces automatic TTL (Time-To-Live) expiration on all firewall drop rules.
Runs a background reaper thread to ensure no temporary block becomes permanent.
"""

import time
import threading
import ipaddress
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from src.mitigation.drivers import BaseFirewallDriver, ActionType
from src.utils.logger import setup_logger

logger = setup_logger("LeaseManager")


@dataclass
class RuleLease:
    ip: str
    created_at: float
    ttl_seconds: int
    expires_at: float
    reason: str
    mitre_stage: str
    risk_score: float
    lead_time_seconds: int
    port: Optional[int] = None
    protocol: Optional[str] = None
    action_type: str = "DROP"
    strike_count: int = 1
    status: str = "ACTIVE"  # "ACTIVE", "EXPIRED", "MANUALLY_REVOKED"

    @property
    def seconds_remaining(self) -> int:
        rem = int(self.expires_at - time.time())
        return max(0, rem)

    def to_dict(self) -> Dict[str, any]:
        d = asdict(self)
        d["seconds_remaining"] = self.seconds_remaining
        return d


class LeaseManager:
    """
    Manages timed mitigation leases, strike escalation, and guarantees graceful unblocking.
    """

    MAX_ACTIVE_RULES: int = 50

    def __init__(self, driver: BaseFirewallDriver, check_interval: float = 2.0, max_active_rules: int = 50):
        self.driver = driver
        self.check_interval = check_interval
        self.max_active_rules = max_active_rules
        self.leases: Dict[str, RuleLease] = {}
        self.history: List[RuleLease] = []
        self.strike_records: Dict[str, List[float]] = {}  # IP -> list of offense timestamps
        self._lock = threading.Lock()
        self._running = False
        self._reaper_thread: Optional[threading.Thread] = None

        self.start_reaper()

    def calculate_strike_escalation(self, ip: str, base_ttl: int = 300) -> Tuple[int, int]:
        """
        Escalates punishment for repeated offenders within a 1-hour window:
        - 1st Strike: Base TTL (5 minutes / 300s)
        - 2nd Strike: 30 minutes (1,800s)
        - 3rd Strike+: 24 hours deep quarantine (86,400s)
        """
        now = time.time()
        with self._lock:
            past_strikes = self.strike_records.get(ip, [])
            # Prune strikes older than 1 hour (3600s)
            recent_strikes = [t for t in past_strikes if (now - t) <= 3600]
            strike_count = len(recent_strikes) + 1
            recent_strikes.append(now)
            self.strike_records[ip] = recent_strikes

        if strike_count == 1:
            escalated_ttl = base_ttl
        elif strike_count == 2:
            escalated_ttl = max(base_ttl * 6, 1800)  # 30 min
        else:
            escalated_ttl = 86400  # 24 hours

        return escalated_ttl, strike_count

    def check_and_apply_subnet_collapsing(self, new_ip: str) -> Optional[str]:
        """
        Detects distributed spoofing or botnet clusters from the same /24 subnet.
        If >= 3 active rules share the same /24 prefix, collapses them into a single subnet rule.
        """
        if "/" in new_ip:
            return None
        try:
            target_net = ipaddress.IPv4Network(f"{new_ip}/24", strict=False)
            subnet_cidr = str(target_net)
        except ValueError:
            return None

        with self._lock:
            if subnet_cidr in self.leases:
                return subnet_cidr

            # Find existing leases in this subnet
            matching_ips = []
            for ip_key in list(self.leases.keys()):
                if "/" not in ip_key:
                    try:
                        if ipaddress.IPv4Address(ip_key) in target_net:
                            matching_ips.append(ip_key)
                    except ValueError:
                        pass

            if len(matching_ips) >= 2:  # with the new one, this makes >= 3
                # Collapse! Unblock individual IPs from driver
                for ip_to_remove in matching_ips:
                    if ip_to_remove in self.leases:
                        del self.leases[ip_to_remove]
                    self.driver.unblock_ip(ip_to_remove)

                logger.warning(f"[SUBNET COLLAPSE] Detected 3+ spoofed hosts in {subnet_cidr}. Collapsing into single /24 rule!")
                return subnet_cidr

        return None

    def start_reaper(self) -> None:
        """Starts the background TTL cleaner thread."""
        if not self._running:
            self._running = True
            self._reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True, name="PRISM-TTL-Reaper")
            self._reaper_thread.start()
            logger.info("Started PRISM TTL Lease Reaper thread")

    def stop_reaper(self) -> None:
        """Stops the reaper thread gracefully."""
        self._running = False
        if self._reaper_thread and self._reaper_thread.is_alive():
            self._reaper_thread.join(timeout=1.0)
            logger.info("Stopped PRISM TTL Lease Reaper thread")

    def register_lease(
        self,
        ip: str,
        ttl_seconds: int = 300,
        reason: str = "",
        mitre_stage: str = "Unknown",
        risk_score: float = 1.0,
        lead_time_seconds: int = 45,
        port: Optional[int] = None,
        protocol: Optional[str] = None,
        action_type: ActionType = ActionType.DROP,
        enable_escalation: bool = True
    ) -> RuleLease:
        """
        Registers an active drop lease with strike escalation, subnet collapsing, and rule ceiling limits.
        """
        # 1. Subnet Auto-Collapsing: Check if 3+ hosts in same /24 are attacking
        collapsed_cidr = self.check_and_apply_subnet_collapsing(ip)
        target_entity = collapsed_cidr if collapsed_cidr else ip
        if collapsed_cidr:
            reason = f"{reason} | Aggregated /24 Subnet Block"

        # 2. Strike Escalation
        if enable_escalation:
            effective_ttl, strike_num = self.calculate_strike_escalation(target_entity, base_ttl=ttl_seconds)
        else:
            effective_ttl = ttl_seconds
            strike_num = 1

        now = time.time()
        expires_at = now + effective_ttl

        lease = RuleLease(
            ip=target_entity,
            created_at=now,
            ttl_seconds=effective_ttl,
            expires_at=expires_at,
            reason=reason,
            mitre_stage=mitre_stage,
            risk_score=risk_score,
            lead_time_seconds=lead_time_seconds,
            port=port,
            protocol=protocol,
            action_type=action_type.value,
            strike_count=strike_num,
            status="ACTIVE"
        )

        with self._lock:
            # 3. Rule Ceiling Protection: If table is full, evict oldest low-risk rule
            if len(self.leases) >= self.max_active_rules and target_entity not in self.leases:
                # Evict lowest risk / oldest rule
                evict_candidate = sorted(self.leases.items(), key=lambda x: (x[1].risk_score, -x[1].created_at))[0][0]
                evicted = self.leases.pop(evict_candidate)
                self.driver.unblock_ip(evict_candidate)
                logger.warning(f"[RULE CEILING] Reached {self.max_active_rules} active rules. Evicted low-priority rule for {evict_candidate} to protect kernel RAM.")

            # Enforce on driver with action_type and surgical port
            self.leases[target_entity] = lease
            self.driver.block_ip(target_entity, port=port, protocol=protocol, reason=reason, action_type=action_type)
            logger.info(f"Registered mitigation lease for {target_entity} | Port: {port or 'ALL'} | Strike #{strike_num} | Action: {action_type.value} | TTL: {effective_ttl}s")

        return lease

    def revoke_lease(self, ip: str, manual: bool = True) -> bool:
        """
        Manually or programmatically revokes a block lease before TTL expiration.
        """
        with self._lock:
            if ip in self.leases:
                lease = self.leases[ip]
                lease.status = "MANUALLY_REVOKED" if manual else "EXPIRED"
                self.history.append(lease)
                del self.leases[ip]
                self.driver.unblock_ip(ip)
                logger.info(f"Revoked mitigation lease for {ip} (Reason: {lease.status})")
                return True
        return False

    def get_active_leases(self) -> List[Dict[str, any]]:
        """Returns all currently active leases with countdown metrics."""
        with self._lock:
            return [lease.to_dict() for lease in self.leases.values()]

    def is_active(self, ip: str) -> bool:
        with self._lock:
            return ip in self.leases

    def _reaper_loop(self) -> None:
        """Periodic loop that finds expired rules and removes them."""
        while self._running:
            time.sleep(self.check_interval)
            now = time.time()
            expired_ips: List[str] = []

            with self._lock:
                for ip, lease in list(self.leases.items()):
                    if now >= lease.expires_at:
                        expired_ips.append(ip)

            for ip in expired_ips:
                logger.info(f"Lease TTL expired for {ip}. Auto-unblocking...")
                self.revoke_lease(ip, manual=False)
