"""
Master Mitigation Engine for PRISM P-IPS.
Coordinates AllowlistGuard, Firewall Drivers, LeaseManager, and EntityAttribution.
Provides dual-mode operation: Autonomous Auto-Pilot vs. Supervised Co-Pilot.
"""

import time
import uuid
import threading
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.mitigation.guard import AllowlistGuard
from src.mitigation.drivers import BaseFirewallDriver, ActionType, get_system_driver
from src.mitigation.lease_manager import LeaseManager, RuleLease
from src.mitigation.attributor import CulpritEntity
from src.mitigation.audit_logger import MitigationAuditLogger
from src.utils.logger import setup_logger

logger = setup_logger("MitigationEngine")


class MitigationMode(str, Enum):
    AUTONOMOUS = "AUTONOMOUS"  # Zero-touch automated kernel/firewall drops
    CO_PILOT = "CO_PILOT"      # 1-click human analyst approval required


@dataclass
class MitigationAction:
    action_id: str
    target_ip: str
    target_port: Optional[int]
    risk_score: float
    lead_time_seconds: int
    ttl_seconds: int
    attack_type: str
    driving_feature: str
    reason: str
    timestamp: float
    mode: MitigationMode
    action_type: str = "DROP"
    strike_count: int = 1
    status: str = "PENDING_APPROVAL"  # "PENDING_APPROVAL", "EXECUTED", "REJECTED_ALLOWLIST", "DISMISSED"

    def to_dict(self) -> Dict[str, any]:
        return asdict(self)


class MitigationEngine:
    """
    Master Orchestrator for Preemptive Intrusion Prevention.
    """

    _instance: Optional["MitigationEngine"] = None
    _lock = threading.Lock()

    def __init__(self, driver: Optional[BaseFirewallDriver] = None, mode: MitigationMode = MitigationMode.CO_PILOT):
        self.guard = AllowlistGuard()
        self.driver = driver or get_system_driver()
        self.lease_manager = LeaseManager(driver=self.driver, check_interval=2.0)
        self.audit_logger = MitigationAuditLogger()
        self.mode = mode

        # Pending approvals for Co-Pilot mode
        self.pending_actions: Dict[str, MitigationAction] = {}
        self.action_history: List[MitigationAction] = []
        self._action_lock = threading.Lock()

        logger.info(f"PRISM MitigationEngine initialized in mode: {self.mode.value} with driver: {type(self.driver).__name__}")

    @classmethod
    def get_instance(cls) -> "MitigationEngine":
        """Singleton accessor to ensure UI and stream share the same state."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = MitigationEngine()
            return cls._instance

    def set_mode(self, mode: MitigationMode) -> None:
        """Switches between Autonomous Auto-Pilot and Supervised Co-Pilot."""
        with self._action_lock:
            self.mode = mode
            logger.info(f"Mitigation mode switched to: {self.mode.value}")

    def evaluate_threat(
        self,
        culprit: CulpritEntity,
        risk_score: float,
        lead_time_seconds: int = 45,
        ttl_seconds: int = 300,
        risk_threshold: float = 0.85,
        total_flows: int = 1,
        ip_flows: int = 1
    ) -> Optional[MitigationAction]:
        """
        Evaluates a predicted threat trajectory and takes immediate autonomous action
        or queues for Co-Pilot human approval with multi-tier graduated response and blast-radius checks.
        """
        if risk_score < risk_threshold:
            return None

        # 1. Determine Graduated Mitigation Strategy
        stage_lower = culprit.stage_name.lower()
        if "recon" in stage_lower or "scan" in stage_lower:
            action_type = ActionType.RATE_LIMIT if risk_score < 0.92 else ActionType.DROP
        elif "brute" in stage_lower or "ssh" in stage_lower or "patator" in stage_lower:
            action_type = ActionType.TCP_RST if risk_score < 0.88 else ActionType.DROP
        else:
            action_type = ActionType.DROP

        # 2. Immutable Allowlist & Blast Radius Check
        can_block, reason = self.guard.is_allowed_to_block(culprit.ip)
        action_id = str(uuid.uuid4())[:8]

        # 3. Dynamic Traffic Ratio Blast Radius Protection
        effective_mode = self.mode
        is_safe_blast, blast_msg, blast_ratio = self.guard.check_blast_radius(
            culprit.ip, ip_flow_count=ip_flows, total_flows=total_flows
        )
        if not is_safe_blast and effective_mode == MitigationMode.AUTONOMOUS:
            effective_mode = MitigationMode.CO_PILOT
            logger.warning(f"[BLAST RADIUS OVERRIDE] {blast_msg}")

        action = MitigationAction(
            action_id=action_id,
            target_ip=culprit.ip,
            target_port=culprit.port,
            risk_score=risk_score,
            lead_time_seconds=lead_time_seconds,
            ttl_seconds=ttl_seconds,
            attack_type=culprit.stage_name,
            driving_feature=culprit.driving_feature,
            reason=culprit.reason if can_block else f"BLOCKED BY GUARD: {reason}",
            timestamp=time.time(),
            mode=effective_mode,
            action_type=action_type.value,
            status="PENDING_APPROVAL" if effective_mode == MitigationMode.CO_PILOT else "EXECUTED"
        )

        if not can_block:
            action.status = "REJECTED_ALLOWLIST"
            logger.warning(f"Mitigation rejected for {culprit.ip}: {reason}")
            self.audit_logger.log_event(
                event_type="MITIGATION_REJECTED_GUARD",
                target_ip=culprit.ip,
                action_type=action_type.value,
                risk_score=risk_score,
                lead_time_seconds=lead_time_seconds,
                ttl_seconds=ttl_seconds,
                reason=reason,
                driver_used=type(self.driver).__name__
            )
            with self._action_lock:
                self.action_history.append(action)
            return action

        # 4. Check if already actively leased
        if self.lease_manager.is_active(culprit.ip):
            logger.debug(f"IP {culprit.ip} is already actively quarantined.")
            return None

        # 5. Route based on Effective Operating Mode
        if effective_mode == MitigationMode.AUTONOMOUS:
            # Autonomous Zero-Touch Execution
            lease = self.lease_manager.register_lease(
                ip=culprit.ip,
                ttl_seconds=ttl_seconds,
                reason=action.reason,
                mitre_stage=culprit.stage_name,
                risk_score=risk_score,
                lead_time_seconds=lead_time_seconds,
                action_type=action_type,
                enable_escalation=True
            )
            action.status = "EXECUTED"
            action.ttl_seconds = lease.ttl_seconds
            action.strike_count = lease.strike_count

            self.audit_logger.log_event(
                event_type="AUTONOMOUS_MITIGATION_EXECUTED",
                target_ip=culprit.ip,
                action_type=action_type.value,
                risk_score=risk_score,
                lead_time_seconds=lead_time_seconds,
                ttl_seconds=lease.ttl_seconds,
                strike_count=lease.strike_count,
                driving_feature=culprit.driving_feature,
                reason=action.reason,
                driver_used=type(self.driver).__name__
            )

            logger.info(f"[AUTONOMOUS AUTO-PILOT] Preemptively applied {action_type.value} on {culprit.ip} (+{lead_time_seconds}s horizon | Strike #{lease.strike_count})")
            with self._action_lock:
                self.action_history.append(action)
            return action

        else:
            # Co-Pilot Mode: Queue for Human Approval
            with self._action_lock:
                # Avoid duplicate pending actions for same IP
                for existing in self.pending_actions.values():
                    if existing.target_ip == culprit.ip:
                        return None
                self.pending_actions[action_id] = action
                logger.info(f"[CO-PILOT QUEUED] Awaiting human approval for {culprit.ip} (Action ID: {action_id} | {action_type.value})")

            self.audit_logger.log_event(
                event_type="CO_PILOT_PROPOSAL_QUEUED",
                target_ip=culprit.ip,
                action_type=action_type.value,
                risk_score=risk_score,
                lead_time_seconds=lead_time_seconds,
                ttl_seconds=ttl_seconds,
                reason=action.reason,
                driver_used=type(self.driver).__name__
            )
            return action

    def approve_action(self, action_id: str) -> bool:
        """Approves a pending Co-Pilot mitigation action."""
        with self._action_lock:
            if action_id not in self.pending_actions:
                return False
            action = self.pending_actions.pop(action_id)

        action_type_enum = ActionType(action.action_type) if action.action_type in ActionType.__members__ else ActionType.DROP

        # Enforce on Lease Manager
        lease = self.lease_manager.register_lease(
            ip=action.target_ip,
            ttl_seconds=action.ttl_seconds,
            reason=action.reason,
            mitre_stage=action.attack_type,
            risk_score=action.risk_score,
            lead_time_seconds=action.lead_time_seconds,
            action_type=action_type_enum,
            enable_escalation=True
        )
        action.status = "EXECUTED"
        action.ttl_seconds = lease.ttl_seconds
        action.strike_count = lease.strike_count

        self.audit_logger.log_event(
            event_type="CO_PILOT_ACTION_APPROVED",
            target_ip=action.target_ip,
            action_type=action.action_type,
            risk_score=action.risk_score,
            lead_time_seconds=action.lead_time_seconds,
            ttl_seconds=lease.ttl_seconds,
            strike_count=lease.strike_count,
            driving_feature=action.driving_feature,
            reason=action.reason,
            driver_used=type(self.driver).__name__
        )

        with self._action_lock:
            self.action_history.append(action)
        logger.info(f"[CO-PILOT APPROVED] Human approved {action.action_type} for {action.target_ip} (Strike #{lease.strike_count})")
        return True

    def dismiss_action(self, action_id: str) -> bool:
        """Dismisses a pending Co-Pilot mitigation proposal."""
        with self._action_lock:
            if action_id in self.pending_actions:
                action = self.pending_actions.pop(action_id)
                action.status = "DISMISSED"
                self.action_history.append(action)

                self.audit_logger.log_event(
                    event_type="CO_PILOT_ACTION_DISMISSED",
                    target_ip=action.target_ip,
                    action_type=action.action_type,
                    risk_score=action.risk_score,
                    reason="Dismissed by human operator",
                    driver_used=type(self.driver).__name__
                )

                logger.info(f"[CO-PILOT DISMISSED] Dismissed mitigation proposal for {action.target_ip}")
                return True
        return False

    def manual_unblock(self, ip: str) -> bool:
        """Immediate manual override to unblock an IP."""
        revoked = self.lease_manager.revoke_lease(ip, manual=True)
        if revoked:
            self.audit_logger.log_event(
                event_type="MANUAL_OVERRIDE_UNBLOCK",
                target_ip=ip,
                action_type="UNBLOCK",
                reason="Manual operator unblock requested from dashboard",
                driver_used=type(self.driver).__name__
            )
        return revoked

    def get_pending_actions(self) -> List[Dict[str, any]]:
        """Returns pending actions for Streamlit UI rendering."""
        with self._action_lock:
            return [act.to_dict() for act in self.pending_actions.values()]

    def get_active_interventions(self) -> List[Dict[str, any]]:
        """Returns all currently active firewall drop leases."""
        return self.lease_manager.get_active_leases()

    def get_status_summary(self) -> Dict[str, any]:
        """Provides status metrics for UI cards."""
        active = self.get_active_interventions()
        pending = self.get_pending_actions()
        return {
            "mode": self.mode.value,
            "driver": type(self.driver).__name__,
            "active_count": len(active),
            "pending_count": len(pending),
            "total_neutralized": len([a for a in self.action_history if a.status == "EXECUTED"])
        }
