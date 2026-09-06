"""
Unit tests for PRISM Preemptive Intrusion Prevention Engine (P-IPS).
Verifies AllowlistGuard, SandboxFirewallDriver, LeaseManager, and MitigationEngine.
"""

import time
import pytest
from src.mitigation.guard import AllowlistGuard
from src.mitigation.drivers import SandboxFirewallDriver
from src.mitigation.lease_manager import LeaseManager
from src.mitigation.attributor import EntityAttributor, CulpritEntity
from src.mitigation.engine import MitigationEngine, MitigationMode


def test_allowlist_guard_immutable_ips():
    """Verify core DNS, localhost, and broadcast are strictly protected."""
    guard = AllowlistGuard()
    
    # Immutable addresses must NEVER be blocked
    protected = ["127.0.0.1", "8.8.8.8", "1.1.1.1", "255.255.255.255"]
    for ip in protected:
        can_block, reason = guard.is_allowed_to_block(ip)
        assert not can_block, f"Guard failed to protect {ip}: {reason}"
        assert "allowlist" in reason.lower() or "loopback" in reason.lower()

    # Normal external attacker IP should be allowed
    can_block, reason = guard.is_allowed_to_block("198.51.100.45")
    assert can_block
    assert reason == "OK"


def test_sandbox_firewall_driver():
    """Verify the SandboxFirewallDriver accurately records rule insertion and deletion."""
    driver = SandboxFirewallDriver()
    assert not driver.is_blocked("192.168.1.105")

    # Block
    ok = driver.block_ip("192.168.1.105", reason="Test Infiltration")
    assert ok
    assert driver.is_blocked("192.168.1.105")
    assert len(driver.get_active_rules()) == 1

    # Unblock
    ok = driver.unblock_ip("192.168.1.105")
    assert ok
    assert not driver.is_blocked("192.168.1.105")
    assert len(driver.get_active_rules()) == 0


def test_lease_manager_auto_reaper():
    """Verify that LeaseManager auto-unblocks expired leases."""
    driver = SandboxFirewallDriver()
    # Check interval 0.2s for rapid testing
    lm = LeaseManager(driver=driver, check_interval=0.2)

    try:
        # Register lease with 1 second TTL
        lease = lm.register_lease("10.0.0.99", ttl_seconds=1, reason="Test short lease")
        assert lm.is_active("10.0.0.99")
        assert driver.is_blocked("10.0.0.99")

        # Wait 1.5 seconds for reaper to fire
        time.sleep(1.5)

        # Should be auto-unblocked
        assert not lm.is_active("10.0.0.99")
        assert not driver.is_blocked("10.0.0.99")
    finally:
        lm.stop_reaper()


def test_mitigation_engine_co_pilot_workflow():
    """Verify Co-Pilot requires human approval before rule enforcement."""
    driver = SandboxFirewallDriver()
    engine = MitigationEngine(driver=driver, mode=MitigationMode.CO_PILOT)

    culprit = CulpritEntity(
        ip="172.16.0.45",
        port=22,
        driving_feature="Max Out-Degree",
        anomaly_intensity=0.92,
        reason="Anomalous fan-out detected",
        stage_name="SSH Brute Force"
    )

    action = engine.evaluate_threat(culprit, risk_score=0.92, lead_time_seconds=45)
    assert action is not None
    assert action.status == "PENDING_APPROVAL"
    # Not yet blocked in driver
    assert not driver.is_blocked("172.16.0.45")
    assert len(engine.get_pending_actions()) == 1

    # Approve
    approved = engine.approve_action(action.action_id)
    assert approved
    assert driver.is_blocked("172.16.0.45")
    assert len(engine.get_pending_actions()) == 0
    assert len(engine.get_active_interventions()) == 1

    # Cleanup
    engine.lease_manager.stop_reaper()


def test_mitigation_engine_autonomous_mode():
    """Verify Autonomous mode executes immediately without human intervention."""
    driver = SandboxFirewallDriver()
    engine = MitigationEngine(driver=driver, mode=MitigationMode.AUTONOMOUS)

    culprit = CulpritEntity(
        ip="192.168.1.200",
        port=80,
        driving_feature="SYN Flood Imbalance",
        anomaly_intensity=0.96,
        reason="DDoS volumetric surge",
        stage_name="Volumetric DDoS"
    )

    action = engine.evaluate_threat(culprit, risk_score=0.96, lead_time_seconds=45)
    assert action is not None
    assert action.status == "EXECUTED"
    # Automatically blocked!
    assert driver.is_blocked("192.168.1.200")
    assert len(engine.get_active_interventions()) == 1

    # Manual unblock
    engine.manual_unblock("192.168.1.200")
    assert not driver.is_blocked("192.168.1.200")

    # Cleanup
    engine.lease_manager.stop_reaper()


def test_blast_radius_protection():
    """Verify that an IP accounting for >15% of active traffic is downgraded to Co-Pilot."""
    driver = SandboxFirewallDriver()
    engine = MitigationEngine(driver=driver, mode=MitigationMode.AUTONOMOUS)

    culprit = CulpritEntity(
        ip="192.168.1.55",
        port=80,
        driving_feature="Heavy Proxy Traffic",
        anomaly_intensity=0.91,
        reason="Suspicious traffic burst",
        stage_name="Infiltration"
    )

    # 30 flows out of 100 total (30% > 15% threshold)
    action = engine.evaluate_threat(culprit, risk_score=0.91, total_flows=100, ip_flows=30)
    assert action is not None
    # Downgraded to Co-Pilot approval instead of immediate drop
    assert action.mode == MitigationMode.CO_PILOT
    assert action.status == "PENDING_APPROVAL"
    assert not driver.is_blocked("192.168.1.55")

    engine.lease_manager.stop_reaper()


def test_strike_escalation_lifecycle():
    """Verify that repeated offenders receive escalating penalties (5m -> 30m -> 24h)."""
    driver = SandboxFirewallDriver()
    lm = LeaseManager(driver=driver, check_interval=10.0)

    try:
        # Strike 1
        ttl1, strike1 = lm.calculate_strike_escalation("198.51.100.77", base_ttl=300)
        assert strike1 == 1
        assert ttl1 == 300

        # Strike 2
        ttl2, strike2 = lm.calculate_strike_escalation("198.51.100.77", base_ttl=300)
        assert strike2 == 2
        assert ttl2 == 1800  # 30 minutes

        # Strike 3
        ttl3, strike3 = lm.calculate_strike_escalation("198.51.100.77", base_ttl=300)
        assert strike3 == 3
        assert ttl3 == 86400  # 24 hours
    finally:
        lm.stop_reaper()


def test_graduated_response_levels():
    """Verify that reconnaissance receives rate-limiting while floods receive drops."""
    driver = SandboxFirewallDriver()
    engine = MitigationEngine(driver=driver, mode=MitigationMode.AUTONOMOUS)

    # Reconnaissance probe (moderate risk -> RATE_LIMIT)
    culprit_scan = CulpritEntity(
        ip="172.16.0.88",
        port=443,
        driving_feature="Port Entropy Surge",
        anomaly_intensity=0.88,
        reason="Early horizontal port sweep",
        stage_name="Reconnaissance PortScan"
    )
    act_scan = engine.evaluate_threat(culprit_scan, risk_score=0.88)
    assert act_scan.action_type == "RATE_LIMIT"

    # Volumetric Flood (high risk -> DROP)
    culprit_flood = CulpritEntity(
        ip="172.16.0.99",
        port=80,
        driving_feature="SYN Imbalance",
        anomaly_intensity=0.98,
        reason="Volumetric connection surge",
        stage_name="Volumetric DDoS"
    )
    act_flood = engine.evaluate_threat(culprit_flood, risk_score=0.98)
    assert act_flood.action_type == "DROP"

    engine.lease_manager.stop_reaper()


def test_audit_logger_cef_format(tmp_path):
    """Verify that SOC audit logger writes valid JSONL and CEF format."""
    from src.mitigation.audit_logger import MitigationAuditLogger

    test_log = str(tmp_path / "test_audit.jsonl")
    logger = MitigationAuditLogger(log_path=test_log)

    rec = logger.log_event(
        event_type="PREEMPTIVE_DROP_ENFORCED",
        target_ip="192.168.10.50",
        action_type="DROP",
        risk_score=0.95,
        lead_time_seconds=45,
        ttl_seconds=300,
        strike_count=2,
        driving_feature="Max Out-Degree",
        reason="Predicted SSH Brute Force",
        driver_used="SANDBOX"
    )

    assert "CEF:0|PRISM|P-IPS|2.0|" in rec["cef_format"]
    assert "src=192.168.10.50" in rec["cef_format"]
    assert "act=DROP" in rec["cef_format"]

    events = logger.get_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["target_ip"] == "192.168.10.50"
    assert events[0]["strike_count"] == 2


def test_subnet_auto_collapsing():
    """Verify that >= 3 spoofed IPs in the same /24 subnet are collapsed into a single CIDR rule."""
    driver = SandboxFirewallDriver()
    lm = LeaseManager(driver=driver, check_interval=10.0)

    try:
        # Register 2 hosts in 198.51.100.0/24
        lm.register_lease("198.51.100.10", ttl_seconds=300, reason="Host 1")
        lm.register_lease("198.51.100.20", ttl_seconds=300, reason="Host 2")
        assert lm.is_active("198.51.100.10")
        assert lm.is_active("198.51.100.20")
        assert not lm.is_active("198.51.100.0/24")

        # Register 3rd host in same /24 -> triggers auto-collapsing
        collapsed = lm.check_and_apply_subnet_collapsing("198.51.100.30")
        assert collapsed == "198.51.100.0/24"

        # Now register lease for the collapsed subnet
        lm.register_lease("198.51.100.0/24", ttl_seconds=300, reason="Collapsed /24")
        assert lm.is_active("198.51.100.0/24")
        # Individual IPs should be purged from leases
        assert not lm.is_active("198.51.100.10")
        assert not lm.is_active("198.51.100.20")
    finally:
        lm.stop_reaper()


def test_max_active_rule_ceiling():
    """Verify that active firewall rules never exceed the kernel ceiling (50 rules)."""
    driver = SandboxFirewallDriver()
    lm = LeaseManager(driver=driver, check_interval=10.0)

    try:
        # Add 55 distinct IPs across distinct /24 subnets so they don't auto-collapse
        for i in range(1, 56):
            lm.register_lease(f"203.0.{i}.1", ttl_seconds=300, reason=f"Flood {i}")

        active_leases = lm.get_active_leases()
        assert len(active_leases) <= lm.MAX_ACTIVE_RULES
        assert len(active_leases) == 50
    finally:
        lm.stop_reaper()


def test_rolling_multi_window_probe_accumulator():
    """Verify that stealthy low-and-slow reconnaissance spaced across multiple windows is flagged."""
    EntityAttributor.clear_probe_cache()
    test_ip = "192.168.1.188"

    # Probing fewer than 5 unique ports -> not yet low-and-slow
    now = time.time() - 50.0  # 50 seconds ago
    EntityAttributor.record_probe(test_ip, 21, timestamp=now)
    EntityAttributor.record_probe(test_ip, 22, timestamp=now + 10)
    EntityAttributor.record_probe(test_ip, 80, timestamp=now + 20)
    EntityAttributor.record_probe(test_ip, 443, timestamp=now + 30)
    assert not EntityAttributor.is_low_and_slow_recon(test_ip, current_time=now + 35)

    # 5th unique port probed 40s after start
    EntityAttributor.record_probe(test_ip, 8080, timestamp=now + 40)
    assert EntityAttributor.is_low_and_slow_recon(test_ip)

    # Verify attribution tags it as Low-and-Slow Reconnaissance
    culprit = EntityAttributor.attribute(
        predicted_stage="Benign",
        risk_score=0.75,
        window_metadata={"culprit_ip": test_ip, "target_port": 8443}
    )
    assert culprit.stage_name == "Low-and-Slow Reconnaissance"
    assert "Rolling Multi-Window" in culprit.driving_feature
    assert culprit.anomaly_intensity >= 0.88

    # Verify MitigationEngine selects RATE_LIMIT graduated response
    driver = SandboxFirewallDriver()
    engine = MitigationEngine(driver=driver, mode=MitigationMode.AUTONOMOUS)
    action = engine.evaluate_threat(culprit, risk_score=culprit.anomaly_intensity)
    assert action is not None
    assert action.action_type == "RATE_LIMIT"

    engine.lease_manager.stop_reaper()
    EntityAttributor.clear_probe_cache()


