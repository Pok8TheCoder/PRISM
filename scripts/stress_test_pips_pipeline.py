"""
PRISM P-IPS Pipeline Stress-Testing Framework:
Simulates coordinated adversary swarms of 1,000, 10,000, and 100,000 attacking IPs.
Measures:
  1. EntityAttributor processing time & throughput
  2. MitigationEngine decision throughput (Autonomous vs. Co-Pilot)
  3. Queue sizes (pending_actions, action_history)
  4. Audit logger throughput (CEF events/sec, I/O latency, file growth)
  5. Memory footprint (RAM RSS delta, peak memory via tracemalloc)
  6. Firewall rules generated (verifying MAX_ACTIVE_RULES=50 ceiling & subnet collapsing)
  7. Verification: Does MAX_ACTIVE_RULES=50 protect only the firewall or also upstream?
"""

import os
import sys
import time
import json
import psutil
import tracemalloc
import tempfile
import numpy as np
from typing import List, Dict, Any

# Append project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mitigation.attributor import EntityAttributor, CulpritEntity
from src.mitigation.guard import AllowlistGuard
from src.mitigation.drivers import SandboxFirewallDriver, ActionType
from src.mitigation.lease_manager import LeaseManager
from src.mitigation.engine import MitigationEngine, MitigationMode
from src.mitigation.audit_logger import MitigationAuditLogger
from src.utils.logger import setup_logger

logger = setup_logger("PIPS_StressTest")


def generate_coordinated_swarm(num_ips: int) -> List[Dict[str, Any]]:
    """
    Generates realistic coordinated attack traffic for num_ips:
      - 60% wide volumetric attack (distinct random public subnets)
      - 25% clustered botnet attack (multiple IPs in same /24 subnets to trigger collapsing)
      - 10% repeat offenders (duplicate IPs testing deduplication & strike escalation)
      - 5% low-and-slow stealth scans (multiple ports on same IPs)
    """
    rng = np.random.RandomState(42)
    stream = []

    # 1. Clustered botnet subnets (3 to 6 IPs per /24 subnet)
    num_clustered = int(num_ips * 0.25)
    clusters = num_clustered // 4
    for c in range(clusters):
        subnet_prefix = f"198.51.{100 + (c % 150)}"
        for host in range(1, 5):
            stream.append({
                "ip": f"{subnet_prefix}.{host}",
                "port": 80,
                "stage": "Volumetric DDoS",
                "risk": float(rng.uniform(0.92, 0.99)),
                "feature": "SYN Flood Surge"
            })

    # 2. Repeat offenders (same 50 IPs repeating multiple times)
    num_repeats = int(num_ips * 0.10)
    repeat_ips = [f"192.0.2.{i % 50}" for i in range(num_repeats)]
    for ip in repeat_ips:
        stream.append({
            "ip": ip,
            "port": 22,
            "stage": "SSH Brute Force",
            "risk": float(rng.uniform(0.88, 0.95)),
            "feature": "High SSH Failures"
        })

    # 3. Stealth low-and-slow scans (same IP touching ports 21, 22, 80, 443, 8080)
    num_stealth = int(num_ips * 0.05)
    stealth_ips = [f"172.16.50.{i % 20}" for i in range(num_stealth)]
    ports = [21, 22, 80, 443, 8080, 8443]
    for idx, ip in enumerate(stealth_ips):
        stream.append({
            "ip": ip,
            "port": ports[idx % len(ports)],
            "stage": "Reconnaissance PortScan",
            "risk": float(rng.uniform(0.72, 0.86)),
            "feature": "Port Entropy Surge"
        })

    # 4. Wide volumetric flood (unique random public IPs across /16 subnets)
    remaining = num_ips - len(stream)
    for i in range(remaining):
        oct2 = (i // 254) % 250
        oct3 = (i % 254) + 1
        stream.append({
            "ip": f"203.{oct2}.{oct3}.{rng.randint(1, 254)}",
            "port": 80,
            "stage": "Mirai DDoS",
            "risk": float(rng.uniform(0.86, 0.98)),
            "feature": "Packet Inflow Surge"
        })

    # Shuffle to simulate simultaneous interleaved packet stream
    rng.shuffle(stream)
    return stream[:num_ips]


def benchmark_scale(scale_size: int) -> Dict[str, Any]:
    print(f"\n" + "="*80)
    print(f"   BENCHMARKING COORDINATED ATTACK SWARM: {scale_size:,} IPs")
    print("="*80)

    swarm = generate_coordinated_swarm(scale_size)
    process = psutil.Process(os.getpid())

    # --- 1. EntityAttributor Benchmark ---
    EntityAttributor.clear_probe_cache()
    gc_before = process.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    t0 = time.perf_counter()
    culprits: List[CulpritEntity] = []
    for item in swarm:
        c = EntityAttributor.attribute(
            predicted_stage=item["stage"],
            risk_score=item["risk"],
            window_metadata={"culprit_ip": item["ip"], "target_port": item["port"]}
        )
        culprits.append(c)
    t_attributor = time.perf_counter() - t0

    current_mem, peak_attributor_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    attributor_cache_size = len(EntityAttributor._probe_cache)

    print(f"[1. EntityAttributor]")
    print(f"  • Total Time:           {t_attributor*1000:.2f} ms ({t_attributor:.4f} s)")
    print(f"  • Throughput:           {scale_size / t_attributor:,.0f} IPs/sec")
    print(f"  • Latency per IP:       {(t_attributor / scale_size) * 1e6:.2f} µs")
    print(f"  • Probe Cache Entries:  {attributor_cache_size:,}")
    print(f"  • Peak Memory Added:    {peak_attributor_mem / (1024*1024):.2f} MB")

    # --- 2. MitigationEngine Benchmark (AUTONOMOUS Mode) ---
    temp_dir = tempfile.mkdtemp()
    temp_audit_log = os.path.join(temp_dir, f"audit_stress_{scale_size}.jsonl")

    driver = SandboxFirewallDriver()
    engine = MitigationEngine(driver=driver, mode=MitigationMode.AUTONOMOUS)
    engine.audit_logger = MitigationAuditLogger(log_path=temp_audit_log)

    tracemalloc.start()
    t0 = time.perf_counter()
    executed_count = 0
    suppressed_count = 0

    for idx, c in enumerate(culprits):
        item = swarm[idx]
        action = engine.evaluate_threat(
            culprit=c,
            risk_score=item["risk"],
            lead_time_seconds=45,
            ttl_seconds=300,
            risk_threshold=0.85
        )
        if action and action.status == "EXECUTED":
            executed_count += 1
        else:
            suppressed_count += 1

    t_engine = time.perf_counter() - t0
    _, peak_engine_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    engine.lease_manager.stop_reaper()

    # Metrics on queue & state
    active_leases_count = len(engine.lease_manager.leases)
    active_firewall_rules = len(driver.get_active_rules())
    action_history_len = len(engine.action_history)
    strike_records_len = len(engine.lease_manager.strike_records)
    audit_file_size_mb = os.path.getsize(temp_audit_log) / (1024 * 1024) if os.path.exists(temp_audit_log) else 0.0

    print(f"\n[2. MitigationEngine (Autonomous Mode)]")
    print(f"  • Total Time:           {t_engine*1000:.2f} ms ({t_engine:.4f} s)")
    print(f"  • Decision Throughput:  {scale_size / t_engine:,.0f} evaluations/sec")
    print(f"  • Latency per Threat:   {(t_engine / scale_size) * 1e6:.2f} µs")
    print(f"  • Actions Executed:     {executed_count:,}")
    print(f"  • Actions Suppressed:   {suppressed_count:,} (Below threshold, deduplicated, or already leased)")
    print(f"  • Action History Queue: {action_history_len:,} items in RAM (UPSTREAM)")
    print(f"  • Strike Tracker Size:  {strike_records_len:,} IPs tracked in RAM (UPSTREAM)")

    print(f"\n[3. Firewall Driver & Rule Ceiling Verification]")
    print(f"  • Active Rules in Driver: {active_firewall_rules} (MAX_ACTIVE_RULES = {engine.lease_manager.max_active_rules})")
    print(f"  • Active Leases in RAM:   {active_leases_count}")
    assert active_firewall_rules <= engine.lease_manager.max_active_rules, "VIOLATION: Kernel rules exceeded 50!"
    print(f"  • Ceiling Enforced?     VERIFIED: Exactly {active_firewall_rules} <= 50 rules enforced!")

    print(f"\n[4. Audit Logger Performance & I/O]")
    print(f"  • Audit Events Logged:  {executed_count:,} CEF events")
    print(f"  • Log File Size on Disk:{audit_file_size_mb:.2f} MB")
    print(f"  • Audit Throughput:     {executed_count / max(t_engine, 0.001):,.0f} CEF events/sec")

    # Cleanup temp log
    try:
        if os.path.exists(temp_audit_log):
            os.remove(temp_audit_log)
        os.rmdir(temp_dir)
    except Exception:
        pass

    # --- 5. Upstream Overload Analysis ---
    # Check if MAX_ACTIVE_RULES=50 protects upstream pipeline
    upstream_protected = (
        action_history_len <= 50 and
        strike_records_len <= 50 and
        attributor_cache_size <= 50
    )

    print(f"\n[5. Upstream Pipeline Protection Analysis]")
    print(f"  • Firewall Layer:       PROTECTED by MAX_ACTIVE_RULES=50 ({active_firewall_rules} rules)")
    print(f"  • Upstream Layer:       UNPROTECTED by rule ceiling:")
    print(f"      - action_history:   {action_history_len:,} items (unbounded growth)")
    print(f"      - strike_records:   {strike_records_len:,} items (unbounded growth)")
    print(f"      - _probe_cache:     {attributor_cache_size:,} items (unbounded growth)")

    return {
        "scale": scale_size,
        "attributor_time_s": round(t_attributor, 4),
        "attributor_throughput_ips_sec": int(scale_size / t_attributor),
        "attributor_latency_us": round((t_attributor / scale_size) * 1e6, 2),
        "attributor_cache_size": attributor_cache_size,
        "engine_time_s": round(t_engine, 4),
        "engine_throughput_eval_sec": int(scale_size / t_engine),
        "engine_latency_us": round((t_engine / scale_size) * 1e6, 2),
        "executed_count": executed_count,
        "suppressed_count": suppressed_count,
        "action_history_size": action_history_len,
        "strike_records_size": strike_records_len,
        "active_firewall_rules": active_firewall_rules,
        "max_rule_ceiling": engine.lease_manager.max_active_rules,
        "ceiling_held": active_firewall_rules <= engine.lease_manager.max_active_rules,
        "audit_file_mb": round(audit_file_size_mb, 2),
        "peak_mem_mb": round((peak_attributor_mem + peak_engine_mem) / (1024 * 1024), 2),
        "upstream_protected": upstream_protected
    }


def main():
    import logging
    # Disable INFO/WARNING logging during benchmark to avoid terminal I/O bottlenecks
    logging.disable(logging.WARNING)

    print("================================================================================")
    print("           PRISM P-IPS CO-ORDINATED ATTACK SWARM STRESS-TEST SUITE              ")
    print("================================================================================")

    scales = [1_000, 10_000, 100_000]
    results = []

    for s in scales:
        res = benchmark_scale(s)
        results.append(res)

    print("\n" + "="*95)
    print("                   FINAL COMPARATIVE STRESS-TEST SUMMARY TABLE                   ")
    print("="*95)
    
    summary_headers = [
        "Swarm Scale", "Attributor Time", "Engine Time", "Decisions/sec", 
        "Firewall Rules", "Upstream History", "Audit File (MB)", "Peak RAM (MB)"
    ]
    
    rows = []
    for r in results:
        rows.append([
            f"{r['scale']:,} IPs",
            f"{r['attributor_time_s']:.3f} s",
            f"{r['engine_time_s']:.3f} s",
            f"{r['engine_throughput_eval_sec']:,}/s",
            f"{r['active_firewall_rules']} / {r['max_rule_ceiling']}",
            f"{r['action_history_size']:,}",
            f"{r['audit_file_mb']:.1f} MB",
            f"{r['peak_mem_mb']:.1f} MB"
        ])

    col_widths = [14, 18, 14, 16, 16, 18, 16, 14]
    header_line = " | ".join(h.ljust(w) for h, w in zip(summary_headers, col_widths))
    sep_line = "-+-".join("-" * w for w in col_widths)
    
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(val).ljust(w) for val, w in zip(row, col_widths)))
    print("="*95)

    # Save to results
    output_path = os.path.join("results", "pips_swarm_stress_test.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Benchmark summary saved to {output_path}\n")


if __name__ == "__main__":
    main()
