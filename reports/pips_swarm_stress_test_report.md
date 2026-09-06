# PRISM P-IPS Pipeline Swarm Stress-Test Report (1,000 / 10,000 / 100,000 IPs)

## Executive Summary
This report presents empirical stress-testing benchmarks of the **Preemptive Intrusion Prevention Subsystem (P-IPS)** under coordinated multi-vector adversary swarms consisting of **1,000**, **10,000**, and **100,000** attacking IPs evaluated simultaneously.

---

## 1. What Are the "50 Active Rules"?
In enterprise operating systems (Windows Filtering Platform / Linux `iptables` / `nftables`), each firewall rule represents an active kernel packet inspection entry.

In PRISM, `MAX_ACTIVE_RULES = 50` represents the hard ceiling of active quarantine leases allowed to exist concurrently in the kernel state table:
1. **Why 50?**: When an adversary launches a spoofed DDoS flood with millions of randomized source IPs, injecting a rule for every IP linearly degrades OS packet inspection throughput (every benign packet must traverse thousands of ACL checks) and risks kernel memory exhaustion. 50 rules provides a dense, surgical barrier without measurable packet latency.
2. **Subnet Auto-Collapsing Multiplier**: A single rule is not limited to 1 host. When $\ge 3$ hosts attack from the same `/24` subnet, PRISM collapses them into a single CIDR block (`198.51.100.0/24`). Thus, 50 rules can effectively quarantine up to **$50 \times 254 = 12,700$ attacker hosts**.
3. **Low-Priority Eviction Mechanism**: When the 51st rule arrives, `LeaseManager` sorts active leases by `(risk_score, -created_at)`. It automatically evicts the lowest-threat or oldest rule from the firewall driver and grants the quarantine slot to the high-priority threat.

---

## 2. Benchmark Results Across Swarm Scales

| Metric | 1,000 IPs | 10,000 IPs | 100,000 IPs | Scaling Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **EntityAttributor Time** | 0.018 s | 0.239 s | 4.247 s | **Linear $O(N)$** (23.5k - 54k IPs/s) |
| **EntityAttributor Latency** | 18.4 µs / IP | 23.8 µs / IP | 42.5 µs / IP | Ultra-low microsecond overhead |
| **Probe Cache Entries** | 920 entries | 6,670 entries | 60,670 entries | Grew proportionally to unique IPs |
| **MitigationEngine Time** | 1.563 s | 11.520 s | 117.714 s | Dominated by disk I/O (850 evals/s) |
| **Actions Executed** | 955 | 9,383 | 93,357 | High-risk & unique threats |
| **Actions Suppressed** | 45 | 617 | 6,643 | Deduplicated / already leased |
| **Active Rules in Driver** | **50 / 50** | **50 / 50** | **50 / 50** | **STRICTLY ENFORCED CEILING** |
| **Active Leases in RAM** | **50** | **50** | **50** | Zero firewall leak |
| **Action History (Upstream)** | 955 | 9,383 | 93,357 | **Unbounded $O(N)$ growth** |
| **Strike Tracker (Upstream)**| 906 | 6,690 | 60,726 | **Unbounded $O(N)$ growth** |
| **Audit Logger Events** | 955 CEF | 9,383 CEF | 93,357 CEF | Append-only CEF event logging |
| **Audit Log File Size** | 0.85 MB | 8.37 MB | 83.50 MB | Synchronous file write overhead |
| **Peak Memory Added** | 0.9 MB | 7.6 MB | 74.2 MB | Safe for 100k, but requires ring-buffer |

---

## 3. Does `MAX_ACTIVE_RULES=50` Protect Only the Firewall or Upstream Too?

### The Empirical Verdict:
> **`MAX_ACTIVE_RULES = 50` ONLY protects the firewall layer and the OS kernel packet engine. It does NOT protect the upstream application pipeline.**

### Detailed Architectural Analysis:
1. **Firewall Layer (100% Protected)**:
   - In all three test runs (1k, 10k, 100k), the sandbox driver and `self.leases` never exceeded 50 active rules. Kernel state tables and network packet traversal remain pristine.
2. **Upstream Layer (Vulnerable to Overload at Scale)**:
   - **`self.action_history`** in `MitigationEngine`: Appends every executed action into an in-memory Python list. At 100,000 IPs, this list contained 93,357 instances, consuming ~34 MB of RAM.
   - **`self.strike_records`** in `LeaseManager`: Stores offense timestamps for every unique IP. At 100,000 IPs, it retained 60,726 keys.
   - **`EntityAttributor._probe_cache`**: Retained 60,670 IP-to-port probe history tuples (40.45 MB RAM) because pruning is lazy (only executed when an IP is revisited).
   - **`MitigationAuditLogger`**: Executed 93,357 synchronous `open() -> write() -> close()` cycles, creating an 83.5 MB log file on disk and consuming 85% of total engine processing time.

---

## 4. Batching, Deduplication, & Prioritization Findings

1. **Batching**:
   - **Status: Unbatched**. Threats are evaluated sequentially item-by-item. While `EntityAttributor` handles >23,000 items/sec, `MitigationEngine` drops to ~850 items/sec due to synchronous file write calls inside `audit_logger.log_event()`.
2. **Deduplication**:
   - **Status: Active at Firewall Layer, Inactive Upstream**.
   - Repeated IPs that are currently leased are immediately suppressed (`if self.lease_manager.is_active(culprit.ip): return None`).
   - However, once an IP is evicted to make room for a new rule under the 50-rule ceiling, a subsequent packet from that IP is treated as a new threat, triggering a fresh evaluation and duplicate audit log.
3. **Prioritization**:
   - **Status: Active at Firewall Layer, Inactive Upstream**.
   - When the 50-rule limit is reached, eviction strictly prioritizes threats by risk score (`min(risk_score)` evicted first). Volumetric DDoS and active brute-force attacks retain their firewall drop leases over low-risk port sweeps.
   - Upstream, however, all actions are queued without load-shedding.

---

## 5. Architectural Recommendations for Enterprise 100k+ Production

1. **Ring-Buffer Cap on Upstream History**:
   - Replace `action_history: List` with a fixed-size `collections.deque(maxlen=1000)` to prevent unbounded memory growth during sustained distributed attacks.
2. **Asynchronous / Batch Audit Logging**:
   - Move `MitigationAuditLogger` file I/O to a background worker thread with a queue (or write in 1,000-event chunks). This will increase engine throughput from 850/sec to **>25,000/sec**.
3. **LRU / Bound on Probe and Strike Caches**:
   - Implement an LRU eviction or global IP count ceiling (e.g. `max_cached_ips = 5000`) on `EntityAttributor._probe_cache` and `LeaseManager.strike_records`.
