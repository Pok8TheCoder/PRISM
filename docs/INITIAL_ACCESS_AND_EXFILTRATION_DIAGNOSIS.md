# Root Cause Analysis: 0% Initial Access & Exfiltration Recall in Lab Evaluation

**Date:** September 12, 2026  
**Target:** `PRISM Laplace Model` (`StreamingLaplaceModel` / Gen 10 Spatio-Temporal World Model + Tier-2 Neuro-Symbolic Engine)  
**Evaluation Reference:** `LAPLACE_LAB_FAIR_EVAL.md` (148 Lab PCAPs, 272 Scored Windows)

---

## 1. Executive Summary

In the fair lab PCAP evaluation across 148 attack captures, the PRISM Laplace Model substantially outperformed previous generations on overall MITRE stage breadth:
- **Laplace MITRE Accuracy:** **14.5%** (vs. **4.0%** for Gen 8 and **2.6%** for Gen 8 + RXI).
- **Binary Attack Accuracy:** **73.9%**.
- **Lateral Movement Recall:** **100.0%**.
- **Impact (DoS) Recall:** **75.0%**.

However, **Initial Access (Stage 2)** and **Exfiltration (Stage 5)** both recorded **0.0% recall**.

This document details the exact mathematical and topological root cause responsible for this failure: **The Docker Subnet Collapse**.

---

## 2. The Root Cause: The "Docker Subnet Collapse"

### Enterprise Topology vs. Containerized Lab Topology
In an enterprise network architecture:
* External attackers originate from public WAN IP addresses (e.g., `203.0.113.x`).
* Corporate servers reside within private RFC 1918 subnets (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`).
* Initial Access represents boundary penetration (`WAN -> LAN`).
* Lateral Movement represents internal pivot traversal (`LAN -> LAN`).
* Exfiltration represents outbound data theft (`LAN -> WAN`).

### What Happens in the Red-Team Lab Testbed
The live IPS lab runs inside Docker Compose (`docker-compose.yml`):
* Attacking Container (`redteam`): `172.18.0.3`
* Target Web Server Container (`target`): `172.18.0.2`
* Background Traffic (`benign-client`): `172.18.0.4`

All containers communicate across the default **Docker Bridge Network (`172.18.0.0/16`)**.

Under standard RFC 1918 private subnet inspection (`ipaddress.ip_address(ip).is_private`):
```python
src_priv = is_private("172.18.0.3")  # -> True
dst_priv = is_private("172.18.0.2")  # -> True
```

This collapses the directional profile across **100% of all 148 lab PCAP files**:
$$\text{is\_wan\_to\_lan} = (\neg \text{src\_priv}) \land \text{dst\_priv} = \textbf{False}$$
$$\text{is\_lan\_to\_wan} = \text{src\_priv} \land (\neg \text{dst\_priv}) = \textbf{False}$$
$$\text{is\_lan\_to\_lan} = \text{src\_priv} \land \text{dst\_priv} = \textbf{True}$$

Because both source and destination IPs are recognized as private subnets, **Tier-2 unconditionally labels every single attack flow as internal East-West (`LAN_TO_LAN`)**.

---

## 3. Why Initial Access Suffered 0% Recall

In [`src/models/streaming_gen10.py`](file:///d:/coding/PRISM/src/models/streaming_gen10.py), Tier-2's attribution rules execute sequentially:

```python
# --- RULE 3: S3 Lateral Movement ---
elif direction == "LAN_TO_LAN" and (
    is_known_lateral_port
    or threat_prob >= suspicion_thresh
    or fwd_bytes > 5000
    or lan_to_lan > 0.50
):
    rule_matched = True
    delta_logits[3] += 12.0
    delta_logits[0] -= 10.0
    delta_logits[2] -= 10.0  # Physically impossible: not ingress
    delta_logits[4] -= 10.0  # Physically impossible: not external C2
    delta_logits[5] -= 10.0  # Physically impossible: not external Exfil
```

### The Failure Cascade
1. **Rule 3 Intercepts the Flow**: Because `direction == "LAN_TO_LAN"` is evaluated *before* port-specific ingress checks, any threat on the Docker network matches Rule 3.
2. **The Hardcoded Penalty**: Rule 3 explicitly assumes that because both endpoints are private, ingress penetration is impossible:
   $$\Delta \text{Logit}(\text{Initial Access}) = -10.0$$
3. **Rule 4 (Initial Access) is Bypassed**: Rule 4 requires `direction == "WAN_TO_LAN"`. Because `direction` is `LAN_TO_LAN`, Rule 4 is never evaluated.

### Empirical Validation
We simulated an active web exploit probe (`T1190` targeting port 80 HTTP) where the Tier-1 neural network gave Initial Access high confidence ($P = 0.85$):
* **Neural Prediction (Tier-1):** Initial Access (Stage 2)
* **Tier-2 Logit Adjustment:** Applied $\Delta = -10.0$ to Stage 2, and $\Delta = +12.0$ to Stage 3.
* **Final Attributed Stage:** **Lateral Movement (Stage 3)** ($P = 0.999987$)
* **Initial Access Probability:** Slashed to $P = 2.06 \times 10^{-9}$ ($0.0000002\%$).

> **Conclusion**: Every external Web Exploit (T1190) and SSH Brute Force (T1110) in the lab was forcibly flipped to Lateral Movement. This explains why **Lateral Movement had 100% recall** while **Initial Access had 0% recall**.

---

## 4. Why Exfiltration Suffered 0% Recall

Exfiltration suffered from two simultaneous blockers:

### A. The `LAN_TO_LAN` Destination Penalty
When the compromised target server exfiltrates stolen credentials or files back to the attacker container (`172.18.0.3`), the destination is still an internal IP on the Docker bridge. Rule 3 triggers and penalizes Exfiltration:
$$\Delta \text{Logit}(\text{Exfiltration}) = -10.0 \quad \text{("Physically impossible: not external Exfil")}$$

### B. Unrealistic Volumetric Thresholds
Rule 5 relies on `has_massive_egress`:
```python
has_massive_egress = bool(
    fwd_bytes > 300_000
    or (fwd_bytes > 40_000 and fwd_bytes > 3.0 * max(bwd_bytes, 1.0))
)
```
In the lab's `T1555_sqli_cred_theft` attack, the data exfiltrated consists of database password hashes or small configuration text files (**typically 2 KB to 5 KB**).
Because $2\text{ KB} \ll 40\text{ KB}$, the volumetric trigger fails to fire, preventing Exfiltration from receiving evidence support.

---

## 5. Engineering Roadmap & Solutions

To bring Initial Access and Exfiltration recall to parity on containerized lab environments without sacrificing enterprise fidelity:

### Fix 1: Decouple Service Ingress from RFC 1918 Address Classification
Instead of assuming all private IPs are internal employee workstations, inspect the **target service role**:
* If `dst_port in {80, 443, 8080, 8443, 8000}` (perimeter web services) or `dst_port in {22, 3389}` (remote ingress gateways) and the incoming traffic exhibits exploit signatures (SQLi, path traversal, brute force), classify as **Initial Access (Stage 2)** regardless of whether the source IP is private.

### Fix 2: Soften Logit Penalties ($\pm 10.0 \to \pm 2.0$)
Dropping logits by $-10.0$ reduces stage probability by a factor of $e^{10} \approx 22,026$, effectively disabling neural predictions.
* Reduce negative evidence offsets to subtle dampening penalties ($\Delta = -1.5$ to $-2.0$).
* This allows strong neural world model activations to override directional heuristics when clear payload signatures are present.

### Fix 3: Implement Micro-Exfiltration Byte Ratios
Lower the volumetric threshold for small credential dumps:
* Trigger Exfiltration if `fwd_bytes > 2,000` (2 KB) and `fwd_bytes > 3.5 * bwd_bytes` when preceded by an access or query stage.
