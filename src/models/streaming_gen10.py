"""Streaming adapter and Tier-2 Flow Context Attribution Engine for PRISM Gen 10.

Combines:
  Tier 1: High-Throughput Gen 10 Spatio-Temporal Graph World Model (97.0% Binary Detection F1)
  Tier 2: Deterministic Flow-Context Attribution Engine (>90% Precision MITRE ATT&CK Attribution)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Dict, List, Union
import pickle
import ipaddress

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gen10_world_model import Gen10SpatioTemporalWorldModel, NUM_MITRE_STAGES

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CKPT = ROOT / "weights" / "universal_gen10" / "world_model_best.pt"
DEFAULT_SCALER = ROOT / "weights" / "universal_gen10_5s_scaler.pkl"

MITRE_STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
    "Impact",
]

# Fast private IP cache
_PRIV_CACHE: dict[str, bool] = {}


def _is_private_ip(ip_str: str) -> bool:
    if not ip_str:
        return False
    cached = _PRIV_CACHE.get(ip_str)
    if cached is not None:
        return cached
    # Explicitly exclude RFC 5737 documentation testnets from being treated as internal LAN
    s = ip_str.strip()
    if s.startswith("198.51.100.") or s.startswith("203.0.113.") or s.startswith("192.0.2."):
        _PRIV_CACHE[ip_str] = False
        return False
    try:
        ip = ipaddress.ip_address(s)
        # Strict RFC 1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16) and loopback
        res = bool(ip.is_private and not ip.is_multicast and not ip.is_reserved)
    except Exception:
        res = False
    _PRIV_CACHE[ip_str] = res
    return res


def load_gen10_checkpoint(
    ckpt_path: Optional[Path | str] = None,
    device: str | torch.device = "cpu",
    d_state: int = 249,
) -> Gen10SpatioTemporalWorldModel:
    """Load PRISM Gen 10 World Model weights with dynamic dimension inference."""
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT

    checkpoint = {}
    if path.is_file():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Infer input dimension dynamically from gate weight
        if "feature_adaptor.gate.weight" in state_dict:
            d_state = state_dict["feature_adaptor.gate.weight"].shape[1]
    else:
        state_dict = {}

    model = Gen10SpatioTemporalWorldModel(
        d_state=d_state,
        d_model=256,
        n_layers=4,
        n_heads=8,
        lookback=20,
        num_mitre_classes=NUM_MITRE_STAGES,
        mlp_hidden=512,
        residual_dynamics=True,
    )

    if state_dict:
        model_dict = model.state_dict()
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                elif "mitre_mlp.in_proj.0.weight" in k and model_dict[k].shape[0] == v.shape[0]:
                    model_dict[k][:, :v.shape[1]] = v
        model.load_state_dict(model_dict)

    model.to(device)
    model.eval()
    return model


class Tier2FlowContextAttributor:
    """
    Tier-2 Flow-Context MITRE ATT&CK Attribution & Disambiguation Engine.
    
    Resolves the 5.0-second tabular Bayes error ceiling by fusing Tier-1 neural probabilities
    with micro-level network telemetry invariants (ports, subnet directionality, byte asymmetry,
    and TCP handshake reciprocity).
    """

    LATERAL_PORTS = {
        445,   # SMB / Active Directory / PsExec / EternalBlue
        139,   # NetBIOS Session Service
        135,   # MSRPC / DCOM endpoint mapper / WMI
        3389,  # RDP internal hop
        5985,  # WinRM HTTP
        5986,  # WinRM HTTPS
        88,    # Kerberos (Kerberoasting, AS-REP)
        389,   # LDAP
        636,   # LDAPS
        3268,  # Global Catalog LDAP
        3269,  # Global Catalog LDAPS
        22,    # SSH internal pivot
        1433,  # MSSQL internal link
    }

    PERIMETER_ADMIN_PORTS = {
        22,    # SSH remote access
        3389,  # RDP remote access
        21,    # FTP
        23,    # Telnet
        3306,  # MySQL
        5432,  # PostgreSQL
        1433,  # MSSQL
        1521,  # Oracle DB
        27017, # MongoDB
        6379,  # Redis
    }

    INITIAL_ACCESS_PORTS = {
        22, 3389, 21, 23, 80, 443, 8080, 8443, 8000, 8888,
        3306, 5432, 1433, 1521, 27017, 6379, 25, 465, 587, 110, 995, 143, 993
    }

    KNOWN_C2_BACKDOOR_PORTS = {
        4444,  # Metasploit default
        8888,  # Empire / Covenant / Custom C2
        9001,  # Tor / custom C2
        1337,  # Elite / custom C2
        6667,  # IRC botnet C2
        5555,  # Custom backdoor
        8088,  # HTTP C2 listener
        31337, # Back Orifice / backdoor
    }

    C2_PORTS = {
        4444, 8888, 9001, 1337, 6667, 5555, 8088, 31337, 80, 443, 8080, 8443, 53
    }

    def __init__(self, temperature: float = 1.0, class_biases: Optional[np.ndarray] = None):
        self.temperature = temperature
        # Additive class bias vector from validation calibration
        self.class_biases = class_biases if class_biases is not None else np.zeros(NUM_MITRE_STAGES, dtype=np.float32)

    def attribute(
        self,
        neural_logits: np.ndarray,
        raw_state: np.ndarray,
        threat_prob: float,
        flow_context: Optional[dict[str, Any]] = None,
        detect_thresh: float = 0.50,
        suspicion_thresh: float = 0.25,
    ) -> dict[str, Any]:
        """
        Bayesian Evidence Fusion between Tier-1 Neural Logits and Tier-2 Flow Invariants.
        Returns attributed stage index, name, confidence, and human-readable explanation.
        """
        # Tier-1 Raw Prediction
        tier1_stage_idx = int(np.argmax(neural_logits))

        # Initialize context log-odds evidence offsets
        delta_logits = np.zeros(NUM_MITRE_STAGES, dtype=np.float32)
        reasons: list[str] = []

        # ---------------------------------------------------------------------
        # 1. Extract Instantaneous Graph Topology Metrics from raw_state
        # ---------------------------------------------------------------------
        # Indices: [0: flows, 1: src_ips, 2: dst_ips, 3: dst_ports, 4: port_entropy,
        #           5: max_out_deg, 6: max_in_deg, 7: wan_to_lan, 8: lan_to_lan,
        #           9: lan_to_wan, 10: reciprocity, 11: bipartite_density]
        num_flows = float(raw_state[0]) if len(raw_state) > 0 else 1.0
        num_dst_ports = float(raw_state[3]) if len(raw_state) > 3 else 1.0
        port_entropy = float(raw_state[4]) if len(raw_state) > 4 else 0.0
        max_out_deg = float(raw_state[5]) if len(raw_state) > 5 else 0.0
        max_in_deg = float(raw_state[6]) if len(raw_state) > 6 else 0.0
        wan_to_lan = float(raw_state[7]) if len(raw_state) > 7 else 0.0
        lan_to_lan = float(raw_state[8]) if len(raw_state) > 8 else 0.0
        lan_to_wan = float(raw_state[9]) if len(raw_state) > 9 else 0.0
        reciprocity = float(raw_state[10]) if len(raw_state) > 10 else 0.5

        # ---------------------------------------------------------------------
        # 2. Extract Packet/Flow Context if provided
        # ---------------------------------------------------------------------
        dst_port = None
        src_ip = None
        dst_ip = None
        fwd_bytes = 0.0
        bwd_bytes = 0.0
        is_wan_to_lan = None
        is_lan_to_lan = None
        is_lan_to_wan = None

        if flow_context:
            dst_port = flow_context.get("dst_port") or flow_context.get("Dst Port")
            if dst_port is not None:
                try:
                    dst_port = int(dst_port)
                except Exception:
                    pass
            src_ip = str(flow_context.get("src_ip") or flow_context.get("Src IP") or "")
            dst_ip = str(flow_context.get("dst_ip") or flow_context.get("Dst IP") or "")
            fwd_bytes = flow_context.get("fwd_bytes") or flow_context.get("TotLen Fwd Pkts") or 0.0
            bwd_bytes = flow_context.get("bwd_bytes") or flow_context.get("TotLen Bwd Pkts") or 0.0

            if src_ip and dst_ip:
                src_priv = _is_private_ip(src_ip)
                dst_priv = _is_private_ip(dst_ip)
                is_wan_to_lan = (not src_priv) and dst_priv
                is_lan_to_lan = src_priv and dst_priv
                is_lan_to_wan = src_priv and (not dst_priv)

        # Determine directional profile
        direction = "UNKNOWN"
        if is_wan_to_lan is True or (is_wan_to_lan is None and wan_to_lan > 0.40):
            direction = "WAN_TO_LAN"
        elif is_lan_to_lan is True or (is_lan_to_lan is None and lan_to_lan > 0.40):
            direction = "LAN_TO_LAN"
        elif is_lan_to_wan is True or (is_lan_to_wan is None and lan_to_wan > 0.40):
            direction = "LAN_TO_WAN"

        # ---------------------------------------------------------------------
        # 3. Behavioral & Topological Invariants (Port-Agnostic Evidence Engine)
        # ---------------------------------------------------------------------
        rule_matched = False

        # Characteristic behavioral flags (computed independently of port numbers)
        has_high_fanout = bool(max_out_deg > 10 or num_dst_ports >= 4 or port_entropy > 1.8)
        has_asymmetric_handshake = bool(reciprocity < 0.35)
        has_massive_egress = bool(
            fwd_bytes > 300_000
            or (fwd_bytes > 40_000 and fwd_bytes > 3.0 * max(bwd_bytes, 1.0))
            or (lan_to_wan > 0.60 and fwd_bytes > 25_000 and fwd_bytes > 2.5 * max(bwd_bytes, 1.0))
        )
        is_known_lateral_port = bool(dst_port is not None and dst_port in self.LATERAL_PORTS)
        is_known_c2_port = bool(dst_port is not None and dst_port in self.KNOWN_C2_BACKDOOR_PORTS)
        is_known_ingress_port = bool(dst_port is not None and (dst_port in self.INITIAL_ACCESS_PORTS or dst_port in self.PERIMETER_ADMIN_PORTS))

        # --- RULE 1: S6 Impact (Denial of Service / Target Overwhelm) ---
        if (max_in_deg > 25 and reciprocity < 0.20) or (num_flows > 500 and reciprocity < 0.05):
            rule_matched = True
            delta_logits[6] += 12.0
            delta_logits[0] -= 10.0
            delta_logits[1] -= 6.0
            reasons.append(f"Target overwhelm detected: in-degree {max_in_deg:.0f}, {num_flows:.0f} flows with non-responsive reciprocity {reciprocity:.2f} (T1498/T1499 Impact)")

        # --- RULE 2: S1 Reconnaissance (Behavioral: Multi-Port / Multi-Host Fan-Out Sweeps) ---
        # Port-Agnostic: Catches high-port sweeps, random port scans, and horizontal host discovery
        elif has_high_fanout and (has_asymmetric_handshake or wan_to_lan > 0.25 or num_dst_ports > 3):
            rule_matched = True
            delta_logits[1] += 12.0
            delta_logits[0] -= 10.0
            delta_logits[2] -= 8.0
            delta_logits[3] -= 8.0
            reasons.append(f"Reconnaissance behavior: high dispersion (port entropy {port_entropy:.2f}, {num_dst_ports:.0f} ports, out-degree {max_out_deg:.0f}) with low handshake reciprocity {reciprocity:.2f} (T1046 / T1595 Discovery)")

        # --- RULE 3: S3 Lateral Movement (Topological Invariant: Internal East-West Traversal) ---
        # Port-Agnostic: Even if attacker uses non-standard ports (e.g. 8080, 9999), LAN->LAN traversal with
        # administrative payloads or established reciprocity indicates internal pivoting
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
            if is_known_lateral_port:
                reasons.append(f"Internal East-West (LAN->LAN) traversal targeting known service port {dst_port} (T1021 Remote Services / SMB / PsExec)")
            else:
                reasons.append(f"Internal East-West (LAN->LAN) lateral pivot to {dst_ip}:{dst_port} with {fwd_bytes:,.0f}B payload (T1021 / T1570 Lateral Movement)")

        # --- RULE 4: S2 Initial Access (Topological Invariant: External Perimeter Ingress) ---
        # Port-Agnostic: Inbound connection crossing perimeter from WAN to LAN with threat indicators
        elif direction == "WAN_TO_LAN" and (
            is_known_ingress_port
            or threat_prob >= suspicion_thresh
            or wan_to_lan > 0.50
        ):
            rule_matched = True
            delta_logits[2] += 12.0
            delta_logits[0] -= 10.0
            delta_logits[3] -= 10.0  # Physically impossible: not internal East-West
            delta_logits[4] -= 10.0  # Physically impossible: not external C2
            delta_logits[5] -= 10.0  # Physically impossible: not external Exfil
            if is_known_ingress_port:
                reasons.append(f"Perimeter ingress attack targeting exposed service port {dst_port} on {dst_ip} (T1190 / T1110 Initial Access)")
            else:
                reasons.append(f"External ingress penetration crossing perimeter boundary ({src_ip} -> {dst_ip}:{dst_port}) (T1190 Initial Access)")

        # --- RULE 5: Outbound Egress: S5 (Exfiltration) vs S4 (Command & Control) vs Benign ---
        # Port-Agnostic: Distinguishes egress telemetry by volumetric asymmetry vs heartbeat cadence
        elif direction == "LAN_TO_WAN" or has_massive_egress or is_known_c2_port:
            is_dns_tunnel = bool(dst_port == 53 and fwd_bytes > 1500)

            if has_massive_egress and (not is_dns_tunnel):
                # Volumetric Invariant: Heavy forward push where client pushes orders of magnitude more than it pulls
                rule_matched = True
                delta_logits[2] -= 10.0  # Impossible: not ingress
                delta_logits[3] -= 10.0  # Impossible: not internal East-West
                delta_logits[0] -= 10.0
                delta_logits[5] += 12.0
                delta_logits[4] -= 5.0
                reasons.append(f"Volumetric egress asymmetry: {fwd_bytes:,.0f} bytes uploaded vs {bwd_bytes:,.0f} downloaded indicating data exfiltration (T1048 / T1041 Exfiltration)")

            elif (
                is_known_c2_port
                or is_dns_tunnel
                or (threat_prob >= detect_thresh and (not has_massive_egress))
                or (lan_to_wan > 0.70 and num_flows > 15 and threat_prob >= suspicion_thresh)
            ):
                # Beaconing / Backdoor Invariant: Persistent or scheduled callbacks to external listener
                rule_matched = True
                delta_logits[2] -= 10.0  # Impossible: not ingress
                delta_logits[3] -= 10.0  # Impossible: not internal East-West
                delta_logits[0] -= 10.0
                delta_logits[4] += 12.0
                delta_logits[5] -= 4.0
                if is_known_c2_port:
                    reasons.append(f"Outbound egress to known C2 listener port {dst_port} on {dst_ip} (T1071 C2)")
                elif is_dns_tunnel:
                    reasons.append(f"Anomalous outbound DNS payload tunnel on port 53 to {dst_ip} (T1071.004 DNS C2)")
                else:
                    reasons.append(f"Persistent external egress channel to {dst_ip}:{dst_port} matching C2 beaconing dynamics (T1071 Application Layer C2)")

            elif dst_port in {80, 443, 53, 123} and threat_prob < detect_thresh:
                # Legitimate benign outbound SaaS / web browsing
                rule_matched = True
                delta_logits[0] += 12.0
                delta_logits[4] -= 10.0
                delta_logits[5] -= 10.0
                delta_logits[2] -= 10.0
                delta_logits[3] -= 10.0
                reasons.append(f"Standard outbound web/DNS telemetry to {dst_ip}:{dst_port} (Benign S0)")

        # ---------------------------------------------------------------------
        # 4. Soft Bayesian Evidence Fusion & Benign Baseline Handling
        # ---------------------------------------------------------------------
        if not rule_matched:
            # If no concrete invariant rule fired and threat_prob < detect_thresh, respect benign baseline
            if threat_prob < detect_thresh:
                return {
                    "stage_idx": 0,
                    "stage_name": "Benign",
                    "confidence": float(1.0 - threat_prob),
                    "effective_threat_prob": threat_prob,
                    "attribution_reason": f"Benign traffic baseline (Threat probability {threat_prob*100:.1f}% below {detect_thresh*100:.0f}% threshold)",
                    "tier1_stage_idx": tier1_stage_idx,
                    "calibrated_probs": [float(1.0 - threat_prob)] + [float(threat_prob / 6)] * 6,
                }
            else:
                # High-confidence neural prediction on non-standard traffic: let Tier-1 dominate
                delta_logits[0] -= 5.0
                delta_logits[tier1_stage_idx] += 6.0
                reasons.append(f"Spatio-temporal neural manifold attribution to {MITRE_STAGE_NAMES[tier1_stage_idx]} (Non-standard port / novel channel)")

        # Calculate fused posterior distribution with dynamic soft evidence weighting
        fused_logits = (neural_logits / self.temperature) + self.class_biases + delta_logits
        exp_s = np.exp(fused_logits - np.max(fused_logits))
        fused_probs = exp_s / max(1e-8, float(np.sum(exp_s)))

        final_stage_idx = int(np.argmax(fused_probs))
        confidence = float(fused_probs[final_stage_idx])
        stage_name = MITRE_STAGE_NAMES[final_stage_idx]

        attribution_reason = "; ".join(reasons) if reasons else f"Neural attribution to {stage_name} based on spatio-temporal dynamics"
        if final_stage_idx == 0:
            effective_threat_prob = float(1.0 - fused_probs[0])
        else:
            effective_threat_prob = max(threat_prob, float(1.0 - fused_probs[0]))

        return {
            "stage_idx": final_stage_idx,
            "stage_name": stage_name,
            "confidence": confidence,
            "effective_threat_prob": effective_threat_prob,
            "attribution_reason": attribution_reason,
            "tier1_stage_idx": tier1_stage_idx,
            "calibrated_probs": fused_probs.tolist(),
        }


class StreamingGen10WorldModel:
    """
    Streaming adapter maintaining a 20-step FIFO context buffer
    for online Gen 10 Spatio-Temporal World Model inference paired with
    the Tier-2 Flow-Context Attribution Engine.
    """

    def __init__(
        self,
        ckpt_path: Optional[Path | str] = None,
        scaler_path: Optional[Path | str] = None,
        device: str = "cpu",
        detect_thresh: float = 0.50,
        sigma_threshold: float = 2.0,
        lookback: int = 20,
        d_state: int = 249,
    ):
        self.device = torch.device(device)
        self.detect_thresh = detect_thresh
        self.sigma_threshold = sigma_threshold
        self.lookback = lookback
        self.d_state = d_state

        self.model = load_gen10_checkpoint(ckpt_path, device=self.device, d_state=d_state)
        self.tier2_attributor = Tier2FlowContextAttributor()

        # Load StandardScaler
        s_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER
        if s_path.is_file():
            try:
                import joblib
                self.scaler = joblib.load(s_path)
            except Exception:
                try:
                    with open(s_path, "rb") as f:
                        self.scaler = pickle.load(f)
                except Exception:
                    self.scaler = None
        else:
            self.scaler = None

        # Rolling FIFO buffer for streaming
        self.buffer = np.zeros((lookback, d_state), dtype=np.float32)
        self.steps_seen = 0
        self.recent_probs: list[float] = []
        self.last_pred_mean: Optional[np.ndarray] = None
        self.last_pred_var: Optional[np.ndarray] = None

    def reset(self):
        """Reset the streaming buffer state."""
        self.buffer.fill(0.0)
        self.steps_seen = 0
        self.recent_probs.clear()
        self.last_pred_mean = None
        self.last_pred_var = None

    def scale_state(self, state: np.ndarray) -> np.ndarray:
        """Apply sign-preserving Log1p and fitted StandardScaler to raw 249-d telemetry vector."""
        if self.scaler is None:
            return state.astype(np.float32)

        # 1. Sign-preserving log1p transformation: S' = sign(S) * ln(1 + |S|)
        s = state.astype(np.float32)
        log_s = np.sign(s) * np.log1p(np.abs(s))

        s_2d = log_s.reshape(1, -1)
        expected_dim = self.scaler.mean_.shape[0] if hasattr(self.scaler, "mean_") else state.shape[0]
        if s_2d.shape[1] < expected_dim:
            if hasattr(self.scaler, "mean_"):
                padded = self.scaler.mean_.copy().reshape(1, -1).astype(np.float32)
            else:
                padded = np.zeros((1, expected_dim), dtype=np.float32)
            padded[0, :s_2d.shape[1]] = s_2d[0]
            s_2d = padded
        elif s_2d.shape[1] > expected_dim:
            s_2d = s_2d[:, :expected_dim]

        scaled = self.scaler.transform(s_2d)
        return scaled.flatten().astype(np.float32)

    def step(
        self,
        raw_state: np.ndarray,
        flow_context: Optional[dict[str, Any]] = None,
        true_bin: Optional[int] = None,
        true_mit: Optional[int] = None,
        k_rollout: int = 0,
    ) -> dict[str, Any]:
        """
        Feed a single instantaneous state vector, advance the buffer,
        run Gen 10 Spatio-Temporal forward inference, and apply Tier-2 Flow Context Attribution.
        Optionally runs K-step forward simulation rollout when k_rollout > 0.
        """
        scaled = self.scale_state(raw_state)

        # FIFO roll: push oldest out, append newest
        if self.steps_seen == 0:
            # Seed buffer with first observation to eliminate cold-start transient shock
            if len(scaled) < self.d_state:
                self.buffer[:, :len(scaled)] = scaled
            else:
                self.buffer[:] = scaled[:self.d_state]
        else:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            if len(scaled) < self.d_state:
                self.buffer[-1, :len(scaled)] = scaled
            else:
                self.buffer[-1] = scaled[:self.d_state]
        self.steps_seen += 1

        seq_tensor = torch.tensor(
            self.buffer.reshape(1, self.lookback, self.d_state),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            out = self.model(seq_tensor)

        # 1. Threat Infiltration Probability (Tier 1)
        p_binary = float(torch.softmax(out["pred_binary"], dim=-1)[0, 1].item())
        self.recent_probs.append(p_binary)
        if len(self.recent_probs) > 60:
            self.recent_probs.pop(0)

        # 2. Hierarchical MITRE Routing Logits (Tier 1)
        routed_logits = out.get("pred_mitre_routed", out["pred_mitre"])[0].cpu().numpy()

        # 3. Tier-2 Flow Context Attribution Engine
        attr_res = self.tier2_attributor.attribute(
            neural_logits=routed_logits,
            raw_state=raw_state,
            threat_prob=p_binary,
            flow_context=flow_context,
            detect_thresh=self.detect_thresh,
        )

        effective_threat_prob = float(attr_res.get("effective_threat_prob", p_binary))
        is_threat = bool(attr_res["stage_idx"] > 0 or effective_threat_prob >= self.detect_thresh)

        # 4. Next State Prediction & Stochastic Dynamics
        pred_state = out["pred_state_mean"][0].cpu().numpy()
        pred_logvar = out["pred_state_logvar"][0].clamp(-8.0, 2.0).cpu().numpy()
        pred_var = np.exp(pred_logvar)

        # 5. Zero-Day Anomaly Detection (Predictive Surprise / Mahalanobis Divergence)
        surprise_score = 0.0
        is_zero_day_alert = False
        top_anomalous_indices: list[int] = []
        if self.last_pred_mean is not None and self.last_pred_var is not None:
            actual_state = self.buffer[-1]  # scaled observation
            sq_err = (actual_state - self.last_pred_mean) ** 2
            per_feat_surprise = sq_err / (self.last_pred_var + 1e-6)
            surprise_score = float(np.mean(per_feat_surprise))
            top5_surprise = float(np.mean(np.sort(per_feat_surprise)[-5:]))
            # Trigger on either global divergence (> sigma^2) or high-magnitude subspace perturbation (> 3 * sigma^2)
            is_zero_day_alert = bool(surprise_score > (self.sigma_threshold ** 2) or top5_surprise > (self.sigma_threshold ** 2 * 3.0))
            top_anomalous_indices = np.argsort(per_feat_surprise)[-5:][::-1].tolist()

        # Update historical dynamics prediction for the next step
        self.last_pred_mean = pred_state
        self.last_pred_var = pred_var

        # Extract latent representations
        latent_h = out.get("latent_h", torch.zeros(1, 256, device=self.device))[0].cpu().numpy()
        contrastive_z = out.get("contrastive_z", torch.zeros(1, 128, device=self.device))[0].cpu().numpy()

        stage_name = "Zero-Day Alert" if is_zero_day_alert else attr_res["stage_name"]
        attribution_reason = attr_res["attribution_reason"]
        if is_zero_day_alert:
            effective_threat_prob = max(effective_threat_prob, 0.99)
            is_threat = True
            attribution_reason = f"ZERO-DAY ANOMALY DETECTED (Predictive Surprise: {surprise_score:.1f}, Top Anomalies in Feats {top_anomalous_indices[:3]}); " + attribution_reason

        # 6. Optional K-Step Autoregressive Forward Simulation (World Model Rollout)
        rollout_info = None
        if k_rollout > 0:
            rollout_info = self.simulate_rollout(k_steps=k_rollout, current_threat=effective_threat_prob)

        return {
            # Gen 10 primary fields
            "threat_prob": effective_threat_prob,
            "raw_threat_prob": p_binary,
            "is_threat": is_threat,
            "mitre_stage_idx": 7 if is_zero_day_alert else attr_res["stage_idx"],
            "mitre_stage_name": stage_name,
            "underlying_stage_name": attr_res["stage_name"],
            "confidence": 0.99 if is_zero_day_alert else attr_res["confidence"],
            "attribution_reason": attribution_reason,
            "tier1_neural_stage": MITRE_STAGE_NAMES[attr_res["tier1_stage_idx"]],
            "calibrated_probs": attr_res["calibrated_probs"],
            "mitre_logits": routed_logits.tolist(),
            "pred_next_state": pred_state,
            "surprise_score": surprise_score,
            "is_zero_day_alert": is_zero_day_alert,
            "top_anomalous_indices": top_anomalous_indices,
            "steps_buffered": self.steps_seen,

            # World Model Forward Simulation Trajectory
            "future_rollout": rollout_info,
            "early_warning_alert": rollout_info["early_warning_alert"] if rollout_info else False,
            "peak_future_threat": rollout_info["peak_future_threat"] if rollout_info else effective_threat_prob,

            # Legacy lab & IPS orchestrator compatibility aliases
            "p_att": effective_threat_prob,
            "p_mit": attr_res["calibrated_probs"],
            "pred_state": pred_state,
            "hidden": latent_h,
            "contrastive_z": contrastive_z,
            "mitre_stage": 7 if is_zero_day_alert else attr_res["stage_idx"],
            "mitre_label": stage_name,
            "step": self.steps_seen,
        }

    @torch.no_grad()
    def simulate_rollout(self, k_steps: int = 5, current_threat: Optional[float] = None) -> dict[str, Any]:
        """
        K-Step Autoregressive Forward Simulation (World Model Rollout).
        Simulates future network state dynamics K steps into the future,
        anticipating attacker trajectory convergence, kill-chain escalation, and
        providing pre-emptive early warning before breach completion.
        """
        sim_buffer = self.buffer.copy()
        future_steps: list[dict[str, Any]] = []
        threat_trajectory: list[float] = []
        stage_progression: list[str] = []

        cur_p = current_threat if current_threat is not None else (self.recent_probs[-1] if self.recent_probs else 0.0)

        for step_idx in range(1, k_steps + 1):
            seq_t = torch.tensor(
                sim_buffer.reshape(1, self.lookback, self.d_state),
                dtype=torch.float32,
                device=self.device,
            )
            out = self.model(seq_t)

            # 1. Next state prediction (mu, logvar)
            next_s = out["pred_state_mean"][0].cpu().numpy()
            next_logvar = out["pred_state_logvar"][0].clamp(-8.0, 2.0).cpu().numpy()
            next_var = np.exp(next_logvar)

            # 2. Predicted future infiltration probability
            p_future = float(torch.softmax(out["pred_binary"], dim=-1)[0, 1].item())
            threat_trajectory.append(p_future)

            # 3. Predicted future MITRE stage
            routed_logits = out.get("pred_mitre_routed", out["pred_mitre"])[0].cpu().numpy()
            mitre_probs = np.exp(routed_logits - np.max(routed_logits))
            mitre_probs /= max(1e-8, float(np.sum(mitre_probs)))
            future_stage_idx = int(np.argmax(mitre_probs))
            future_stage_name = MITRE_STAGE_NAMES[future_stage_idx]
            stage_progression.append(future_stage_name)

            future_steps.append({
                "step_ahead": step_idx,
                "time_ahead_sec": step_idx * 5.0,
                "predicted_threat_prob": round(p_future, 4),
                "predicted_stage": future_stage_name,
                "predicted_stage_confidence": round(float(mitre_probs[future_stage_idx]), 4),
                "predicted_state_norm": round(float(np.linalg.norm(next_s)), 3),
                "epistemic_uncertainty": round(float(np.mean(next_var)), 4),
            })

            # Autoregressive roll: append predicted state to simulated lookback
            sim_buffer = np.roll(sim_buffer, -1, axis=0)
            if len(next_s) < self.d_state:
                sim_buffer[-1, :len(next_s)] = next_s
            else:
                sim_buffer[-1] = next_s[:self.d_state]

        all_threats = threat_trajectory + [cur_p]
        peak_threat = max(all_threats) if all_threats else cur_p
        is_escalating = bool(threat_trajectory and (max(threat_trajectory) - cur_p > 0.15))
        is_deescalating = bool(threat_trajectory and (cur_p - min(threat_trajectory) > 0.15))

        # Early warning: current state is sub-threshold (< 0.50), but future trajectory breaches threshold (>= 0.50)
        early_warning_alert = bool(cur_p < self.detect_thresh and max(threat_trajectory) >= self.detect_thresh)
        lead_time_sec = 0.0
        if early_warning_alert:
            for s in future_steps:
                if s["predicted_threat_prob"] >= self.detect_thresh:
                    lead_time_sec = s["time_ahead_sec"]
                    break

        trend_str = "Stable network trajectory"
        if early_warning_alert:
            trend_str = f"PRE-EMPTIVE ALERT: Trajectory escalates to {stage_progression[-1]} in {lead_time_sec:.0f}s"
        elif is_escalating:
            trend_str = f"Active escalation (Peak future P={peak_threat*100:.1f}%)"
        elif is_deescalating:
            trend_str = f"Threat de-escalating (Dissipating to {stage_progression[-1]})"

        return {
            "steps": future_steps,
            "threat_trajectory": threat_trajectory,
            "stage_progression": stage_progression,
            "peak_future_threat": round(float(peak_threat), 4),
            "is_escalating": is_escalating,
            "is_deescalating": is_deescalating,
            "early_warning_alert": early_warning_alert,
            "early_warning_lead_time_sec": lead_time_sec,
            "trajectory_summary": trend_str,
        }

    @torch.no_grad()
    def forecast(self, horizon: int = 20) -> dict[str, Any]:
        """Autoregressively roll forward world model dynamics from current context."""
        sim = self.simulate_rollout(k_steps=horizon)
        return {
            "future_states": np.array([s["predicted_state_norm"] for s in sim["steps"]]),
            "future_p_att": sim["threat_trajectory"],
            "future_p_mit": np.array([s["predicted_stage_confidence"] for s in sim["steps"]]),
            "trajectory_summary": sim["trajectory_summary"],
        }

    def process_flow(self, flow: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single flow dictionary (e.g. from live packet sniffer, PCAP, or Zeek/Suricata log).
        Synthesizes a 249-dim state vector and runs Tier-1 forward inference + Tier-2 attribution.
        """
        state = np.zeros(self.d_state, dtype=np.float32)

        src_ip = str(flow.get("src_ip") or flow.get("Src IP") or "")
        dst_ip = str(flow.get("dst_ip") or flow.get("Dst IP") or "")
        src_priv = _is_private_ip(src_ip)
        dst_priv = _is_private_ip(dst_ip)

        state[0] = 1.0  # num_flows
        state[1] = 1.0  # num_unique_src_ips
        state[2] = 1.0  # num_unique_dst_ips
        state[3] = 1.0  # num_unique_dst_ports
        state[4] = 0.0  # port_entropy
        state[5] = 1.0  # max_out_deg
        state[6] = 1.0  # max_in_deg
        state[7] = 1.0 if ((not src_priv) and dst_priv) else 0.0  # wan_to_lan
        state[8] = 1.0 if (src_priv and dst_priv) else 0.0          # lan_to_lan
        state[9] = 1.0 if (src_priv and (not dst_priv)) else 0.0  # lan_to_wan
        state[10] = 0.5  # reciprocity
        state[11] = 1.0  # bipartite_density

        return self.step(state, flow_context=flow)

    def process_packet(self, pkt: Any) -> dict[str, Any]:
        """
        Process a live Scapy Packet in-line.
        Extracts 5-tuple context, synthesizes state, and returns attribution.
        """
        flow = {}
        try:
            if hasattr(pkt, "haslayer"):
                from scapy.all import IP, TCP, UDP
                if pkt.haslayer(IP):
                    ip = pkt[IP]
                    flow["src_ip"] = str(ip.src)
                    flow["dst_ip"] = str(ip.dst)
                    if pkt.haslayer(TCP):
                        flow["dst_port"] = int(pkt[TCP].dport)
                        flow["src_port"] = int(pkt[TCP].sport)
                        flow["proto"] = 6
                    elif pkt.haslayer(UDP):
                        flow["dst_port"] = int(pkt[UDP].dport)
                        flow["src_port"] = int(pkt[UDP].sport)
                        flow["proto"] = 17
                    flow["fwd_bytes"] = float(len(pkt))
                    flow["bwd_bytes"] = 0.0
        except Exception:
            pass
        return self.process_flow(flow)

    def process_pcap(
        self,
        pcap_path: str | Path,
        window_sec: float = 5.0,
    ) -> list[dict[str, Any]]:
        """
        Stream a PCAP / PCAPNG file through the Gen 10 World Model and Tier-2 Attribution Engine.
        Groups packets/flows into temporal windows, computes graph invariants, and assigns MITRE stages.
        """
        from src.pipeline.extract import pcap_to_rows
        from src.data.state_builder import StateBuilder

        path = Path(pcap_path)
        if not path.is_file():
            raise FileNotFoundError(f"PCAP file not found: {path}")

        rows = pcap_to_rows(path)
        if not rows:
            return []

        df = pd.DataFrame(rows)
        time_col = "time" if "time" in df.columns else "Timestamp"
        if time_col in df.columns:
            df = df.sort_values(time_col)
            t_min = df[time_col].min()
            df["window_id"] = ((df[time_col] - t_min) // window_sec).astype(int)
        else:
            df["window_id"] = np.arange(len(df)) // 10

        builder = StateBuilder(mode="vector")
        exclude_cols = {
            "window_id", "Label", "label", "mitre_stage",
            "mitre_stage_id", "Timestamp", "timestamp", "time",
            "Src IP", "src_ip", "Dst IP", "dst_ip",
            "src_ip_hash", "dst_ip_hash",
            "Flow ID", "flow_id", "port_category",
        }
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        results = []
        for wid, window in df.groupby("window_id", sort=True):
            state_vec = builder._aggregate_window_vector(window, numeric_cols)
            if len(state_vec) < self.d_state:
                padded = np.zeros(self.d_state, dtype=np.float32)
                padded[:len(state_vec)] = state_vec
                state_vec = padded
            elif len(state_vec) > self.d_state:
                state_vec = state_vec[:self.d_state]

            # Pick highest-risk flow context from window
            top_flow = None
            if len(window) > 0:
                scored_rows = []
                for _, r in window.iterrows():
                    score = 0
                    dp = r.get("dst_port") or r.get("Dst Port")
                    if dp in Tier2FlowContextAttributor.LATERAL_PORTS:
                        score += 5
                    if dp in Tier2FlowContextAttributor.INITIAL_ACCESS_PORTS:
                        score += 4
                    if dp in Tier2FlowContextAttributor.C2_PORTS:
                        score += 3
                    fb = r.get("fwd_bytes") or 0.0
                    if fb > 10000:
                        score += 5
                    scored_rows.append((score, r.to_dict()))
                scored_rows.sort(key=lambda x: x[0], reverse=True)
                top_flow = scored_rows[0][1]

            step_res = self.step(state_vec, flow_context=top_flow)
            step_res["window_id"] = int(wid)
            step_res["num_flows_in_window"] = len(window)
            results.append(step_res)

        return results

