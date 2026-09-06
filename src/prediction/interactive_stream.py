"""
Interactive Stream Generator (V2 Architecture - 286 Dimensions):
Extracts realistic 15-second multi-phase attack sequences
(Benign -> Suspicious Onset -> Active Attack -> Post-Attack Recovery)
for the Live Interactive Network Stream Simulator across Enterprise & IoT environments.
"""

import numpy as np
import os
from typing import Dict, List, Tuple

class ScenarioGenerator:
    """Provides curated attack scenarios with realistic temporal progression."""

    @staticmethod
    def get_available_scenarios() -> List[str]:
        return [
            "Scenario A: Stealth Reconnaissance → SSH Brute Force Infiltration (CIC-IDS2018)",
            "Scenario B: Low-and-Slow PortScan → Volumetric DDoS Attack (CIC-IDS2017)",
            "Scenario C: IoT Botnet Infiltration → Mirai DDoS Surge (CICIoT2023)",
            "Scenario D: Web Application Exploit → Data Exfiltration (UNSW-NB15)",
            "Scenario E: Pure Benign Enterprise Day (Baseline Stability)",
            "Scenario F: Custom Interactive Attack Injector (User Controlled)"
        ]

    @staticmethod
    def generate_scenario_states(
        scenario_name: str,
        base_states: np.ndarray,
        base_atks: np.ndarray,
        base_mitres: np.ndarray,
        user_attack_start: int = 20,
        user_attack_intensity: float = 0.85,
        user_attack_stage: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, any]]:
        """
        Returns a sequence of 50 fifteen-second states with smooth progression.
        """
        seq_len = 50
        d_state = base_states.shape[1] if len(base_states) > 0 else 292
        
        # Default benign background from earlier slice of real states
        if len(base_states) >= seq_len:
            benign_idx = np.where(base_atks == 0)[0]
            if len(benign_idx) >= seq_len:
                bg_states = base_states[benign_idx[:seq_len]].copy()
            else:
                bg_states = base_states[:seq_len].copy()
        else:
            bg_states = np.random.randn(seq_len, d_state).astype(np.float32)

        states = bg_states.copy()
        atks = np.zeros(seq_len, dtype=np.int64)
        mitres = np.zeros(seq_len, dtype=np.int64)
        metadata = {}

        if "Scenario A" in scenario_name:
            # Real SSH Brute Force (Patator from 02-14-2018)
            metadata = {
                "name": "SSH Brute Force Infiltration (Real CIC-IDS2018 Capture)",
                "recon_window": (12, 18),
                "attack_window": (19, 36),
                "target_port": 22,
                "target_protocol": "SSH",
                "attacker_ip": "192.168.10.50",
                "description": "Real SSH dictionary brute-force capture: Attacker probes port 22, attempts rapid credential spray, and breaches the host."
            }
            # Find real Stage 2 (Initial Access) windows in base_states
            stage2_idx = np.where(base_mitres == 2)[0]
            if len(stage2_idx) >= 18:
                real_atk_slice = base_states[stage2_idx[:18]]
                states[19:37] = real_atk_slice
            atks[19:37] = 1
            mitres[12:19] = 1
            mitres[19:37] = 2

        elif "Scenario B" in scenario_name:
            # Real Volumetric DDoS Flood (CIC-IDS2017)
            metadata = {
                "name": "Volumetric DDoS Packet Flood (Real Friday-DDos Capture)",
                "recon_window": (10, 16),
                "attack_window": (17, 38),
                "target_port": 80,
                "target_protocol": "HTTP",
                "attacker_ip": "192.168.10.14",
                "description": "Real LOIC/HOIC volumetric packet flood capture: Extreme SYN burst converging on web server gateway."
            }
            stage6_idx = np.where(base_mitres == 6)[0]
            if len(stage6_idx) >= 22:
                real_atk_slice = base_states[stage6_idx[:22]]
                states[17:39] = real_atk_slice
            atks[17:39] = 1
            mitres[10:17] = 1
            mitres[17:39] = 6

        elif "Scenario C" in scenario_name:
            # Real IoT Botnet Mirai Flood (CICIoT2023)
            metadata = {
                "name": "IoT Botnet Mirai & Flood Surge (Real CICIoT2023 Capture)",
                "recon_window": (8, 14),
                "attack_window": (15, 35),
                "target_port": 23,
                "target_protocol": "Telnet/Mirai",
                "attacker_ip": "192.168.1.105",
                "description": "Real Mirai IoT botnet swarm: Compromised devices launch coordinated high-rate UDP flood against edge gateway."
            }
            iot_idx = np.where(base_atks > 0)[0]
            if len(iot_idx) >= 21:
                real_atk_slice = base_states[iot_idx[:21]]
                states[15:36] = real_atk_slice
            atks[15:36] = 1
            mitres[8:15] = 1
            mitres[15:36] = 4

        elif "Scenario D" in scenario_name:
            # Real PortScan & Stealth Reconnaissance
            metadata = {
                "name": "Stealth Subnet PortScan (Real Friday-PortScan Capture)",
                "recon_window": (10, 30),
                "attack_window": (15, 32),
                "target_port": 443,
                "target_protocol": "TCP",
                "attacker_ip": "172.16.0.1",
                "description": "Real Nmap stealth port-sweep: Horizontal host traversal searching for open vulnerable listening ports."
            }
            recon_idx = np.where(base_mitres == 1)[0]
            if len(recon_idx) >= 15:
                real_atk_slice = base_states[recon_idx[:15]]
                states[15:30] = real_atk_slice
            atks[15:32] = 1
            mitres[10:32] = 1

        elif "Scenario E" in scenario_name:

            # Pure Benign
            metadata = {
                "name": "Normal Enterprise Traffic",
                "recon_window": None,
                "attack_window": None,
                "target_port": 80,
                "target_protocol": "Standard Mix",
                "description": "Routine business day with regular employee browsing, cloud syncing, and zero malicious activity."
            }

        else:
            # Custom User-controlled scenario
            metadata = {
                "name": "Custom User Injected Attack",
                "recon_window": (max(0, user_attack_start - 5), user_attack_start - 1),
                "attack_window": (user_attack_start, min(seq_len - 1, user_attack_start + 12)),
                "target_port": 8080,
                "target_protocol": "Custom",
                "description": f"User injected attack with intensity {user_attack_intensity:.0%} starting at step {user_attack_start}."
            }
            r_start, r_end = metadata["recon_window"]
            for t in range(r_start, r_end + 1):
                states[t, 8] += np.random.uniform(10, 25) * user_attack_intensity
                mitres[t] = 1

            a_start, a_end = metadata["attack_window"]
            for t in range(a_start, a_end + 1):
                states[t, 8] += np.random.uniform(100, 500) * user_attack_intensity
                states[t, 10] += np.random.uniform(10000, 80000) * user_attack_intensity
                states[t, 41] += np.random.uniform(20, 100) * user_attack_intensity
                atks[t] = 1
                mitres[t] = user_attack_stage

        return states, atks, mitres, metadata
