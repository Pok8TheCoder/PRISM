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
            "⚡ Scenario A: Stealth Reconnaissance → SSH Brute Force Infiltration (CIC-IDS2018)",
            "⚡ Scenario B: Low-and-Slow PortScan → Volumetric DDoS Attack (CIC-IDS2017)",
            "⚡ Scenario C: IoT Botnet Infiltration → Mirai DDoS Surge (CICIoT2023)",
            "⚡ Scenario D: Web Application Exploit → Data Exfiltration (UNSW-NB15)",
            "⚡ Scenario E: Pure Benign Enterprise Day (Baseline Stability)",
            "⚡ Scenario F: Custom Interactive Attack Injector (User Controlled)"
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
            # SSH Brute Force (Patator)
            metadata = {
                "name": "SSH Brute Force Infiltration (Patator)",
                "recon_window": (12, 18),
                "attack_window": (19, 36),
                "target_port": 22,
                "target_protocol": "SSH",
                "description": "Attacker scans SSH port 22, attempts dictionary password spraying, and successfully breaches the server."
            }
            # Reconnaissance probing
            for t in range(12, 19):
                states[t, 0] += np.random.uniform(50, 150)
                states[t, 8] += np.random.uniform(5, 15)  # tot_fwd_pkts
                states[t, 10] += np.random.uniform(200, 800)  # tot_fwd_bytes
                mitres[t] = 1  # Recon

            # Active SSH Brute Force
            for t in range(19, 37):
                states[t, 0] += np.random.uniform(500, 2000)  # duration
                states[t, 8] += np.random.uniform(40, 120)  # packet bursts
                states[t, 10] += np.random.uniform(2000, 8000)
                states[t, 41] += np.random.uniform(10, 40)  # syn flags
                atks[t] = 1
                mitres[t] = 2  # Initial Access

        elif "Scenario B" in scenario_name:
            # Volumetric DDoS
            metadata = {
                "name": "Volumetric DDoS Flood (LOIC / HOIC)",
                "recon_window": (10, 16),
                "attack_window": (17, 38),
                "target_port": 80,
                "target_protocol": "HTTP",
                "description": "Distributed botnet performs rapid SYN scan followed by massive volumetric packet flood on web server port 80."
            }
            for t in range(10, 17):
                states[t, 8] += np.random.uniform(15, 30)
                mitres[t] = 1

            for t in range(17, 39):
                states[t, 8] += np.random.uniform(300, 1200)  # Massive packets
                states[t, 10] += np.random.uniform(50000, 250000)  # Massive bytes
                states[t, 41] += np.random.uniform(100, 500)  # Extreme SYN count
                atks[t] = 1
                mitres[t] = 6  # Impact / DDoS

        elif "Scenario C" in scenario_name:
            # IoT Botnet / Mirai Flood
            metadata = {
                "name": "IoT Botnet Mirai & Flood Surge (CICIoT2023)",
                "recon_window": (8, 14),
                "attack_window": (15, 35),
                "target_port": 23,
                "target_protocol": "Telnet/Mirai",
                "description": "Compromised IoT camera swarm executes high-rate UDP & GRE flood against edge gateway."
            }
            for t in range(8, 15):
                states[t, 5] += np.random.uniform(20, 50)  # rate
                mitres[t] = 1

            for t in range(15, 36):
                states[t, 5] += np.random.uniform(500, 3000)  # extreme rate
                states[t, 8] += np.random.uniform(400, 1500)
                states[t, 10] += np.random.uniform(80000, 400000)
                atks[t] = 1
                mitres[t] = 4  # C2 / Botnet

        elif "Scenario D" in scenario_name:
            # Web Exploit -> Exfiltration
            metadata = {
                "name": "Web Exploit & Data Exfiltration (UNSW-NB15)",
                "recon_window": (12, 17),
                "attack_window": (18, 34),
                "target_port": 443,
                "target_protocol": "HTTPS",
                "description": "SQL Injection & Cross-Site Scripting exploit followed by encrypted exfiltration of database records."
            }
            for t in range(12, 18):
                states[t, 8] += np.random.uniform(10, 25)
                mitres[t] = 1

            for t in range(18, 35):
                states[t, 9] += np.random.uniform(80, 300)  # bwd packets (exfil response)
                states[t, 11] += np.random.uniform(40000, 180000)  # bwd bytes out
                states[t, 43] += np.random.uniform(20, 80)  # PSH flags
                atks[t] = 1
                mitres[t] = 5  # Exfiltration

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
