"""Attacker bot orchestrator — wraps catalog bots with evasion chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.adversarial.bots import run_bot, resolve_class_id, list_bots
from src.model.attack_catalog import get_evasion_chains, get_legacy_strategy_map


class EvasionTactic(Enum):
    NONE = "none"
    RANDOM_TIMING = "random_timing"
    SLOW_TIMING = "slow_timing"
    DECOY_SOURCES = "decoy_sources"
    SOURCE_PORT_MANIPULATION = "source_port_manipulation"
    RANDOMIZE_PORT_ORDER = "randomize_port_order"
    FRAGMENT_PACKETS = "fragment_packets"


@dataclass
class AttackResult:
    class_id: str
    evasion: EvasionTactic
    detected: bool
    duration_sec: float
    flows_generated: int
    metadata: dict = field(default_factory=dict)


class AttackStrategy(Enum):
    """Legacy enum kept for training_loop compatibility."""

    SSH_BRUTEFORCE = "ssh_bruteforce"
    PORT_SCAN_SEQUENTIAL = "port_scan_sequential"
    PORT_SCAN_RANDOM = "port_scan_random"
    HTTP_FLOOD = "http_flood"
    SLOW_LORIS = "slow_loris"
    SYN_SCAN_STEALTH = "syn_scan_stealth"


def _build_strategy_chain() -> dict[str, list[EvasionTactic]]:
    chains = get_evasion_chains()
    legacy = get_legacy_strategy_map()
    out: dict[str, list[EvasionTactic]] = {}

    for legacy_name, class_id in legacy.items():
        raw = chains.get(class_id, [])
        out[legacy_name] = [EvasionTactic(v) for v in raw if v in EvasionTactic._value2member_map_]

    for class_id, raw in chains.items():
        out[class_id] = [EvasionTactic(v) for v in raw if v in EvasionTactic._value2member_map_]

    return out


STRATEGY_CHAIN = _build_strategy_chain()


class AttackerBot:
    def __init__(self, target_ip: str, target_ports: list[int] | None = None):
        self.target_ip = target_ip
        self.target_ports = target_ports or [22, 80, 443, 21, 25, 3389, 8080, 8443, 53, 110]
        self._evasion_index: dict[str, int] = {}

    def get_next_evasion(self, strategy_key: str) -> Optional[EvasionTactic]:
        class_id = resolve_class_id(strategy_key)
        chain = STRATEGY_CHAIN.get(class_id) or STRATEGY_CHAIN.get(strategy_key, [])
        idx = self._evasion_index.get(class_id, 0)
        if idx >= len(chain):
            return None
        self._evasion_index[class_id] = idx + 1
        return chain[idx]

    def reset_strategy(self, strategy_key: str):
        class_id = resolve_class_id(strategy_key)
        self._evasion_index[class_id] = 0

    def execute_attack(
        self,
        class_id: str,
        evasion: EvasionTactic | str = EvasionTactic.NONE,
    ) -> AttackResult:
        import time

        resolved = resolve_class_id(class_id)
        evasion_str = evasion.value if isinstance(evasion, EvasionTactic) else str(evasion)
        start = time.time()
        flows = run_bot(resolved, self.target_ip, evasion_str)
        duration = time.time() - start

        return AttackResult(
            class_id=resolved,
            evasion=EvasionTactic(evasion_str) if evasion_str in EvasionTactic._value2member_map_ else EvasionTactic.NONE,
            detected=False,
            duration_sec=duration,
            flows_generated=flows,
            metadata={"target": self.target_ip},
        )

    @staticmethod
    def available_classes() -> list[str]:
        return list_bots()
