"""Attack bot registry — one lab-safe bot per network-distinct class."""

from __future__ import annotations

from typing import Callable

from src.adversarial.bots import (
    lab_cred_theft,
    lab_defacement,
    lab_key_theft,
    t1020_automated_exfil,
    t1021_remote_services,
    t1030_exfil_size_limit,
    t1040_network_sniffing,
    t1041_exfil_c2,
    t1046_service_scan,
    t1048_exfil_alt_protocol,
    t1049_connections_discovery,
    t1071_dns_tunnel,
    t1071_http_beacon,
    t1090_proxy,
    t1095_non_app_protocol,
    t1102_web_service,
    t1104_multistage_channel,
    t1110_ssh_bruteforce,
    t1110_web_bruteforce,
    t1133_external_remote,
    t1135_share_discovery,
    t1187_password_spray,
    t1190_web_exploit_probe,
    t1205_traffic_signaling,
    t1210_exploit_remote,
    t1498_network_dos,
    t1499_http_flood,
    t1499_slowloris,
    t1568_dynamic_resolution,
    t1570_lateral_transfer,
    t1571_non_standard_port,
    t1572_protocol_tunnel,
    t1573_encrypted_channel,
    t1595_active_scan,
    t1018_remote_discovery,
)

RunFn = Callable[[str, str], int]

_BOT_MODULES = [
    t1046_service_scan,
    t1595_active_scan,
    t1018_remote_discovery,
    t1040_network_sniffing,
    t1110_ssh_bruteforce,
    t1110_web_bruteforce,
    t1187_password_spray,
    t1133_external_remote,
    t1190_web_exploit_probe,
    t1021_remote_services,
    t1210_exploit_remote,
    t1135_share_discovery,
    t1049_connections_discovery,
    t1071_http_beacon,
    t1071_dns_tunnel,
    t1571_non_standard_port,
    t1572_protocol_tunnel,
    t1573_encrypted_channel,
    t1095_non_app_protocol,
    t1102_web_service,
    t1104_multistage_channel,
    t1041_exfil_c2,
    t1048_exfil_alt_protocol,
    t1030_exfil_size_limit,
    t1020_automated_exfil,
    t1498_network_dos,
    t1499_http_flood,
    t1499_slowloris,
    t1090_proxy,
    t1568_dynamic_resolution,
    t1205_traffic_signaling,
    t1570_lateral_transfer,
]

BOT_REGISTRY: dict[str, RunFn] = {m.CLASS_ID: m.run for m in _BOT_MODULES}

# --- Objective bots (real requests-based HTTP client, structured results) --
# Kept in a separate registry from BOT_REGISTRY: these have a richer contract
# (`run_objective(target_ip, evasion, clock, event_log, round_id) -> dict`)
# than the recon bots' `RunFn -> int`. The live orchestrator
# (scripts/live_attack_lab.py) calls `run_objective` directly; `run_bot`/
# `BOT_REGISTRY` above are untouched for the existing recon-bot rotation.
_OBJECTIVE_BOT_MODULES = [
    lab_defacement,
    lab_key_theft,
    lab_cred_theft,
]

ObjectiveRunFn = Callable[..., dict]

OBJECTIVE_BOT_REGISTRY: dict[str, ObjectiveRunFn] = {
    m.CLASS_ID: m.run_objective for m in _OBJECTIVE_BOT_MODULES
}

OBJECTIVE_BOT_TACTICS: dict[str, str] = {
    m.CLASS_ID: m.MITRE_TACTIC for m in _OBJECTIVE_BOT_MODULES
}

# Legacy PoC strategy names -> catalog class IDs
LEGACY_ALIASES: dict[str, str] = {
    "ssh_bruteforce": "T1110_ssh_bruteforce",
    "port_scan_sequential": "T1046_service_scan",
    "port_scan_random": "T1046_service_scan",
    "http_flood": "T1499_http_flood",
    "slow_loris": "T1499_slowloris",
    "syn_scan_stealth": "T1046_service_scan",
}


def resolve_class_id(name: str) -> str:
    if name in BOT_REGISTRY:
        return name
    return LEGACY_ALIASES.get(name, name)


def run_bot(class_id: str, target_ip: str, evasion: str = "none") -> int:
    resolved = resolve_class_id(class_id)
    if resolved not in BOT_REGISTRY:
        raise KeyError(f"Unknown attack class: {class_id} (resolved: {resolved})")
    return BOT_REGISTRY[resolved](target_ip, evasion)


def list_bots() -> list[str]:
    return sorted(BOT_REGISTRY.keys())


def list_objective_bots() -> list[str]:
    return sorted(OBJECTIVE_BOT_REGISTRY.keys())


def run_objective_bot(class_id: str, target_ip: str, evasion: str, **kwargs) -> dict:
    if class_id not in OBJECTIVE_BOT_REGISTRY:
        raise KeyError(f"Unknown objective class: {class_id}")
    return OBJECTIVE_BOT_REGISTRY[class_id](target_ip, evasion, **kwargs)
