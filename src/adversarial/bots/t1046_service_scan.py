from src.adversarial.bots.base import DEFAULT_PORTS, scan_ports

CLASS_ID = "T1046_service_scan"


def run(target_ip: str, evasion: str) -> int:
    return scan_ports(target_ip, DEFAULT_PORTS, evasion, sequential=True)
