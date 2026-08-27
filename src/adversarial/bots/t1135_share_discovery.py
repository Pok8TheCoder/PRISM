import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1135_share_discovery"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for port in [445, 139, 2049, 111]:
        tcp_probe(target_ip, port, evasion, timeout=2.0)
        flows += 1
        time.sleep(get_timing(evasion))
    return flows
