import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1498_network_dos"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for port in [80, 443, 22]:
        for _ in range(40):
            tcp_probe(target_ip, port, evasion, timeout=0.2)
            flows += 1
            time.sleep(get_timing(evasion) * 0.1)
    return flows
