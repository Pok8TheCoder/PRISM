import random
import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1595_active_scan"


def run(target_ip: str, evasion: str) -> int:
    ports = list(range(1, 1025))
    if evasion != "slow_timing":
        ports = random.sample(ports, min(120, len(ports)))
    flows = 0
    for port in ports:
        tcp_probe(target_ip, port, evasion, timeout=0.5)
        flows += 1
        time.sleep(get_timing(evasion))
    return flows
