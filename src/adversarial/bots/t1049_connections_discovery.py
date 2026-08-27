import random
import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1049_connections_discovery"


def run(target_ip: str, evasion: str) -> int:
    ports = random.sample(range(1024, 65535), 30)
    flows = 0
    for port in ports:
        tcp_probe(target_ip, port, evasion, timeout=0.3)
        flows += 1
        time.sleep(get_timing(evasion))
    return flows
