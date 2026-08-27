import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1205_traffic_signaling"


def run(target_ip: str, evasion: str) -> int:
    """Port-knock sequence followed by connection on signaled port."""
    knock_ports = [7000, 8000, 9000]
    flows = 0
    for port in knock_ports:
        tcp_probe(target_ip, port, evasion, timeout=0.5)
        flows += 1
        time.sleep(get_timing(evasion))
    time.sleep(0.5)
    tcp_probe(target_ip, 22, evasion, timeout=2.0)
    flows += 1
    return flows
