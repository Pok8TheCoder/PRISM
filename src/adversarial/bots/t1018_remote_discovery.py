import time

from src.adversarial.bots.base import DEFAULT_PORTS, get_timing, icmp_echo, tcp_probe

CLASS_ID = "T1018_remote_discovery"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for _ in range(5):
        flows += icmp_echo(target_ip)
        time.sleep(get_timing(evasion))
    for port in [22, 80, 443, 445, 3389]:
        tcp_probe(target_ip, port, evasion)
        flows += 1
        time.sleep(get_timing(evasion))
    flows += len(DEFAULT_PORTS) // 2
    for port in DEFAULT_PORTS[::2]:
        tcp_probe(target_ip, port, evasion, timeout=1.0)
        time.sleep(get_timing(evasion))
    return flows
