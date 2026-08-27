import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1133_external_remote"


def run(target_ip: str, evasion: str) -> int:
    remote_ports = [22, 3389, 5900, 5985, 8443]
    flows = 0
    for port in remote_ports:
        tcp_probe(target_ip, port, evasion, timeout=3.0)
        flows += 1
        time.sleep(get_timing(evasion))
    return flows
