import time

from src.adversarial.bots.base import get_timing, icmp_echo

CLASS_ID = "T1095_non_app_protocol"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(10):
        flows += icmp_echo(target_ip, payload=f"prism-{i}".encode())
        time.sleep(get_timing(evasion))
    return flows
