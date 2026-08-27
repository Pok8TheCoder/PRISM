import random
import time

from src.adversarial.bots.base import dns_query, get_timing

CLASS_ID = "T1071_dns_tunnel"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(20):
        label = "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=12))
        domain = f"{label}.tunnel.lab"
        flows += dns_query(target_ip, domain)
        time.sleep(get_timing(evasion))
    return flows
