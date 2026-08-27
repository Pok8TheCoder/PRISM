import random
import time

from src.adversarial.bots.base import dns_query, get_timing

CLASS_ID = "T1568_dynamic_resolution"


def run(target_ip: str, evasion: str) -> int:
    domains = [
        "cdn{}.malware.lab".format(random.randint(1, 9999)),
        "update{}.service.lab".format(random.randint(1, 9999)),
        "api{}.cloud.lab".format(random.randint(1, 9999)),
    ]
    flows = 0
    for _ in range(15):
        domain = random.choice(domains)
        flows += dns_query(target_ip, domain, qtype=1)
        flows += dns_query(target_ip, domain, qtype=16)  # TXT
        time.sleep(get_timing(evasion))
    return flows
