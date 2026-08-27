import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1571_non_standard_port"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for port in [8080, 8443, 9000, 4444, 31337]:
        flows += http_request(target_ip, port, "GET", "/status", evasion)
        time.sleep(get_timing(evasion))
    return flows
