import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1499_http_flood"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for _ in range(50):
        flows += http_request(target_ip, 80, "GET", "/", evasion)
    return flows
