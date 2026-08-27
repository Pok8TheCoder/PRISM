import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1572_protocol_tunnel"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for _ in range(10):
        flows += http_request(
            target_ip, 80, "CONNECT", "internal.target:443",
            evasion, headers={"Proxy-Connection": "keep-alive"},
        )
        time.sleep(get_timing(evasion))
    return flows
