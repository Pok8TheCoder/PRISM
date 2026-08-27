import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1090_proxy"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(10):
        flows += http_request(
            target_ip, 80, "GET", f"/resource/{i}",
            evasion,
            headers={
                "Via": "1.1 proxy.lab",
                "X-Forwarded-For": f"10.0.0.{i}",
            },
        )
        time.sleep(get_timing(evasion))
    return flows
