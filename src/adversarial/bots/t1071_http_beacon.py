import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1071_http_beacon"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(15):
        flows += http_request(
            target_ip, 80, "GET", f"/beacon?id=session1&seq={i}",
            evasion, headers={"User-Agent": "Mozilla/5.0 PRISMBeacon"},
        )
        time.sleep(get_timing(evasion))
    return flows
