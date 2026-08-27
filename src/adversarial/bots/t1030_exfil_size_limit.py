import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1030_exfil_size_limit"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(25):
        chunk = b"a" * 64
        flows += http_request(
            target_ip, 80, "POST", f"/chunk/{i}",
            evasion, body=chunk,
            headers={"Content-Type": "application/octet-stream"},
        )
        time.sleep(get_timing(evasion))
    return flows
