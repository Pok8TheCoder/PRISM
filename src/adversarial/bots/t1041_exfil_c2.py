import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1041_exfil_c2"


def run(target_ip: str, evasion: str) -> int:
    payload = b"x" * 4096
    flows = 0
    for i in range(8):
        flows += http_request(
            target_ip, 80, "POST", "/upload",
            evasion,
            headers={"Content-Type": "application/octet-stream"},
            body=payload,
        )
        time.sleep(get_timing(evasion))
    return flows
