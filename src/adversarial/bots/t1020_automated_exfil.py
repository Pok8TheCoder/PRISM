import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1020_automated_exfil"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(30):
        body = (f"record={i}&data={'x'*512}").encode()
        flows += http_request(
            target_ip, 80, "POST", "/sync",
            evasion, body=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        time.sleep(get_timing(evasion) * 0.5)
    return flows
