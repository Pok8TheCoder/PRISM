import time

from src.adversarial.bots.base import get_timing, http_request

CLASS_ID = "T1102_web_service"


def run(target_ip: str, evasion: str) -> int:
    paths = [
        "/raw.githubusercontent.com/user/repo/main/config.json",
        "/api.github.com/repos/user/repo/contents/",
        "/pastebin.com/raw/abc123",
    ]
    flows = 0
    for path in paths:
        flows += http_request(
            target_ip, 80, "GET", path, evasion,
            headers={"User-Agent": "curl/8.0 PRISM"},
        )
        time.sleep(get_timing(evasion))
    return flows
