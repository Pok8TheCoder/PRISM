import time

from src.adversarial.bots.base import dns_query, get_timing, http_request

CLASS_ID = "T1104_multistage_channel"


def run(target_ip: str, evasion: str) -> int:
    flows = 0
    for i in range(5):
        flows += http_request(target_ip, 80, "GET", f"/stage1/{i}", evasion)
        time.sleep(get_timing(evasion))
        flows += dns_query(target_ip, f"stage2-{i}.lab.local")
        time.sleep(get_timing(evasion))
        flows += http_request(target_ip, 443, "GET", f"/stage3/{i}", evasion)
        time.sleep(get_timing(evasion))
    return flows
