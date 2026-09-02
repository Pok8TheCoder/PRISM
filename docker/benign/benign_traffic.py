#!/usr/bin/env python3
"""Generate benign background traffic inside the isolated lab.

The live adversarial lab (`scripts/live_attack_lab.py`) feeds raw captured
traffic straight into ARY.01's 242-d StateBuilder features -- no
per-lab-run rescaling -- so absolute counts (num_flows etc.) need to be in
the same ballpark ARY.01 saw during CIC-IDS-2018 training (median ~98-207
flows per 30s window), not the ~1 request/4s a single lab client would
naturally produce. A single slow client made every window look emptier
than *any* CIC-IDS window regardless of attack/benign, which flattened
P(attack) everywhere. `BENIGN_WORKERS` concurrent loops approximate a
busier background network without needing dozens of real containers.
"""

import os
import random
import socket
import threading
import time

TARGET = os.environ.get("TARGET_HOST", "target-server")
INTERVAL = float(os.environ.get("BENIGN_INTERVAL", "1.0"))
WORKERS = int(os.environ.get("BENIGN_WORKERS", "4"))


def http_get(path="/"):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    try:
        sock.connect((TARGET, 80))
        sock.send(f"GET {path} HTTP/1.1\r\nHost: {TARGET}\r\nConnection: close\r\n\r\n".encode())
        sock.recv(2048)
    except OSError:
        pass
    finally:
        sock.close()


def ssh_banner():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((TARGET, 22))
        sock.recv(1024)
    except OSError:
        pass
    finally:
        sock.close()


def worker_loop(worker_id: int) -> None:
    paths = ["/", "/index.html", "/admin", "/api/health"]
    while True:
        action = random.choice(["http", "http", "http", "ssh"])
        if action == "http":
            http_get(random.choice(paths))
        else:
            ssh_banner()
        time.sleep(random.uniform(INTERVAL * 0.5, INTERVAL * 1.5))


def main():
    print(f"Benign client targeting {TARGET} with {WORKERS} concurrent workers", flush=True)
    threads = [
        threading.Thread(target=worker_loop, args=(i,), daemon=True)
        for i in range(WORKERS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
