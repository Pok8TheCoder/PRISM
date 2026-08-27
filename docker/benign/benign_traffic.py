#!/usr/bin/env python3
"""Generate benign background traffic inside the isolated lab."""

import os
import random
import socket
import time

TARGET = os.environ.get("TARGET_HOST", "target-server")
INTERVAL = float(os.environ.get("BENIGN_INTERVAL", "3.0"))


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


def main():
    paths = ["/", "/index.html", "/admin", "/api/health"]
    print(f"Benign client targeting {TARGET}", flush=True)
    while True:
        action = random.choice(["http", "http", "ssh"])
        if action == "http":
            http_get(random.choice(paths))
        else:
            ssh_banner()
        time.sleep(random.uniform(INTERVAL * 0.5, INTERVAL * 1.5))


if __name__ == "__main__":
    main()
