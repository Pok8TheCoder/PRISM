#!/usr/bin/env python3
"""Generate benign background traffic inside the isolated lab.

Profiles (BENIGN_PROFILE):
  default  — original HTTP/SSH mix for quiet 1× lab
  internet — multi-persona mix: browser sessions, API polling, search, assets
  browser  — logged-in browsing, search, static assets, keep-alive bursts
  api      — rapid JSON health checks and HEAD probes
  crawler  — varied search queries and path enumeration

The live adversarial lab feeds captured traffic into PRISM feature builders, so
absolute flow counts should stay in a CIC-like ballpark. Use BENIGN_WORKERS and
BENIGN_INTERVAL (or the lab_v2 realistic overlay) to tune density.
"""

from __future__ import annotations

import os
import random
import socket
import threading
import time
import urllib.parse

TARGET = os.environ.get("TARGET_HOST", "target-server")
INTERVAL = float(os.environ.get("BENIGN_INTERVAL", "1.0"))
WORKERS = int(os.environ.get("BENIGN_WORKERS", "4"))
DNS_NOISE = os.environ.get("BENIGN_DNS_NOISE", "0") == "1"
JITTER_MS = int(os.environ.get("BENIGN_API_JITTER_MS", "0"))
PROFILE = os.environ.get("BENIGN_PROFILE", "default").strip().lower()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]

SEARCH_TERMS = [
    "security", "network", "monitoring", "blog", "update", "release",
    "product", "pricing", "support", "documentation", "api", "health",
]

STATIC_PATHS = [
    "/favicon.ico", "/static/style.css", "/static/app.js", "/static/logo.png",
    "/robots.txt", "/sitemap.xml",
]

API_PATHS = ["/api/health", "/api/health", "/api/health", "/"]


def _sleep() -> None:
    jitter = (random.randint(0, JITTER_MS) / 1000.0) if JITTER_MS else 0.0
    if PROFILE in ("internet", "browser"):
        # Log-normal-ish pauses mimic human browsing better than uniform sleep.
        base = random.lognormvariate(0.0, 0.35) * INTERVAL
    else:
        base = random.uniform(INTERVAL * 0.5, INTERVAL * 1.5)
    time.sleep(max(0.05, base + jitter))


def http_request(
    method: str = "GET",
    path: str = "/",
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    recv_bytes: int = 4096,
) -> None:
    hdrs = {
        "Host": TARGET,
        "Connection": "close",
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if headers:
        hdrs.update(headers)
    req = f"{method} {path} HTTP/1.1\r\n"
    req += "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
    if body:
        req += f"Content-Length: {len(body)}\r\n"
    req += "\r\n"
    payload = req.encode() + (body or b"")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(4)
    try:
        sock.connect((TARGET, 80))
        sock.sendall(payload)
        sock.recv(recv_bytes)
    except OSError:
        pass
    finally:
        sock.close()


def http_burst(paths: list[str]) -> None:
    """Several requests on one TCP connection — common on real web pages."""
    hdrs = {
        "Host": TARGET,
        "Connection": "keep-alive",
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/json,*/*;q=0.8",
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(4)
    try:
        sock.connect((TARGET, 80))
        for path in paths:
            req = f"GET {path} HTTP/1.1\r\n"
            req += "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
            req += "\r\n"
            sock.sendall(req.encode())
            sock.recv(2048)
            time.sleep(random.uniform(0.02, 0.12))
    except OSError:
        pass
    finally:
        sock.close()


def dns_query() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1)
    try:
        qname = TARGET.encode()
        sock.sendto(
            b"\x00\x00\x01\x00\x00\x01" + bytes([len(TARGET)]) + qname + b"\x00\x00\x01\x00\x01",
            (TARGET, 53),
        )
        sock.recv(512)
    except OSError:
        pass
    finally:
        sock.close()


def ssh_banner() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((TARGET, 22))
        sock.recv(1024)
    except OSError:
        pass
    finally:
        sock.close()


def browser_session() -> None:
    user = f"user{random.randint(1000, 9999)}"
    password = f"pass{random.randint(1000, 9999)}"
    body = urllib.parse.urlencode({"username": user, "password": password}).encode()
    http_request("POST", "/register", headers={"Content-Type": "application/x-www-form-urlencoded"}, body=body)
    body = urllib.parse.urlencode({"username": user, "password": password}).encode()
    http_request("POST", "/login", headers={"Content-Type": "application/x-www-form-urlencoded"}, body=body)
    http_burst(["/", "/search", random.choice(STATIC_PATHS)])
    q = random.choice(SEARCH_TERMS)
    http_request("GET", f"/search?q={urllib.parse.quote(q)}")
    if random.random() < 0.2:
        http_request("GET", "/admin/edit_page")


def api_poll() -> None:
    path = random.choice(API_PATHS)
    if random.random() < 0.3:
        http_request("HEAD", path, recv_bytes=512)
    else:
        http_request("GET", path, headers={"Accept": "application/json"})


def crawler_step() -> None:
    paths = ["/", "/login", "/register", "/search", "/products", "/admin"]
    http_request("GET", random.choice(paths))
    q = random.choice(SEARCH_TERMS) + str(random.randint(1, 50))
    http_request("GET", f"/search?q={urllib.parse.quote(q)}")
    if random.random() < 0.15:
        ssh_banner()


def default_step() -> None:
    paths = ["/", "/index.html", "/admin", "/api/health", "/products", "/login"]
    action = random.choice(["http", "http", "http", "ssh", "dns" if DNS_NOISE else "http"])
    if action == "http":
        http_request("GET", random.choice(paths))
    elif action == "dns":
        dns_query()
    else:
        ssh_banner()


def internet_step(worker_id: int) -> None:
    persona = ["browser", "browser", "api", "crawler"][worker_id % 4]
    if persona == "browser":
        browser_session() if random.random() < 0.35 else http_burst(
            ["/", random.choice(STATIC_PATHS), "/search"]
        )
    elif persona == "api":
        api_poll()
    else:
        crawler_step()
    if DNS_NOISE and random.random() < 0.1:
        dns_query()
    if random.random() < 0.05:
        ssh_banner()


def demo_step(worker_id: int) -> None:
    """Lightweight browsing for the dual-IPS demo (~1000 identities, no mass register)."""
    uid = 1000 + (worker_id % 1000)
    paths = ["/", "/", "/search", "/api/health", "/login", "/register"]
    http_request("GET", random.choice(paths), headers={"X-Demo-User": f"user{uid}"})
    if random.random() < 0.45:
        q = random.choice(SEARCH_TERMS)
        http_request("GET", f"/search?q={urllib.parse.quote(q)}")
    if random.random() < 0.2:
        http_burst(["/", random.choice(STATIC_PATHS), "/api/health"])


def worker_loop(worker_id: int) -> None:
    profile = PROFILE
    if profile == "internet":
        step_fn = lambda: internet_step(worker_id)  # noqa: E731
    elif profile == "browser":
        step_fn = lambda: browser_session() if random.random() < 0.5 else http_burst(  # noqa: E731
            ["/", "/search", random.choice(STATIC_PATHS)]
        )
    elif profile == "api":
        step_fn = api_poll
    elif profile == "crawler":
        step_fn = crawler_step
    elif profile == "demo":
        step_fn = lambda: demo_step(worker_id)  # noqa: E731
    else:
        step_fn = default_step

    while True:
        step_fn()
        _sleep()


def main() -> None:
    print(
        f"Benign client targeting {TARGET} profile={PROFILE} workers={WORKERS} interval={INTERVAL}s",
        flush=True,
    )
    threads = [threading.Thread(target=worker_loop, args=(i,), daemon=True) for i in range(WORKERS)]
    for i, t in enumerate(threads):
        t.start()
        if i % 50 == 49:
            time.sleep(0.05)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
