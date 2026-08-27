"""Shared utilities for lab-safe network attack bots."""

from __future__ import annotations

import random
import socket
import struct
import time

DEFAULT_PORTS = [21, 22, 25, 53, 80, 110, 443, 445, 8080, 8443, 3389]


def get_timing(evasion: str) -> float:
    if evasion == "slow_timing":
        return random.uniform(2.0, 8.0)
    if evasion == "random_timing":
        return random.uniform(0.1, 3.0)
    return 0.05


def tcp_probe(
    target_ip: str,
    port: int,
    evasion: str,
    payload: bytes = b"",
    timeout: float = 2.0,
    source_port: int | None = None,
) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        if source_port:
            try:
                sock.bind(("", source_port))
            except OSError:
                pass
        sock.connect_ex((target_ip, port))
        if payload:
            sock.send(payload)
            try:
                sock.recv(1024)
            except OSError:
                pass
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def scan_ports(
    target_ip: str,
    ports: list[int],
    evasion: str,
    sequential: bool = True,
) -> int:
    port_list = list(ports)
    if not sequential or evasion == "randomize_port_order":
        random.shuffle(port_list)

    source_port = None
    if evasion == "source_port_manipulation":
        source_port = random.randint(40000, 60000)

    flows = 0
    for port in port_list:
        tcp_probe(target_ip, port, evasion, source_port=source_port)
        flows += 1
        if source_port:
            source_port += 1
        time.sleep(get_timing(evasion))
    return flows


def http_request(
    target_ip: str,
    port: int,
    method: str,
    path: str,
    evasion: str,
    headers: dict | None = None,
    body: bytes = b"",
) -> int:
    hdrs = headers or {}
    hdr_lines = "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
    req = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: {target_ip}\r\n"
        f"{hdr_lines}"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + body
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    flows = 0
    try:
        sock.connect((target_ip, port))
        sock.send(req)
        try:
            sock.recv(4096)
        except OSError:
            pass
        flows = 1
    except OSError:
        flows = 1
    finally:
        sock.close()
    time.sleep(get_timing(evasion))
    return flows


def dns_query(target_ip: str, domain: str, qtype: int = 1) -> int:
    """Send a UDP DNS query to target (port 53) or resolver-style to target."""
    tid = random.randint(0, 65535)
    header = struct.pack("!HHHHHH", tid, 0x0100, 1, 0, 0, 0)
    parts = domain.strip(".").split(".")
    question = b""
    for part in parts:
        question += bytes([len(part)]) + part.encode()
    question += b"\x00" + struct.pack("!HH", qtype, 1)
    packet = header + question

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2)
    try:
        sock.sendto(packet, (target_ip, 53))
        try:
            sock.recvfrom(512)
        except OSError:
            pass
        return 1
    except OSError:
        return 1
    finally:
        sock.close()


def icmp_echo(target_ip: str, payload: bytes = b"prism-lab") -> int:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
    except PermissionError:
        # Fallback: TCP probe on common ports if raw ICMP unavailable
        return tcp_probe(target_ip, 80, "none") and 1 or 1

    sock.settimeout(2)
    checksum = 0
    header = struct.pack("!BBHHH", 8, 0, checksum, random.randint(1, 65535), 1)
    packet = header + payload
    try:
        sock.sendto(packet, (target_ip, 0))
        try:
            sock.recvfrom(1024)
        except OSError:
            pass
        return 1
    except OSError:
        return 1
    finally:
        sock.close()
