import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1187_password_spray"


def run(target_ip: str, evasion: str) -> int:
    users = ["admin", "root", "user", "guest", "operator", "backup"]
    password = "Spring2024!"
    flows = 0
    for user in users:
        sock = None
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((target_ip, 22))
            sock.recv(1024)
            sock.send(f"{user} {password}\r\n".encode())
            flows += 1
        except OSError:
            flows += 1
        finally:
            if sock:
                sock.close()
        time.sleep(get_timing(evasion))
    for port in [80, 443]:
        tcp_probe(target_ip, port, evasion)
        flows += 1
    return flows
