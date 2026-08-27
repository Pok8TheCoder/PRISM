import time

from src.adversarial.bots.base import get_timing, tcp_probe

CLASS_ID = "T1110_ssh_bruteforce"


def run(target_ip: str, evasion: str) -> int:
    passwords = ["admin", "password", "123456", "root", "toor", "test", "guest"]
    flows = 0
    for pwd in passwords:
        sock = None
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((target_ip, 22))
            sock.recv(1024)
            sock.send(f"root {pwd}\r\n".encode())
            flows += 1
        except OSError:
            flows += 1
        finally:
            if sock:
                sock.close()
        time.sleep(get_timing(evasion))
    return flows
