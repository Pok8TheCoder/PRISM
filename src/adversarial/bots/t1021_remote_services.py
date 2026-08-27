import time

from src.adversarial.bots.base import get_timing

CLASS_ID = "T1021_remote_services"


def run(target_ip: str, evasion: str) -> int:
    import socket
    flows = 0
    for port in [22, 445, 5985]:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((target_ip, port))
            if port == 22:
                banner = sock.recv(1024)
                sock.send(b"SSH-2.0-PRISMClient\r\n")
                sock.recv(1024)
            elif port == 445:
                sock.send(b"\x00")
            else:
                sock.send(b"")
            flows += 1
        except OSError:
            flows += 1
        finally:
            if sock:
                sock.close()
        time.sleep(get_timing(evasion))
    return flows
