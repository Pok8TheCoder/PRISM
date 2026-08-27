import socket
import time

from src.adversarial.bots.base import get_timing

CLASS_ID = "T1570_lateral_transfer"


def run(target_ip: str, evasion: str) -> int:
    """Simulate bulk tool transfer over SSH/FTP-like TCP sessions."""
    flows = 0
    for port in [22, 21]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        try:
            sock.connect((target_ip, port))
            chunk = b"TOOLTRANSFER" + b"x" * 8192
            for _ in range(5):
                sock.send(chunk)
                flows += 1
                time.sleep(get_timing(evasion))
        except OSError:
            flows += 1
        finally:
            sock.close()
    return flows
