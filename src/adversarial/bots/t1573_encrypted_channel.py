import socket
import time

from src.adversarial.bots.base import get_timing

CLASS_ID = "T1573_encrypted_channel"


def run(target_ip: str, evasion: str) -> int:
    """TLS ClientHello probes to simulate encrypted C2 channel setup."""
    flows = 0
    client_hello = bytes([
        0x16, 0x03, 0x01, 0x00, 0x2f, 0x01, 0x00, 0x00, 0x2b,
        0x03, 0x03,
    ] + [0x00] * 32 + [0x00, 0x02, 0x00, 0x00])
    for port in [443, 8443, 4433]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect((target_ip, port))
            sock.send(client_hello)
            try:
                sock.recv(1024)
            except OSError:
                pass
            flows += 1
        except OSError:
            flows += 1
        finally:
            sock.close()
        time.sleep(get_timing(evasion))
    return flows
