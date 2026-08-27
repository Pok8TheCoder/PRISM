import socket
import time

from src.adversarial.bots.base import get_timing

CLASS_ID = "T1048_exfil_alt_protocol"


def run(target_ip: str, evasion: str) -> int:
    """Simulate FTP-style bulk transfer over TCP port 21."""
    flows = 0
    commands = [b"USER anonymous\r\n", b"PASS guest@\r\n", b"STOR dump.bin\r\n"]
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((target_ip, 21))
        for cmd in commands:
            sock.send(cmd)
            try:
                sock.recv(1024)
            except OSError:
                pass
            flows += 1
            time.sleep(get_timing(evasion))
        sock.send(b"x" * 2048)
        flows += 1
    except OSError:
        flows += 1
    finally:
        sock.close()
    return flows
