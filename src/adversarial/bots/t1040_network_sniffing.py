import socket
import struct
import time

from src.adversarial.bots.base import get_timing

CLASS_ID = "T1040_network_sniffing"


def run(target_ip: str, evasion: str) -> int:
    """Simulate discovery traffic (ARP-style broadcast probes via UDP/TCP)."""
    flows = 0
    for port in [67, 68, 137, 138, 5353]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1)
        try:
            sock.sendto(b"PRISM-DISCOVERY", (target_ip, port))
            flows += 1
        except OSError:
            flows += 1
        finally:
            sock.close()
        time.sleep(get_timing(evasion))

    # mDNS-like multicast probe attempt (may fail in container; still generates flows)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(struct.pack("!H", 0) + b"\x00\x00\x01", ("224.0.0.251", 5353))
        flows += 1
        sock.close()
    except OSError:
        flows += 1
    return flows
