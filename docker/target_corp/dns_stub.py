#!/usr/bin/env python3
"""UDP/53 stub so DNS-looking exfil actually hits the target."""

from __future__ import annotations

import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 53))
while True:
    try:
        data, addr = sock.recvfrom(2048)
        if not data:
            continue
        # Echo a FORMERR-ish 12-byte header so clients stop retrying hard.
        txid = data[:2] if len(data) >= 2 else b"\x00\x00"
        sock.sendto(txid + b"\x81\x81\x00\x00\x00\x00\x00\x00\x00\x00", addr)
    except OSError:
        continue
