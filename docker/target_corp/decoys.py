#!/usr/bin/env python3
"""Listen on extra TCP ports so a service-scan is a real multi-port event."""

from __future__ import annotations

import socket
import threading

PORTS = (21, 443, 3306, 5432, 6379, 8080, 8443, 9090, 9200, 27017)


def serve(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.listen(32)
    except OSError:
        return
    while True:
        try:
            conn, _ = sock.accept()
            try:
                conn.sendall(b"harborline-decoy\n")
            finally:
                conn.close()
        except OSError:
            break


def main() -> None:
    for port in PORTS:
        threading.Thread(target=serve, args=(port,), daemon=True).start()
    threading.Event().wait()


if __name__ == "__main__":
    main()
