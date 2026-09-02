"""Scale quiet lab PCAP flow rows toward CIC-like traffic density."""

from __future__ import annotations

import pandas as pd

PRISM_SCALE_KEYS = [
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "FIN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
]

# Default matches scale_sensitivity "CIC-like" preset (10× rates + 5× flows).
DEFAULT_SCALE_FACTOR = 10.0
DEFAULT_REPLICATE = 5


def scale_prism_rows(
    rows: list[dict],
    factor: float = DEFAULT_SCALE_FACTOR,
    replicate: int = DEFAULT_REPLICATE,
) -> list[dict]:
    """Multiply rate/count features and replicate flows with tiny timestamp jitter."""
    if not rows:
        return []
    out: list[dict] = []
    base = [dict(r) for r in rows]
    for rep in range(max(1, int(replicate))):
        for r in base:
            nr: dict = {}
            for k, v in r.items():
                if k in PRISM_SCALE_KEYS:
                    nr[k] = float(v or 0) * factor
                else:
                    nr[k] = v
            if rep and "Timestamp" in nr and nr["Timestamp"] is not None:
                nr["Timestamp"] = pd.Timestamp(nr["Timestamp"]) + pd.Timedelta(milliseconds=rep * 50)
            out.append(nr)
    return out
