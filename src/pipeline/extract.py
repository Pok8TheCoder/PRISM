"""PCAP and CIC CSV → normalized feature matrix (flow + packet levels)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features import FEATURE_COLS, NUM_FEATURES, PACKET_FEATURE_COLS


def _empty_row() -> dict:
    return {c: 0.0 for c in FEATURE_COLS}


def pcap_to_rows(pcap_path: str | Path) -> list[dict]:
    """Parse a PCAP into one feature dict per bidirectional-ish 5-tuple flow."""
    from scapy.all import IP, TCP, UDP, rdpcap

    try:
        packets = rdpcap(str(pcap_path))
    except Exception:
        return []

    flows: dict = defaultdict(list)
    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        ttl = int(ip.ttl)
        frag = 1 if (int(ip.flags) & 0x1) or int(getattr(ip, "frag", 0) or 0) else 0
        proto = "TCP" if TCP in pkt else ("UDP" if UDP in pkt else "Other")
        win = 0
        seq = None
        payload_len = 0
        flags = 0
        if TCP in pkt:
            l4 = pkt[TCP]
            flags = int(l4.flags)
            win = int(l4.window)
            seq = int(l4.seq)
            payload_len = len(bytes(l4.payload)) if l4.payload else 0
            key = (ip.src, int(l4.sport), ip.dst, int(l4.dport), proto)
        elif UDP in pkt:
            l4 = pkt[UDP]
            payload_len = len(bytes(l4.payload)) if l4.payload else 0
            key = (ip.src, int(l4.sport), ip.dst, int(l4.dport), proto)
        else:
            key = (ip.src, 0, ip.dst, 0, proto)
            payload_len = max(0, int(ip.len) - (ip.ihl * 4 if ip.ihl else 20))

        flows[key].append({
            "time": float(pkt.time),
            "len": len(pkt),
            "flags": flags,
            "dport": key[3],
            "proto": proto,
            "src": key[0],
            "ttl": ttl,
            "win": win,
            "frag": frag,
            "payload": payload_len,
            "seq": seq,
        })

    rows: list[dict] = []
    for key, pkts in flows.items():
        pkts.sort(key=lambda p: p["time"])
        fwd_src = pkts[0]["src"]
        fwd = [p for p in pkts if p["src"] == fwd_src]
        bwd = [p for p in pkts if p["src"] != fwd_src]
        times = [p["time"] for p in pkts]
        iats = np.diff(times) if len(times) > 1 else np.array([0.0])
        dur = times[-1] - times[0] if len(times) > 1 else 0.0
        fwd_lens = [p["len"] for p in fwd] or [0]
        bwd_lens = [p["len"] for p in bwd] or [0]
        fl = [p["flags"] for p in pkts]
        ttls = [p["ttl"] for p in pkts]
        wins = [p["win"] for p in pkts if p["win"]]
        payloads = [p["payload"] for p in pkts]
        seqs = [p["seq"] for p in pkts if p["seq"] is not None]
        retrans = max(0, len(seqs) - len(set(seqs))) if seqs else 0

        row = _empty_row()
        row.update({
            "Dst Port": pkts[0]["dport"],
            "Protocol": 6 if pkts[0]["proto"] == "TCP" else (17 if pkts[0]["proto"] == "UDP" else 0),
            "Flow Duration": dur * 1e6,
            "Tot Fwd Pkts": len(fwd),
            "Tot Bwd Pkts": len(bwd),
            "Fwd Pkt Len Max": max(fwd_lens),
            "Fwd Pkt Len Min": min(fwd_lens),
            "Fwd Pkt Len Mean": float(np.mean(fwd_lens)),
            "Bwd Pkt Len Max": max(bwd_lens) if bwd else 0,
            "Bwd Pkt Len Min": min(bwd_lens) if bwd else 0,
            "Bwd Pkt Len Std": float(np.std(bwd_lens)) if len(bwd) > 1 else 0,
            "Flow Byts/s": sum(p["len"] for p in pkts) / (dur + 1e-9),
            "Flow Pkts/s": len(pkts) / (dur + 1e-9),
            "Flow IAT Mean": float(np.mean(iats)),
            "Flow IAT Std": float(np.std(iats)),
            "SYN Flag Cnt": sum(1 for f in fl if f & 0x02),
            "FIN Flag Cnt": sum(1 for f in fl if f & 0x01),
            "RST Flag Cnt": sum(1 for f in fl if f & 0x04),
            "PSH Flag Cnt": sum(1 for f in fl if f & 0x08),
            "ACK Flag Cnt": sum(1 for f in fl if f & 0x10),
            "TTL Mean": float(np.mean(ttls)),
            "TTL Std": float(np.std(ttls)),
            "TCP Win Mean": float(np.mean(wins)) if wins else 0.0,
            "IP Frag Cnt": float(sum(p["frag"] for p in pkts)),
            "Payload Len Mean": float(np.mean(payloads)),
            "Payload Len Std": float(np.std(payloads)),
            "Retrans Cnt": float(retrans),
        })
        rows.append(row)
    return rows


def rows_to_matrix(rows: list[dict]) -> np.ndarray:
    if not rows:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32)
    df = pd.DataFrame(rows)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    return (
        df[FEATURE_COLS]
        .fillna(0)
        .replace([np.inf, -np.inf], 0)
        .values.astype(np.float32)
    )


def csv_to_matrix(csv_path: str | Path, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a CIC-style or PRISM feature CSV. Packet columns default to 0 if absent.

    Returns (X, y_or_none) where y is 1 for non-benign Label if present.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    if "Label" in df.columns:
        df = df[df["Label"] != "Label"]
        df["Label"] = df["Label"].astype(str).str.strip()
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows]

    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0 if c in PACKET_FEATURE_COLS else 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = None
    if "Label" in df.columns:
        y = (df["Label"].str.lower() != "benign").astype(np.int64).values
    return X, y


def load_traffic_file(path: str | Path, max_rows: int | None = 50_000) -> np.ndarray:
    """Dispatch PCAP or CSV to a feature matrix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".pcap", ".pcapng"}:
        return rows_to_matrix(pcap_to_rows(path))
    if suffix in {".csv", ".tsv"}:
        X, _ = csv_to_matrix(path, max_rows=max_rows)
        return X
    raise ValueError(f"Unsupported traffic file: {path}")
