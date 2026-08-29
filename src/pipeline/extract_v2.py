"""Build v2 (YMT.01) state sequences from PCAPs and CIC-IDS-2018 CSVs.

Pipeline: raw source -> base flow matrix (N, 24) -> contiguous blocks of
SEQ_LEN flows -> per-timestep causal trailing-window aggregation -> (T, 64).

The aggregation at timestep ``t`` only ever looks at flows ``t-7 .. t``, never
forward, so the representation is computable by a live sensor with no lookahead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.features_v2 import (
    BASE_IDX,
    DB_PORTS,
    NUM_BASE,
    NUM_FEATURES_V2,
    REMOTE_ADMIN_PORTS,
    WEB_PORTS,
    WINDOW_FLOWS,
)

EPS = 1e-6


def slog(x: np.ndarray) -> np.ndarray:
    """Signed log compression for heavy-tailed rate columns."""
    return np.sign(x) * np.log1p(np.abs(x))


# ── Source readers → base flow matrix ─────────────────────────────────────────
def pcap_to_base(pcap_path: str | Path) -> np.ndarray:
    """Parse a PCAP into the (N, 24) base flow matrix, ordered by flow start."""
    from collections import defaultdict

    from scapy.all import IP, TCP, UDP, rdpcap

    try:
        packets = rdpcap(str(pcap_path))
    except Exception:
        return np.zeros((0, NUM_BASE), dtype=np.float32)

    flows: dict = defaultdict(list)
    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        frag = 1 if (int(ip.flags) & 0x1) or int(getattr(ip, "frag", 0) or 0) else 0
        if TCP in pkt:
            l4, proto = pkt[TCP], 6
            key = (ip.src, int(l4.sport), ip.dst, int(l4.dport), proto)
            flags, win = int(l4.flags), int(l4.window)
            seq = int(l4.seq)
            payload = len(bytes(l4.payload)) if l4.payload else 0
        elif UDP in pkt:
            l4, proto = pkt[UDP], 17
            key = (ip.src, int(l4.sport), ip.dst, int(l4.dport), proto)
            flags, win, seq = 0, 0, None
            payload = len(bytes(l4.payload)) if l4.payload else 0
        else:
            proto = 1 if ip.proto == 1 else 0
            key = (ip.src, 0, ip.dst, 0, proto)
            flags, win, seq = 0, 0, None
            payload = max(0, int(ip.len) - (ip.ihl * 4 if ip.ihl else 20))

        flows[key].append({
            "t": float(pkt.time), "len": len(pkt), "flags": flags,
            "src": ip.src, "ttl": int(ip.ttl), "win": win,
            "frag": frag, "payload": payload, "seq": seq,
            "dport": key[3], "proto": proto,
        })

    records = []
    for pkts in flows.values():
        pkts.sort(key=lambda p: p["t"])
        fwd_src = pkts[0]["src"]
        fwd = [p for p in pkts if p["src"] == fwd_src]
        bwd = [p for p in pkts if p["src"] != fwd_src]
        times = [p["t"] for p in pkts]
        iats = np.diff(times) if len(times) > 1 else np.array([0.0])
        dur = times[-1] - times[0] if len(times) > 1 else 0.0
        fl = [p["flags"] for p in pkts]
        wins = [p["win"] for p in pkts if p["win"]]
        seqs = [p["seq"] for p in pkts if p["seq"] is not None]
        fwd_lens = [p["len"] for p in fwd] or [0]
        bwd_lens = [p["len"] for p in bwd] or [0]

        row = np.zeros(NUM_BASE, dtype=np.float64)
        row[BASE_IDX["dst_port"]] = pkts[0]["dport"]
        row[BASE_IDX["protocol"]] = pkts[0]["proto"]
        row[BASE_IDX["duration_us"]] = dur * 1e6
        row[BASE_IDX["fwd_pkts"]] = len(fwd)
        row[BASE_IDX["bwd_pkts"]] = len(bwd)
        row[BASE_IDX["fwd_bytes"]] = sum(fwd_lens) if fwd else 0
        row[BASE_IDX["bwd_bytes"]] = sum(bwd_lens) if bwd else 0
        row[BASE_IDX["byts_s"]] = sum(p["len"] for p in pkts) / (dur + 1e-9)
        row[BASE_IDX["pkts_s"]] = len(pkts) / (dur + 1e-9)
        row[BASE_IDX["iat_mean"]] = float(np.mean(iats)) * 1e6
        row[BASE_IDX["iat_std"]] = float(np.std(iats)) * 1e6
        row[BASE_IDX["fwd_len_mean"]] = float(np.mean(fwd_lens))
        row[BASE_IDX["bwd_len_max"]] = max(bwd_lens) if bwd else 0
        row[BASE_IDX["syn_cnt"]] = sum(1 for f in fl if f & 0x02)
        row[BASE_IDX["ack_cnt"]] = sum(1 for f in fl if f & 0x10)
        row[BASE_IDX["fin_cnt"]] = sum(1 for f in fl if f & 0x01)
        row[BASE_IDX["rst_cnt"]] = sum(1 for f in fl if f & 0x04)
        row[BASE_IDX["psh_cnt"]] = sum(1 for f in fl if f & 0x08)
        row[BASE_IDX["payload_mean"]] = float(np.mean([p["payload"] for p in pkts]))
        row[BASE_IDX["ttl_mean"]] = float(np.mean([p["ttl"] for p in pkts]))
        row[BASE_IDX["ttl_std"]] = float(np.std([p["ttl"] for p in pkts]))
        row[BASE_IDX["tcpwin_mean"]] = float(np.mean(wins)) if wins else 0.0
        row[BASE_IDX["retrans_cnt"]] = max(0, len(seqs) - len(set(seqs))) if seqs else 0
        row[BASE_IDX["frag_cnt"]] = float(sum(p["frag"] for p in pkts))
        row[BASE_IDX["fwd_len_max"]] = max(fwd_lens)
        row[BASE_IDX["fwd_len_min"]] = min(fwd_lens)
        row[BASE_IDX["bwd_len_min"]] = min(bwd_lens) if bwd else 0
        row[BASE_IDX["bwd_len_std"]] = float(np.std(bwd_lens)) if len(bwd) > 1 else 0.0
        row[BASE_IDX["payload_std"]] = float(np.std([p["payload"] for p in pkts]))
        records.append((times[0], row))

    if not records:
        return np.zeros((0, NUM_BASE), dtype=np.float32)
    records.sort(key=lambda r: r[0])
    out = np.stack([r[1] for r in records]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def cic_df_to_base(df: pd.DataFrame) -> np.ndarray:
    """Map CIC-IDS-2018 CSV columns onto the base flow matrix.

    TTL, retransmission and fragmentation have no CSV equivalent and stay zero.
    TCP window is recovered from ``Init Fwd Win Byts``, and payload size from
    ``Pkt Size Avg``, so only 3 of 24 base columns are PCAP-exclusive.
    """
    n = len(df)
    out = np.zeros((n, NUM_BASE), dtype=np.float32)

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.zeros(n, dtype=np.float64)
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(np.float64)

    fwd_pkts, bwd_pkts = col("Tot Fwd Pkts"), col("Tot Bwd Pkts")
    out[:, BASE_IDX["dst_port"]] = col("Dst Port")
    out[:, BASE_IDX["protocol"]] = col("Protocol")
    out[:, BASE_IDX["duration_us"]] = col("Flow Duration")
    out[:, BASE_IDX["fwd_pkts"]] = fwd_pkts
    out[:, BASE_IDX["bwd_pkts"]] = bwd_pkts
    out[:, BASE_IDX["fwd_bytes"]] = col("TotLen Fwd Pkts")
    out[:, BASE_IDX["bwd_bytes"]] = col("TotLen Bwd Pkts")
    out[:, BASE_IDX["byts_s"]] = col("Flow Byts/s")
    out[:, BASE_IDX["pkts_s"]] = col("Flow Pkts/s")
    out[:, BASE_IDX["iat_mean"]] = col("Flow IAT Mean")
    out[:, BASE_IDX["iat_std"]] = col("Flow IAT Std")
    out[:, BASE_IDX["fwd_len_mean"]] = col("Fwd Pkt Len Mean")
    out[:, BASE_IDX["bwd_len_max"]] = col("Bwd Pkt Len Max")
    out[:, BASE_IDX["syn_cnt"]] = col("SYN Flag Cnt")
    out[:, BASE_IDX["ack_cnt"]] = col("ACK Flag Cnt")
    out[:, BASE_IDX["fin_cnt"]] = col("FIN Flag Cnt")
    out[:, BASE_IDX["rst_cnt"]] = col("RST Flag Cnt")
    out[:, BASE_IDX["psh_cnt"]] = col("PSH Flag Cnt")
    out[:, BASE_IDX["payload_mean"]] = col("Pkt Size Avg")
    out[:, BASE_IDX["tcpwin_mean"]] = np.clip(col("Init Fwd Win Byts"), 0, None)
    out[:, BASE_IDX["fwd_len_max"]] = col("Fwd Pkt Len Max")
    out[:, BASE_IDX["fwd_len_min"]] = col("Fwd Pkt Len Min")
    out[:, BASE_IDX["bwd_len_min"]] = col("Bwd Pkt Len Min")
    out[:, BASE_IDX["bwd_len_std"]] = col("Bwd Pkt Len Std")
    out[:, BASE_IDX["payload_std"]] = col("Pkt Len Std")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def base_to_v1(base: np.ndarray) -> np.ndarray:
    """Project base flow records onto the 27-column v1 (ZMT.01) schema.

    Deriving v1 here rather than from ``pipeline.extract`` guarantees both
    models are fed the same flows in the same order. One deviation from the
    shipped v1 reader is intentional: ``pcap_to_rows`` reports Flow IAT in
    seconds while the CIC CSVs report microseconds, so v1 silently mixes units
    across its two sources. Base records are microseconds throughout, which
    hands that fix to the baseline rather than crediting it to v2.
    """
    from src.pipeline.features import FEATURE_COLS

    mapping = {
        "Dst Port": "dst_port", "Protocol": "protocol",
        "Flow Duration": "duration_us", "Tot Fwd Pkts": "fwd_pkts",
        "Tot Bwd Pkts": "bwd_pkts", "Fwd Pkt Len Max": "fwd_len_max",
        "Fwd Pkt Len Min": "fwd_len_min", "Fwd Pkt Len Mean": "fwd_len_mean",
        "Bwd Pkt Len Max": "bwd_len_max", "Bwd Pkt Len Min": "bwd_len_min",
        "Bwd Pkt Len Std": "bwd_len_std", "Flow Byts/s": "byts_s",
        "Flow Pkts/s": "pkts_s", "Flow IAT Mean": "iat_mean",
        "Flow IAT Std": "iat_std", "SYN Flag Cnt": "syn_cnt",
        "FIN Flag Cnt": "fin_cnt", "RST Flag Cnt": "rst_cnt",
        "PSH Flag Cnt": "psh_cnt", "ACK Flag Cnt": "ack_cnt",
        "TTL Mean": "ttl_mean", "TTL Std": "ttl_std",
        "TCP Win Mean": "tcpwin_mean", "IP Frag Cnt": "frag_cnt",
        "Payload Len Mean": "payload_mean", "Payload Len Std": "payload_std",
        "Retrans Cnt": "retrans_cnt",
    }
    idx = [BASE_IDX[mapping[c]] for c in FEATURE_COLS]
    return base[..., idx].astype(np.float32)


# ── Window aggregation ────────────────────────────────────────────────────────
def _counts_uniq_entropy(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct count and Shannon entropy over a small window, vectorized.

    For each element i let c_i be how many window entries equal it. Then
    sum_j p_j log p_j == (1/w) sum_i log(c_i / w), which avoids materialising
    a per-value histogram.
    """
    w = vals.shape[1]
    eq = vals[:, :, None] == vals[:, None, :]
    c = eq.sum(axis=2).astype(np.float64)
    uniq = (1.0 / c).sum(axis=1)
    entropy = -(np.log2(c / w)).sum(axis=1) / w
    return uniq, entropy


def _window_features(win: np.ndarray) -> np.ndarray:
    """Aggregate one trailing window (n, w, 24) into (n, 64) v2 features."""
    n, w, _ = win.shape
    g = lambda name: win[:, :, BASE_IDX[name]].astype(np.float64)  # noqa: E731

    ports, proto = g("dst_port"), g("protocol")
    fwd_pkts, bwd_pkts = g("fwd_pkts"), g("bwd_pkts")
    fwd_bytes, bwd_bytes = g("fwd_bytes"), g("bwd_bytes")
    dur = g("duration_us")

    feats: list[np.ndarray] = []

    # ── Block A: window composition ──
    uniq_ports, port_ent = _counts_uniq_entropy(ports)
    _, proto_ent = _counts_uniq_entropy(proto)
    sorted_ports = np.sort(ports, axis=1)
    seq_hits = (np.diff(sorted_ports, axis=1) == 1.0).sum(axis=1)
    total_dur_s = dur.sum(axis=1) / 1e6
    feats += [
        np.full(n, float(w)),
        uniq_ports,
        port_ent,
        uniq_ports / w,
        seq_hits / max(w - 1, 1),
        slog(sorted_ports[:, -1] - sorted_ports[:, 0]),
        proto_ent,
        slog(w / (total_dur_s + EPS)),
    ]

    # ── Block B: protocol mix ──
    is_tcp, is_udp, is_icmp = (proto == 6), (proto == 17), (proto == 1)
    feats += [
        is_tcp.mean(axis=1),
        is_udp.mean(axis=1),
        is_icmp.mean(axis=1),
        (~(is_tcp | is_udp | is_icmp)).mean(axis=1),
    ]

    # ── Block C: destination port class mix ──
    feats += [
        (ports < 1024).mean(axis=1),
        ((ports >= 1024) & (ports < 49152)).mean(axis=1),
        (ports >= 49152).mean(axis=1),
        np.isin(ports, WEB_PORTS).mean(axis=1),
        np.isin(ports, REMOTE_ADMIN_PORTS).mean(axis=1),
        np.isin(ports, DB_PORTS).mean(axis=1),
    ]

    # ── Block D: dispersion (mean + std of 13 derived quantities) ──
    tot_pkts = fwd_pkts + bwd_pkts
    tot_bytes = fwd_bytes + bwd_bytes
    derived = np.stack([
        slog(dur),
        fwd_pkts,
        bwd_pkts,
        fwd_pkts / (bwd_pkts + 1.0),
        slog(g("byts_s")),
        slog(g("pkts_s")),
        slog(g("iat_mean")),
        slog(g("iat_std")),
        g("fwd_len_mean"),
        g("bwd_len_max"),
        tot_bytes / (tot_pkts + 1.0),
        g("payload_mean"),
        (fwd_bytes - bwd_bytes) / (tot_bytes + 1.0),
    ], axis=2)
    feats += list(derived.mean(axis=1).T)
    feats += list(derived.std(axis=1).T)

    # ── Block E: TCP flag profile ──
    syn, ack = g("syn_cnt").sum(axis=1), g("ack_cnt").sum(axis=1)
    fin, rst = g("fin_cnt").sum(axis=1), g("rst_cnt").sum(axis=1)
    psh = g("psh_cnt").sum(axis=1)
    pkt_tot = tot_pkts.sum(axis=1) + 1.0
    flagless = (
        g("syn_cnt") + g("ack_cnt") + g("fin_cnt") + g("rst_cnt") + g("psh_cnt")
    ) == 0
    feats += [
        syn / pkt_tot, ack / pkt_tot, fin / pkt_tot, rst / pkt_tot, psh / pkt_tot,
        syn / (ack + 1.0),
        rst / w,
        flagless.mean(axis=1),
    ]

    # ── Block F: packet-level anomaly ──
    ttl = g("ttl_mean")
    tcpwin = g("tcpwin_mean")
    feats += [
        ttl.mean(axis=1),
        ttl.std(axis=1),
        ttl.max(axis=1) - ttl.min(axis=1),
        slog(tcpwin.mean(axis=1)),
        slog(tcpwin.std(axis=1)),
        g("retrans_cnt").sum(axis=1) / pkt_tot,
        g("frag_cnt").sum(axis=1) / pkt_tot,
        (g("payload_mean") == 0).mean(axis=1),
    ]

    # ── Block G: current-flow passthrough ──
    cur = win[:, -1, :].astype(np.float64)
    feats += [
        slog(cur[:, BASE_IDX["dst_port"]]),
        cur[:, BASE_IDX["protocol"]],
        slog(cur[:, BASE_IDX["duration_us"]]),
        cur[:, BASE_IDX["payload_mean"]],
    ]

    out = np.stack(feats, axis=1).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def blocks_to_v2(blocks: np.ndarray, window: int = WINDOW_FLOWS) -> np.ndarray:
    """Convert (n, T, 24) flow blocks into (n, T, 64) v2 state sequences.

    Timestep ``t`` aggregates flows ``max(0, t-window+1) .. t`` only.
    """
    n, T, _ = blocks.shape
    out = np.zeros((n, T, NUM_FEATURES_V2), dtype=np.float32)
    for t in range(T):
        lo = max(0, t - window + 1)
        out[:, t, :] = _window_features(blocks[:, lo:t + 1, :])
    return out
