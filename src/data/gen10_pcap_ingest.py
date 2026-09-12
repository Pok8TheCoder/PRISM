"""PCAP → raw 249-d Gen10/Laplace states + per-window flow context.

Uses StateBuilder (graph topology + flow stats) on 5s windows — the native
feature path Laplace was trained on. No pre-baked NPZ splits required.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.state_builder import StateBuilder, _is_private_ip_fast

GEN10_STATE_DIM = 249
DEFAULT_WINDOW_SEC = 5.0


def pcap_to_flow_dataframe(pcap_path: str | Path) -> pd.DataFrame:
    """Parse PCAP into per-flow rows with IPs, ports, and CIC-style columns."""
    from scapy.all import IP, TCP, UDP, rdpcap

    path = Path(pcap_path)
    try:
        packets = rdpcap(str(path))
    except Exception:
        return pd.DataFrame()

    flows: dict[tuple, list[dict]] = defaultdict(list)
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
        sport = 0
        dport = 0
        if TCP in pkt:
            l4 = pkt[TCP]
            flags = int(l4.flags)
            win = int(l4.window)
            seq = int(l4.seq)
            sport = int(l4.sport)
            dport = int(l4.dport)
            payload_len = len(bytes(l4.payload)) if l4.payload else 0
        elif UDP in pkt:
            l4 = pkt[UDP]
            sport = int(l4.sport)
            dport = int(l4.dport)
            payload_len = len(bytes(l4.payload)) if l4.payload else 0
        else:
            payload_len = max(0, int(ip.len) - (ip.ihl * 4 if ip.ihl else 20))

        flows[(str(ip.src), sport, str(ip.dst), dport, proto)].append({
            "time": float(pkt.time),
            "len": len(pkt),
            "flags": flags,
            "ttl": ttl,
            "frag": frag,
            "win": win,
            "payload": payload_len,
            "seq": seq,
            "src_ip": str(ip.src),
            "dst_ip": str(ip.dst),
            "sport": sport,
            "dport": dport,
            "proto": proto,
        })

    rows: list[dict[str, Any]] = []
    for (_src, _sport, _dst, _dport, _proto), pkts in flows.items():
        pkts.sort(key=lambda p: p["time"])
        fwd_src = pkts[0]["src_ip"]
        fwd = [p for p in pkts if p["src_ip"] == fwd_src]
        bwd = [p for p in pkts if p["src_ip"] != fwd_src]
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
        fwd_bytes = sum(p["len"] for p in fwd)
        bwd_bytes = sum(p["len"] for p in bwd)

        rows.append({
            "Timestamp": times[0],
            "Src IP": pkts[0]["src_ip"],
            "Dst IP": pkts[0]["dst_ip"],
            "Dst Port": int(pkts[0]["dport"]),
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
            "Flow Byts/s": (fwd_bytes + bwd_bytes) / (dur + 1e-9),
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
            "_fwd_bytes": float(fwd_bytes),
            "_bwd_bytes": float(bwd_bytes),
            "_src_priv": _is_private_ip_fast(pkts[0]["src_ip"]),
            "_dst_priv": _is_private_ip_fast(pkts[0]["dst_ip"]),
        })

    return pd.DataFrame(rows)


def _assign_windows(df: pd.DataFrame, window_sec: float) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_numeric(df["Timestamp"], errors="coerce")
    if ts.notna().any():
        t0 = float(ts.dropna().iloc[0])
        df["window_id"] = ((ts.fillna(t0) - t0) // window_sec).astype(int)
    else:
        df["window_id"] = np.arange(len(df)) // 32
    return df


def _flow_context_for_window(window: pd.DataFrame) -> dict[str, Any]:
    """Dominant flow in window — feeds Laplace Tier-2 attribution."""
    if window.empty:
        return {}
    vol = window["_fwd_bytes"].fillna(0) + window["_bwd_bytes"].fillna(0)
    row = window.iloc[int(vol.argmax())]
    return {
        "src_ip": str(row.get("Src IP", "")),
        "dst_ip": str(row.get("Dst IP", "")),
        "dst_port": int(row.get("Dst Port", 0) or 0),
        "fwd_bytes": float(row.get("_fwd_bytes", 0) or 0),
        "bwd_bytes": float(row.get("_bwd_bytes", 0) or 0),
        "flow_count": int(len(window)),
        "unique_dst_ports": int(window["Dst Port"].nunique()) if "Dst Port" in window.columns else 1,
    }


_SCALER_MEAN: np.ndarray | None = None


def _get_neutral_tail(start_idx: int, target_dim: int) -> np.ndarray:
    """Return expm1 of scaler training mean for unobserved feature columns.
    
    Guarantees that log1p + StandardScaler produces exactly 0.0 Z-score (neutral baseline)
    rather than artificial negative bias.
    """
    global _SCALER_MEAN
    pad_len = max(0, target_dim - start_idx)
    if pad_len == 0:
        return np.zeros(0, dtype=np.float32)

    if _SCALER_MEAN is None:
        scaler_path = Path(__file__).resolve().parent.parent.parent / "weights" / "universal_gen10_5s_scaler.pkl"
        if scaler_path.is_file():
            try:
                import joblib
                s = joblib.load(scaler_path)
                if hasattr(s, "mean_"):
                    _SCALER_MEAN = s.mean_.astype(np.float32)
            except Exception:
                pass

    if _SCALER_MEAN is not None and len(_SCALER_MEAN) >= target_dim:
        return np.expm1(_SCALER_MEAN[start_idx:target_dim]).astype(np.float32)
    return np.zeros(pad_len, dtype=np.float32)


def pcap_to_gen10_windows(
    pcap_path: str | Path,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
) -> dict[str, Any]:
    """
    Build raw 249-d states and flow contexts from a PCAP.

    Returns dict with keys: states (T, 249), flow_contexts (list[dict]), feature_names.
    """
    df = pcap_to_flow_dataframe(pcap_path)
    if df.empty:
        return {
            "states": np.zeros((0, GEN10_STATE_DIM), dtype=np.float32),
            "flow_contexts": [],
            "feature_names": [],
            "window_ids": np.array([], dtype=np.int64),
        }

    df = _assign_windows(df, window_sec)
    builder = StateBuilder(window_size_seconds=int(window_sec))
    built = builder.build_states(df, label_col="mitre_stage_id")

    states = built["states"]
    if states.size and states.shape[1] < GEN10_STATE_DIM:
        tail = _get_neutral_tail(states.shape[1], GEN10_STATE_DIM)
        pad_matrix = np.tile(tail, (states.shape[0], 1))
        states = np.hstack([states, pad_matrix])
    elif states.size and states.shape[1] > GEN10_STATE_DIM:
        states = states[:, :GEN10_STATE_DIM]

    contexts: list[dict[str, Any]] = []
    grouped = df.groupby("window_id", sort=True)
    for wid in sorted(df["window_id"].unique()):
        contexts.append(_flow_context_for_window(grouped.get_group(wid)))

    return {
        "states": states.astype(np.float32),
        "flow_contexts": contexts,
        "feature_names": built.get("feature_names", builder.feature_names),
        "window_ids": built.get("window_ids", np.arange(len(states))),
    }
