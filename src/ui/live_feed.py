"""Live packet / flow feed for the SOC visualizer."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.adversarial.lab_config import MISSED_DIR, SAVE_DIR

LIVE_STREAM_PATH = SAVE_DIR / "live_stream.jsonl"
LOOP_STATUS_PATH = SAVE_DIR / "loop_status.json"
MAX_EVENTS = 400
MAX_PACKETS = 64

VERDICT_COLORS = {
    "intrusion": "#f85149",
    "evasion": "#a371f7",
    "misclass": "#d29922",
    "benign": "#3fb950",
    "unknown": "#8b949e",
}
VERDICT_LABELS = {
    "intrusion": "INTRUSION",
    "evasion": "EVASION",
    "misclass": "WRONG-CLASS",
    "benign": "BENIGN",
    "unknown": "UNSCORED",
}


def classify_verdict(true_class: str | None, prediction: dict | None) -> str:
    if not prediction:
        return "unknown"
    pred = prediction.get("pred_class", "Benign")
    detected = bool(prediction.get("detected"))
    if pred == "Benign" or not detected:
        if true_class and true_class != "Benign":
            return "evasion"
        return "benign"
    if true_class and pred == true_class:
        return "intrusion"
    return "misclass"


def summarize_packets(pcap_path: str | Path | None, limit: int = MAX_PACKETS) -> list[dict]:
    if not pcap_path:
        return []
    path = Path(pcap_path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        from scapy.all import ICMP, IP, TCP, UDP, rdpcap  # type: ignore
    except Exception:
        return []
    try:
        pkts = rdpcap(str(path))
    except Exception:
        return []
    rows = []
    for pkt in pkts[:limit]:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        proto = "IP"
        sport = dport = 0
        flags = ""
        if TCP in pkt:
            proto = "TCP"
            sport = int(pkt[TCP].sport)
            dport = int(pkt[TCP].dport)
            flags = str(pkt[TCP].flags)
        elif UDP in pkt:
            proto = "UDP"
            sport = int(pkt[UDP].sport)
            dport = int(pkt[UDP].dport)
        elif ICMP in pkt:
            proto = "ICMP"
        rows.append({
            "src": ip.src,
            "dst": ip.dst,
            "sport": sport,
            "dport": dport,
            "proto": proto,
            "len": int(len(pkt)),
            "flags": flags,
        })
    return rows


def _flow_ports(features: np.ndarray | None, limit: int = 24) -> list[int]:
    if features is None or len(features) == 0:
        return []
    col = features[-limit:, 0]
    return [int(v) for v in col]


def append_live_event(
    *,
    true_class: str,
    evasion: str,
    prediction: dict | None,
    pcap_path: str | Path | None,
    features: np.ndarray | None,
    flow_count: int,
    round_num: int = 0,
    attempt: int = 0,
) -> dict:
    pred = prediction or {}
    verdict = classify_verdict(true_class, prediction)
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "true_class": true_class,
        "pred_class": pred.get("pred_class", "—"),
        "confidence": float(pred.get("confidence", 0) or 0),
        "attack_prob": float(pred.get("attack_prob", 0) or 0),
        "detected": bool(pred.get("detected", False)),
        "evasion": evasion or "none",
        "verdict": verdict,
        "flow_count": int(flow_count),
        "round": round_num,
        "attempt": attempt,
        "flow_ports": _flow_ports(features),
        "packets": summarize_packets(pcap_path),
    }
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_STREAM_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    _trim_stream()
    return event


def _trim_stream():
    if not LIVE_STREAM_PATH.exists():
        return
    lines = LIVE_STREAM_PATH.read_text(encoding="utf-8").splitlines()
    if len(lines) <= MAX_EVENTS:
        return
    LIVE_STREAM_PATH.write_text("\n".join(lines[-MAX_EVENTS:]) + "\n", encoding="utf-8")


def load_live_events(limit: int = 200) -> list[dict]:
    if not LIVE_STREAM_PATH.exists():
        bootstrap_from_missed()
    if not LIVE_STREAM_PATH.exists():
        return []
    events = []
    for line in LIVE_STREAM_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def bootstrap_from_missed(limit: int = 80):
    if LIVE_STREAM_PATH.exists() and LIVE_STREAM_PATH.stat().st_size > 0:
        return
    if not MISSED_DIR.exists():
        return
    metas = sorted(MISSED_DIR.rglob("*.json"))[-limit:]
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LIVE_STREAM_PATH, "a", encoding="utf-8") as f:
        for meta_path in metas:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            true_c = meta.get("true_class_id", "")
            pred = {
                "pred_class": meta.get("predicted_class", "Benign"),
                "confidence": meta.get("confidence", 0),
                "attack_prob": meta.get("attack_prob", 0),
                "detected": meta.get("reason") == "misclassified"
                or (
                    meta.get("predicted_class") == true_c
                    and (meta.get("confidence") or 0) >= 0.55
                ),
            }
            reason = meta.get("reason", "")
            if reason == "evaded":
                verdict = "evasion"
            elif reason == "misclassified":
                verdict = "misclass"
            else:
                verdict = classify_verdict(true_c, pred)
            event = {
                "ts": meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "true_class": true_c,
                "pred_class": pred["pred_class"],
                "confidence": float(pred["confidence"] or 0),
                "attack_prob": float(pred["attack_prob"] or 0),
                "detected": bool(pred["detected"]),
                "evasion": meta.get("evasion", "none"),
                "verdict": verdict,
                "flow_count": 0,
                "round": meta.get("round", 0),
                "attempt": meta.get("attempt", 0),
                "flow_ports": [],
                "packets": [],
            }
            f.write(json.dumps(event) + "\n")


def loop_status() -> dict[str, Any]:
    if not LOOP_STATUS_PATH.exists():
        return {"status": "idle"}
    try:
        return json.loads(LOOP_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "idle"}


def write_loop_status(status: str, extra: dict | None = None):
    payload = {"status": status, "ts": datetime.now(timezone.utc).isoformat()}
    if extra:
        payload.update(extra)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOOP_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _counts(events: list[dict]) -> dict[str, int]:
    out = {k: 0 for k in VERDICT_COLORS}
    for e in events:
        out[e.get("verdict", "unknown")] = out.get(e.get("verdict", "unknown"), 0) + 1
    return out


def render_packet_visualizer():
    events = load_live_events()
    status = loop_status()
    running = status.get("status") == "running"
    counts = _counts(events)
    latest = events[-1] if events else None

    st.markdown(
        f'<div class="wire-banner">'
        f'<span class="wire-live {"on" if running else "off"}">'
        f'{"● LIVE LOOP" if running else "○ IDLE"}</span>'
        f'<span class="wire-legend">'
        f'<i class="dot intrusion"></i> Intrusion '
        f'<i class="dot evasion"></i> Evasion '
        f'<i class="dot misclass"></i> Wrong-class'
        f'</span></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Intrusions", counts["intrusion"])
    m2.metric("Evasions", counts["evasion"])
    m3.metric("Wrong-class", counts["misclass"])
    m4.metric("Events", len(events))

    if not events:
        st.info("No scored traffic yet. Run an attack or the 20-min loop.")
        return

    left, right = st.columns([3, 2])
    with left:
        _plot_packet_sky(events)
    with right:
        _plot_verdict_mix(counts)
        if latest:
            color = VERDICT_COLORS.get(latest["verdict"], "#8b949e")
            label = VERDICT_LABELS.get(latest["verdict"], latest["verdict"])
            st.markdown(
                f'<div class="verdict-card" style="border-color:{color}">'
                f'<div class="verdict-tag" style="color:{color}">{html.escape(label)}</div>'
                f'<div class="verdict-true">true <code>{html.escape(str(latest["true_class"]))}</code></div>'
                f'<div class="verdict-pred">model <code>{html.escape(str(latest["pred_class"]))}</code>'
                f' · {latest["confidence"]:.0%} · p(attack) {latest["attack_prob"]:.0%}</div>'
                f'<div class="verdict-meta">evasion={html.escape(str(latest["evasion"]))} · '
                f'{latest["flow_count"]} flows · {len(latest.get("packets") or [])} pkts</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.subheader("Packet wire")
    _render_packet_ticker(events)


def _plot_packet_sky(events: list[dict]):
    xs, ys, colors, sizes, hover = [], [], [], [], []
    x = 0
    for ev in events[-80:]:
        pkts = ev.get("packets") or []
        ports = ev.get("flow_ports") or []
        verdict = ev.get("verdict", "unknown")
        color = VERDICT_COLORS.get(verdict, "#8b949e")
        label = VERDICT_LABELS.get(verdict, verdict)
        if pkts:
            for pkt in pkts:
                xs.append(x)
                ys.append(pkt.get("dport") or 0)
                colors.append(color)
                sizes.append(max(6, min(18, (pkt.get("len") or 40) / 40)))
                hover.append(
                    f"{label}<br>{pkt.get('src')}:{pkt.get('sport')} → "
                    f"{pkt.get('dst')}:{pkt.get('dport')} {pkt.get('proto')} "
                    f"{pkt.get('len')}B {pkt.get('flags')}<br>"
                    f"true {ev.get('true_class')} · pred {ev.get('pred_class')}"
                )
                x += 1
        elif ports:
            for port in ports:
                xs.append(x)
                ys.append(port)
                colors.append(color)
                sizes.append(8)
                hover.append(
                    f"{label}<br>flow dst_port={port}<br>"
                    f"true {ev.get('true_class')} · pred {ev.get('pred_class')}"
                )
                x += 1
        else:
            xs.append(x)
            ys.append(hash(ev.get("true_class") or "") % 1024)
            colors.append(color)
            sizes.append(10)
            hover.append(
                f"{label}<br>true {ev.get('true_class')} · pred {ev.get('pred_class')}"
            )
            x += 1

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.85, line=dict(width=0)),
        text=hover, hoverinfo="text",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=360,
        margin=dict(l=40, r=10, t=36, b=36),
        title="Packet sky — dest port over capture order",
        xaxis_title="Packet / flow (time →)",
        yaxis_title="Destination port",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#21262d", zeroline=False)
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    st.plotly_chart(fig, width="stretch")


def _plot_verdict_mix(counts: dict[str, int]):
    labels = ["intrusion", "evasion", "misclass"]
    fig = go.Figure(go.Pie(
        labels=[VERDICT_LABELS[k] for k in labels],
        values=[counts[k] for k in labels],
        hole=0.62,
        marker=dict(colors=[VERDICT_COLORS[k] for k in labels]),
        textinfo="label+value",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        height=220,
        margin=dict(l=10, r=10, t=30, b=10),
        title="Verdict mix",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")


def _render_packet_ticker(events: list[dict]):
    rows = []
    for ev in reversed(events[-12:]):
        pkts = ev.get("packets") or []
        verdict = ev.get("verdict", "unknown")
        color = VERDICT_COLORS.get(verdict, "#8b949e")
        tag = VERDICT_LABELS.get(verdict, verdict)
        if pkts:
            for pkt in pkts[-8:]:
                line = (
                    f"{pkt.get('src')}:{pkt.get('sport')} → "
                    f"{pkt.get('dst')}:{pkt.get('dport')} "
                    f"{pkt.get('proto')} {pkt.get('len')}B {pkt.get('flags')}"
                )
                rows.append(_pkt_row(color, tag, line, ev))
        else:
            line = (
                f"capture  true={ev.get('true_class')}  "
                f"pred={ev.get('pred_class')}  evasion={ev.get('evasion')}"
            )
            rows.append(_pkt_row(color, tag, line, ev))
    st.markdown(
        '<div class="pkt-feed">' + "".join(rows[:40]) + "</div>",
        unsafe_allow_html=True,
    )


def _pkt_row(color: str, tag: str, line: str, ev: dict) -> str:
    return (
        f'<div class="pkt-row" style="border-left-color:{color}">'
        f'<span class="pkt-tag" style="color:{color}">{html.escape(tag)}</span>'
        f'<span class="pkt-line">{html.escape(line)}</span>'
        f'<span class="pkt-cls">{html.escape(str(ev.get("true_class")))}</span>'
        f"</div>"
    )
