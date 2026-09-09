"""Forecast Player tab: scrub through a synthetic real-data kill-chain
timeline while Aryan's real model (frozen and/or RAM-A.01) forecasts ahead
of a moving "now" line, self-healing as ground truth arrives.

Backed by `src/ui/forecast_sessions.py`. See
`docs/ARY01_VS_ARY02.md` and `results/forecast/v8_ram_aryan_killchain/`
for the research this visualizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import forecast_sessions as fs

TIMELINE_CHOICES = [500, 1000, 2000]
PAST_WINDOW = 80
FUTURE_WINDOW = 25
TICK_SECONDS = 0.4             # fragment refresh interval. Each tick rebuilds 3 Plotly
                                # figures + the error table, which isn't free -- too low a
                                # value here saturates a CPU core continuously (measured
                                # ~100% of one core at 0.15s) and starves the rest of the
                                # app/session, which is what made Play "not do much".
TRANSITION_MS = 140            # plotly animates each redraw over this long instead of a hard cut

SLOT_LINE_COLORS = {1: "#27ae60", 2: "#8e44ad", 3: "#e67e22"}

# Shading palette — past vs future use different hues so overlaps read at a glance.
# Overview chart uses GT_ATTACK_* only (labels, not model output).
GT_ATTACK_PAST_FILL = "rgba(248,81,73,0.14)"
GT_ATTACK_PAST_EDGE = "#f85149"
GT_ATTACK_FUTURE_EDGE = "rgba(248,81,73,0.45)"   # dashed hint only, no fill

PAST_ANOMALY_FILL = "rgba(56,139,253,0.32)"
PAST_ANOMALY_EDGE = "rgba(56,139,253,0.55)"
FUTURE_ANOMALY_FILL = "rgba(125,211,252,0.16)"
FUTURE_ANOMALY_EDGE = "rgba(125,211,252,0.45)"

PAST_INTRUSION_FILL = "rgba(248,81,73,0.38)"
PAST_INTRUSION_EDGE = "rgba(248,81,73,0.75)"
FUTURE_INTRUSION_FILL = "rgba(255,166,87,0.22)"   # orange = model forecast suspicion
FUTURE_INTRUSION_EDGE = "rgba(255,166,87,0.55)"

MEMORY_FILL = "rgba(255,193,7,0.28)"
MEMORY_EDGE = "rgba(255,193,7,0.55)"

NOW_LINE = dict(color="#c9d1d9", width=1.6, dash="dot")
CHART_LAYOUT_BASE = dict(
    template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
    transition=dict(duration=TRANSITION_MS, easing="cubic-in-out"),
    uirevision="fp-const",
)


def _init_state():
    defaults = {
        "fp_timeline_len": 1000,
        "fp_feature_idx": None,
        "fp_slot1": "ary_01",
        "fp_slot2": "ary_02",
        "fp_slot3": "xmt_01" if (fs.CKPT_XMT01.exists()) else "none",
        "fp_playhead": None,
        "fp_playing": False,
        "fp_speed": 8,       # steps / second
        "fp_accum": 0.0,     # fractional step accumulator for smooth low speeds
        "fp_source": "synthetic",
        "fp_upload_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def _get_recording(model_choice: str, timeline_len: int, source: str, upload_key: str,
                   _build_version: int = fs.RECORDING_BUILD_VERSION) -> fs.Recording:
    upload_path = None
    if source == "upload" and upload_key:
        upload_path = upload_key
    return fs.build_recording(model_choice, timeline_len, source=source, uploaded_path=upload_path)


@st.cache_data(show_spinner=False)
def _get_feature_options() -> list[int]:
    return fs.top_variance_features(12)


def _render_top_controls():
    st.subheader("Forecast Player")
    st.caption(
        "Replays a synthetic multi-attack timeline built from Aryan's real held-out "
        "CIC-IDS-2018 windows through his real trained checkpoint (no retraining). "
        "**Overview (top):** red shading = ground-truth attack labels (not model output) — only "
        "filled up to the playhead; future attacks show as a faint dashed line at their start. "
        "**Model slots:** solid blue/red = model signals on **revealed** real data; pale "
        "cyan/orange = the model's own **forecast** cone ahead of now (no ground truth used). "
        "**Yellow** = RAM memory writes (revealed steps only). "
        "ARY.01 = CIC epoch 13. ARY.02 = CIC retrain epoch 22. "
        "XMT.01 = same 242-d features, trained on PRISM lab PCAPs (slot 3)."
    )
    src, c1, c2 = st.columns([1.2, 1, 1])
    with src:
        st.selectbox(
            "Traffic source",
            ["synthetic", "xmt_lab", "upload", "live"],
            format_func=lambda k: {
                "synthetic": "Synthetic CIC kill-chain",
                "xmt_lab": "XMT lab test timeline",
                "upload": "Uploaded PCAP / CIC CSV",
                "live": "Lab live captures",
            }[k],
            key="fp_source",
        )
    with c1:
        st.selectbox("Timeline length (steps)", TIMELINE_CHOICES, key="fp_timeline_len")
    with c2:
        feat_opts = _get_feature_options()
        if st.session_state.fp_feature_idx not in feat_opts:
            st.session_state.fp_feature_idx = feat_opts[0]
        st.selectbox("Feature to plot (top-variance raw dims)", feat_opts,
                     format_func=lambda i: f"feature[{i}]", key="fp_feature_idx")

    if st.session_state.fp_source == "upload":
        up = st.file_uploader("PCAP or CIC CSV", type=["pcap", "pcapng", "csv"], key="fp_upload")
        if up is not None:
            dest = Path("data") / "raw" / "player_uploads"
            dest.mkdir(parents=True, exist_ok=True)
            saved = dest / up.name
            saved.write_bytes(up.getvalue())
            st.session_state.fp_upload_name = str(saved)
            st.caption(f"Using {saved}")
    elif st.session_state.fp_source == "live":
        st.caption("Newest lab pcaps under data/raw/adversarial are windowed into 242-d states. Run an attack if this is empty.")

    s1, s2, s3 = st.columns(3)
    choices = list(fs.MODEL_CHOICES.keys())
    with s1:
        st.selectbox("Model slot 1", choices, format_func=lambda k: fs.MODEL_CHOICES[k], key="fp_slot1")
    with s2:
        st.selectbox("Model slot 2", choices, format_func=lambda k: fs.MODEL_CHOICES[k], key="fp_slot2")
    with s3:
        st.selectbox("Model slot 3 (reserved for a future candidate)", choices,
                     format_func=lambda k: fs.MODEL_CHOICES[k], key="fp_slot3")


def _clip_mask_side(mask: np.ndarray, playhead: int, side: str) -> np.ndarray:
    out = mask.copy()
    idx = np.arange(len(mask))
    if side == "past":
        out[idx > playhead] = False
    else:
        out[idx <= playhead] = False
    return out


def _intrusion_mask(probs: np.ndarray) -> np.ndarray:
    return np.nan_to_num(probs, nan=0.0) > fs.INTRUSION_PROB_THRESH


def _mask_runs(mask: np.ndarray, lo: int, hi: int) -> list[tuple[int, int]]:
    idxs = np.nonzero(mask[lo:hi])[0] + lo
    if len(idxs) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idxs[0])
    for i in idxs[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
            continue
        runs.append((start, prev))
        start = prev = i
    runs.append((start, prev))
    return runs


def _attack_label_for_run(rec: fs.Recording, playhead: int, a: int, b: int, *, future: bool):
    best_attack, best_conf = "Unknown", 0.0
    max_p = 0.0
    tech = None
    for s in range(a, b + 1):
        if future or s > playhead:
            mitre = rec.prospective_mitre_prob[s] if 0 <= s < rec.n_steps else None
            p_intr = rec.prospective_intrusion_prob[s] if 0 <= s < rec.n_steps else float("nan")
        else:
            mitre = rec.retro_mitre_prob[s]
            p_intr = rec.retro_intrusion_prob[s]
        if mitre is not None and not np.all(np.isnan(mitre)):
            attack, conf = fs.attack_label_from_probs(mitre)
            if attack != "Benign" and conf >= best_conf:
                best_attack, best_conf = attack, conf
            elif best_attack == "Unknown":
                best_attack, best_conf = attack, conf
        if np.isfinite(p_intr):
            max_p = max(max_p, float(p_intr))
        if rec.technique_by_step and 0 <= s < len(rec.technique_by_step):
            tech = rec.technique_by_step[s] or tech
    if tech:
        best_attack = tech.split(" (p=")[0]
    return best_attack, best_conf, max_p


def _intrusion_band_labels(rec: fs.Recording, playhead: int, lo: int, hi: int,
                           past_intr: np.ndarray, future_intr: np.ndarray):
    annotations: list[dict] = []
    caption_bits: list[str] = []
    for runs, future, color in (
        (_mask_runs(past_intr, lo, hi), False, "#ff7b72"),
        (_mask_runs(future_intr, lo, hi), True, "#ffa657"),
    ):
        for a, b in runs:
            attack, conf, p_intr = _attack_label_for_run(rec, playhead, a, b, future=future)
            prefix = "Forecast " if future else ""
            text = prefix + attack
            if np.isfinite(conf) and attack != "Benign":
                text += f" {conf:.0%}"
            elif np.isfinite(p_intr):
                text += f" · {p_intr:.0%}"
            annotations.append(dict(
                x=(a + b) / 2, y=1.04, yref="paper", text=text, showarrow=False,
                font=dict(size=10, color=color),
                bgcolor="rgba(13,17,23,0.9)", borderpad=3, xanchor="center", yanchor="bottom",
            ))
            tag = "forecast" if future else "detected"
            cap = f"steps {a}–{b}: **{attack}** ({tag}"
            if np.isfinite(conf) and attack != "Benign":
                cap += f", {conf:.0%}"
            if np.isfinite(p_intr):
                cap += f", attack {p_intr:.0%}"
            cap += ")"
            caption_bits.append(cap)
    return annotations, caption_bits


def _live_playhead_attack(rec: fs.Recording, playhead: int) -> dict | None:
    """Badge at the playhead: what attack the model is calling right now."""
    if not (0 <= playhead < rec.n_steps):
        return None
    p = rec.retro_intrusion_prob[playhead]
    if not (np.isfinite(p) and p > fs.INTRUSION_PROB_THRESH):
        return None
    attack, conf = fs.predicted_attack(rec, playhead)
    text = f"LIVE: {attack}"
    if np.isfinite(conf):
        text += f" {conf:.0%}"
    text += f" · attack {p:.0%}"
    return dict(
        x=playhead, y=1.14, yref="paper", text=text, showarrow=True,
        arrowhead=2, arrowsize=0.8, arrowwidth=1.2, arrowcolor="#3fb950",
        ax=0, ay=-28, font=dict(size=11, color="#3fb950"),
        bgcolor="rgba(13,17,23,0.95)", bordercolor="#3fb950", borderwidth=1, borderpad=4,
        xanchor="center", yanchor="bottom",
    )


def _mask_regions(mask: np.ndarray, lo: int, hi: int, fill: str, layer: str = "below",
                  edge: str | None = None) -> list[dict]:
    """Collapses a boolean mask into contiguous-run rectangles instead of one
    shape per flagged index -- far fewer shapes to build/serialize/diff each
    frame, which is most of what made playback feel janky."""
    runs = _mask_runs(mask, lo, hi)
    if not runs:
        return []
    line = dict(width=0)
    if edge:
        line = dict(color=edge, width=1)
    return [
        dict(type="rect", xref="x", yref="paper", x0=a - 0.5, x1=b + 0.5, y0=0, y1=1,
             fillcolor=fill, line=line, layer=layer)
        for a, b in runs
    ]


def _now_line(playhead: int) -> dict:
    return dict(type="line", xref="x", yref="paper", x0=playhead, x1=playhead, y0=0, y1=1, line=NOW_LINE)


def _render_overview(recordings: list[tuple[int, fs.Recording]], feature_idx: int, playhead: int, n_steps: int):
    """Ground-truth-only full replay -- no model, no forecast. The real
    signal is revealed progressively up to "now" (like live footage, not a
    pre-rendered snapshot) with attack windows in red for context, so you
    can always see where "now" sits relative to the whole timeline."""
    base_rec = recordings[0][1]
    x = np.arange(n_steps)
    y_full = base_rec.full_actual[:n_steps, feature_idx].astype(np.float64)
    y = np.where(x <= playhead, y_full, np.nan)  # hide the future -- this is playback, not a spoiler

    shapes = []
    annotations: list[dict] = []
    for lbl, a, b in base_rec.segments:
        if lbl == "Benign":
            continue
        fill_end = min(b, playhead + 1)
        if a < fill_end:
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=a, x1=fill_end, y0=0, y1=1,
                fillcolor=GT_ATTACK_PAST_FILL,
                line=dict(color=GT_ATTACK_PAST_EDGE, width=1),
                layer="below",
            ))
            annotations.append(dict(
                x=(a + fill_end - 1) / 2, y=1.03, yref="paper", text=lbl,
                showarrow=False, font=dict(size=10, color="#ff7b72"),
                bgcolor="rgba(13,17,23,0.85)", borderpad=2, xanchor="center", yanchor="bottom",
            ))
        if a > playhead:
            shapes.append(dict(
                type="line", xref="x", yref="paper", x0=a, x1=a, y0=0, y1=1,
                line=dict(color=GT_ATTACK_FUTURE_EDGE, width=1.2, dash="dash"),
                layer="below",
            ))
    shapes.append(_now_line(playhead))

    gt_now = fs.ground_truth_attack(base_rec, playhead)
    if gt_now:
        annotations.append(dict(
            x=playhead, y=1.12, yref="paper", text=f"LIVE: {gt_now}",
            showarrow=True, arrowhead=2, ax=0, ay=-24,
            font=dict(size=11, color="#58a6ff"),
            bgcolor="rgba(13,17,23,0.95)", bordercolor="#58a6ff", borderwidth=1, borderpad=3,
            xanchor="center", yanchor="bottom",
        ))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="ground truth",
                              line=dict(color="#58a6ff", width=1.1), hoverinfo="skip"))
    fig.update_layout(
        **CHART_LAYOUT_BASE,
        height=190, margin=dict(l=40, r=10, t=34, b=25),
        title=dict(text=f"Ground truth — full replay — feature[{feature_idx}]", font=dict(size=14)),
        showlegend=False, shapes=shapes, annotations=annotations,
    )
    fig.update_xaxes(range=[0, max(1, n_steps - 1)], gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True, key="fp_overview_chart")
    st.caption(
        "Filled red = real attack in timeline (SSH-Bruteforce, Infilteration, DoS-Hulk, …). "
        "Dashed red line = upcoming attack. Blue ▶ = attack active at playhead."
    )


def _render_error_ranking(recordings: list[tuple[int, fs.Recording]], playhead: int):
    rows = []
    for slot_idx, rec in recordings:
        err = rec.cumulative_error_at(playhead)
        rows.append({"Rank": 0, "Slot": f"Slot {slot_idx}", "Model": rec.display_name,
                     "Mean standardized MSE so far": err})
    rows.sort(key=lambda r: (np.isnan(r["Mean standardized MSE so far"]), r["Mean standardized MSE so far"]))
    for i, r in enumerate(rows, start=1):
        r["Rank"] = i

    st.caption(
        "Current model error — mean standardized MSE across all revealed steps and all "
        "242 features so far, best (lowest error) first. Already known for recorded/analysed data."
    )
    # Keep the error column numeric (float, possibly NaN) instead of mixing in strings --
    # a column_config formatter handles display, which is far more robust for Streamlit/Arrow.
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "Mean standardized MSE so far": st.column_config.NumberColumn(
                "Mean standardized MSE so far", format="%.4f", help="NaN = nothing revealed at this step yet",
            )
        },
    )


def _render_model_row(rec: fs.Recording, feature_idx: int, playhead: int, slot_idx: int):
    x, actual_y, pred_y = rec.displayed_series(feature_idx, playhead, PAST_WINDOW, FUTURE_WINDOW)
    lo, hi = int(x[0]), int(x[-1]) + 1
    color = SLOT_LINE_COLORS[slot_idx]

    past_anom = _clip_mask_side(rec.retro_anomaly_mask, playhead, "past")
    future_anom = _clip_mask_side(rec.prospective_anomaly_mask, playhead, "future")
    past_intr = _clip_mask_side(_intrusion_mask(rec.retro_intrusion_prob), playhead, "past")
    future_intr = _clip_mask_side(_intrusion_mask(rec.prospective_intrusion_prob), playhead, "future")

    shapes = []
    shapes += _mask_regions(future_anom, lo, hi, FUTURE_ANOMALY_FILL, layer="below", edge=FUTURE_ANOMALY_EDGE)
    shapes += _mask_regions(past_anom, lo, hi, PAST_ANOMALY_FILL, layer="below", edge=PAST_ANOMALY_EDGE)
    if rec.has_memory:
        mem_past = _clip_mask_side(rec.memory_write_mask, playhead, "past")
        shapes += _mask_regions(mem_past, lo, hi, MEMORY_FILL, layer="below", edge=MEMORY_EDGE)
    shapes += _mask_regions(future_intr, lo, hi, FUTURE_INTRUSION_FILL, layer="above", edge=FUTURE_INTRUSION_EDGE)
    shapes += _mask_regions(past_intr, lo, hi, PAST_INTRUSION_FILL, layer="above", edge=PAST_INTRUSION_EDGE)
    shapes.append(_now_line(playhead))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=actual_y, mode="lines", name="actual",
                              line=dict(color="#c9d1d9", width=1.6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=pred_y, mode="lines", name="forecast",
                              line=dict(color=color, width=1.8, dash="dash"), hoverinfo="skip"))

    band_ann, band_caps = _intrusion_band_labels(rec, playhead, lo, hi, past_intr, future_intr)
    live = _live_playhead_attack(rec, playhead)
    if live:
        band_ann.append(live)

    fig.update_layout(
        **CHART_LAYOUT_BASE,
        height=250, margin=dict(l=40, r=10, t=34, b=55),
        title=dict(text=f"Slot {slot_idx} — {rec.display_name} — feature[{feature_idx}]", font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.28, x=0, yanchor="top"),
        shapes=shapes,
        annotations=band_ann,
    )
    fig.update_xaxes(range=[lo, hi], gridcolor="#21262d", title="step (dotted line = now)")
    fig.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True, key=f"fp_chart_slot{slot_idx}")
    cap = (
        "Red = model detecting a named attack on revealed traffic; orange = forecast ahead. "
        "Green ▶ at playhead = live call as playback reaches that step. "
        + ("Yellow = RAM memory writes. " if rec.has_memory else "")
        + "Dashed line = forecast trajectory."
    )
    if band_caps:
        cap += "  **In view:** " + " · ".join(band_caps)
    st.caption(cap)
    attack, conf = fs.predicted_attack(rec, playhead)
    fut, fut_p = fs.predicted_attack(rec, playhead, future=True)
    gt = fs.ground_truth_attack(rec, playhead)
    live_line = f"At playhead — Model: **{attack}**"
    if np.isfinite(conf):
        live_line += f" ({conf:.0%})"
    live_line += f"  ·  Forecast: **{fut}**"
    if np.isfinite(fut_p):
        live_line += f" ({fut_p:.0%})"
    if gt:
        live_line += f"  ·  True attack in timeline: **{gt}**"
    st.caption(live_line)


def _tick(max_step: int):
    if not st.session_state.fp_playing:
        return
    st.session_state.fp_accum += st.session_state.fp_speed * TICK_SECONDS
    step = int(st.session_state.fp_accum)
    if step < 1:
        return
    st.session_state.fp_accum -= step
    nxt = min(st.session_state.fp_playhead + step, max_step)
    st.session_state.fp_playhead = nxt
    if nxt >= max_step:
        st.session_state.fp_playing = False


def _render_player_body(recordings: list[tuple[int, fs.Recording]], n_steps: int, feature_idx: int):
    max_step = n_steps - 1
    _tick(max_step)
    playhead = st.session_state.fp_playhead

    _render_overview(recordings, feature_idx, playhead, n_steps)
    _render_error_ranking(recordings, playhead)

    st.divider()
    b1, b2, b3, b4, b5, b6 = st.columns([0.7, 0.7, 1.1, 0.7, 0.7, 2.2])
    if b1.button("|<<", use_container_width=True, help="Jump to start"):
        st.session_state.fp_playhead = recordings[0][1].context - 1
        st.session_state.fp_playing = False
    if b2.button("<-", use_container_width=True, help="Step back 1"):
        st.session_state.fp_playhead = max(0, st.session_state.fp_playhead - 1)
        st.session_state.fp_playing = False
    if b3.button("Pause" if st.session_state.fp_playing else "Play",
                 use_container_width=True, type="primary"):
        st.session_state.fp_playing = not st.session_state.fp_playing
        st.session_state.fp_accum = 0.0
    if b4.button("->", use_container_width=True, help="Step forward 1"):
        st.session_state.fp_playhead = min(max_step, st.session_state.fp_playhead + 1)
        st.session_state.fp_playing = False
    if b5.button(">>|", use_container_width=True, help="Jump to end"):
        st.session_state.fp_playhead = max_step
        st.session_state.fp_playing = False
    with b6:
        st.slider("Speed (steps/sec)", 1, 40, key="fp_speed", label_visibility="visible")

    st.slider("Scrub", 0, max_step, key="fp_playhead", label_visibility="collapsed")

    playhead = st.session_state.fp_playhead
    pct = 100.0 * playhead / max_step if max_step else 0.0
    status = "\u25b6 Playing" if st.session_state.fp_playing else "\u23f8 Paused"
    st.caption(f"**{status}** — step {playhead} / {max_step} ({pct:.1f}%) — "
               f"{st.session_state.fp_speed} steps/sec — {len(recordings)} model slot(s) below")

    for slot_idx, rec in recordings:
        _render_model_row(rec, feature_idx, playhead, slot_idx)

    _render_labels_and_shap(recordings, playhead)


def _render_labels_and_shap(recordings: list[tuple[int, fs.Recording]], playhead: int):
    st.divider()
    st.subheader("Suspicion / attack label")
    for slot_idx, rec in recordings:
        st.markdown(f"**Slot {slot_idx} — {rec.display_name}**")
        st.write(rec.suspicion_text(playhead))

    rec = recordings[0][1]
    st.subheader("Feature attribution at playhead")
    st.caption("Integrated Gradients on P(attack) for the last 20 real steps. Optional Kernel SHAP is slower.")
    win = rec.window_at(playhead)
    if win is None or rec.adapter is None:
        st.info("Need at least 20 revealed steps before attribution.")
        return
    if st.session_state.fp_playing:
        st.caption("Pause playback to run Integrated Gradients / SHAP.")
        return
    from src.explain.aryan_attribution import integrated_gradients_infiltration, shap_infiltration
    ig = integrated_gradients_infiltration(rec.adapter, win)
    ig_df = pd.DataFrame(ig, columns=["feature", "score"]).set_index("feature")
    st.bar_chart(ig_df, height=220)
    if st.button("Compute SHAP at playhead (slow)", key="fp_shap"):
        bg = rec.full_actual[: min(64, rec.n_steps)]
        shap_rows = shap_infiltration(rec.adapter, win, bg)
        if shap_rows:
            st.bar_chart(pd.DataFrame(shap_rows, columns=["feature", "score"]).set_index("feature"), height=220)
        else:
            st.warning("SHAP unavailable (package missing or explainer failed). IG above is the fallback.")


try:
    _render_player_fragment = st.fragment(run_every=TICK_SECONDS)(_render_player_body)
except Exception:
    _render_player_fragment = _render_player_body


def render_forecast_player():
    _init_state()
    if st.session_state.get("fp_build_version") != fs.RECORDING_BUILD_VERSION:
        _get_recording.clear()
        st.session_state.fp_build_version = fs.RECORDING_BUILD_VERSION
    _render_top_controls()

    timeline_len = st.session_state.fp_timeline_len
    feature_idx = st.session_state.fp_feature_idx
    slot_choices = [st.session_state.fp_slot1, st.session_state.fp_slot2, st.session_state.fp_slot3]
    active = [(i + 1, c) for i, c in enumerate(slot_choices) if c != "none"]
    if not active:
        st.info("Select at least one model slot above to start the player.")
        return

    source = st.session_state.fp_source
    upload_key = st.session_state.fp_upload_name or ""
    if source == "upload" and not upload_key:
        st.info("Upload a PCAP or CIC CSV above to drive the player from a real file.")
        return
    if source == "live":
        from src.adversarial.lab_config import SAVE_DIR
        newest = max((p.stat().st_mtime for p in SAVE_DIR.glob("*.pcap")), default=0)
        upload_key = f"live:{newest}"

    recordings = []
    for slot_idx, choice in active:
        with st.spinner(f"Preparing slot {slot_idx}: {fs.MODEL_CHOICES[choice]} "
                         f"({source}, {timeline_len}-step, one-time per selection)..."):
            try:
                rec = _get_recording(choice, timeline_len, source, upload_key)
            except Exception as exc:
                st.error(f"Slot {slot_idx} failed: {exc}")
                continue
        recordings.append((slot_idx, rec))
    if not recordings:
        return

    n_steps = min(rec.n_steps for _, rec in recordings)
    if st.session_state.fp_playhead is None or st.session_state.fp_playhead >= n_steps:
        st.session_state.fp_playhead = min(recordings[0][1].context + 5, n_steps - 1)

    _render_player_fragment(recordings, n_steps, feature_idx)

    with st.expander("Session details"):
        for slot_idx, rec in recordings:
            st.write(f"**Slot {slot_idx} — {rec.display_name}**")
            st.json(rec.meta)
            if rec.retrievals:
                st.write("Memory retrievals:", rec.retrievals)
