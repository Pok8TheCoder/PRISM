#!/usr/bin/env python3
"""Forecast demo: Harborline site + dual-model retrospective / prospective charts + IPS.

Dashboard layout (per model):
  - Left of center: black solid line = re-analyzed past traffic
  - Right of center: dashed line = K-step forecast
  - White dashed vertical = NOW
  - Blue bands = suspicious, red bands = attack (+ predicted class label)

IPS: first model with 2 consecutive windows P>=0.5 blocks host + attacker-bot (+ redteam).
Recon is observe-only (no IPS) so the model can label port-scan before enum/spray/loot.

Usage:
  python scripts/demo_forecast.py
  python scripts/demo_forecast.py --no-up --warmup-windows 60
  python scripts/demo_forecast.py --no-up --auto-attack
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.ips_controller import (  # noqa: E402
    block_demo_attackers,
    clear_blocks,
    get_bridge_gateway,
    get_container_ip,
)
from src.adversarial.lab_config import ATTACKER_CONTAINER, REDTEAM_CONTAINER, TARGET_CONTAINER  # noqa: E402
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.explain.forecast_block_explain import explain_ips_block  # noqa: E402
from src.hx.causal import StreamingHXC, load_hxc_bundle  # noqa: E402
from src.hx.lab_adapt import (  # noqa: E402
    FORECAST_PHASE_LABELS,
    append_run_samples,
    finetune_lab_adapt,
    phase_to_class_id,
    resolve_hxc_ckpt,
)
from src.shaun.streaming import StreamingShaunRamxV3, load_shaun_bundle, pcap_to_shaun_state  # noqa: E402

COMPOSE_BASE = ROOT / "docker" / "docker-compose.yml"
COMPOSE_FC = ROOT / "docker" / "docker-compose.forecast.yml"
W5S = ROOT.parent / "PRISM-shaun" / "weights" / "w5s" / "world_model.pt"
EVENTS = ROOT / "data" / "lab_events"
STATE_PATH = EVENTS / "demo_forecast.json"
HIST_PATH = EVENTS / "demo_forecast.jsonl"
PID_PATH = EVENTS / "demo_forecast.pid"
PCAP_DIR = EVENTS / "demo_forecast_pcaps"
CONFIG_PATH = EVENTS / "lab_console_config.json"
ATTACK_THRESHOLD = 0.5
SUSPICIOUS_THRESHOLD = 0.35
WINDOW_SEC = 1.0
HORIZON = 6
HISTORY_MAX = 120
CONFIRM_WINDOWS = 2
IPS_ATTACK_CONFIRM = 1  # faster block once enum/spray/loot starts (recon still observe-only)
REFRESH_MS = 1000
FORECAST_DEMO_VERSION = "v2.2"
# IPS blocks only during post-recon kill-chain phases (enum/spray/loot).
RECON_PHASES = frozenset({"recon"})
ATTACK_PHASES = frozenset({"enum", "spray", "loot"})
IPS_ARM_PHASES = ATTACK_PHASES
DASH_PORT = 8788
SITE_URL = "http://127.0.0.1:8080"

DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PRISM Labs — Forecast</title>
  <style>
    :root { color-scheme: light; }
    body { margin: 0; font-family: "Segoe UI", Consolas, ui-monospace, sans-serif; background: #f4f5f7; color: #1a1a1a; }
    header { padding: 14px 20px; border-bottom: 1px solid #d8dce3; display: flex; justify-content: space-between; align-items: center; background: #fff; }
    h1 { font-size: 16px; margin: 0; color: #5b21b6; letter-spacing: 0.08em; }
    .ok { color: #15803d; } .hot { color: #b91c1c; } .muted { color: #64748b; } .warn { color: #a16207; }
    .banner { margin: 12px 20px; padding: 12px 14px; border-radius: 8px; border: 1px solid #d8dce3; background: #fff; }
    .panel { margin: 0 20px 18px; }
    .panel h2 { font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin: 0 0 8px; }
    canvas { width: 100%; height: 220px; background: #fff; border: 1px solid #d8dce3; border-radius: 8px; display: block; }
    .legend { font-size: 11px; color: #64748b; margin-top: 8px; }
    .legend span { margin-right: 14px; }
    .swatch { display: inline-block; width: 10px; height: 10px; margin-right: 4px; vertical-align: -1px; }
    a { color: #5b21b6; }
    .tabbar { display: flex; gap: 0; margin: 0 20px; border-bottom: 1px solid #d8dce3; }
    .tabbar button { background: #eef1f5; border: 1px solid #d8dce3; border-bottom: none; color: #64748b;
      padding: 8px 16px; font: inherit; cursor: pointer; border-radius: 8px 8px 0 0; margin-right: 4px; }
    .tabbar button.active { background: #fff; color: #5b21b6; border-color: #5b21b6; }
    .layout { margin: 0 20px 16px; }
    .pane-terminal { display: none; }
    body.tab-terminal .pane-forecast { display: none; }
    body.tab-terminal .pane-terminal { display: block; }
    body.view-dual .tabbar button { pointer-events: none; opacity: 0.85; }
    body.view-dual .tabbar button[data-tab="forecast"],
    body.view-dual .tabbar button[data-tab="terminal"] { background: #fff; color: #5b21b6; border-color: #5b21b6; }
    body.view-dual .layout { display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 12px; align-items: start; }
    body.view-dual .pane-forecast, body.view-dual .pane-terminal { display: block !important; }
    body.view-dual canvas { height: 180px; }
    body.view-dual .panel { margin: 0 0 12px; }
    body.view-dual .legend { display: none; }
    #term { background: #fff; border: 1px solid #d8dce3; border-radius: 8px; height: 420px; overflow: auto;
      padding: 10px 12px; font-size: 11px; line-height: 1.45; margin: 0; }
    #term .ln { white-space: pre-wrap; word-break: break-word; }
    #term .ln.cmd { color: #a16207; }
    #term .ln.attack { color: #2563eb; }
    #term .ln.scorer { color: #64748b; }
    #term .ln.alert { color: #b91c1c; font-weight: bold; }
    #term .ln.phase { color: #15803d; }
    .term-head { font-size: 11px; color: #64748b; letter-spacing: 0.1em; text-transform: uppercase; margin: 0 0 8px; }
    .status { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 20px 12px; }
    .status .card { background: #fff; border: 1px solid #d8dce3; border-radius: 8px; padding: 10px 12px; }
    .status .k { color: #64748b; font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; }
    .status .v { font-size: 18px; margin-top: 4px; color: #5b21b6; }
    .shap-panel { margin: 0 20px 20px; padding: 14px 16px; background: #fff; border: 1px solid #d8dce3; border-radius: 8px; display: none; }
    .shap-panel.visible { display: block; }
    .shap-panel h2 { font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #b91c1c; margin: 0 0 10px; }
    .shap-panel h3 { font-size: 11px; color: #64748b; margin: 0 0 8px; text-transform: uppercase; letter-spacing: 0.08em; }
    #shap-narrative { font-size: 12px; line-height: 1.5; color: #334155; margin: 0 0 12px; }
    .shap-meta { font-size: 11px; color: #64748b; margin-bottom: 12px; }
    .shap-grid { display: grid; grid-template-columns: 1.4fr 0.6fr; gap: 16px; }
    .shap-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin: 12px 0; }
    .shap-section { margin-top: 14px; }
    .shap-waterfall { font-size: 11px; }
    .wf-row { display: grid; grid-template-columns: 140px 1fr 56px 56px; gap: 8px; align-items: center; margin: 4px 0; }
    .wf-bar-wrap { background: #eef1f5; border-radius: 3px; height: 10px; position: relative; }
    .wf-bar { height: 10px; border-radius: 3px; position: absolute; top: 0; }
    .wf-bar.pos { background: #dc2626; left: 50%; }
    .wf-bar.neg { background: #2563eb; right: 50%; }
    #shap-temporal { width: 100%; max-width: 100%; background: #f8fafc; border-radius: 6px; margin-top: 6px; border: 1px solid #e2e8f0; }
    .shap-tag { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-right: 6px; }
    .shap-tag.pred { background: #fee2e2; color: #b91c1c; }
    .shap-tag.gt { background: #dcfce7; color: #15803d; }
    .shap-tag.miss { background: #fef3c7; color: #a16207; }
    .shap-table { width: 100%; border-collapse: collapse; font-size: 11px; }
    .shap-table th, .shap-table td { text-align: left; padding: 5px 8px; border-bottom: 1px solid #e2e8f0; }
    .shap-bar-wrap { background: #eef1f5; border-radius: 3px; height: 8px; width: 100%; }
    .shap-bar { height: 8px; background: #7c3aed; border-radius: 3px; }
    body.view-dual .shap-panel { grid-column: 1 / -1; }
  </style>
</head>
<body>
  <header>
    <h1>PRISM LABS // FORECAST</h1>
    <div id="clock" class="muted"></div>
  </header>
  <div id="banner" class="banner muted">Warming up on Harborline benign traffic.</div>
  <div class="status">
    <div class="card"><div class="k">IPS blocker</div><div class="v" id="blocker">none</div></div>
    <div class="card"><div class="k">Block time</div><div class="v" id="blockt">—</div></div>
    <div class="card"><div class="k">v3 streak ≥0.5</div><div class="v" id="stv3">0 / 2</div></div>
    <div class="card"><div class="k">HX streak ≥0.5</div><div class="v" id="sthx">0 / 2</div></div>
  </div>
  <div class="tabbar">
    <button type="button" class="active" data-tab="forecast">Forecast charts</button>
    <button type="button" data-tab="terminal">Attack terminal</button>
  </div>
  <div class="layout">
  <div class="pane-forecast">
  <div class="panel">
    <h2>Shaun v3 — retrospective (solid) / forecast (dashed)</h2>
    <canvas id="chart-v3" width="1200" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>HX-C — retrospective (solid) / forecast (dashed)</h2>
    <canvas id="chart-hx" width="1200" height="220"></canvas>
  </div>
  <p class="legend" style="padding: 0 20px 20px;">
    <span><span class="swatch" style="background:#334155;border:1px dashed #334155"></span>now</span>
    <span><span class="swatch" style="background:#111;border:1px solid #64748b"></span>past (re-analyzed)</span>
    <span><span class="swatch" style="background:transparent;border:1px dashed #64748b"></span>forecast</span>
    <span><span class="swatch" style="background:rgba(255,214,0,0.45)"></span>ground truth: recon (past + forecast)</span>
    <span><span class="swatch" style="background:rgba(147,51,234,0.35)"></span>ground truth: attack (past + forecast)</span>
    <span><span class="swatch" style="background:rgba(59,130,246,0.25)"></span>model suspicious</span>
    <span><span class="swatch" style="background:rgba(220,38,38,0.22)"></span>model attack</span>
    · Site: <a href="http://127.0.0.1:8080" target="_blank">http://127.0.0.1:8080</a>
  </p>
  </div>
  <div class="pane-terminal">
    <div class="term-head">Live attack + scorer log</div>
    <div id="term"></div>
  </div>
  </div>
  <div id="shap-panel" class="shap-panel">
    <h2>Why it blocked — full SHAP report</h2>
    <p id="shap-narrative" class="muted">Waiting for IPS block…</p>
    <div id="shap-meta" class="shap-meta"></div>
    <div class="shap-section">
      <h3>Waterfall — P(attack)</h3>
      <div id="shap-waterfall" class="shap-waterfall muted">—</div>
    </div>
    <div class="shap-section">
      <h3>Temporal lookback (which past seconds drove the score)</h3>
      <canvas id="shap-temporal" width="1100" height="72"></canvas>
    </div>
    <div class="shap-section shap-grid-3" id="shap-class-section" style="display:none">
      <div>
        <h3>Class probabilities</h3>
        <table class="shap-table"><thead><tr><th>Class</th><th>P</th></tr></thead><tbody id="shap-classes"></tbody></table>
      </div>
      <div>
        <h3>SHAP — predicted class</h3>
        <table class="shap-table"><thead><tr><th>Feature</th><th>SHAP</th></tr></thead><tbody id="shap-pred-class"></tbody></table>
      </div>
      <div>
        <h3>SHAP — ground truth class</h3>
        <table class="shap-table"><thead><tr><th>Feature</th><th>SHAP</th></tr></thead><tbody id="shap-gt-class"></tbody></table>
      </div>
    </div>
    <div class="shap-section" id="shap-contrast-section" style="display:none">
      <h3>Class contrast (why predicted ≠ ground truth)</h3>
      <table class="shap-table"><thead><tr><th>Feature</th><th>Predicted</th><th>Ground truth</th><th>Δ</th></tr></thead><tbody id="shap-contrast"></tbody></table>
    </div>
    <div class="shap-grid">
      <div>
        <h3>Top traffic features — P(attack)</h3>
        <table class="shap-table"><thead><tr><th>Feature</th><th>Share</th><th>SHAP</th><th>Value</th><th></th></tr></thead><tbody id="shap-features"></tbody></table>
      </div>
      <div>
        <h3>Feature blocks</h3>
        <table class="shap-table"><thead><tr><th>Block</th><th>Share</th><th></th></tr></thead><tbody id="shap-blocks"></tbody></table>
      </div>
    </div>
    <p id="shap-report-path" class="shap-meta" style="margin-top:10px"></p>
  </div>
<script>
const params = new URLSearchParams(location.search);
const dual = params.get('view') === 'dual';
const tabParam = params.get('tab');
if (dual) document.body.classList.add('view-dual');
else if (tabParam === 'terminal') document.body.classList.add('tab-terminal');
else document.body.classList.add('tab-forecast');
document.querySelectorAll('.tabbar button').forEach(btn => {
  const on = dual || btn.dataset.tab === (tabParam === 'terminal' ? 'terminal' : 'forecast');
  if (!dual) btn.classList.toggle('active', on);
  btn.addEventListener('click', () => {
    if (dual) return;
    document.body.className = 'tab-' + btn.dataset.tab;
    document.querySelectorAll('.tabbar button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});
let logCursor = 0;
async function pollLogs() {
  try {
    const j = await (await fetch('/api/logs?since=' + logCursor)).json();
    const box = document.getElementById('term');
    for (const ln of (j.lines || [])) {
      const div = document.createElement('div');
      div.className = 'ln ' + (ln.kind || 'scorer');
      div.textContent = ln.text;
      box.appendChild(div);
      logCursor = Math.max(logCursor, ln.id);
    }
    if (j.lines && j.lines.length) box.scrollTop = box.scrollHeight;
  } catch (e) {}
}
setInterval(pollLogs, 500);
pollLogs();
const SUSP = 0.35, ATK = 0.5;
function level(p) {
  if (p >= ATK) return 'attack';
  if (p >= SUSP) return 'suspicious';
  return 'benign';
}
function yPos(p, h, padT, padB) {
  return padT + (1 - Math.min(1, Math.max(0, p))) * (h - padT - padB);
}
function retroIndexForWindow(retro, w) {
  let idx = 0;
  for (let i = 0; i < retro.length; i++) {
    if (retro[i].w <= w) idx = i;
  }
  return idx;
}
function drawGroundTruth(ctx, retro, pro, regions, xRetro, xPro, padT, h, padB, midX, canvasW, padR) {
  if (!regions || !regions.length) return;
  for (const reg of regions) {
    const col = reg.kind === 'recon'
      ? 'rgba(255, 214, 0, 0.42)'
      : 'rgba(147, 51, 234, 0.32)';
    const i0 = retroIndexForWindow(retro, reg.start_w);
    const x0 = xRetro(i0);
    let x1;
    if (reg.open) {
      x1 = pro.length ? (canvasW - padR) : midX + 24;
    } else {
      const i1 = retroIndexForWindow(retro, reg.end_w);
      x1 = xRetro(i1) + 8;
      for (let j = 0; j < pro.length; j++) {
        const pw = pro[j].w;
        if (pw >= reg.start_w && pw <= reg.end_w) {
          x1 = Math.max(x1, xPro(j) + 8);
        }
      }
    }
    ctx.fillStyle = col;
    ctx.fillRect(x0, padT, Math.max(x1 - x0, 3), h - padT - padB);
    ctx.fillStyle = reg.kind === 'recon' ? '#92400e' : '#6b21a8';
    ctx.font = '9px Consolas';
    const lbl = reg.label || reg.phase;
    ctx.fillText(lbl, x0 + 4, padT + 11);
    if (reg.open && x1 > midX + 12) {
      ctx.fillText(lbl + ' →', midX + 6, padT + 11);
    }
  }
}
function drawModel(canvasId, model, windowSec, horizon, phaseRegions) {
  const c = document.getElementById(canvasId);
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  const padL = 48, padR = 16, padT = 18, padB = 28;
  const midX = Math.floor(w * 0.5);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);
  const retro = model.retrospective || [];
  const pro = model.prospective || [];
  if (!retro.length) return;
  const pastSec = Math.max(windowSec, retro.length * windowSec);
  const futSec = Math.max(windowSec, (pro.length || 1) * windowSec);
  const leftW = midX - padL;
  const rightW = w - padR - midX;
  function xRetro(i) {
    const t = retro[i].t_rel;
    return midX + (t / pastSec) * leftW;
  }
  function xPro(i) {
    const t = pro[i].t_rel;
    return midX + (t / futSec) * rightW;
  }
  drawGroundTruth(ctx, retro, pro, phaseRegions, xRetro, xPro, padT, h, padB, midX, w, padR);
  function bandColor(lvl) {
    if (lvl === 'attack') return 'rgba(220, 38, 38, 0.18)';
    if (lvl === 'suspicious') return 'rgba(59, 130, 246, 0.2)';
    return null;
  }
  function drawBands(points, xFn) {
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i], b = points[i + 1];
      const lvl = level(Math.max(a.p, b.p));
      const col = bandColor(lvl);
      if (!col) continue;
      const x0 = xFn(i), x1 = xFn(i + 1);
      ctx.fillStyle = col;
      ctx.fillRect(x0, padT, x1 - x0, h - padT - padB);
      if (lvl === 'attack') {
        const lbl = a.label || b.label;
        if (lbl) {
          ctx.fillStyle = '#b91c1c';
          ctx.font = '10px Consolas';
          const tx = (x0 + x1) / 2;
          ctx.fillText(lbl, tx - ctx.measureText(lbl).width / 2, padT + 22);
        }
      }
    }
  }
  drawBands(retro, xRetro);
  drawBands(pro, xPro);
  ctx.strokeStyle = '#cbd5e1';
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, h - padB);
  ctx.lineTo(w - padR, h - padB);
  ctx.stroke();
  ctx.setLineDash([2, 4]);
  ctx.strokeStyle = 'rgba(202, 138, 4, 0.75)';
  ctx.beginPath();
  ctx.moveTo(padL, yPos(ATK, h, padT, padB));
  ctx.lineTo(w - padR, yPos(ATK, h, padT, padB));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = '#334155';
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(midX, padT);
  ctx.lineTo(midX, h - padB);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#334155';
  ctx.font = '10px Consolas';
  ctx.fillText('NOW', midX + 4, padT + 10);
  if (retro.length >= 2) {
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 2;
    ctx.beginPath();
    retro.forEach((pt, i) => {
      const x = xRetro(i), y = yPos(pt.p, h, padT, padB);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.lineWidth = 1;
  }
  if (pro.length) {
    const startY = yPos(retro[retro.length - 1].p, h, padT, padB);
    ctx.strokeStyle = '#64748b';
    ctx.setLineDash([7, 5]);
    ctx.beginPath();
    ctx.moveTo(midX, startY);
    pro.forEach((pt, i) => {
      const x = xPro(i), y = yPos(pt.p, h, padT, padB);
      ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = '#64748b';
  ctx.font = '10px Consolas';
  ctx.fillText('0', midX - 8, h - 8);
  ctx.fillText('-' + Math.round(pastSec) + 's', padL, h - 8);
  ctx.fillText('+' + Math.round(futSec) + 's', w - padR - 28, h - 8);
  ctx.fillText('1.0', 8, padT + 4);
  ctx.fillText('0.0', 8, h - padB);
}
function renderWaterfall(wf) {
  const el = document.getElementById('shap-waterfall');
  if (!wf || !wf.steps || !wf.steps.length) { el.textContent = '—'; return; }
  const maxAbs = Math.max(...wf.steps.map(s => Math.abs(s.shap_value)), 0.0001);
  el.innerHTML = '<div class="wf-row muted"><span>base</span><span></span><span></span><span>' +
    (wf.base_value != null ? wf.base_value.toFixed(3) : '?') + '</span></div>' +
    wf.steps.map(s => {
      const w = Math.max(2, Math.round(Math.abs(s.shap_value) / maxAbs * 48));
      const cls = s.shap_value >= 0 ? 'pos' : 'neg';
      const style = cls === 'pos' ? 'width:' + w + '%;left:50%' : 'width:' + w + '%;right:50%';
      return '<div class="wf-row"><span>' + s.feature + '</span>' +
        '<div class="wf-bar-wrap"><div class="wf-bar ' + cls + '" style="' + style + '"></div></div>' +
        '<span>' + (s.shap_value >= 0 ? '+' : '') + s.shap_value.toFixed(4) + '</span>' +
        '<span>' + (s.cumulative != null ? s.cumulative.toFixed(3) : '') + '</span></div>';
    }).join('') +
    '<div class="wf-row muted"><span>output</span><span></span><span></span><span>' +
    (wf.output_value != null ? wf.output_value.toFixed(3) : '?') + '</span></div>';
}
function renderTemporal(temporal) {
  const c = document.getElementById('shap-temporal');
  const ctx = c.getContext('2d');
  const steps = (temporal && temporal.steps) || [];
  ctx.clearRect(0, 0, c.width, c.height);
  if (!steps.length) return;
  const pad = 24, barW = Math.max(4, (c.width - pad * 2) / steps.length - 2);
  const maxI = Math.max(...steps.map(s => s.importance), 0.001);
  steps.forEach((s, i) => {
    const h = Math.max(2, (s.importance / maxI) * (c.height - 22));
    const x = pad + i * (barW + 2);
    ctx.fillStyle = s.t_rel_sec === 0 ? '#b91c1c' : '#7c3aed';
    ctx.fillRect(x, c.height - 14 - h, barW, h);
    if (i % 5 === 0 || s.t_rel_sec === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '9px Consolas';
      ctx.fillText(s.label, x, c.height - 2);
    }
  });
}
function renderShapReport(s) {
  const panel = document.getElementById('shap-panel');
  const rep = s.block_explanation;
  if (!s.blocked) {
    panel.classList.remove('visible');
    return;
  }
  panel.classList.add('visible');
  if (!rep || !rep.top_features) {
    document.getElementById('shap-narrative').textContent =
      'IPS blocked — computing full SHAP report…';
    document.getElementById('shap-meta').textContent = '';
    document.getElementById('shap-features').innerHTML = '';
    document.getElementById('shap-blocks').innerHTML = '';
    document.getElementById('shap-waterfall').textContent = 'Computing…';
    return;
  }
  document.getElementById('shap-narrative').textContent = rep.narrative || '';
  let tags = (rep.blocker_label || rep.blocker) + ' · method=' + (rep.method || '?') +
    ' · P=' + (rep.p_attack != null ? rep.p_attack.toFixed(3) : '?');
  if (rep.predicted_class) tags += ' · <span class="shap-tag pred">' + rep.predicted_class + '</span>';
  if (rep.ground_truth_class) {
    const miss = rep.class_match === false;
    tags += ' · <span class="shap-tag ' + (miss ? 'miss' : 'gt') + '">GT ' + rep.ground_truth_class + '</span>';
  }
  if (rep.runner_up) tags += ' · runner-up ' + rep.runner_up + ' P=' + rep.runner_up_p;
  document.getElementById('shap-meta').innerHTML = tags;
  renderWaterfall(rep.waterfall);
  renderTemporal(rep.temporal);
  const ca = rep.class_analysis || {};
  const hasClass = (ca.top_probs && ca.top_probs.length);
  document.getElementById('shap-class-section').style.display = hasClass ? 'grid' : 'none';
  if (hasClass) {
    document.getElementById('shap-classes').innerHTML = ca.top_probs.map(r =>
      '<tr><td>' + r.class + '</td><td>' + (r.p * 100).toFixed(1) + '%</td></tr>').join('');
    document.getElementById('shap-pred-class').innerHTML = (ca.predicted_class_shap || []).map(r =>
      '<tr><td>' + r.feature + '</td><td>' + (r.shap_value >= 0 ? '+' : '') + r.shap_value.toFixed(4) + '</td></tr>').join('');
    document.getElementById('shap-gt-class').innerHTML = (ca.ground_truth_class_shap || []).map(r =>
      '<tr><td>' + r.feature + '</td><td>' + (r.shap_value >= 0 ? '+' : '') + r.shap_value.toFixed(4) + '</td></tr>').join('') || '<tr><td colspan="2" class="muted">—</td></tr>';
  }
  const contrast = ca.contrast || [];
  document.getElementById('shap-contrast-section').style.display = contrast.length ? 'block' : 'none';
  document.getElementById('shap-contrast').innerHTML = contrast.map(r =>
    '<tr><td>' + r.feature + '</td><td>' + r.predicted_shap.toFixed(4) + '</td><td>' +
    r.ground_truth_shap.toFixed(4) + '</td><td>' + (r.delta >= 0 ? '+' : '') + r.delta.toFixed(4) + '</td></tr>').join('');
  const maxF = Math.max(...rep.top_features.map(r => r.importance), 0.001);
  document.getElementById('shap-features').innerHTML = rep.top_features.map(r => {
    const pct = (r.importance * 100).toFixed(1);
    const w = Math.max(2, Math.round((r.importance / maxF) * 100));
    const sh = r.shap_value != null ? ((r.shap_value >= 0 ? '+' : '') + r.shap_value.toFixed(4)) : '—';
    return '<tr><td>' + r.feature + '</td><td>' + pct + '%</td><td>' + sh + '</td><td>' + r.value + '</td>' +
      '<td><div class="shap-bar-wrap"><div class="shap-bar" style="width:' + w + '%"></div></div></td></tr>';
  }).join('');
  const maxB = Math.max(...(rep.top_blocks || []).map(r => r.importance), 0.001);
  document.getElementById('shap-blocks').innerHTML = (rep.top_blocks || []).map(r => {
    const pct = (r.importance * 100).toFixed(1);
    const w = Math.max(2, Math.round((r.importance / maxB) * 100));
    return '<tr><td>' + r.block + '</td><td>' + pct + '%</td>' +
      '<td><div class="shap-bar-wrap"><div class="shap-bar" style="width:' + w + '%"></div></div></td></tr>';
  }).join('');
  const paths = rep.report_paths || {};
  document.getElementById('shap-report-path').textContent =
    paths.markdown ? ('Saved: ' + paths.markdown) : '';
}
async function tick() {
  const s = await (await fetch('/api/state')).json();
  const ap = s.attack_phase || 'idle';
  document.getElementById('clock').textContent =
    s.phase.toUpperCase() + '  w' + s.window + '  t+' + s.elapsed_sec.toFixed(0) + 's' +
    (ap !== 'idle' ? '  attack:' + ap : '');
  const b = document.getElementById('banner');
  document.getElementById('blocker').textContent = s.first_blocker || 'none';
  document.getElementById('blockt').textContent = s.block_at_sec == null ? '—' : (s.block_at_sec.toFixed(1) + 's');
  document.getElementById('stv3').textContent = (s.pending_sn2rx3 || 0) + ' / ' + (s.confirm_need || 2);
  document.getElementById('sthx').textContent = (s.pending_hx_c || 0) + ' / ' + (s.confirm_need || 2);
  if (s.blocked) {
    b.className = 'banner hot';
    b.textContent = 'IPS BLOCK by ' + (s.first_blocker || '?') + ' at t+' + s.block_at_sec.toFixed(1) +
      's (window ' + s.block_window + '). Host + attacker traffic dropped — benign Docker users stay up.';
  } else if (s.forecast_alert) {
    b.className = 'banner hot';
    b.textContent = 'FORECAST ALERT at t+' + s.forecast_at_sec.toFixed(1) + 's — HX now ' +
      s.p_hx_c.toFixed(3) + ', cone max ' + s.forecast_max.toFixed(3) + '. Loot may not have happened yet.';
  } else if (ap === 'recon') {
    b.className = 'banner warn';
    b.textContent = 'RECON (yellow band) — model scores live; IPS observe-only until enum/spray/loot.';
  } else if (ap && ap !== 'idle' && ap !== 'complete') {
    b.className = 'banner warn';
    b.textContent = 'ATTACK PHASE: ' + ap.toUpperCase() +
      ' (purple band) — IPS armed; 1 score >= 0.5 in purple will block.';
  } else if (s.phase === 'live') {
    b.className = 'banner ok';
    b.textContent = 'LIVE — 1s windows · yellow=recon · purple=attack phases · IPS blocks only in purple.';
  } else {
    b.className = 'banner muted';
    b.textContent = 'WARMUP — Harborline benign users only. Wait for LIVE.';
  }
  const models = s.models || {};
  const regions = s.phase_regions || [];
  drawModel('chart-v3', models.shaun_v3 || {}, s.window_sec || 1, s.horizon || 6, regions);
  drawModel('chart-hx', models.hx_c || {}, s.window_sec || 1, s.horizon || 6, regions);
  renderShapReport(s);
}
setInterval(tick, 1000);
tick();
</script>
</body></html>
"""


def bump_phase_regions(
    regions: list[dict[str, Any]],
    window_idx: int,
    attack_phase: str,
    prev_phase: str,
) -> list[dict[str, Any]]:
    """UI-only ground-truth bands: yellow=recon, purple=enum/spray/loot."""
    tracked = RECON_PHASES | ATTACK_PHASES
    if attack_phase not in tracked:
        return regions
    kind = "recon" if attack_phase in RECON_PHASES else "attack"
    if not regions or regions[-1].get("phase") != attack_phase:
        if regions and regions[-1].get("open"):
            regions[-1]["end_w"] = max(regions[-1]["start_w"], window_idx - 1)
            regions[-1]["open"] = False
        regions.append({
            "phase": attack_phase,
            "kind": kind,
            "start_w": window_idx,
            "end_w": window_idx,
            "label": attack_phase.upper(),
            "open": True,
        })
    else:
        regions[-1]["end_w"] = window_idx
    return regions


def read_horizon_windows(window_sec: float, fallback: int = HORIZON) -> int:
    if not CONFIG_PATH.exists():
        return fallback
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        sec = float(cfg.get("horizonSec", fallback * window_sec))
        windows = max(1, int(round(sec / max(window_sec, 0.1))))
        # Rollout is O(horizon); cap to keep live scoring responsive on CPU.
        return min(windows, 15)
    except Exception:
        return fallback


def bump_memory_regions(
    regions: list[dict[str, Any]],
    window_idx: int,
    *,
    system: str,
) -> list[dict[str, Any]]:
    if regions and regions[-1].get("system") == system and int(regions[-1]["end_w"]) == window_idx - 1:
        regions[-1]["end_w"] = window_idx
        return regions
    regions.append({
        "start_w": window_idx,
        "end_w": window_idx,
        "system": system,
        "label": "RAMX episodic",
    })
    return regions


def level_from_p(p: float) -> str:
    if p >= ATTACK_THRESHOLD:
        return "attack"
    if p >= SUSPICIOUS_THRESHOLD:
        return "suspicious"
    return "benign"


def point(
    *,
    t_rel: float,
    w: int,
    p: float,
    label: str | None,
) -> dict[str, Any]:
    return {
        "t_rel": round(t_rel, 2),
        "w": w,
        "p": round(p, 4),
        "level": level_from_p(p),
        "label": label if level_from_p(p) == "attack" else None,
    }


class TermLog:
    """Ring buffer of terminal lines for the Attack terminal pane."""

    def __init__(self, maxlen: int = 800) -> None:
        self.lock = threading.Lock()
        self.lines: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._seq = 0

    def append(self, text: str, *, kind: str = "scorer") -> None:
        line = text.rstrip()
        if not line:
            return
        with self.lock:
            self._seq += 1
            self.lines.append({"id": self._seq, "text": line, "kind": kind, "t": time.time()})

    def since(self, after_id: int) -> list[dict[str, Any]]:
        with self.lock:
            return [ln for ln in self.lines if ln["id"] > after_id]


TLOG = TermLog()


def tlog(text: str, *, kind: str = "scorer") -> None:
    print(text)
    TLOG.append(text, kind=kind)


class LiveState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "phase": "starting",
            "attack_phase": "idle",
            "window": 0,
            "elapsed_sec": 0.0,
            "window_sec": WINDOW_SEC,
            "horizon": HORIZON,
            "p_sn2rx3": 0.0,
            "p_hx_c": 0.0,
            "forecast_max": 0.0,
            "class_shift": 0.0,
            "relative_anomaly": 0.0,
            "forecast_alert": False,
            "forecast_at_sec": None,
            "forecast_window": None,
            "site_url": SITE_URL,
            "demo_version": FORECAST_DEMO_VERSION,
            "blocked": False,
            "first_blocker": None,
            "block_at_sec": None,
            "block_window": None,
            "block_ips": [],
            "pending_sn2rx3": 0,
            "pending_hx_c": 0,
            "confirm_need": CONFIRM_WINDOWS,
            "same_window_tie": False,
            "scores_at_block": None,
            "block_explanation": None,
            "phase_regions": [],
            "memory_regions": [],
            "ips_armed": False,
            "models": {"shaun_v3": {"retrospective": [], "prospective": []}, "hx_c": {"retrospective": [], "prospective": []}},
        }

    def update(self, **kwargs: Any) -> dict[str, Any]:
        with self.lock:
            self.data.update(kwargs)
            snap = dict(self.data)
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        with HIST_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snap) + "\n")
        return snap

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.data)


STATE = LiveState()


class DashHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/logs"):
            since = 0
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for part in q.split("&"):
                    if part.startswith("since="):
                        try:
                            since = int(part.split("=", 1)[1])
                        except ValueError:
                            since = 0
            lines = TLOG.since(since)
            body = json.dumps({"lines": lines}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except ConnectionError:
                return
            return
        if self.path.startswith("/api/state"):
            body = json.dumps(STATE.snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except ConnectionError:
                return
            return
        body = DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:
            return


def compose(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_BASE), "-f", str(COMPOSE_FC), "-p", "prism", *args]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    r = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, timeout=10,
    )
    line = (r.stdout or "").strip().lower()
    return bool(line) and str(pid) in line and "python" in line


def acquire_pid_lock() -> None:
    EVENTS.mkdir(parents=True, exist_ok=True)
    if PID_PATH.exists():
        try:
            old = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            old = 0
        if old != os.getpid() and _pid_alive(old):
            raise SystemExit(f"Forecast scorer already running as pid {old}")
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")


def up_lab() -> None:
    (ROOT / "data" / "lab_events").mkdir(parents=True, exist_ok=True)
    print("Starting Harborline forecast lab (site on 127.0.0.1:8080)...")
    subprocess.run(compose("up", "-d", "--build", "--remove-orphans"), cwd=str(ROOT), check=True)


def wait_http(url: str, timeout: float = 90.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status < 500:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(1.5)
    raise RuntimeError(f"timeout waiting for {url}: {last}")


def ingest_pcap(pcap: Path, last: np.ndarray | None, *, window_sec: float = WINDOW_SEC) -> np.ndarray:
    if pcap is None or not pcap.exists():
        return last if last is not None else np.zeros(292, np.float32)
    try:
        st = pcap_to_shaun_state(pcap, last_state=last, window_sec=window_sec)
        return np.asarray(st, dtype=np.float32)
    except Exception as exc:
        print(f"  ingest failed: {exc}", file=sys.stderr)
        return last if last is not None else np.zeros(292, np.float32)


def build_timeline(
    history: deque[dict[str, Any]],
    roll: dict[str, Any],
    *,
    window_idx: int,
    window_sec: float,
    horizon: int,
) -> dict[str, list[dict[str, Any]]]:
    retro: list[dict[str, Any]] = []
    for i, row in enumerate(history):
        age_windows = len(history) - 1 - i
        t_rel = -age_windows * window_sec
        retro.append(
            point(
                t_rel=t_rel,
                w=int(row["w"]),
                p=float(row["p"]),
                label=row.get("label"),
            )
        )
    if retro:
        retro[-1]["t_rel"] = 0.0
    forecast = roll.get("forecast") or []
    labels = roll.get("forecast_labels") or roll.get("forecast_tech") or []
    pro: list[dict[str, Any]] = []
    for j, p in enumerate(forecast):
        lbl = labels[j] if j < len(labels) else None
        if lbl in (None, "Benign") and float(p) >= ATTACK_THRESHOLD:
            lbl = "Attack (forecast)"
        pro.append(
            point(
                t_rel=(j + 1) * window_sec,
                w=window_idx + j + 1,
                p=float(p),
                label=lbl,
            )
        )
    return {"retrospective": retro, "prospective": pro}


def apply_ips_block(
    t0: float,
    window_idx: int,
    fired: list[str],
    scores: dict[str, float],
    *,
    sn: StreamingShaunRamxV3,
    hx: StreamingHXC,
    out_v3: dict[str, Any],
    out_hx: dict[str, Any],
    ground_truth_class: str | None = None,
) -> None:
    extra: list[str] = []
    try:
        extra.append(get_container_ip(ATTACKER_CONTAINER))
    except RuntimeError as exc:
        tlog(f"  IPS: could not resolve {ATTACKER_CONTAINER}: {exc}", kind="alert")
    results = block_demo_attackers(
        include_redteam=True,
        include_host_gateway=True,
        extra_ips=extra,
    )
    ips = [r.source_ip for r in results if r.blocked]
    first = fired[0] if fired else None
    tie = len(fired) > 1
    tlog(
        f"\n*** IPS BLOCK by {first} at window {window_idx} "
        f"t+{time.time()-t0:.1f}s  fired={fired} scores={scores} ips={ips} tie={tie} ***\n",
        kind="alert",
    )
    STATE.update(
        blocked=True,
        first_blocker=first,
        block_at_sec=time.time() - t0,
        block_window=window_idx,
        block_ips=ips,
        same_window_tie=tie,
        scores_at_block=scores,
        block_explanation=None,
    )

    def _shap_job() -> None:
        try:
            report = explain_ips_block(
                first_blocker=first or "hx_c",
                sn=sn,
                hx=hx,
                scores=scores,
                sn_meta=out_v3,
                hx_meta=out_hx,
                ground_truth_class=ground_truth_class,
                window_idx=window_idx,
            )
            if report:
                STATE.update(block_explanation=report)
                tlog(f"  SHAP report ready ({report.get('method')}) — see dashboard", kind="phase")
                top = report.get("top_features") or []
                if top:
                    tlog(
                        "  top features: " + ", ".join(
                            f"{r['feature']} ({r['importance']*100:.1f}%)" for r in top[:5]
                        ),
                        kind="phase",
                    )
        except Exception as exc:
            tlog(f"  SHAP report failed: {exc}", kind="alert")

    threading.Thread(target=_shap_job, daemon=True).start()


def do_lab_adapt(
    hx: StreamingHXC,
    hxc_bundle: dict[str, Any],
    adapt_samples: list[dict[str, Any]],
    *,
    lab_adapt: bool,
    reason: str,
) -> None:
    if not lab_adapt or not adapt_samples:
        return
    batch = list(adapt_samples)
    adapt_samples.clear()
    class_names = list(hx.class_names)

    def _job() -> None:
        try:
            n = append_run_samples(batch, class_names=class_names, run_id=reason)
            if n == 0:
                return
            stats = finetune_lab_adapt(hxc_bundle, reason=reason)
            STATE.update(lab_adapt=stats)
            if stats.get("ok"):
                tlog(
                    f"  lab adapt ({reason}): +{n} windows, buffer={stats['buffer_n']}, "
                    f"replay acc={stats['buffer_acc']:.1%} -> {stats['ckpt']}",
                    kind="phase",
                )
            else:
                tlog(f"  lab adapt skipped: {stats.get('message', stats)}", kind="phase")
        except Exception as exc:
            tlog(f"  lab adapt failed: {exc}", kind="alert")

    threading.Thread(target=_job, daemon=True).start()


def _run_attack_cmd(name: str, cmd: list[str]) -> int:
    tlog(f"$ {' '.join(cmd)}", kind="cmd")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout:
        for line in proc.stdout:
            tlog(line.rstrip(), kind="attack")
    return proc.wait()


def run_auto_attack() -> None:
    """Run kill-chain phases after LIVE; updates attack_phase in dashboard state."""
    killchain = [sys.executable, str(ROOT / "scripts" / "demo_killchain.py")]
    phases: list[tuple[str, list[str]]] = [
        ("recon", killchain + ["--phase", "recon"]),
        ("enum", killchain + ["--phase", "enum"]),
        ("spray", killchain + ["--phase", "spray"]),
        ("loot", killchain + ["--phase", "loot"]),
    ]
    tlog("  [auto-attack] waiting for LIVE...", kind="phase")
    while True:
        if STATE.snapshot().get("phase") == "live":
            break
        time.sleep(1.0)
    time.sleep(3.0)
    for name, cmd in phases:
        if STATE.snapshot().get("blocked"):
            tlog(f"  [auto-attack] stopped — IPS already blocked by {STATE.snapshot().get('first_blocker')}", kind="alert")
            break
        STATE.update(attack_phase=name)
        tlog(f"  [auto-attack] phase={name}", kind="phase")
        rc = _run_attack_cmd(name, cmd)
        if rc != 0:
            tlog(f"  [auto-attack] phase {name} exited {rc}", kind="alert")
        time.sleep(6.0)
    STATE.update(attack_phase="complete")
    tlog("  [auto-attack] complete", kind="phase")


def run_loop(
    *,
    warmup_windows: int,
    no_up: bool,
    horizon: int,
    auto_attack: bool,
    window_sec: float,
    confirm_windows: int,
    lab_adapt: bool = True,
) -> None:
    if not no_up:
        up_lab()
        wait_http(SITE_URL)
    else:
        try:
            wait_http(SITE_URL, timeout=8.0)
        except RuntimeError as exc:
            tlog(f"  WARNING: {exc} — continuing without live site traffic", kind="alert")
    clear_blocks()
    if HIST_PATH.exists():
        HIST_PATH.write_text("", encoding="utf-8")
    PCAP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gw = get_bridge_gateway()
        rt = get_container_ip(REDTEAM_CONTAINER)
        ab = get_container_ip(ATTACKER_CONTAINER)
        tlog(f"  IPS will drop: gateway={gw or '?'} redteam={rt} attacker-bot={ab}")
    except RuntimeError as exc:
        tlog(f"  WARNING: {exc}", kind="alert")

    if auto_attack:
        threading.Thread(target=run_auto_attack, daemon=True).start()

    shaun_bundle = load_shaun_bundle(ckpt_path=W5S if W5S.exists() else None)
    ckpt_path = resolve_hxc_ckpt()
    hxc_bundle = load_hxc_bundle(ckpt_path=ckpt_path)
    if ckpt_path.name.endswith("_lab.pt"):
        tlog(f"  HX-C using lab-adapted weights: {ckpt_path}", kind="phase")
    sn = StreamingShaunRamxV3(shaun_bundle, context_skip_steps=10_000)
    hx = StreamingHXC(hxc_bundle, context_skip_steps=10_000)
    sn.begin_live_phase()
    hx.begin_live_phase()

    capture = TrafficCapture(container_name=TARGET_CONTAINER, output_dir=PCAP_DIR)
    last = None
    t0 = time.time()
    window_idx = 0
    warmup_n = 0
    phase = "warmup"
    forecast_alert = False
    blocked = False
    pending = {"sn2rx3": 0, "hx_c": 0}
    v3_hist: deque[dict[str, Any]] = deque(maxlen=HISTORY_MAX)
    hx_hist: deque[dict[str, Any]] = deque(maxlen=HISTORY_MAX)
    tlog(f"  warmup {warmup_windows} x {window_sec:.0f}s windows then LIVE. Dashboard http://127.0.0.1:{DASH_PORT}")
    tlog(f"  site {SITE_URL}  IPS confirm={confirm_windows} consecutive >={ATTACK_THRESHOLD}")
    if auto_attack:
        tlog("  auto-attack enabled — kill chain will run after LIVE")
    if lab_adapt:
        tlog("  lab adapt ON — HX-C finetunes after block or run end (ground-truth phases)")

    adapt_samples: list[dict[str, Any]] = []
    prev_attack_phase = "idle"
    phase_regions: list[dict[str, Any]] = []
    memory_regions: list[dict[str, Any]] = []
    try:
        while True:
            horizon = read_horizon_windows(window_sec, horizon)
            window_idx += 1
            if phase == "warmup" and warmup_n >= warmup_windows:
                phase = "live"
                sn.set_context_skip(warmup_n)
                hx.set_context_skip(warmup_n)
                tlog("  LIVE — scoring on; IPS arms for enum/spray/loot only (recon = observe).", kind="phase")
            fname = f"fc_w{window_idx:05d}.pcap"
            capture.start_capture(fname)
            time.sleep(window_sec)
            pcap = capture.stop_capture()
            last = ingest_pcap(pcap, last, window_sec=window_sec)
            if pcap and pcap.exists():
                try:
                    pcap.unlink()
                except OSError:
                    pass
            out_v3 = sn.step(last)
            out_hx = hx.step(last)
            roll_v3 = sn.rollout(horizon)
            roll_hx = hx.rollout(horizon)
            p3 = float(out_v3["p_att"])
            ph = float(out_hx["p_att"])
            fc = max(float(roll_hx.get("forecast_max") or 0.0), ph)
            rel = float(out_hx.get("relative_anomaly") or 0.0)
            hx_label = out_hx.get("technique") or out_hx.get("mitre_stage_name")
            if level_from_p(ph) != "attack":
                hx_label = None
            v3_label = out_v3.get("mitre_stage_name")
            if level_from_p(p3) != "attack":
                v3_label = None
            v3_hist.append({"w": window_idx, "p": p3, "label": v3_label})
            hx_hist.append({"w": window_idx, "p": ph, "label": hx_label})
            if out_hx.get("memory_written"):
                memory_regions = bump_memory_regions(memory_regions, window_idx, system="hx_c")
            models = {
                "shaun_v3": build_timeline(v3_hist, roll_v3, window_idx=window_idx, window_sec=window_sec, horizon=horizon),
                "hx_c": build_timeline(hx_hist, roll_hx, window_idx=window_idx, window_sec=window_sec, horizon=horizon),
            }
            if phase == "warmup":
                warmup_n += 1
            attack_phase = STATE.snapshot().get("attack_phase", "idle")
            phase_regions = bump_phase_regions(
                phase_regions, window_idx, attack_phase, prev_attack_phase,
            )
            if attack_phase in RECON_PHASES | IPS_ARM_PHASES and attack_phase != prev_attack_phase:
                pending = {"sn2rx3": 0, "hx_c": 0}
            ips_armed = phase == "live" and not blocked and attack_phase in IPS_ARM_PHASES
            ips_confirm = IPS_ATTACK_CONFIRM if ips_armed else confirm_windows
            if phase == "live" and attack_phase in FORECAST_PHASE_LABELS and lab_adapt:
                adapt_samples.append({
                    "seq": hx._scaled_window(hx.buf),
                    "phase": attack_phase,
                    "window": window_idx,
                })
            if ips_armed:
                for name, p in (("sn2rx3", p3), ("hx_c", ph)):
                    pending[name] = pending[name] + 1 if p >= ATTACK_THRESHOLD else 0
                fired = [n for n, c in pending.items() if c >= ips_confirm]
                fired.sort(key=lambda n: {"sn2rx3": p3, "hx_c": ph}[n], reverse=True)
                if fired:
                    blocked = True
                    gt = phase_to_class_id(attack_phase)
                    apply_ips_block(
                        t0,
                        window_idx,
                        fired,
                        {"sn2rx3": p3, "hx_c": ph},
                        sn=sn,
                        hx=hx,
                        out_v3=out_v3,
                        out_hx=out_hx,
                        ground_truth_class=gt,
                    )
                    do_lab_adapt(hx, hxc_bundle, adapt_samples, lab_adapt=lab_adapt, reason=f"block_w{window_idx}")
            elif phase == "live" and attack_phase in RECON_PHASES:
                pending = {"sn2rx3": 0, "hx_c": 0}
            if (
                lab_adapt
                and prev_attack_phase != "complete"
                and attack_phase == "complete"
            ):
                do_lab_adapt(hx, hxc_bundle, adapt_samples, lab_adapt=lab_adapt, reason="killchain_complete")
            prev_attack_phase = attack_phase
            if phase == "live" and not forecast_alert and ph >= ATTACK_THRESHOLD and rel >= 0.30:
                forecast_alert = True
                tlog(
                    f"\n*** FORECAST ALERT by HX-C at window {window_idx} "
                    f"t+{time.time()-t0:.1f}s  now={ph:.3f} cone={fc:.3f} rel={rel:.3f} ***\n",
                    kind="alert",
                )
                STATE.update(
                    forecast_alert=True,
                    forecast_at_sec=time.time() - t0,
                    forecast_window=window_idx,
                )
            STATE.update(
                phase=phase,
                window=window_idx,
                elapsed_sec=time.time() - t0,
                horizon=horizon,
                window_sec=window_sec,
                p_sn2rx3=p3,
                p_hx_c=ph,
                forecast_max=fc,
                class_shift=float(out_hx.get("class_shift") or 0.0),
                relative_anomaly=rel,
                pending_sn2rx3=pending["sn2rx3"],
                pending_hx_c=pending["hx_c"],
                confirm_need=ips_confirm if ips_armed else confirm_windows,
                ips_confirm=IPS_ATTACK_CONFIRM,
                ips_armed=ips_armed,
                phase_regions=phase_regions,
                memory_regions=memory_regions,
                models=models,
            )
            tlog(
                f"  w{window_idx:03d} {phase:7s}  hx={ph:.3f}  cone={fc:.3f}  "
                f"v3={p3:.3f}  blocked={blocked}  attack={attack_phase}"
            )
    finally:
        do_lab_adapt(hx, hxc_bundle, adapt_samples, lab_adapt=lab_adapt, reason="run_end")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-up", action="store_true")
    p.add_argument("--warmup-windows", type=int, default=60, help="Warmup windows (1s each by default)")
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="Capture/score window length (default 1s)")
    p.add_argument("--confirm-windows", type=int, default=CONFIRM_WINDOWS, help="Consecutive alert windows before IPS DROP")
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--dash-port", type=int, default=DASH_PORT)
    p.add_argument("--auto-attack", action="store_true", help="Run kill-chain automatically after LIVE")
    p.add_argument(
        "--no-lab-adapt",
        action="store_true",
        help="Disable post-run HX-C finetune from kill-chain ground truth",
    )
    args = p.parse_args()
    acquire_pid_lock()
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(("127.0.0.1", args.dash_port), DashHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Labs forecast dashboard: http://127.0.0.1:{args.dash_port}  pid={os.getpid()}")
    try:
        run_loop(
            warmup_windows=args.warmup_windows,
            no_up=args.no_up,
            horizon=args.horizon,
            auto_attack=args.auto_attack,
            window_sec=args.window_sec,
            confirm_windows=args.confirm_windows,
            lab_adapt=not args.no_lab_adapt,
        )
    except KeyboardInterrupt:
        print("\nStopping forecast scorer (lab containers stay up).")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
