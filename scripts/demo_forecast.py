#!/usr/bin/env python3
"""Forecast demo: Harborline site + HX-C K-step cone + Shaun v3 now-P.

This is IDS/forecast, not delayed IPS. The dashboard shows:
  - now: fused P on the window that just closed
  - forecast: HX-C world-model rollout of the next K windows

Usage:
  python scripts/demo_forecast.py
  python scripts/demo_forecast.py --no-up --warmup-windows 15
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adversarial.lab_config import TARGET_CONTAINER  # noqa: E402
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.hx.causal import StreamingHXC, load_hxc_bundle  # noqa: E402
from src.shaun.streaming import StreamingShaunRamxV3, load_shaun_bundle, pcap_to_shaun_state  # noqa: E402

COMPOSE_BASE = ROOT / "docker" / "docker-compose.yml"
COMPOSE_FC = ROOT / "docker" / "docker-compose.forecast.yml"
W5S = ROOT.parent / "PRISM-shaun" / "weights" / "w5s" / "world_model.pt"
EVENTS = ROOT / "data" / "lab_events"
STATE_PATH = EVENTS / "demo_forecast.json"
HIST_PATH = EVENTS / "demo_forecast.jsonl"
PID_PATH = EVENTS / "demo_forecast.pid"
PCAP_DIR = EVENTS / "demo_forecast_pcaps"
DETECT = 0.5
WINDOW_SEC = 5.0
HORIZON = 6
DASH_PORT = 8788
SITE_URL = "http://127.0.0.1:8080"

DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PRISM Labs — Forecast</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: Consolas, ui-monospace, sans-serif; background: #0d1117; color: #c9d1d9; }
    header { padding: 14px 20px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; }
    h1 { font-size: 16px; margin: 0; color: #3fb950; letter-spacing: 0.08em; }
    .ok { color: #3fb950; } .hot { color: #f85149; } .muted { color: #8b949e; } .warn { color: #d29922; }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 12px; padding: 16px 20px; }
    .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 14px; }
    .k { color: #8b949e; font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; }
    .v { font-size: 22px; margin-top: 6px; color: #58a6ff; }
    canvas { width: 100%; height: 280px; background: #010409; border: 1px solid #30363d; border-radius: 8px; }
    .wrap { padding: 0 20px 20px; }
    .banner { margin: 12px 20px; padding: 12px 14px; border-radius: 8px; border: 1px solid #30363d; background: #010409; }
    a { color: #58a6ff; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    td, th { text-align: left; padding: 6px 8px; border-bottom: 1px solid #30363d; }
  </style>
</head>
<body>
  <header>
    <h1>PRISM LABS // FORECAST</h1>
    <div id="clock" class="muted"></div>
  </header>
  <div id="banner" class="banner muted">Warming up on Harborline benign traffic. Do not start recon yet.</div>
  <div class="grid">
    <div class="card"><div class="k">HX-C now</div><div class="v" id="p_hx">—</div></div>
    <div class="card"><div class="k">HX-C forecast max (next 30s)</div><div class="v" id="p_fc">—</div></div>
    <div class="card"><div class="k">Shaun v3 now</div><div class="v" id="p_v3">—</div></div>
    <div class="card"><div class="k">Class shift / anomaly</div><div class="v" id="shift">—</div></div>
  </div>
  <div class="wrap">
    <canvas id="chart" width="1200" height="280"></canvas>
    <p class="muted">Site: <a href="http://127.0.0.1:8080" target="_blank">http://127.0.0.1:8080</a>
      · solid = now · dashed orange = forecast cone · ~80 benign users · kill chain: recon → enum → IDOR → loot</p>
    <table>
      <thead><tr><th>w</th><th>phase</th><th>HX now</th><th>HX forecast</th><th>v3 now</th><th>shift</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
<script>
const histHx = [], histFc = [], histV3 = [];
function draw() {
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.fillStyle = '#010409'; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = '#30363d';
  ctx.beginPath(); ctx.moveTo(40, 20); ctx.lineTo(40, h-24); ctx.lineTo(w-10, h-24); ctx.stroke();
  const y = (v) => (h-24) - v * (h-44);
  ctx.strokeStyle = '#d29922'; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(40, y(0.5)); ctx.lineTo(w-10, y(0.5)); ctx.stroke(); ctx.setLineDash([]);
  function series(data, color, dash) {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.setLineDash(dash || []);
    ctx.beginPath();
    data.forEach((v,i) => {
      const x = 40 + i * ((w-50) / Math.max(data.length-1, 1));
      if (i===0) ctx.moveTo(x, y(v)); else ctx.lineTo(x, y(v));
    });
    ctx.stroke(); ctx.setLineDash([]);
  }
  series(histV3, '#8b949e');
  series(histHx, '#58a6ff');
  series(histFc, '#d29922', [6,4]);
  ctx.fillStyle = '#8b949e'; ctx.font = '11px Consolas';
  ctx.fillText('Shaun now', 50, 16);
  ctx.fillStyle = '#58a6ff'; ctx.fillText('HX now', 130, 16);
  ctx.fillStyle = '#d29922'; ctx.fillText('HX forecast max', 200, 16);
}
async function tick() {
  const s = await (await fetch('/api/state')).json();
  document.getElementById('p_hx').textContent = s.p_hx_c.toFixed(3);
  document.getElementById('p_fc').textContent = s.forecast_max.toFixed(3);
  document.getElementById('p_v3').textContent = s.p_sn2rx3.toFixed(3);
  document.getElementById('shift').textContent = (s.class_shift||0).toFixed(3) + ' / ' + (s.relative_anomaly||0).toFixed(3);
  document.getElementById('clock').textContent = s.phase.toUpperCase() + '  w' + s.window + '  t+' + s.elapsed_sec.toFixed(0) + 's';
  const b = document.getElementById('banner');
  if (s.forecast_alert) {
    b.className = 'banner hot';
    b.textContent = 'FORECAST ALERT at t+' + s.forecast_at_sec.toFixed(1) +
      's window ' + s.forecast_window + ' — cone max ' + s.forecast_max.toFixed(3) +
      ' (HX now ' + s.p_hx_c.toFixed(3) + '). Loot may not have happened yet.';
  } else if (s.phase === 'live') {
    b.className = 'banner ok';
    b.textContent = 'LIVE — start recon (scan) from a terminal. Forecast should rise before payroll download.';
  } else {
    b.className = 'banner muted';
    b.textContent = 'WARMUP — Harborline benign users only. Wait for LIVE before recon.';
  }
  histHx.push(s.p_hx_c); histFc.push(s.forecast_max); histV3.push(s.p_sn2rx3);
  if (histHx.length > 60) { histHx.shift(); histFc.shift(); histV3.shift(); }
  draw();
  const tr = document.createElement('tr');
  tr.innerHTML = '<td>'+s.window+'</td><td>'+s.phase+'</td><td>'+s.p_hx_c.toFixed(3)+'</td><td>'+s.forecast_max.toFixed(3)+'</td><td>'+s.p_sn2rx3.toFixed(3)+'</td><td>'+(s.class_shift||0).toFixed(3)+'</td>';
  const tb = document.getElementById('rows');
  tb.prepend(tr);
  while (tb.children.length > 24) tb.removeChild(tb.lastChild);
}
setInterval(tick, 1000); tick();
</script>
</body></html>
"""


class LiveState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, Any] = {
            "phase": "starting",
            "window": 0,
            "elapsed_sec": 0.0,
            "p_sn2rx3": 0.0,
            "p_hx_c": 0.0,
            "forecast_max": 0.0,
            "forecast": [],
            "class_shift": 0.0,
            "relative_anomaly": 0.0,
            "forecast_alert": False,
            "forecast_at_sec": None,
            "forecast_window": None,
            "site_url": SITE_URL,
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


def ingest_pcap(pcap: Path, last: np.ndarray | None) -> np.ndarray:
    if pcap is None or not pcap.exists():
        return last if last is not None else np.zeros(292, np.float32)
    try:
        st = pcap_to_shaun_state(pcap, last_state=last, window_sec=WINDOW_SEC)
        return np.asarray(st, dtype=np.float32)
    except Exception as exc:
        print(f"  ingest failed: {exc}", file=sys.stderr)
        return last if last is not None else np.zeros(292, np.float32)


def run_loop(*, warmup_windows: int, no_up: bool, horizon: int) -> None:
    if not no_up:
        up_lab()
    wait_http(SITE_URL)
    if HIST_PATH.exists():
        HIST_PATH.write_text("", encoding="utf-8")
    PCAP_DIR.mkdir(parents=True, exist_ok=True)

    shaun_bundle = load_shaun_bundle(ckpt_path=W5S if W5S.exists() else None)
    hxc_bundle = load_hxc_bundle()
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
    print(f"  warmup {warmup_windows} windows then LIVE. Dashboard http://127.0.0.1:{DASH_PORT}")
    print(f"  site {SITE_URL}")

    while True:
        window_idx += 1
        if phase == "warmup" and warmup_n >= warmup_windows:
            phase = "live"
            sn.set_context_skip(warmup_n)
            hx.set_context_skip(warmup_n)
            print(f"  LIVE - start recon. Forecast cone is the orange dashed line.")
        fname = f"fc_w{window_idx:05d}.pcap"
        capture.start_capture(fname)
        time.sleep(WINDOW_SEC)
        pcap = capture.stop_capture()
        last = ingest_pcap(pcap, last)
        if pcap and pcap.exists():
            try:
                pcap.unlink()
            except OSError:
                pass
        out_v3 = sn.step(last)
        out_hx = hx.step(last)
        roll = hx.rollout(horizon)
        p3 = float(out_v3["p_att"])
        ph = float(out_hx["p_att"])
        fc = max(float(roll.get("forecast_max") or 0.0), ph)
        rel = float(out_hx.get("relative_anomaly") or 0.0)
        if phase == "warmup":
            warmup_n += 1
        if phase == "live" and not forecast_alert and ph >= DETECT and rel >= 0.30:
            forecast_alert = True
            print(
                f"\n*** FORECAST ALERT by HX-C at window {window_idx} "
                f"t+{time.time()-t0:.1f}s  now={ph:.3f} cone={fc:.3f} rel={rel:.3f} "
                f"(recon-scale; loot not required) ***\n"
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
            p_sn2rx3=p3,
            p_hx_c=ph,
            forecast_max=fc,
            forecast=roll.get("forecast") or [],
            class_shift=float(out_hx.get("class_shift") or 0.0),
            relative_anomaly=float(out_hx.get("relative_anomaly") or 0.0),
        )
        print(
            f"  w{window_idx:03d} {phase:7s}  hx={ph:.3f}  cone={fc:.3f}  "
            f"v3={p3:.3f}  shift={float(out_hx.get('class_shift') or 0):.3f}  "
            f"rel={float(out_hx.get('relative_anomaly') or 0):.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-up", action="store_true")
    p.add_argument("--warmup-windows", type=int, default=20)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--dash-port", type=int, default=DASH_PORT)
    args = p.parse_args()
    acquire_pid_lock()
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(("127.0.0.1", args.dash_port), DashHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Labs forecast dashboard: http://127.0.0.1:{args.dash_port}  pid={os.getpid()}")
    try:
        run_loop(warmup_windows=args.warmup_windows, no_up=args.no_up, horizon=args.horizon)
    except KeyboardInterrupt:
        print("\nStopping forecast scorer (lab containers stay up).")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
