#!/usr/bin/env python3
"""Live dual IPS demo: Shaun RAMX v3 + HX-C, site on :8080, Labs UI on :8787.

Simulates ~1000 benign users in Docker, scores both models on 5s captures,
and the first model to reach P>=0.5 blocks the host browser + red-team IP.

Usage:
  python scripts/demo_dual_ips.py              # bring up lab + score forever
  python scripts/demo_dual_ips.py --no-up      # lab already running
  python scripts/demo_dual_ips.py --warmup 20
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

from src.adversarial.ips_controller import (  # noqa: E402
    block_demo_attackers,
    clear_blocks,
    get_bridge_gateway,
    get_container_ip,
)
from src.adversarial.lab_config import REDTEAM_CONTAINER, TARGET_CONTAINER  # noqa: E402
from src.adversarial.traffic_capture import TrafficCapture  # noqa: E402
from src.hx.causal import StreamingHXC, load_hxc_bundle  # noqa: E402
from src.shaun.streaming import StreamingShaunRamxV3, load_shaun_bundle, pcap_to_shaun_state  # noqa: E402

COMPOSE_BASE = ROOT / "docker" / "docker-compose.yml"
COMPOSE_DEMO = ROOT / "docker" / "docker-compose.demo.yml"
W5S = ROOT.parent / "PRISM-shaun" / "weights" / "w5s" / "world_model.pt"
EVENTS = ROOT / "data" / "lab_events"
STATE_PATH = EVENTS / "demo_live.json"
HIST_PATH = EVENTS / "demo_live.jsonl"
PID_PATH = EVENTS / "demo_dual_ips.pid"
PCAP_DIR = EVENTS / "demo_pcaps"
DETECT = 0.5
WINDOW_SEC = 5.0
DASH_PORT = 8787
SITE_URL = "http://127.0.0.1:8080"

DASHBOARD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>PRISM Labs — Live dual IPS</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; font-family: Consolas, ui-monospace, sans-serif; background: #0d1117; color: #c9d1d9; }
    header { padding: 14px 20px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
    h1 { font-size: 16px; margin: 0; color: #58a6ff; letter-spacing: 0.08em; }
    .ok { color: #3fb950; } .hot { color: #f85149; } .muted { color: #8b949e; }
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
    <h1>PRISM LABS // DUAL IPS</h1>
    <div id="clock" class="muted"></div>
  </header>
  <div id="banner" class="banner muted">Warming up — browsing the site is safe. Attack when the phase reads LIVE.</div>
  <div class="grid">
    <div class="card"><div class="k">Shaun RAMX v3</div><div class="v" id="p_v3">—</div></div>
    <div class="card"><div class="k">HX-C</div><div class="v" id="p_hxc">—</div></div>
    <div class="card"><div class="k">First blocker</div><div class="v" id="blocker">none</div></div>
    <div class="card"><div class="k">Block time</div><div class="v" id="blockt">—</div></div>
  </div>
  <div class="grid" style="padding-top:0">
    <div class="card"><div class="k">v3 streak ≥0.5</div><div class="v" id="st_v3">0 / 2</div></div>
    <div class="card"><div class="k">HX-C streak ≥0.5</div><div class="v" id="st_hx">0 / 2</div></div>
  </div>
  <div class="wrap">
    <canvas id="chart" width="1200" height="280"></canvas>
    <p class="muted">Site: <a href="http://127.0.0.1:8080" target="_blank">http://127.0.0.1:8080</a>
      · threshold 0.5 · 5s windows · ~1000 simulated users in Docker</p>
    <table>
      <thead><tr><th>window</th><th>phase</th><th>v3 P</th><th>HX-C P</th><th>blocked</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
<script>
const histV3 = [], histHx = [], labels = [];
function draw() {
  const c = document.getElementById('chart');
  const ctx = c.getContext('2d');
  const w = c.width, h = c.height;
  ctx.fillStyle = '#010409'; ctx.fillRect(0,0,w,h);
  ctx.strokeStyle = '#30363d';
  ctx.beginPath(); ctx.moveTo(40, 20); ctx.lineTo(40, h-24); ctx.lineTo(w-10, h-24); ctx.stroke();
  const thr = 0.5;
  const y = (v) => (h-24) - v * (h-44);
  ctx.strokeStyle = '#d29922'; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(40, y(thr)); ctx.lineTo(w-10, y(thr)); ctx.stroke(); ctx.setLineDash([]);
  function series(data, color) {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.beginPath();
    data.forEach((v,i) => {
      const x = 40 + i * ((w-50) / Math.max(data.length-1, 1));
      if (i===0) ctx.moveTo(x, y(v)); else ctx.lineTo(x, y(v));
    });
    ctx.stroke();
  }
  series(histV3, '#8b949e');
  series(histHx, '#58a6ff');
  ctx.fillStyle = '#8b949e'; ctx.font = '11px Consolas';
  ctx.fillText('Shaun v3', 50, 16); ctx.fillStyle = '#58a6ff'; ctx.fillText('HX-C', 130, 16);
}
async function tick() {
  const s = await (await fetch('/api/state')).json();
  document.getElementById('p_v3').textContent = s.p_sn2rx3.toFixed(3);
  document.getElementById('p_hxc').textContent = s.p_hx_c.toFixed(3);
  document.getElementById('blocker').textContent = s.first_blocker || 'none';
  document.getElementById('blockt').textContent = s.block_at_sec == null ? '—' : (s.block_at_sec.toFixed(1) + 's');
  document.getElementById('st_v3').textContent = (s.pending_sn2rx3 || 0) + ' / ' + (s.confirm_need || 2);
  document.getElementById('st_hx').textContent = (s.pending_hx_c || 0) + ' / ' + (s.confirm_need || 2);
  document.getElementById('clock').textContent = s.phase.toUpperCase() + '  w' + s.window + '  t+' + s.elapsed_sec.toFixed(0) + 's';
  const b = document.getElementById('banner');
  if (s.blocked) {
    b.className = 'banner hot';
    b.textContent = 'IPS BLOCK by ' + s.first_blocker + ' at t+' + s.block_at_sec.toFixed(1) +
      's (window ' + s.block_window + '). Host browser to :8080 should now fail. Benign Docker users stay up.';
  } else if (s.phase === 'live') {
    b.className = 'banner ok';
        b.textContent = 'LIVE — site is open. Attack from the :8080 tab. First model with 2 consecutive windows ≥ 0.5 issues the iptables DROP.';
  } else {
    b.className = 'banner muted';
    b.textContent = 'WARMUP — models are calibrating on simulated users. Wait until LIVE before attacking.';
  }
  histV3.push(s.p_sn2rx3); histHx.push(s.p_hx_c);
  if (histV3.length > 60) { histV3.shift(); histHx.shift(); }
  draw();
  const tr = document.createElement('tr');
  tr.innerHTML = '<td>'+s.window+'</td><td>'+s.phase+'</td><td>'+s.p_sn2rx3.toFixed(3)+'</td><td>'+s.p_hx_c.toFixed(3)+'</td><td>'+(s.blocked?'YES':'')+'</td>';
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
            "blocked": False,
            "first_blocker": None,
            "block_at_sec": None,
            "block_window": None,
            "block_ips": [],
            "same_window_tie": False,
            "site_url": SITE_URL,
            "pending_sn2rx3": 0,
            "pending_hx_c": 0,
            "confirm_need": 2,
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
    return ["docker", "compose", "-f", str(COMPOSE_BASE), "-f", str(COMPOSE_DEMO), "-p", "prism", *args]


def up_lab() -> None:
    (ROOT / "data" / "lab_events").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    print("Starting demo lab (site on 127.0.0.1:8080, ~1000 simulated users)...")
    subprocess.run(compose("up", "-d", "--build"), cwd=str(ROOT), check=True, env=env)
    subprocess.run(compose("ps"), cwd=str(ROOT))


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


def apply_block(t0: float, window_idx: int, fired: list[str], scores: dict[str, float]) -> None:
    results = block_demo_attackers(include_redteam=True, include_host_gateway=True)
    ips = [r.source_ip for r in results if r.blocked]
    first = fired[0] if fired else None
    tie = len(fired) > 1
    print(
        f"\n*** IPS BLOCK by {first} at window {window_idx} "
        f"t+{time.time()-t0:.1f}s  fired={fired} scores={scores} ips={ips} tie={tie} ***\n"
    )
    STATE.update(
        blocked=True,
        first_blocker=first,
        block_at_sec=time.time() - t0,
        block_window=window_idx,
        block_ips=ips,
        same_window_tie=tie,
        fired=fired,
        scores_at_block=scores,
    )


def run_loop(*, warmup_sec: float, warmup_windows: int, no_up: bool) -> None:
    if not no_up:
        up_lab()
    wait_http(SITE_URL)
    clear_blocks()
    if HIST_PATH.exists():
        HIST_PATH.write_text("", encoding="utf-8")
    PCAP_DIR.mkdir(parents=True, exist_ok=True)

    try:
        gw = get_bridge_gateway()
        rt = get_container_ip(REDTEAM_CONTAINER)
        print(f"  host-NAT gateway (browser source): {gw or '?'}")
        print(f"  redteam IP: {rt}")
    except RuntimeError as exc:
        print(f"  WARNING: {exc}")

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
    pending = {"sn2rx3": 0, "hx_c": 0}
    confirm_windows = 2
    blocked = False
    phase = "warmup"
    print(f"  warmup {warmup_windows} windows then LIVE. Dashboard http://127.0.0.1:{DASH_PORT}")
    print(f"  site {SITE_URL}")

    while True:
        window_idx += 1
        elapsed = time.time() - t0
        if phase == "warmup" and warmup_n >= warmup_windows and elapsed >= warmup_sec:
            phase = "live"
            sn.set_context_skip(warmup_n)
            hx.set_context_skip(warmup_n)
            print(f"  LIVE - context gate closed after {warmup_n} warmup windows (need {confirm_windows} consecutive alerts to IPS-block)")
        fname = f"demo_w{window_idx:05d}.pcap"
        capture.start_capture(fname)
        time.sleep(WINDOW_SEC)
        pcap = capture.stop_capture()
        last = ingest_pcap(pcap, last)
        sys.path.insert(0, str(ROOT))
        if pcap and pcap.exists():
            try:
                pcap.unlink()
            except OSError:
                pass
        out_v3 = sn.step(last)
        out_hx = hx.step(last)
        p3 = float(out_v3["p_att"])
        ph = float(out_hx["p_att"])
        raw3 = float(out_v3.get("raw_p_att") or p3)
        rawh = float(out_hx.get("raw_p_att") or ph)
        if phase == "warmup":
            warmup_n += 1
        STATE.update(
            phase=phase,
            window=window_idx,
            elapsed_sec=time.time() - t0,
            p_sn2rx3=p3,
            p_hx_c=ph,
            raw_sn2rx3=raw3,
            raw_hx_c=rawh,
            pending_sn2rx3=pending["sn2rx3"],
            pending_hx_c=pending["hx_c"],
            confirm_need=confirm_windows,
        )
        print(f"  w{window_idx:03d} {phase:7s}  v3={p3:.3f} (raw {raw3:.3f})  hx_c={ph:.3f} (raw {rawh:.3f})  blocked={blocked}")
        if phase == "live" and not blocked:
            for name, p in (("sn2rx3", p3), ("hx_c", ph)):
                pending[name] = pending[name] + 1 if p >= DETECT else 0
            STATE.update(pending_sn2rx3=pending["sn2rx3"], pending_hx_c=pending["hx_c"])
            fired = [n for n, c in pending.items() if c >= confirm_windows]
            fired.sort(key=lambda n: {"sn2rx3": p3, "hx_c": ph}[n], reverse=True)
            if fired:
                blocked = True
                apply_block(t0, window_idx, fired, {"sn2rx3": p3, "hx_c": ph})


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    r = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
        timeout=10,
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
            raise SystemExit(
                f"Labs scorer already running as pid {old}. "
                f"Stop it first (Stop-Process -Id {old}) so :{DASH_PORT} is not shared."
            )
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-up", action="store_true")
    p.add_argument("--warmup", type=float, default=0.0, help="Extra wall-seconds after min warmup windows")
    p.add_argument("--warmup-windows", type=int, default=20, help="Min gated windows so RAMX offset is ready")
    p.add_argument("--dash-port", type=int, default=DASH_PORT)
    args = p.parse_args()

    acquire_pid_lock()
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(("127.0.0.1", args.dash_port), DashHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"Labs dashboard: http://127.0.0.1:{args.dash_port}  pid={os.getpid()}")
    try:
        run_loop(warmup_sec=args.warmup, warmup_windows=args.warmup_windows, no_up=args.no_up)
    except KeyboardInterrupt:
        print("\nStopping demo scorer (lab containers stay up).")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
