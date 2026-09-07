#!/usr/bin/env python3
"""FastAPI BFF for PRISM Lab Console (phase 2 stubs).

Run:
  pip install fastapi uvicorn
  python -m apps.lab_console.api.main

Serves on http://127.0.0.1:8790 — proxied by Vite dev server at /api.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[3]
MOCKS = ROOT / "apps" / "lab-console" / "mocks"
sys.path.insert(0, str(ROOT))

app = FastAPI(title="PRISM Lab Console API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOG_LINES: list[dict] = [
    {"ts": 1, "kind": "phase", "text": "BFF stub — connect scorer in phase 2"},
]


def _load_mock(name: str) -> dict:
    path = MOCKS / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


@app.get("/api/dashboard")
def get_dashboard() -> dict:
    try:
        from src.adversarial.lab_manager import lab_status

        return {"containers": lab_status(), "stub": False}
    except Exception:
        return _load_mock("dashboard")


@app.get("/api/sessions/{session_id}/state")
def get_session_state(session_id: str) -> dict:
    try:
        from src.lab.forecast_session import ForecastSession

        sess = ForecastSession()
        return _load_mock("session") | {"id": session_id, "bff": sess.state.snapshot()}
    except Exception:
        return _load_mock("session") | {"id": session_id}


@app.patch("/api/sessions/{session_id}/policy")
def patch_policy(session_id: str, body: dict) -> dict:
    return {"id": session_id, "policyMode": body.get("policyMode", "ids")}


@app.get("/api/logs")
def get_logs(since: int = 0) -> dict:
    return {"lines": [l for l in _LOG_LINES if l.get("ts", 0) >= since]}


@app.post("/api/scripts/run")
def run_script(body: dict) -> dict:
    script_id = body.get("script_id", "unknown")
    _LOG_LINES.append({"ts": 0, "kind": "phase", "text": f"[script] queued {script_id}"})
    return {"job_id": f"job-{script_id}", "status": "queued"}


@app.get("/api/scripts/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return {"job_id": job_id, "status": "completed", "output": ["mock output"]}


@app.get("/api/models")
def list_models() -> dict:
    return {"models": []}


@app.get("/api/recordings")
def list_recordings() -> dict:
    rec_dir = ROOT / "reports" / "lab" / "hx" / "recordings"
    items = []
    if rec_dir.is_dir():
        for p in sorted(rec_dir.glob("*.json"))[:50]:
            try:
                items.append(json.loads(p.read_text(encoding="utf-8")) | {"path": str(p)})
            except Exception:
                pass
    return {"recordings": items}


@app.websocket("/api/terminal")
async def terminal_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_text("Terminal PTY not wired yet — phase 2\r\n")
    await ws.close()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8790, reload=False)


if __name__ == "__main__":
    main()
