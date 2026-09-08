#!/usr/bin/env python3
"""Stub BFF for UI/UX branch — rich mocks only, no Docker/models/scorer.

Run:
  python apps/lab-console/api/stub_main.py

Or from repo root:
  python run_lab_console_ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(API_DIR))

from stub_data import (  # noqa: E402
    dashboard,
    get_config,
    logs,
    models_registry,
    patch_config,
    run_script,
    scripts,
    session,
    set_policy,
)

API_VERSION = "0.3.0-ui-stub"

app = FastAPI(title="PRISM Lab Console API (UI stub)", version=API_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PolicyBody(BaseModel):
    policyMode: str


class ScriptBody(BaseModel):
    script_id: str
    delay_sec: float | None = None


class LabConfigBody(BaseModel):
    horizonSec: float | None = None
    attackDelaySec: float | None = None


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "apiVersion": API_VERSION,
        "features": {"terminalPty": False, "liveCharts": True, "stub": True},
    }


@app.get("/api/dashboard")
def get_dashboard() -> dict:
    return dashboard()


@app.get("/api/sessions/{session_id}/state")
def get_session_state(session_id: str, mode: str = "recorded") -> dict:
    return session(mode)


@app.patch("/api/sessions/{session_id}/policy")
def patch_policy(session_id: str, body: PolicyBody) -> dict:
    mode = set_policy(body.policyMode)
    return {"id": session_id, "policyMode": mode}


@app.get("/api/logs")
def api_logs(since: int = 0) -> dict:
    return {"lines": logs(since)}


@app.get("/api/scripts")
def api_scripts() -> dict:
    return {"scripts": scripts()}


@app.post("/api/scripts/run")
def api_run_script(body: ScriptBody) -> dict:
    return run_script(body.script_id, delay_sec=body.delay_sec)


@app.get("/api/lab-config")
def api_get_lab_config() -> dict:
    return get_config()


@app.patch("/api/lab-config")
def api_patch_lab_config(body: LabConfigBody) -> dict:
    return patch_config(**body.model_dump(exclude_none=True))


@app.get("/api/models")
def api_models() -> dict:
    return {"models": models_registry()}


@app.get("/api/recordings")
def list_recordings() -> dict:
    return {
        "recordings": [
            {
                "id": "rec-demo-001",
                "title": "Harborline kill-chain — IPS block @ enum",
                "durationSec": 92,
                "createdAt": "2026-09-05T14:22:00Z",
                "models": ["shaun_v3", "hx_c"],
            },
            {
                "id": "rec-demo-002",
                "title": "Benign baseline + RAMX episodic",
                "durationSec": 120,
                "createdAt": "2026-09-04T09:10:00Z",
                "models": ["hx_c"],
            },
        ]
    }


@app.websocket("/api/terminal")
async def terminal_ws(ws: WebSocket) -> None:
    await ws.accept()
    await ws.send_text("\r\n\x1b[33m[stub]\x1b[0m Shell disabled in UI demo mode.\r\n")
    await ws.send_text("Use Log tab or run full BFF for PTY.\r\n")
    await ws.close()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8790, reload=False)


if __name__ == "__main__":
    main()
