#!/usr/bin/env python3
"""FastAPI BFF for PRISM Lab Console.

Run:
  pip install fastapi uvicorn
  python -m apps.lab_console.api.main

Serves on http://127.0.0.1:8790 — proxied by Vite dev server at /api.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
API_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(API_DIR))

from services.dashboard_service import build_dashboard  # noqa: E402
from services.lab_config_service import get_config, patch_config  # noqa: E402
from services.models_service import list_model_registry  # noqa: E402
from services.scripts_service import append_log, get_logs, list_scripts, run_script  # noqa: E402
from services.session_service import build_session, set_policy_mode  # noqa: E402
from services.terminal_service import relay_terminal  # noqa: E402
from services.attack_scheduler import set_log_fn, start_scheduler  # noqa: E402

API_VERSION = "0.3.0"

app = FastAPI(title="PRISM Lab Console API", version=API_VERSION)
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


@app.on_event("startup")
def _startup() -> None:
    set_log_fn(append_log)
    start_scheduler()


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "apiVersion": API_VERSION,
        "features": {"terminalPty": True, "liveCharts": True},
    }


@app.get("/api/dashboard")
def get_dashboard() -> dict:
    return build_dashboard()


@app.get("/api/sessions/{session_id}/state")
def get_session_state(session_id: str, mode: str = "recorded") -> dict:
    return build_session(session_id, mode=mode)


@app.patch("/api/sessions/{session_id}/policy")
def patch_policy(session_id: str, body: PolicyBody) -> dict:
    mode = set_policy_mode(body.policyMode)
    return {"id": session_id, "policyMode": mode}


@app.get("/api/logs")
def api_logs(since: int = 0) -> dict:
    return {"lines": get_logs(since)}


@app.get("/api/scripts")
def api_scripts() -> dict:
    return {"scripts": list_scripts()}


@app.post("/api/scripts/run")
def api_run_script(body: ScriptBody) -> dict:
    return run_script(body.script_id, delay_sec=body.delay_sec)


@app.get("/api/lab-config")
def api_get_lab_config() -> dict:
    return get_config()


@app.patch("/api/lab-config")
def api_patch_lab_config(body: LabConfigBody) -> dict:
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    return patch_config(**updates)


@app.get("/api/scripts/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return {"job_id": job_id, "status": "completed", "output": []}


@app.get("/api/models")
def api_models() -> dict:
    return {"models": list_model_registry()}


@app.get("/api/recordings")
def list_recordings() -> dict:
    import json

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
    await relay_terminal(ws)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8790, reload=False)


if __name__ == "__main__":
    main()
