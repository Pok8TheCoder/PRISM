# Lab Console BFF (Backend-for-Frontend)

FastAPI service that bridges the React UI to PRISM lab scripts, scorer state, and Docker helpers.

## Install

```bash
pip install -r apps/lab-console/requirements-bff.txt
```

From repo root (so Python can import `services.*` and PRISM `scripts/`):

```bash
python apps/lab-console/api/main.py
```

Listens on **http://127.0.0.1:8790**.

## Requirements

- Python **3.11+**
- Windows: `pywinpty` for interactive shell WebSocket (optional on Linux — uses subprocess PTY)

## Not required for UI mock mode

Frontend developers can skip this entirely when using `npm run dev:mock`.

## Endpoints

See `../docs/API.md` and `../contracts/openapi.yaml`.

## Structure

```
api/
  main.py              FastAPI app + routes
  requirements.txt     Same pins as ../requirements-bff.txt
  services/
    dashboard_service.py
    session_service.py   Live charts ← demo_forecast.json
    scripts_service.py   Script launcher + scorer subprocess
    attack_scheduler.py  Delayed kill-chain
    terminal_service.py  WebSocket PTY
    lab_config_service.py
    models_service.py
```

## Run with UI

```bash
# Repo root
python run_lab_console.py --api
```

Starts BFF + Vite dev server.

## Tests

```bash
python scripts/lab_console_smoke_test.py   # requires BFF running
```
