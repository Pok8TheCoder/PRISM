# Backend integration map

For agents wiring UI to real lab infrastructure vs staying on stubs.

---

## Three backend modes

| Mode | Command | BFF | Scorer | Docker |
|------|---------|-----|--------|--------|
| **UI demo** | `python run_lab_console_ui.py` | `stub_main.py` | Fake | No |
| **Mock only** | `npm run dev:mock` | None | None | No |
| **Full stack** | `python run_lab_console.py --api` | `main.py` | Real | Optional |

---

## FastAPI BFF structure

```
apps/lab-console/api/
  main.py              Real BFF entry
  stub_main.py         UI/UX demo entry
  stub_data.py         Rich fake payloads
  services/
    dashboard_service.py   GET /dashboard — containers, metrics
    session_service.py     GET /sessions/.../state — charts
    scripts_service.py     POST /scripts/run — subprocess launcher
    attack_scheduler.py    Delayed kill-chain jobs
    lab_config_service.py  Horizon + attack delay
    models_service.py      Model registry
    terminal_service.py    WebSocket PTY
```

### Key files on disk (real mode)

| Path | Reader | Content |
|------|--------|---------|
| `data/lab_events/demo_forecast.json` | `session_service` | Live chart state |
| `data/lab_events/demo_forecast.jsonl` | `dashboard_service` | Metrics history |
| `data/lab_events/demo_forecast.pid` | `scripts_service`, `session_service` | Scorer PID |
| `data/lab_events/lab_console_config.json` | `lab_config_service`, `demo_forecast.py` | Horizon, delays |
| `reports/lab/hx/recordings/*.json` | `main.py` (partial) | Recording manifests |
| `reports/lab/hx/shap/*` | *not wired to BFF yet* | SHAP reports |

---

## Script launcher → subprocess

`POST /api/scripts/run` maps `script_id` → command:

| script_id | Real command | Stub behavior |
|-----------|--------------|---------------|
| `scorer-start` | `demo_forecast.py --no-up` | Log "already live" |
| `scorer-stop` | `taskkill` PID | Log no-op |
| `killchain-*` | `attack_scheduler` → `demo_killchain.py` | Schedule GT band + log |
| `lab-up` | `lab_ctl.py up` | Log no-op |
| `auto-attack` | `demo_forecast.py --auto-attack` | Launch detached |
| `forecast-record` | `demo_forecast_record.py` | Log queued |

---

## Session state builder (real)

`session_service._live_from_forecast()`:

1. Read `demo_forecast.json`.
2. Map `models.shaun_v3` / `models.hx_c` timelines → `SeriesPoint[]`.
3. Convert `phase_regions` → `groundTruthRegions`.
4. Merge `attack_scheduler.ui_attack_regions()` for manual kill-chain.
5. Expose `playheadSec`, `elapsedSec`, `windowSec`.

---

## Python modules (do not call from frontend)

| Module | Purpose |
|--------|---------|
| `scripts/demo_forecast.py` | Scorer + legacy HTML dashboard |
| `scripts/demo_killchain.py` | Harborline attack phases |
| `scripts/lab_ctl.py` | Docker compose up/down |
| `src/explain/forecast_block_explain.py` | SHAP on IPS block |
| `src/ui/forecast_sessions.py` | Offline recording builder (Streamlit) |
| `src/adversarial/lab_manager.py` | Container status |

---

## Adding a new API endpoint (checklist)

1. Implement service function in `api/services/`.
2. Add route in `main.py` **and** `stub_main.py`.
3. Add stub payload in `stub_data.py`.
4. Add `fetchX()` in `frontend/src/api/client.ts`.
5. Add hook in `useLabApi.ts` or feature hook.
6. Update `docs/API.md`, `contracts/openapi.yaml`.
7. Add fixture JSON if complex.

---

## WebSocket terminal

Real: `terminal_service.py` — `pywinpty` on Windows, subprocess on Unix.

Stub: sends message that shell is disabled; Log tab still works.

---

## Environment variables (frontend)

| Var | Effect |
|-----|--------|
| `VITE_FORCE_MOCK` | Skip all fetch; use `src/mocks/` |
| `VITE_API_BASE_URL` | REST prefix (default `/api`) |
| `VITE_PROXY_TARGET` | Vite proxy → BFF (default `:8790`) |

---

## Smoke test

```bash
python run_lab_console.py --api   # or stub
python scripts/lab_console_smoke_test.py
```

18 checks against real BFF. Stub passes health + dashboard + session + scripts.
