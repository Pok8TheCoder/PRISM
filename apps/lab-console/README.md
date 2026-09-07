# PRISM Lab Console

New lab UI replacing the Streamlit dashboard (`run_dashboard.py`). UI-first phase uses in-app mocks; phase 2 connects to Python lab code via FastAPI BFF.

## Quick start

```bash
# From repo root
python run_lab_console.py

# With API stubs (proxied at /api in dev)
python run_lab_console.py --api
```

Frontend only: `cd apps/lab-console/frontend && npm install && npm run dev`

## Structure

```
apps/lab-console/
  frontend/     React + Vite + TypeScript + Tailwind
  api/          FastAPI BFF (port 8790)
  mocks/        Shared JSON contracts for API + docs
```

## Tabs

| Route | Tab |
|-------|-----|
| `/` | Dashboard — containers, training, datasets, device, networks, models |
| `/session` | Lab Session — multi-model charts, IDS/IPS, playback, terminal rail |
| `/adversarial` | Model registry (agent-oriented stub) |
| `/explain` | SHAP on blocks (placeholder) |
| `/recordings` | WebM manifest library (placeholder) |

## Themes

Dark and light via top-nav toggle (`data-theme` CSS variables).

## Lab Session chart semantics

- Green line: model predicted path
- Black/white solid: actual (theme-dependent)
- Dotted vertical: playhead (rAF-smoothed)
- Yellow: model suspicious · Red: model attack
- Cyan/Navy: resolved wrong (after playhead passes)
- Purple: ground truth · Pink: overlap

## API (phase 2)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/dashboard` | Ops overview |
| `GET /api/sessions/:id/state` | Chart + model data |
| `PATCH /api/sessions/:id/policy` | IDS / IPS toggle |
| `GET /api/logs` | Terminal log stream |
| `POST /api/scripts/run` | Killchain, lab_ctl, bench scripts |
| `WS /api/terminal` | xterm PTY |

Run API: `python apps/lab-console/api/main.py`

## Migration from Streamlit

See [MIGRATION.md](./MIGRATION.md). `run_dashboard.py` remains for now but Lab Console is the target primary UI.

## Ports

- Frontend dev: **5173**
- FastAPI BFF: **8790**
- Legacy forecast demo: **8788** (unchanged)
