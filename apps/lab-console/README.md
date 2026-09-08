# PRISM Lab Console

New lab UI replacing the Streamlit dashboard (`run_dashboard.py`).

## UI/UX demo (stub backend)

Design review — no Docker, models, or scorer:

```bash
python run_lab_console_ui.py
```

Opens Vite on LAN + stub BFF. Try **Live** mode and kill-chain scripts.

## For UI developers

**Start here:** [HANDOFF.md](./HANDOFF.md) → `frontend/README.md`

```bash
cd apps/lab-console/frontend
npm install
npm run dev:mock    # no Python/Docker needed
```

## Full stack (lab team)

```bash
# From repo root — UI + FastAPI BFF
python run_lab_console.py --api
```

| Service | Port |
|---------|------|
| Frontend (Vite) | 5173 |
| FastAPI BFF | 8790 |
| Legacy forecast demo | 8788 |

## Structure

```
apps/lab-console/
  frontend/           React UI (standalone npm package)
  api/                FastAPI BFF
  docs/               API + architecture docs
  contracts/          OpenAPI + JSON fixtures
  mocks/              Legacy shared mocks (also copied to contracts/fixtures)
  requirements-bff.txt
  HANDOFF.md
  MISSING.md
```

## Docs

| Doc | Audience |
|-----|----------|
| [HANDOFF.md](./HANDOFF.md) | UI dev onboarding |
| [frontend/README.md](./frontend/README.md) | Install, scripts, env vars |
| [docs/API.md](./docs/API.md) | REST contract |
| [docs/UI-ARCHITECTURE.md](./docs/UI-ARCHITECTURE.md) | Frontend conventions |
| [api/README.md](./api/README.md) | BFF setup |
| [MISSING.md](./MISSING.md) | Feature backlog |

## BFF install only

```bash
pip install -r apps/lab-console/requirements-bff.txt
python apps/lab-console/api/main.py
```

## Tabs

| Route | Tab |
|-------|-----|
| `/` | Dashboard |
| `/session` | Lab Session |
| `/adversarial` | Model registry |
| `/explain` | Placeholder |
| `/recordings` | Placeholder |

## Migration from Streamlit

See [MIGRATION.md](./MIGRATION.md).
