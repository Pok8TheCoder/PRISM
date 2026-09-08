# PRISM Lab Console — UI package

React + Vite + TypeScript + Tailwind. **Runs standalone** with in-app mocks; optionally connects to the FastAPI BFF.

## Requirements

| Tool | Version |
|------|---------|
| Node.js | **20+** |
| npm | 10+ |

No Python or Docker needed for UI-only work.

## Install & run (mock mode)

```bash
cd apps/lab-console/frontend
npm install
cp .env.example .env
npm run dev:mock
```

Open **http://localhost:5173**. The header badge shows **mock** — all data comes from `src/mocks/`.

## Install & run (with backend)

**Terminal 1 — BFF** (from PRISM repo root):

```bash
pip install -r apps/lab-console/requirements-bff.txt
python apps/lab-console/api/main.py
```

**Terminal 2 — UI:**

```bash
cd apps/lab-console/frontend
npm install
npm run dev
```

Vite proxies `/api` → `http://127.0.0.1:8790`. Badge shows **API** when healthy.

Or use the all-in-one launcher from repo root:

```bash
python run_lab_console.py --api
```

## Environment variables

Copy `.env.example` → `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `VITE_FORCE_MOCK` | `false` | Skip network; always use mocks |
| `VITE_API_BASE_URL` | `/api` | REST prefix (use full URL if no proxy) |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8790` | BFF target for Vite proxy |
| `VITE_DEV_PORT` | `5173` | Dev server port |

`npm run dev:mock` loads `.env.mock` (`VITE_FORCE_MOCK=true`).

## Project layout

```
src/
  api/           HTTP client + env config (only place that talks to BFF)
  app/           Shell: routes, sidebar, layout
  components/    Shared UI (charts, terminal, dashboard widgets)
  features/      Page-level screens (dashboard, lab-session, adversarial)
  hooks/         Data hooks (useLabApi, useLivePlayback, useTheme)
  mocks/         Fallback fixtures when API offline or FORCE_MOCK
  styles/        Theme tokens (themes.css)
  types/         TypeScript contracts (mirror API + contracts/)
```

**Rule of thumb:** pages in `features/`, reusable pieces in `components/`, no `fetch` outside `api/`.

## Routes

| Path | Screen |
|------|--------|
| `/` | Dashboard |
| `/session` | Lab Session (charts, IDS/IPS, scripts, output dock) |
| `/adversarial` | Model registry |
| `/explain` | Placeholder |
| `/recordings` | Placeholder |

## API contract

Types live in `src/types/`. JSON fixtures and OpenAPI spec:

- `../contracts/fixtures/` — sample responses
- `../docs/API.md` — endpoint reference
- `../contracts/openapi.yaml` — machine-readable schema

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Dev server + API proxy |
| `npm run dev:mock` | Dev server, mocks only |
| `npm run build` | Production build → `dist/` |
| `npm run preview` | Serve `dist/` |
| `npm run typecheck` | `tsc` without emit |
| `npm run lint` | Oxlint |

## Handoff bundle

Minimum files for a UI-only developer:

```
frontend/          (this package)
docs/API.md
docs/UI-ARCHITECTURE.md
contracts/
```

See `../HANDOFF.md` in the parent folder.

## Tech stack

- React 19, React Router 7
- Vite 8, TypeScript 6
- Tailwind CSS 4 (`@tailwindcss/vite`)
- xterm.js (shell tab)
- lucide-react icons

## Chart semantics (Lab Session)

Documented in parent `README.md` — green = predicted, solid = actual, purple = ground truth, yellow/red = model regions.
