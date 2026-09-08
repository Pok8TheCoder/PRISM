# Handoff — Lab Console UI

Quick guide for handing the **UI package** to another developer.

## What to share

| Path | Role |
|------|------|
| `apps/lab-console/frontend/` | **Standalone npm app** — all UI work happens here |
| `apps/lab-console/docs/API.md` | REST contract the UI expects |
| `apps/lab-console/docs/UI-ARCHITECTURE.md` | Folder conventions & extension points |
| `apps/lab-console/contracts/` | OpenAPI + JSON fixtures |

They do **not** need the full PRISM repo, Docker, or ML code to iterate on layout/visuals.

## 5-minute setup (UI only)

```bash
cd apps/lab-console/frontend
npm install
npm run dev:mock
```

→ http://localhost:5173 with mock data.

## When they need live data

1. Install BFF deps: `pip install -r apps/lab-console/requirements-bff.txt`
2. Run BFF: `python apps/lab-console/api/main.py` (port **8790**)
3. Run UI: `npm run dev` in `frontend/`

## Boundaries

| Layer | Owner | Location |
|-------|-------|----------|
| **UI** | Frontend dev | `frontend/src/` |
| **API client** | Shared contract | `frontend/src/api/` + `contracts/` |
| **BFF** | Backend / lab team | `api/` |
| **Scorer / Docker** | Lab team | `scripts/demo_forecast.py`, `run_lab_console.py` |

UI dev should **not** import from `src/` (Python) or change scorer logic. New screens → add route in `app/routes.tsx`, page in `features/`, types in `types/`, optional mock in `mocks/`.

## Adding a new tab

1. `features/my-tab/MyTabPage.tsx`
2. Route in `app/routes.tsx`
3. Nav item in `app/Sidebar.tsx`
4. Types + mock data if needed
5. `api/client.ts` + `docs/API.md` when backend exists

## Status badge

`ApiSourceBadge` shows **API** vs **mock**. Controlled by `useLabApi` hooks — if BFF is down or `VITE_FORCE_MOCK=true`, mocks are used automatically.

## Questions?

- API shapes → `docs/API.md`, `contracts/openapi.yaml`
- UI structure → `docs/UI-ARCHITECTURE.md`
- Missing features → `MISSING.md`
