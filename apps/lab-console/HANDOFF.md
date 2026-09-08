# Handoff — Lab Console UI

**For AI agents:** start at [docs/AGENT-HANDOFF.md](./docs/AGENT-HANDOFF.md) (full read order + specs).

**For humans:** [QUICKSTART-UI.md](../../QUICKSTART-UI.md) at repo root (install + run).

## UI/UX demo (this branch)

```bash
cd apps/lab-console/frontend && npm install && cd ../../..
python run_lab_console_ui.py
```

Stub backend + Vite on LAN. No Docker or models. Open http://localhost:5173/session

## UI-only (mocks, no Python)

```bash
cd apps/lab-console/frontend
npm install
npm run dev:mock
```

## Full stack (lab team)

```bash
python run_lab_console.py --api
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/README.md](./docs/README.md) | **Index of all docs** |
| [docs/AGENT-HANDOFF.md](./docs/AGENT-HANDOFF.md) | Master agent onboarding |
| [docs/PRODUCT-SPEC.md](./docs/PRODUCT-SPEC.md) | Every tab + user journeys |
| [docs/FUTURE-TABS.md](./docs/FUTURE-TABS.md) | Explain, Recordings, MITRE, Missed specs |
| [docs/DOMAIN-CONCEPTS.md](./docs/DOMAIN-CONCEPTS.md) | Charts, IDS/IPS, scorer glossary |
| [frontend/README.md](./frontend/README.md) | npm install, env vars |

## Boundaries

| Layer | Location |
|-------|----------|
| UI | `frontend/src/` |
| Stub API | `api/stub_*.py` |
| Real API | `api/main.py` |
| Contracts | `contracts/` |

## Adding a tab

1. `features/<name>/<Name>Page.tsx`
2. `app/routes.tsx` + `app/Sidebar.tsx`
3. Types + mocks + stub endpoint
4. Document in `docs/PRODUCT-SPEC.md`
