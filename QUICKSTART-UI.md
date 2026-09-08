# Lab Console UI — install & run

Branch: **`UI/UX`**

## Requirements

- **Node.js 20+** and npm
- **Python 3.11+** (for stub API — recommended)

## Install & run (recommended)

```bash
git checkout UI/UX

# One-time: frontend dependencies
cd apps/lab-console/frontend
npm install
cd ../../..

# Stub backend + UI on LAN
python run_lab_console_ui.py
```

Open: **http://localhost:5173/session**

Try **Live** mode and the kill-chain script buttons.

| Service | Port |
|---------|------|
| UI (Vite) | 5173 |
| Stub API | 8790 |

## UI only (no Python)

```bash
cd apps/lab-console/frontend
npm install
npm run dev:mock
```

Open: **http://localhost:5173/session** — uses in-app mocks (badge shows **mock**).

## More docs

- [apps/lab-console/HANDOFF.md](apps/lab-console/HANDOFF.md) — UI developer onboarding
- [apps/lab-console/docs/UI-BRANCH.md](apps/lab-console/docs/UI-BRANCH.md) — branch guide & feature matrix
