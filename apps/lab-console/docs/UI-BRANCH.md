# UI/UX branch guide

Branch: **`UI/UX`** on `origin` (github.com/Pok8TheCoder/PRISM)

**Quick start:** [QUICKSTART-UI.md](../../../QUICKSTART-UI.md) at repo root.

## Purpose

Isolate **visual design and UX iteration** from backend/scorer work. Models and Docker are **not required**.

## What's different on this branch

| Item | UI/UX branch | `main` |
|------|--------------|--------|
| `run_lab_console_ui.py` | ✅ One-click demo host | May not exist |
| `api/stub_main.py` | ✅ Fake BFF | — |
| `api/stub_data.py` | ✅ Animated live session | — |
| `docs/*` | ✅ Agent handoff docs | Partial |
| Real scorer integration | Same as main when using `main.py` | ✅ |

## Run locally

See [QUICKSTART-UI.md](../../../QUICKSTART-UI.md) for the full install steps. Short version:

```bash
git checkout UI/UX
cd apps/lab-console/frontend && npm install && cd ../../..
python run_lab_console_ui.py
```

UI only (no Python):

```bash
cd apps/lab-console/frontend
npm install
npm run dev:mock
```

### URLs

- UI: http://localhost:5173/session
- LAN: printed on startup (`--host`)
- Stub API: http://127.0.0.1:8790/api/health → `"stub": true`

## What works in demo mode

| Feature | Works? |
|---------|--------|
| Dashboard KPIs + chart | ✅ Rich fake data |
| Lab Session recorded | ✅ Static timeline |
| Lab Session live | ✅ Playhead advances in real time |
| IDS/IPS toggle | ✅ Updates policy in stub |
| Kill-chain scripts | ✅ GT bands + log lines |
| Log autoscroll | ✅ |
| Shell terminal | ❌ Message only |
| Explain tab | ❌ Placeholder — see FUTURE-TABS.md |
| Recordings tab | ❌ Placeholder — stub lists 2 fake items via API only if you add UI |

## What to improve (intended agent tasks)

1. **Visual polish** — spacing, typography, chart aesthetics, dark/mono themes.
2. **Build Explain tab** — use `reports/lab/hx/shap/*.json` as fixtures.
3. **Build Recordings tab** — video player + manifest card.
4. **Placeholder → designed empty states** for `/explain`, `/recordings`.
5. **Motion** — playhead, region fades, page transitions.
6. **Responsive** — minimum 1280px desktop first; tablet optional.

## Commit scope

Keep UI/UX commits under:

- `apps/lab-console/frontend/`
- `apps/lab-console/api/stub_*.py`
- `apps/lab-console/docs/`
- `run_lab_console_ui.py`

Avoid mixing ML changes (`src/hx/`, `scripts/demo_forecast.py`) into UI/UX PRs.

## Merging back to main

When UX stabilizes:

1. Merge `frontend/` changes to `main`.
2. Keep `stub_*` on branch or behind `--stub` flag.
3. Wire new tabs to real BFF endpoints as backend lands.

## Related docs

- [AGENT-HANDOFF.md](./AGENT-HANDOFF.md) — master index
- [PRODUCT-SPEC.md](./PRODUCT-SPEC.md) — screen-by-screen spec
- [FUTURE-TABS.md](./FUTURE-TABS.md) — Explain & Recordings deep spec
