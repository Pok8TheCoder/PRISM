# Agent handoff — Lab Console (UI/UX branch)

**Audience:** Another AI agent or human developer taking over UI/UX work on branch `UI/UX`.

**Read order:**

1. [PROJECT-OVERVIEW.md](./PROJECT-OVERVIEW.md) — What PRISM is and where Lab Console fits
2. [PRODUCT-SPEC.md](./PRODUCT-SPEC.md) — Every screen, behavior, and user journey
3. [DOMAIN-CONCEPTS.md](./DOMAIN-CONCEPTS.md) — Forecast charts, IDS/IPS, models, MITRE, regions
4. [FUTURE-TABS.md](./FUTURE-TABS.md) — **Explain**, **Recordings**, MITRE, Missed (not built yet — full spec)
5. [UI-ARCHITECTURE.md](./UI-ARCHITECTURE.md) — Code layout and conventions
6. [API.md](./API.md) — REST contract
7. [BACKEND-INTEGRATION.md](./BACKEND-INTEGRATION.md) — Real BFF vs stub; Python integration map
8. [UI-BRANCH.md](./UI-BRANCH.md) — How to run demo mode on this branch

**Quick start (no backend):**

```bash
python run_lab_console_ui.py
# → http://localhost:5173/session
```

**Repo paths:**

| Path | Role |
|------|------|
| `apps/lab-console/frontend/` | React UI — **your primary workspace** |
| `apps/lab-console/api/stub_*.py` | Fake API for design review (UI/UX branch) |
| `apps/lab-console/api/` | Real FastAPI BFF (needs full PRISM + Docker for live scoring) |
| `apps/lab-console/contracts/` | OpenAPI + JSON fixtures |
| `src/ui/app.py` | Legacy Streamlit dashboard (reference for missing features) |
| `scripts/demo_forecast.py` | Live scorer + legacy HTML dashboard (:8788) |
| `scripts/demo_forecast_record.py` | WebM recorder → `reports/lab/hx/recordings/` |

**What you should NOT change** (unless explicitly asked):

- `src/shaun/`, `src/hx/` — model inference code
- `scripts/demo_forecast.py` scorer loop (lab team owns this)
- Docker compose under `docker/`

**What you SHOULD change:**

- Layout, typography, motion, chart polish in `frontend/`
- Placeholder pages → real UX for `/explain`, `/recordings`
- Mock data richness in `frontend/src/mocks/` and `api/stub_data.py`
- Component decomposition in `components/` and `features/`

**Status badge:** Top-right **API** = BFF reachable; **mock** = in-app fixtures only.

**Open backlog:** [MISSING.md](../MISSING.md)
