# Lab Console — missing features (vs old Streamlit UI)

Tracked gaps between the **current Lab Console** and the legacy `run_dashboard.py` / `src/ui/app.py` dashboard.  
**Priority 1 (in progress):** wire existing tabs to the FastAPI BFF so Dashboard, Lab Session, and Adversarial Lab use real backend data.

## Priority 1 — Backend wiring (current sprint)

- [x] `GET /api/dashboard` — containers, device, datasets, models, metrics
- [x] `GET /api/sessions/:id/state` — chart payload (live from `demo_forecast.json` when scorer running, else recorded mock)
- [x] `PATCH /api/sessions/:id/policy` — IDS / IPS mode
- [x] `GET /api/logs` + `POST /api/scripts/run` — terminal log stream & script launcher
- [x] `GET /api/models` — model registry for Adversarial Lab
- [x] Frontend hooks + API badge (`API` vs `mock` fallback)
- [ ] WebSocket `/api/terminal` — real xterm PTY (still stub)
- [ ] Live session: start/stop scorer from UI without manual `demo_forecast.py`
- [ ] Recorded mode: load timeline from `reports/lab/hx/recordings/*.json` manifests

## Priority 2 — Tabs not built yet

| Old Streamlit tab | Lab Console route | Status |
|-------------------|-------------------|--------|
| MITRE Catalog | `/mitre` (new) | Not started |
| Missed & Retrain | `/missed` (new) | Not started |
| Explain / SHAP | `/explain` | Placeholder only |
| Recordings library | `/recordings` | Placeholder only |
| PS Demo (upload PCAP/CSV) | `/analyze` or Dashboard drawer | Not started |

## Priority 3 — Features inside existing tabs

- [ ] **Suspect triage** — Confirm / Dismiss / Save-to-missed workflow (old Live Monitor)
- [ ] **MITRE tactic bar chart** — aggregated tactic scores from softmax
- [ ] **Live packet wire** — per-packet color stream (intrusion / evasion / misclass)
- [ ] **Flow feature table** — raw model input inspector
- [ ] **Forensic JSON export** — timeline + alerts bundle
- [ ] **Attack Runbook wizard** — 5-step guided flow (scripts exist as flat grid)
- [ ] **Configurable model slots** — pick checkpoint per chart row (old Forecast Player dropdowns)
- [ ] **Multi-feature charts** — plot raw dims beyond P(attack)
- [ ] **SHAP at playhead** — Integrated Gradients / Kernel SHAP panel on IPS block

## How to run with backend

```bash
# From repo root — starts Vite + FastAPI BFF on :8790
python run_lab_console.py --api
```

Without `--api`, the UI falls back to in-app mocks (badge shows **mock**).

## API endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | BFF liveness |
| `GET /api/dashboard` | Dashboard tab |
| `GET /api/sessions/:id/state?mode=live\|recorded` | Lab Session charts |
| `PATCH /api/sessions/:id/policy` | IDS / IPS |
| `GET /api/logs?since=` | Terminal log poll |
| `GET /api/scripts` | Script launcher catalog |
| `POST /api/scripts/run` | Run killchain / lab_ctl / scorer |
| `GET /api/models` | Adversarial Lab registry |
| `GET /api/recordings` | Recording manifests (for future tab) |
| `WS /api/terminal` | PTY (stub) |
