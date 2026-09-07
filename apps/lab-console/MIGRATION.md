# Streamlit → Lab Console migration

The new **Lab Console** (`run_lab_console.py`) replaces `run_dashboard.py` / `src/ui/app.py`.

## Tab mapping

| Streamlit tab | Lab Console |
|---------------|-------------|
| PS Demo | Lab Session (live mode + scripts) |
| Live Monitor | Dashboard + Lab Session |
| Attack Runbook | Lab Session terminal rail + script launcher |
| Missed & Retrain | Phase 2 — Adversarial Lab or dedicated drawer |
| MITRE Catalog | Phase 2 — placeholder nav exists |
| Forecast Player | Lab Session (charts + playback) |

## Legacy demos

- `scripts/demo_forecast.py` (:8788) — logic to be extracted into `src/lab/forecast_session.py` for BFF
- `scripts/demo_forecast_record.py` — wired via `POST /api/sessions/:id/record` in phase 2

## Retiring Streamlit

When BFF endpoints are fully wired:

1. Default lab entrypoint becomes `python run_lab_console.py --api`
2. `run_dashboard.py` prints deprecation warning and exits unless `--streamlit-legacy`
3. Remove `streamlit` from primary docs (optional dependency)

## Phase 2 checklist

- [ ] Extract scorer from `demo_forecast.py` → `src/lab/forecast_session.py`
- [ ] Wire `GET /api/sessions/:id/state` to live scorer
- [ ] WebSocket stream for live playhead + model ticks
- [ ] PTY bridge for `/api/terminal`
- [ ] Script runner for `demo_killchain.py`, `lab_ctl.py`, bench scripts
- [ ] SHAP panel on IPS block → `/explain` route
- [ ] Recordings scanner → `/recordings` route
