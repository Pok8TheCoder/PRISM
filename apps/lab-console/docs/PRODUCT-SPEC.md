# Lab Console — product specification

Per-screen behavior: **current implementation** vs **intended end state**.

---

## Global shell

### Sidebar navigation

| Route | Label | Status |
|-------|-------|--------|
| `/` | Dashboard | Implemented |
| `/session` | Lab Session | Implemented (primary screen) |
| `/adversarial` | Adversarial Lab | Partial (registry browser) |
| `/explain` | Explain | Placeholder |
| `/recordings` | Recordings | Placeholder |

Placeholder routes show `PlaceholderPage` with title + one-line description.

### Theme

- **Dark / light** toggle (sidebar footer).
- **Mono / Aurora** style presets (`useTheme` → `data-theme` + `themes.css`).
- Design tokens: `--accent`, `--region-*`, `--chart-*`, glass panels.

### API source badge

- **API** (green): BFF responded to `/api/health`.
- **mock** (muted): Using `src/mocks/*` or `VITE_FORCE_MOCK=true`.

---

## Dashboard (`/`)

### Purpose

SOC-style **ops overview** when the lab is running: containers, GPU, datasets, loaded models, and a coarse attack-probability sparkline.

### Current UI

- KPI tiles: containers running, CUDA device, aggregate P(attack), models loaded.
- Area chart: `metricsTimeseries` (flow rate + P(attack) over last ~60 windows).
- Cards: container list, training status, datasets, protected networks, model summaries.

### Intended behavior

| Element | Spec |
|---------|------|
| Containers | Live from `lab_status()` / Docker; green=running, red=stopped |
| Training | Active when HX-C lab-adapt or catalog retrain running; show progress % |
| Datasets | Scan `data/` paths; show sample counts and sizes |
| Networks | Lab CIDRs with protected flag (IPS applies to protected nets) |
| Metrics chart | Prefer live tail from `demo_forecast.jsonl` when scorer running |
| Click-through | Container row → logs; model row → Adversarial detail |

### Data source

- `GET /api/dashboard` → `DashboardState` (`types/dashboard.ts`)
- Stub: `api/stub_data.py::dashboard()`

---

## Lab Session (`/session`)

### Purpose

**Primary forecast player** — multi-model time-series charts, IDS/IPS policy, script launcher, log/shell output. Replaces Streamlit "Forecast Player" + "Live Monitor" + "Attack Runbook".

### Toolbar

| Control | Behavior |
|---------|----------|
| **Live / Recorded** | `mode=live\|recorded` on session API poll |
| **IDS / IPS** | `PATCH /api/sessions/:id/policy` — IPS enables auto-block when models exceed threshold |
| **Model chips** | Toggle which `ModelChartRow` charts are visible |
| **Record** | Runs `forecast-record` script (WebM capture — backend) |

### Charts area

- Up to **2 model rows visible**; scroll for more (`max-h-[34rem]`).
- Each row: model name, active region badge, MAE/P/R badges, `TimeSeriesChart`.
- **Legend** above charts explains line and region colors.

### Chart semantics (must preserve)

| Visual | Meaning |
|--------|---------|
| Solid filled line | Actual / retrospective P(attack) |
| Dashed / glow line | Model forecast (prospective) |
| Center vertical | **Now** playhead — data scrolls underneath in live mode |
| Yellow band | Model: suspicious |
| Red band | Model: attack |
| Purple band | Ground truth (kill-chain phase or scheduled manual attack) |
| Pink band | Overlap (model region ∩ ground truth) |
| White / episodic band | RAMX episodic memory window (HX-C) |
| Cyan/navy resolved | Past region judged wrong after playhead passes |

### Playback

| Mode | Playhead |
|------|----------|
| Recorded | `usePlaybackClock` — scrub full timeline, speed control |
| Live | `useLivePlayback` — smooth extrapolation between scorer polls; never runs ahead of data |

### Script rail (right)

- **Layout toggles:** charts-only | split | terminal-focus (legacy; dock is now page-bottom).
- **Lab config:** horizon (seconds), attack delay (seconds) → `PATCH /api/lab-config`.
- **Script launcher:** kill-chain phases, scorer start/stop, lab up/down, bench, record.

### Output dock (page bottom)

- **Log tab:** polled `/api/logs` with autoscroll.
- **Shell tab:** WebSocket PTY (`/api/terminal`) — disabled in stub mode.

### Live session data flow (real BFF)

1. `scripts/demo_forecast.py` scores 1s windows → writes `data/lab_events/demo_forecast.json`.
2. BFF `session_service.build_session(mode=live)` reads JSON, maps windows → chart series.
3. UI polls every **250ms** in live mode.

### Recorded session (intended)

Load manifest + timeline from `reports/lab/hx/recordings/*.json` and replay without scorer. **Not wired yet** — currently uses static mock timeline.

---

## Adversarial Lab (`/adversarial`)

### Purpose

**Model registry** for agents and humans: checkpoint paths, dimensions, class lists, tags. Future home for **missed-sample retrain** and **catalog attack** workflows.

### Current UI

- Card grid of models from `GET /api/models`.
- Side drawer: parameters, classes, tags, checkpoint path.

### Intended end state

| Feature | Spec |
|---------|------|
| Registry | All checkpoints under `models/`, `Automode/train/`, `PRISM-shaun/weights/` |
| Exists badge | Green if file on disk; red if missing |
| Actions | "Retrain on missed", "Run catalog bench", "Adversarial loop 20min" (sidebar in Streamlit) |
| Results | Link to `reports/lab/*` markdown reports |
| Agent API | Structured JSON for autonomous agents picking checkpoints |

---

## Explain (`/explain`) — NOT BUILT

See [FUTURE-TABS.md](./FUTURE-TABS.md#explain-tab) for full spec.

**One-line:** SHAP / Integrated Gradients waterfall at IPS block or playhead — why the model fired.

---

## Recordings (`/recordings`) — NOT BUILT

See [FUTURE-TABS.md](./FUTURE-TABS.md#recordings-tab) for full spec.

**One-line:** Library of WebM screen captures + JSON manifests from forecast demo runs.

---

## User journeys

### Journey A — Watch live forecast

1. Operator starts lab (`lab up` or script).
2. Clicks **Start scorer** in Lab Session.
3. Switches to **Live** — charts tick forward, playhead moves smoothly.
4. Clicks **Kill-chain: recon** — purple GT band appears after delay.
5. Watches HX-C / Shaun regions turn yellow → red as attack probability rises.
6. In **IPS** mode, sees armed badge and block streak counters.

### Journey B — Review a recorded take

1. Operator opens **Recordings**.
2. Selects `forecast_dual_v2.2_take03_*.webm`.
3. Manifest shows blocker, block window, link to SHAP JSON.
4. Opens in player OR loads timeline into Lab Session **Recorded** mode.

### Journey C — Explain an IPS block

1. During or after live run, IPS blocks at window N.
2. Operator opens **Explain** (or inline drawer on Lab Session).
3. Sees SHAP waterfall: top flow features, class probabilities, predicted vs ground-truth contrast.
4. Exports markdown report (already generated on disk by backend).

---

## Non-goals (UI/UX branch)

- Real model inference in browser.
- Real Docker control in stub mode.
- Replacing Suricata or Harborline apps.
