# UI architecture

Conventions for the Lab Console React app (`frontend/`).

## Layers

```
features/     Pages & screen-specific logic
components/   Reusable UI (no direct API calls)
hooks/        Stateful data + playback (calls api/)
api/          HTTP + env config only
types/        Shared TypeScript contracts
mocks/        Offline fixtures
styles/       Global CSS variables
app/          Router, layout shell, sidebar
```

**Dependency rule:** `features` → `components`, `hooks`, `api`, `types`.  
`components` must not import from `features`.

## Key screens

### Lab Session (`features/lab-session/LabSessionPage.tsx`)

- Toolbar: Live/Recorded, IDS/IPS, model chips, record
- Charts: `ModelChartRow` → `TimeSeriesChart` (canvas)
- Playback: `usePlaybackClock` (recorded) / `useLivePlayback` (live)
- Scripts rail: `TerminalRail` → `ScriptLauncher`
- Output dock: `TerminalDock` → `LogStream` + `LabTerminal`

### Dashboard (`features/dashboard/DashboardPage.tsx`)

KPI tiles, area chart, container/model cards.

### Adversarial (`features/adversarial/AdversarialPage.tsx`)

Model registry table from `/api/models`.

## Data hooks (`hooks/useLabApi.ts`)

| Hook | Purpose |
|------|---------|
| `useDashboardData` | Dashboard tab |
| `useSessionData(id, mode)` | Lab session state |
| `useLogStream` | Terminal log lines |
| `useScripts` | Script catalog + `launch()` |
| `useLabConfig` | Horizon + attack delay |
| `useModelRegistry` | Adversarial tab |
| `useApiAvailable` | Health check |

All hooks fall back to mocks on error unless `FORCE_MOCK`.

## Theming

- `hooks/useTheme.ts` — `data-theme` on `<html>`
- `styles/themes.css` — CSS variables (`--accent`, `--chart-*`, `--region-*`)

## Charts

`TimeSeriesChart` is canvas-based. Props:

- `actual`, `predicted` — `{ t, y }[]`
- `modelRegions`, `groundTruthRegions`, `memoryRegions`
- `playheadSec` — smooth in live mode via `useLivePlayback`
- `windowSec` — visible time span (default 24s centered on playhead)

## Terminal

- **Log tab:** polled `/api/logs`
- **Shell tab:** WebSocket `/api/terminal` + xterm.js (`LabTerminal.tsx`)

## Adding API integration

1. Add types in `types/`
2. Add `fetchX()` in `api/client.ts`
3. Add hook in `useLabApi.ts` (or dedicated hook file)
4. Update `docs/API.md` + `contracts/openapi.yaml`
5. Add mock in `mocks/` for offline dev

## Build & deploy

```bash
npm run build   # → dist/
npm run preview # static preview
```

Production: serve `dist/` and reverse-proxy `/api` to BFF, or set `VITE_API_BASE_URL` at build time.
