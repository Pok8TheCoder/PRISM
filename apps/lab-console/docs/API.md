# Lab Console API (BFF)

Base URL: `http://127.0.0.1:8790/api` (dev: proxied via Vite at `/api`).

TypeScript types: `frontend/src/types/`. Fixtures: `contracts/fixtures/`.

## Health

### `GET /health`

```json
{ "ok": true, "apiVersion": "0.3.0", "features": { "terminalPty": true, "liveCharts": true } }
```

## Dashboard

### `GET /dashboard`

→ `DashboardState` (`types/dashboard.ts`)

Fixture: `contracts/fixtures/dashboard.json` (partial; full shape in TS mocks).

## Lab Session

### `GET /sessions/{id}/state?mode=live|recorded`

→ `SessionState` (`types/session.ts`)

| Field | Description |
|-------|-------------|
| `playheadSec` | Scored timeline position (seconds) |
| `elapsedSec` | Wall elapsed since scorer start (live) |
| `windowSec` | Scorer window length (live, default 1) |
| `durationSec` | Timeline extent |
| `models[]` | Per-model predicted series + regions |
| `actual[]` | Ground-truth probability track |
| `groundTruthRegions[]` | Purple attack-phase bands |
| `memoryRegions[]` | RAMX episodic bands (live) |
| `scorerRunning` | Whether `demo_forecast.py` is active |

Fixture: `contracts/fixtures/session_full.json`

### `PATCH /sessions/{id}/policy`

Body: `{ "policyMode": "ids" | "ips" }`

Response: `{ "id": "...", "policyMode": "ids" }`

## Logs & scripts

### `GET /logs?since={ts}`

```json
{ "lines": [{ "ts": 1, "kind": "info|phase|alert|scorer|attack", "text": "..." }] }
```

### `GET /scripts`

```json
{ "scripts": [{ "id": "scorer-start", "label": "...", "description": "..." }] }
```

### `POST /scripts/run`

Body: `{ "script_id": "killchain-recon", "delay_sec": 45 }`

Response: `{ "job_id": "...", "status": "queued|already_running|error" }`

Script IDs: `killchain-recon`, `killchain-enum`, `killchain-spray`, `killchain-loot`, `killchain-all`, `scorer-start`, `scorer-stop`, `auto-attack`, `lab-up`, `lab-down`, `bench-fair-ids`, `forecast-record`.

## Lab config

### `GET /lab-config`

```json
{ "horizonSec": 60, "attackDelaySec": 45, "attackDelayMinSec": 30, "attackDelayMaxSec": 60 }
```

### `PATCH /lab-config`

Body: `{ "horizonSec": 60, "attackDelaySec": 30 }` (partial)

## Models

### `GET /models`

```json
{ "models": [ModelRegistryEntry] }
```

## Recordings (stub)

### `GET /recordings`

```json
{ "recordings": [] }
```

## WebSocket

### `WS /terminal`

Binary/text PTY stream. Resize: `{ "type": "resize", "cols": 120, "rows": 24 }`.

Client helper: `terminalWsUrl()` in `api/client.ts`.

## Polling conventions (UI)

| Hook | Interval | Endpoint |
|------|----------|----------|
| Dashboard | 5s | `/dashboard` |
| Session live | 250ms | `/sessions/default/state?mode=live` |
| Session recorded | 5s | `/sessions/default/state?mode=recorded` |
| Logs | 1.5s | `/logs?since=` |

## Mock fallback

If BFF is unreachable or `VITE_FORCE_MOCK=true`, hooks use `src/mocks/*` and badge shows **mock**.
