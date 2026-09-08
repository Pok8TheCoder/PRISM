# Future tabs — detailed specifications

Tabs that exist in navigation but are **not implemented** (or only placeholders). Use this doc to build them without guessing intent.

---

## Explain tab (`/explain`)

### Problem statement

When IPS blocks or the operator pauses at a suspicious playhead, they need to answer: **"Why did the model think this was an attack?"** — with feature-level evidence suitable for a SOC analyst or judge demo.

### Legacy reference

| Source | What it did |
|--------|-------------|
| Streamlit PS Demo tab | Upload PCAP/CSV → inference → attention/gradient/SHAP expander |
| Streamlit Live Monitor | Z-score proxy attribution bar chart on latest flow window |
| `src/explain/forecast_block_explain.py` | **Real** IG + SHAP on IPS block; writes markdown + JSON |
| `reports/lab/hx/shap/*.md` | Example output artifacts |

### Intended UX (React)

```
┌─────────────────────────────────────────────────────────────┐
│ Explain — IPS block @ window 38 · HX-C · P(attack)=0.59     │
├──────────────────────┬──────────────────────────────────────┤
│ Context card         │  SHAP waterfall (horizontal bars)    │
│ · Blocker model      │  Top 12 flow features                │
│ · Block time         │  Base → cumulative f(x)              │
│ · Predicted class    │                                      │
│ · Ground truth phase │  Class probability table             │
│ · Link to session    │  Predicted vs GT contrast table      │
├──────────────────────┴──────────────────────────────────────┤
│ Feature group breakdown (flow_mins, flow_means, …)          │
│ [ Export MD ] [ Open in Lab Session @ playhead ]            │
└─────────────────────────────────────────────────────────────┘
```

### Entry points

1. **Global nav** — browse latest block reports from disk.
2. **Lab Session** — "Explain this moment" when `ips.blocker` set or region active.
3. **Recordings** — open SHAP from manifest `block_explanation`.

### API (to implement)

```
GET /api/explain/blocks?limit=20
→ [{ window, blocker, pAttack, recordedAt, summaryPath, jsonPath }]

GET /api/explain/blocks/{id}
→ { markdown, json, window, sessionSnapshot? }

POST /api/explain/at-playhead
Body: { sessionId, playheadSec, modelId? }
→ runs or fetches cached attribution for that window
```

**Phase 1 (UI-only):** Load static JSON from `reports/lab/hx/shap/forecast_shap_w00035_*.json` in stub BFF.

**Phase 2:** BFF calls `explain_ips_block()` or reads precomputed files keyed by `demo_forecast.json` `block_window`.

### Data shape (from existing JSON reports)

```json
{
  "blocker": "hx_c",
  "method": "integrated_gradients",
  "p_attack": 0.589,
  "predicted_class": "Lateral Movement",
  "ground_truth": "T1190_web_exploit_probe",
  "waterfall": [{ "feature": "flow_41_min", "shap": -0.007, "value": 1.0 }],
  "class_probs": [{ "label": "T1021_remote_services", "p": 0.571 }],
  "contrast": [{ "feature": "...", "predicted_shap": 0.01, "gt_shap": -0.02 }]
}
```

### Visual design notes

- Use same glass panel / badge language as Lab Session.
- Waterfall: diverging bars (red = pushes toward attack, blue = away).
- Mono theme must remain readable — no color-only encoding.
- Show **method** badge (IG vs Kernel SHAP).

### Non-goals

- Running PyTorch in the browser.
- Explaining every window — focus on **blocks** and **user-selected playhead**.

---

## Recordings tab (`/recordings`)

### Problem statement

Demo recordings (WebM + manifest) are how PRISM is shown to judges and stakeholders. Operators need a **library** to browse, preview, and reload a take into the Forecast Player.

### Legacy reference

| Source | What it did |
|--------|-------------|
| `scripts/demo_forecast_record.py` | Playwright records `:8788` dashboard during auto-attack |
| Output dir | `reports/lab/hx/recordings/` |
| Filename pattern | `forecast_{layout}_{version}_take{N}_{timestamp}.webm` |
| Sidecar | Same basename `.json` manifest |

### Intended UX

```
┌─────────────────────────────────────────────────────────────┐
│ Recordings library                    [ Record new ▾ ]      │
├──────────────┬──────────────────────────────────────────────┤
│ Filter       │  ┌─────────────────────────────────────────┐ │
│ · layout     │  │         HTML5 video player              │ │
│ · version    │  │         (WebM)                          │ │
│ · blocked?   │  └─────────────────────────────────────────┘ │
│ · date       │  Manifest summary card                       │
│              │  [ Load in Lab Session ] [ Open SHAP ]       │
│  take list   │                                              │
│  (scroll)    │  Timeline scrubber (optional phase 2)        │
└──────────────┴──────────────────────────────────────────────┘
```

### List item fields

| Field | Source |
|-------|--------|
| Title | `forecast_dual_v2.2_take03` or user alias |
| Thumbnail | First frame or static placeholder |
| Duration | From video metadata or manifest estimate |
| Blocker | `manifest.blocker` |
| Block @ | `manifest.block_at_sec` |
| Attack complete | `manifest.attack_phase === "complete"` |

### API (to implement)

```
GET /api/recordings
→ { recordings: [{ id, basename, webmPath, manifestPath, ...manifest fields }] }

GET /api/recordings/{id}/manifest
→ full JSON

GET /api/recordings/{id}/video
→ stream WebM (or static file URL in dev)

POST /api/recordings/record
→ queues demo_forecast_record.py (already stubbed in script list)
```

**Stub BFF** already returns 2 fake recordings in `stub_main.py` — expand with realistic manifests.

### "Load in Lab Session"

1. Parse manifest + optional full timeline export (future: `.npz` or session JSON).
2. Navigate to `/session?recording={id}`.
3. Switch to **Recorded** mode with `playheadSec` at `block_at_sec` or 0.

**Current gap:** `session_service` does not load from `reports/lab/hx/recordings/` yet.

### Recording workflow (operator)

1. Lab Session → **Record** button (or script).
2. Backend starts scorer with `--auto-attack` if needed.
3. Playwright opens `?view=dual` (charts + terminal).
4. Waits for `attack_phase=complete` or IPS block + SHAP.
5. Saves WebM + manifest; appends to library.

---

## MITRE Catalog tab (future `/mitre`)

### Purpose

Browse **222 MITRE ATT&CK techniques** with PRISM-specific metadata: network-detectable vs host-only, bench pass/fail, link to lab PCAPs.

### Legacy

Streamlit **MITRE Catalog** tab + `scripts/bench_mitre222_ary5_ramx.py` results.

### Intended UX

- Searchable table: technique ID, name, tactic, network-visible flag.
- Status column: ✅ detected in lab / ❌ missed / ⚠️ host-only.
- Drill-down: description, example bot class, link to run bench for one technique.

### API sketch

```
GET /api/mitre/techniques?network_only=true
GET /api/mitre/bench-status  → merges bench JSON from results/
```

Not in sidebar yet — add route when building.

---

## Missed & Retrain tab (future `/missed`)

### Purpose

**Adversarial loop** triage: attacks that evaded or were misclassified get saved to `data/raw/adversarial/missed/`. Analyst confirms, dismisses, or triggers retrain.

### Legacy

Streamlit tab + sidebar buttons:

- "Retrain on missed samples"
- "Train on ALL catalog attacks"
- "Adversarial loop 20 min"

Suspect review box in Live Monitor:

- **Confirm alert** → SOC timeline
- **Dismiss** → false positive
- **Save to missed** → training folder
- **Retrain now** → GPU job

### Intended UX

```
┌─────────────────────────────────────────────────────────────┐
│ Missed samples (37)              [ Retrain all ] [ Loop ]   │
├─────────────────────────────────────────────────────────────┤
│ Queue table: PCAP, predicted class, true class, evasion,    │
│              timestamp, [Confirm] [Dismiss] [Retrain]       │
├─────────────────────────────────────────────────────────────┤
│ Training job status / log stream                            │
└─────────────────────────────────────────────────────────────┘
```

Could live under **Adversarial Lab** as a second panel instead of separate route — product decision pending.

### API sketch

```
GET /api/missed
PATCH /api/missed/{id}  { action: "confirm"|"dismiss" }
POST /api/train/retrain-missed
POST /api/train/catalog
POST /api/train/adversarial-loop
```

---

## PS Demo / Analyze (future `/analyze`)

Not in sidebar. Streamlit **PS Demo** tab:

- Upload `.pcap` or `.csv`
- Run world model inference
- Show P(attack), MITRE tactic, SHAP expander

Could be:

- Dashboard drawer, or
- Lab Session upload button, or
- Dedicated route.

Spec deferred until Explain tab ships (shared SHAP components).

---

## Implementation priority (suggested)

| Priority | Tab | Rationale |
|----------|-----|-----------|
| P0 | Recordings | Unblocks demo replay; manifest API half exists |
| P1 | Explain | Judge-facing differentiator; artifacts already on disk |
| P2 | Missed | Training loop; smaller audience |
| P3 | MITRE | Reference data; large table UX |
| P4 | Analyze | Overlaps Explain + Session upload |

---

## Stub data guidance

When building UI before backend:

1. Add fixtures under `contracts/fixtures/explain_block.json`, `recordings_list.json`.
2. Extend `api/stub_data.py` with matching endpoints.
3. Keep TypeScript types in `src/types/explain.ts`, `src/types/recordings.ts` (create when starting).
