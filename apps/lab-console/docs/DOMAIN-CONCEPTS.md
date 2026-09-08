# Domain concepts — Lab Console

Glossary for agents implementing UI without reading all of `src/`.

---

## Forecast timeline

### Window

- Default **1 second** of captured traffic (`window_sec` in scorer state).
- Each window: PCAP → feature vector → model `step()` → P(attack) point on chart.
- **Window index** `w` maps to chart time `t = w × window_sec`.

### Playhead

- Vertical **now** line on charts.
- **Recorded:** user-controlled scrubber.
- **Live:** advances smoothly between API updates; capped at latest scored data.

### Retrospective vs prospective

| Series | Chart side | Meaning |
|--------|------------|---------|
| Retrospective | Left of now | Re-analyzed past windows (solid actual line) |
| Prospective | Right of now | K-step rollout forecast (dashed predicted cone) |

`horizonSec` in lab config → number of forecast steps (capped at 15 windows in scorer for CPU).

---

## IDS vs IPS

| Mode | Behavior |
|------|----------|
| **IDS** | Detect and alert only — log lines, chart regions, no packet drop |
| **IPS** | Auto-block when model sees **N consecutive** windows with P(attack) ≥ 0.5 |

IPS nuances in Harborline demo:

- **Recon** is observe-only (port scan should not trigger drop).
- **Enum / spray / loot** phases arm IPS.
- First model to confirm blocks Docker bridge traffic to attacker containers.
- UI shows: `ips.armed`, `ips.blocker`, `ips.streaks`, `blockAtSec`.

---

## Model regions (chart bands)

Generated from P(attack) thresholds on the predicted series:

| Kind | Threshold | Color (theme) |
|------|-----------|---------------|
| `suspicious` | ≥ 0.35 | Yellow |
| `attack` | ≥ 0.5 | Red |

Optional metadata: `resolved`, `correct` — for post-playhead styling (wrong prediction).

---

## Ground truth regions

**Purple bands** — operator-known attack phases (not model output):

- From scorer `phase_regions` when auto-attack runs inside `demo_forecast.py`.
- From **attack scheduler** when user clicks kill-chain scripts in UI (UI-only overlay on stub/real BFF).

Labels: `RECON`, `ENUM`, `SPRAY`, `LOOT` with MITRE class IDs (T1046, T1190, …).

---

## Memory regions (episodic)

**White bands** — RAMX episodic memory active windows on HX-C / Shaun v3. Indicates the model is using retrieved past attack context, not just the current window.

---

## Kill-chain phases

Harborline scripted attack (`scripts/demo_killchain.py`):

| Phase | MITRE-ish | What happens |
|-------|-----------|--------------|
| `recon` | T1046 | TCP port scan from attacker-bot |
| `enum` | T1190 | Directory bust, robots.txt, tickets |
| `spray` | T1110 | Credential spray / guest IDOR |
| `loot` | T1041 | Ops login, payroll download |

Phases can be run individually or as `all`. UI schedules with configurable **attack delay** (30–60s default).

---

## Scorer

Background process: `scripts/demo_forecast.py`

- Writes `data/lab_events/demo_forecast.json` (latest state).
- Writes `data/lab_events/demo_forecast.jsonl` (history).
- PID file: `demo_forecast.pid`.

UI **Start scorer** → BFF launches scorer detached (real) or logs stub message (demo).

States: `warmup` → `live`. Warmup collects benign baseline before IPS arms.

---

## Log line kinds

| `kind` | UI treatment |
|--------|--------------|
| `info` | Neutral monospace |
| `phase` | Phase transitions, script lifecycle |
| `scorer` | Window scoring lines (`w003 hx=0.42 …`) |
| `alert` | FORECAST ALERT, IPS block, attack scheduled |
| `attack` | Kill-chain script stdout |
| `block` | IPS drop events |

---

## Model registry fields

See `types/models.ts`:

- `checkpointPath`, `sizeMb`, `parameters`
- `featureDim`, `seqLen` — input tensor shape
- `classes[]` — MITRE or bot class names
- `tags[]` — e.g. `forecast`, `ips`, `lab-adapt`

---

## SHAP / explainability (backend exists, UI does not)

When IPS blocks, `demo_forecast.py` calls `explain_ips_block()`:

- Integrated Gradients on attack probability w.r.t. flow features.
- Writes `reports/lab/hx/shap/forecast_shap_w{window}_*.md` + `.json`.
- Included in scorer state as `block_explanation` (summary text + paths).

Explain tab should render this — see [FUTURE-TABS.md](./FUTURE-TABS.md).

---

## Recording manifest

Produced by `scripts/demo_forecast_record.py` alongside `.webm`:

```json
{
  "demo_version": "v2.2",
  "layout": "dual",
  "take": 3,
  "recorded_at": "2026-09-06T14:22:00",
  "blocker": "hx_c",
  "block_at_sec": 42.1,
  "block_window": 38,
  "blocked": true,
  "block_explanation": { ... },
  "attack_phase": "complete",
  "p_hx_c": 0.59,
  "p_sn2rx3": 0.21
}
```

Recordings tab lists these; player loads WebM from same directory.

---

## MITRE in PRISM

- `data/mitre_attack.json` — 222 Enterprise techniques.
- `data/mitre_network_catalog.json` — which techniques are visible on **network telemetry only** (vs host-only).
- HX-C outputs 33-class softmax → mapped to tactics for bar charts (Streamlit Live Monitor).

Future MITRE tab: catalog browser + per-technique bench status.

---

## Adversarial / missed samples

When an attack **evades** detection or is **misclassified**, samples go to:

`data/raw/adversarial/missed/`

Retrain scripts consume this folder. Streamlit "Missed & Retrain" tab managed triage (confirm/dismiss/save). Not ported to React yet.
