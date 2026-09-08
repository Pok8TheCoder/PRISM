# PRISM & Lab Console — project overview

## What is PRISM?

**PRISM** (Predictive Risk Intelligence for Security Monitoring) is a research prototype for **proactive** network defense. Instead of only classifying past traffic (traditional IDS), PRISM trains **world models** that:

1. Ingest flow/packet features from live or recorded traffic (292-dim Shaun states, 128-dim HX states).
2. Predict **P(attack)** and future trajectory **K steps ahead** (forecast cone).
3. Map behavior to **MITRE ATT&CK** tactics/techniques.
4. Optionally **block** attackers in IPS mode when confidence crosses thresholds.

The flagship demo is **Harborline Internal** — a fake corporate intranet in Docker where a kill-chain (recon → enum → spray → loot) runs against a vulnerable ticketing app while Suricata and PRISM models score traffic.

## Repository layout (high level)

```
PRISM/
├── apps/lab-console/     ← NEW React UI + FastAPI BFF (replaces Streamlit)
├── src/                  ← ML: shaun, hx, explain, adversarial, pipeline
├── scripts/              ← Lab control, demos, benchmarks, training helpers
├── docker/               ← Harborline lab compose stacks
├── data/                 ← MITRE JSON, processed datasets, lab_events/
├── reports/lab/          ← Experiment writeups, SHAP reports, recordings
└── run_lab_console.py    ← Launch full UI + real BFF
    run_lab_console_ui.py ← Launch UI + stub BFF (UI/UX branch)
```

## What is Lab Console?

Lab Console is the **operator-facing web app** for running and observing PRISM lab experiments. It consolidates what used to be scattered across:

| Legacy (Streamlit `src/ui/app.py`) | Lab Console route |
|-----------------------------------|-------------------|
| Live Monitor | `/` Dashboard + `/session` Live |
| Forecast Player | `/session` Recorded |
| Attack Runbook | `/session` script rail |
| PS Demo (upload PCAP) | *Not ported* — future `/analyze` or session drawer |
| Missed & Retrain | *Not ported* — future `/missed` or Adversarial drawer |
| MITRE Catalog | *Not ported* — future `/mitre` |

## Runtime services (when full stack is up)

| Port | Service | Purpose |
|------|---------|---------|
| **5173** | Vite (React) | Lab Console UI |
| **8790** | FastAPI BFF | API for UI; proxies lab scripts & reads scorer state |
| **8788** | `demo_forecast.py` HTTP | Legacy forecast HTML dashboard (being replaced by Lab Console) |
| **8080** | Harborline site | Target web app in Docker |

## Models in the lab demo

| ID | Name | Role in UI |
|----|------|------------|
| `shaun_v3` | Shaun v3 | Primary forecast track; RAMX episodic memory; IPS candidate |
| `hx_c` | HX-C | Causal 33-class head; often first to forecast enum/spray; SHAP on block |
| `ary_5s` | ARY 5s + RAMX | Multiclass baseline; fair IDS bench comparisons |
| `xmt_01` | XMT.01 | Experimental world model; registry only |

Charts in Lab Session show **P(attack)** over time — not raw feature dimensions (yet).

## UI/UX branch purpose

Branch **`UI/UX`** isolates **visual and interaction design** from backend reliability:

- `api/stub_main.py` returns rich fake data — no Docker, GPU, or weights.
- `run_lab_console_ui.py` hosts Vite on LAN for design review.
- All product intent is documented in `apps/lab-console/docs/` for the next agent.

## Success criteria (product vision)

An operator should be able to:

1. See lab health (containers, GPU, datasets) on Dashboard.
2. Start/watch a **live forecast session** with synchronized charts, playhead, and IPS state.
3. Fire kill-chain scripts and see **ground-truth bands** vs model predictions.
4. Review **why** IPS blocked (Explain / SHAP) at the playhead moment.
5. Browse **recorded WebM sessions** with manifests tying block events to SHAP JSON.
6. Manage model registry and adversarial retrain loop (Adversarial / Missed).

Steps 4–6 are **specified but not fully built** in the React UI yet.
