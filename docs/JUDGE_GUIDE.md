# PRISM — Plain-English Guide for Judges (and Non-Infiltrators)

> **Who this is for:** You understand AI, agents, and systems — but you're not a red-team operator. That's fine. This doc explains the problem statement, what we built, and how to talk about it confidently.

---

## The one-sentence pitch

**PRISM watches network traffic like a security camera that doesn't just say "bad guy" — it learns how attacks *unfold over time* and tries to warn you *before* the damage is done.**

---

## Part 1: What the problem statement (PS) is actually asking

### The old way (what judges are tired of)

Most security ML works like this:

```
One network flow  →  Model  →  "Benign" or "Attack"
```

That's like watching one frame of a movie and deciding the whole plot. You miss the story:

- Port scan *then* brute force *then* data upload
- Slow, sneaky traffic that looks fine in isolation
- The **order** and **progression** of an attack

### The new way (what the PS wants)

The PS asks for a **World Model** approach:

```
Past traffic windows  →  Model learns "what usually happens next"
                      →  Roll forward in time (K steps)
                      →  "This path is heading toward infiltration"
```

**World Model** in simple terms: the AI builds a mental simulation of how the network behaves. Not just "is this packet bad?" but "given where we are now, where are we likely to be in 5 steps?"

### The PS checklist (what they want to see)

| # | They want… | In plain English |
|---|------------|------------------|
| 1 | **Traffic ingestion** | Read real network data (CSV flows or PCAP captures) |
| 2 | **Two feature levels** | Flow stats (volume, ports, flags) *and* packet details (timing, TTL, etc.) |
| 3 | **World model** | Learn how network state *changes over time*, not just label one row |
| 4 | **K-step forecast** | Predict attack probability for the *next several time windows* |
| 5 | **MITRE ATT&CK mapping** | Say *which attack stage* you're in (Recon, Initial Access, C2, Exfil…) |
| 6 | **Explainability** | Show *why* the model flagged something (not a black box) |
| 7 | **Demo UI** | Working app: upload traffic → see timeline + stages |
| 8 | **Beat a baseline** | Prove temporal learning beats dumb static classification |

---

## Part 2: The analogy that always works with judges

Think of a hospital **sepsis early-warning system**:

| Hospital | PRISM |
|----------|-------|
| Vital signs over time (heart rate, temp, BP) | Network flows over time (ports, flags, bytes, timing) |
| "Patient looks fine right now" | "This one connection looks fine" |
| Model sees the *trajectory* worsening | Model sees attack *progression* (scan → login attempts → exfil) |
| Alert *before* organ failure | Alert *before* full compromise |
| Doctor asks "what vitals drove this?" | Analyst asks "what traffic patterns drove this?" |

You're not building a hacker tool. You're building **early warning for defenders**.

---

## Part 3: What PRISM is (our answer to the PS)

**PRISM** = **P**redictive **R**isk **I**ntelligence for **S**ecurity **M**onitoring

It's a research prototype that:

1. **Ingests** network traffic (CIC-IDS-2018 dataset + live PCAP from a Docker lab)
2. **Learns** temporal patterns with a **Temporal Transformer World Model**
3. **Classifies** traffic into **33 types** (Benign + 32 attack behaviors)
4. **Maps** predictions to **MITRE ATT&CK** (222 techniques in the knowledge base)
5. **Forecasts** attack probability **K steps ahead**
6. **Trains itself** when attacks evade detection (adversarial loop)
7. **Shows everything** in a **Streamlit SOC dashboard**

---

## Part 4: How the pieces fit together (the architecture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR LAPTOP (HOST)                              │
│                                                                         │
│   ┌──────────────┐    ┌─────────────────┐    ┌──────────────────────┐    │
│   │  Streamlit   │───▶│  World Model    │───▶│  MITRE Knowledge     │    │
│   │  Dashboard   │    │  (PyTorch GPU)  │    │  Base (222 techniques)│   │
│   └──────┬───────┘    └────────▲────────┘    └──────────────────────┘    │
│          │                     │                                        │
│          │ start/stop          │ PCAP features                          │
│          ▼                     │                                        │
│   ┌──────────────┐    ┌────────┴────────┐                               │
│   │  Training    │    │  Traffic Capture │◀── tcpdump PCAP files       │
│   │  Loop        │    │  (Scapy parser)  │                             │
│   └──────────────┘    └──────────────────┘                               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ docker exec / docker cp
┌──────────────────────────────▼──────────────────────────────────────────┐
│              DOCKER LAB (isolated — cannot touch your real network)        │
│                                                                          │
│   ┌─────────────┐      attacks       ┌─────────────┐                     │
│   │ attacker-bot│ ─────────────────▶ │target-server│                     │
│   │ (32 bots)   │                    │ SSH+Apache  │                     │
│   └─────────────┘                    └──────▲──────┘                     │
│                                             │ normal traffic               │
│                                    ┌────────┴──────┐                     │
│                                    │ benign-client │                     │
│                                    └───────────────┘                     │
│                                                                          │
│   Network: prism-lab (internal only — no internet, no host ports)      │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key idea:** Attacks only happen *inside the plastic sandbox*. Your real machine is never the target.

---

## Part 5: What each component does (no jargon version)

### 5.1 Network traffic → numbers the AI can read

Computers don't send "attacks." They send **packets** and **flows** (conversations between two IPs/ports).

We turn each flow into a **feature vector** — a row of numbers like:

| Feature | What it means |
|---------|---------------|
| Dst Port | Which door they knocked on (22=SSH, 80=web, 443=HTTPS…) |
| SYN Flag Cnt | How many "hello, let's connect" packets |
| Flow IAT Std | How irregular the timing is (bots vs humans) |
| Flow Byts/s | How fast data is moving |

We use **19 flow-level features** today. The PS also wants deeper packet features (TTL, window size) — that's on our gap list.

### 5.2 The World Model (the brain)

**File:** `src/model/world_model_multiclass.py`  
**Architecture:** Temporal Transformer (3 layers, 4 attention heads)

It has **two jobs** (two "heads"):

```
Input: last 10 time windows of traffic features
         │
         ├──▶ State Head:  "What will traffic look like NEXT?"
         │                  (learns P(S_t+1 | S_t) — the world model part)
         │
         └──▶ Class Head:   "What attack type is this?"
                            (Benign, port scan, SSH brute force, HTTP flood…)
```

**Why a Transformer?** Same family as GPT, but here it's reading *sequences of network windows* instead of words. Attention lets it weigh which past moments matter most.

**33 output classes:** Benign + 32 attack types that map to MITRE techniques we can actually see in network traffic.

### 5.3 MITRE ATT&CK (the shared language)

MITRE ATT&CK is like a **periodic table of hacking** — every technique has an ID (T1046, T1110…).

We have:
- **222 techniques** in `data/mitre_attack.json` (reference + dashboard)
- **32 lab bots** that generate traffic for techniques visible on the wire
- **161 "host-only" techniques** (credential dumping, registry edits…) — **cannot** be detected from PCAP alone; we don't pretend we can

When the model says `T1110_ssh_bruteforce`, the dashboard shows: **Credential Access → T1110 Brute Force**.

### 5.4 The 32 attack bots (the red team in a box)

**Folder:** `src/adversarial/bots/`

Each bot is a **small Python script** that generates *lab-safe* traffic patterns:

| Bot | What it simulates |
|-----|-------------------|
| `T1046_service_scan` | Knocking on many ports to see what's open |
| `T1110_ssh_bruteforce` | Repeated SSH login attempts |
| `T1071_http_beacon` | Periodic HTTP check-ins (C2-style) |
| `T1499_http_flood` | Flooding a web server |
| … | 32 total |

These are **not exploits**. They're traffic *shapes* that real attacks produce — safe to run in Docker.

### 5.5 Docker lab (the sandbox)

**Folder:** `docker/`

Three containers on an **internal network** (`prism-lab`):

| Container | Role |
|-----------|------|
| `target-server` | Fake victim (SSH + Apache + tcpdump) |
| `attacker-bot` | Runs our attack scripts |
| `benign-client` | Generates normal background traffic |

**No ports published to your host.** The attacker cannot reach your real network or the internet.

**Start it:** `python scripts/lab_ctl.py up`

### 5.6 Adversarial training loop (self-healing)

**File:** `src/adversarial/training_loop.py`

This is the coolest part for judges:

```
1. Bot attacks target
2. tcpdump captures traffic → PCAP file
3. Model scores the traffic
4. If DETECTED correctly → bot tries evasion (slower timing, scrambled ports…)
5. If EVADED or WRONG → save to "missed" dataset → retrain model
```

The system **learns from its mistakes**. Red team vs blue team, automated.

### 5.7 The dashboard (what you show live)

**File:** `src/ui/app.py`  
**Launch:** `python run_dashboard.py`

Four tabs:

| Tab | What it does |
|-----|--------------|
| **Live Monitor** | Threat level, attack probability chart, K-step forecast, flow table |
| **Attack Runbook** | 5-step wizard: check lab → pick attack → run → review → save/retrain |
| **Missed & Retrain** | Samples the model got wrong; one-click retrain |
| **MITRE Catalog** | Browse all 222 techniques with detectability tags |

**Review-before-alert:** When something looks suspicious, you get a **Suspect card** first — you confirm before it becomes an alert. That's SOC realism.

### 5.8 Baseline (proving we're better than "dumb ML")

**File:** `src/baseline.py`

Logistic Regression on the **same features**, but **one flow at a time** (no temporal context).

| Model | F1 Score | False Positive Rate |
|-------|----------|---------------------|
| Logistic Regression (static) | ~54% | ~41% |
| World Model (temporal) | ~85% | ~17% |

**Talking point:** "+31% F1 by modeling *time*, not just snapshots."

---

## Part 6: A full attack session (step by step)

Here's what happens when you click "Run Attack" in the dashboard:

```
Step 1   You pick: T1046_service_scan, evasion: none
           │
Step 2   target-server starts tcpdump (records all packets)
           │
Step 3   attacker-bot runs: python attack_script.py target-server T1046_service_scan none
           │                 (connects to ports 22, 80, 443, 21… one by one)
           │
Step 4   tcpdump stops → PCAP copied to host → data/raw/adversarial/
           │
Step 5   Scapy reads PCAP → groups packets into flows → 19 features per flow
           │
Step 6   World Model takes last 10 flow windows → outputs:
           │   • Attack type probabilities (33 classes)
           │   • Attack probability timeline
           │   • K-step forecast (rolls state head forward 5 times)
           │
Step 7   Dashboard shows results. If evaded → saved to missed/ for retraining.
```

**You don't need to understand packets.** You need to understand: **traffic in → features → sequence model → prediction + forecast out.**

---

## Part 7: What we achieved vs what the PS still wants

### Done (say these confidently)

- [x] World Model architecture (Transformer, dual head, temporal sequences)
- [x] Learns P(S_{t+1} | S_t) via state prediction head
- [x] K-step forward simulation (dashboard)
- [x] MITRE ATT&CK mapping (222-technique KB + 33-class detection)
- [x] 32 lab attack bots + isolated Docker sandbox
- [x] Adversarial self-fortification loop
- [x] Streamlit SOC dashboard with live lab control
- [x] Logistic regression baseline + benchmark improvement
- [x] Fully offline, open-source stack
- [x] GPU-accelerated inference (PyTorch + CUDA)

### Partial (be honest if asked)

- [~] **Packet-level features** (TTL, window size, fragments) — flow-level done, packet depth not yet
- [~] **SHAP / attention explainability** — dashboard uses z-score proxy; real SHAP module stubbed
- [~] **PCAP/CSV file upload demo** — lab-capture works; drag-and-drop upload tab not primary yet
- [~] **All 32 classes trained equally** — architecture supports 33 classes; need more labeled PCAP per bot
- [~] **Graph Neural Network** — on roadmap; vectors only today

### Not done (don't claim it)

- [ ] 222 separate classifier outputs (physically impossible from NetFlow alone)
- [ ] Host-side attack detection (T1003 credential dump, etc.) without host logs
- [ ] CTU-13 dataset integration
- [ ] Production eBPF/firewall auto-block

---

## Part 8: Judge Q&A cheat sheet

### "What problem are you solving?"

> Static IDS tools label individual flows and miss attack *progression*. PRISM models how network state evolves over time and forecasts whether traffic is heading toward compromise — giving defenders a head start.

### "What's a World Model here?"

> Instead of classifying one snapshot, we learn the transition rule: given the last N traffic windows, what's the distribution over the *next* window? We can roll that forward K steps to simulate "if nothing changes, where does this go?"

### "Why not detect all 222 MITRE techniques?"

> Only ~60 of 222 techniques leave visible network signatures. 161 are host-only (registry, memory, process injection). We detect 32 distinct network patterns, map 32 more via families, and keep all 222 in the knowledge base for analyst context.

### "How do you know it works?"

> Binary PoC benchmark: 85% F1 vs 54% logistic regression on CIC-IDS-2018. Live Docker lab captures real PCAP from 32 attack bots. Adversarial loop measures detection vs evasion rate and retrains on misses.

### "Is this safe to demo?"

> Yes. Attacks run inside an internal Docker network with no published ports. Bots generate traffic *patterns*, not real exploits. The host machine is never targeted.

### "What's novel about your approach?"

> Three things: (1) temporal world model instead of static classification, (2) closed-loop adversarial retraining when bots evade detection, (3) high-resolution MITRE-mapped class catalog tied to executable lab bots — not just a PDF mapping slide.

### "What would you do next?"

> SHAP explainability, full packet-level features, PCAP upload in the UI, batch-generate labeled data for all 32 bots, and hybrid Temporal-GNN for lateral movement across network topology.

---

## Part 9: File map (where everything lives)

```
PRISM/
├── idea.txt                          ← Original problem statement
├── docs/JUDGE_GUIDE.md               ← You are here
├── configs/
│   ├── attack_catalog.yaml           ← 32 bots + MITRE family mappings
│   └── default.yaml                  ← Model hyperparameters
├── data/
│   ├── mitre_attack.json             ← 222 MITRE techniques
│   ├── mitre_network_catalog.json    ← Generated: what's detectable vs host-only
│   └── raw/adversarial/              ← Captured PCAPs + missed samples
├── docker/                           ← Isolated lab (target, attacker, benign)
├── models/checkpoints/
│   └── world_model_multiclass.pth    ← Trained 33-class weights
├── scripts/
│   ├── lab_ctl.py                    ← start/stop/verify lab
│   └── build_mitre_network_catalog.py
├── src/
│   ├── baseline.py                   ← Logistic regression benchmark
│   ├── model/
│   │   ├── world_model_multiclass.py ← THE brain
│   │   └── attack_catalog.py         ← Class/MMITRE loader
│   ├── adversarial/
│   │   ├── bots/                     ← 32 attack scripts
│   │   ├── training_loop.py        ← Adversarial self-healing
│   │   └── traffic_capture.py      ← tcpdump wrapper
│   └── ui/
│       ├── app.py                    ← Streamlit dashboard
│       └── lab_controller.py         ← Dashboard backend
└── run_dashboard.py                  ← Launch the UI
```

---

## Part 10: Demo script for judges (5 minutes)

1. **30 sec — Problem:** "Attacks are stories, not snapshots. We forecast the story."
2. **30 sec — Architecture:** Show the diagram from Part 4. Point at World Model + Docker lab.
3. **1 min — Live lab:** `python scripts/lab_ctl.py up` → show 3 containers running.
4. **2 min — Dashboard:** `python run_dashboard.py` → Attack Runbook → run `T1046_service_scan` → show timeline, MITRE panel, suspect review.
5. **30 sec — Self-healing:** Show Missed tab → explain retrain on evasions.
6. **30 sec — Numbers:** "85% F1 vs 54% baseline. 33 classes. 32 bots. 222-technique MITRE KB."
7. **30 sec — Honest gaps:** "SHAP and packet-level features are next. Host-only techniques need host telemetry."

---

## Glossary (10 terms you'll hear)

| Term | Simple meaning |
|------|----------------|
| **Flow** | A conversation between two IPs/ports (like one phone call) |
| **PCAP** | A recording file of raw network packets |
| **NetFlow** | Summarized flow stats (not every packet, just aggregates) |
| **Feature vector** | A row of numbers describing one flow |
| **Transformer** | AI architecture that reads sequences and learns what to pay attention to |
| **World Model** | AI that predicts "what happens next" in an environment |
| **MITRE ATT&CK** | Industry-standard catalog of attacker techniques (T-codes) |
| **Kill chain** | Stages of an attack: recon → access → move → steal |
| **False positive** | Crying wolf — flagging benign traffic as attack |
| **Evasion** | Attacker changing behavior to avoid detection (slower, randomized) |

---

*You don't need to be a network infiltrator. You built an AI system that watches traffic stories, speaks MITRE, fights back against evasion, and proves temporal learning beats static classifiers. That's the pitch.*
