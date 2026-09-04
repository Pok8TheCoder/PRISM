# Harborline forecast demo (stepwise steal)

The old Flask SQLi box was one request. This lab is **Harborline Internal**: payroll is behind ops role + a ticket IDOR. HX-C now uses class-shift + anomaly (the binary head was saturating at 1.0) and **rolls the state head 6 windows / ~30s ahead**.

| URL | What |
|-----|------|
| http://127.0.0.1:8788 | Labs **forecast** (HX now + orange cone, Shaun v3 now) |
| http://127.0.0.1:8080 | Harborline site (same container the bots hit) |

Start:

```powershell
cd D:\Cursor\PRISM
.\venv\Scripts\python.exe scripts\demo_forecast.py
```

If the lab is already up: `--no-up`. Wait until Labs says **LIVE** (~20 × 5s windows).

## Video layout

1. Tab A: Labs `:8788` — solid blue = HX now, dashed orange = forecast max, gray = Shaun now.
2. Tab B: site `:8080` — browse like a normal employee (no steal yet).
3. After LIVE, run **recon only**. The cone should move **before** you download payroll.

## Stepwise attack

Do these in order. Search UNION will not dump users (parameterized).

### 1. Recon (this is what HX actually sees)

From a terminal, while Labs is LIVE:

```powershell
.\venv\Scripts\python.exe scripts\demo_killchain.py --phase recon
```

That docker-execs into `attacker-bot` on the lab network and TCP-scans Harborline (SSH, decoy 3306/6379/…). Takes ~15s. Watch the orange cone.

### 2. Enum (browser or helper)

```powershell
.\venv\Scripts\python.exe scripts\demo_killchain.py --phase enum
```

Or in the browser: `/robots.txt`, `/internal/runbook` (401 until login), `/tickets`.

### 3. Guest + ticket IDOR

Register a guest, sign in, then open `/tickets/1`, `/tickets/2`, … until **`/tickets/4412`**. That ticket holds `ops.monitor` / `Harborline!4412`.

Helper: `--phase spray`

### 4. Loot

Log in as `ops.monitor`, open `/files/jobs`, download the artifact. CSV contains `PAYROLL_SECRET=harborline-hl4412-2026`.

Helper: `--phase loot`

Optional DNS burst: `--phase exfil`

Full scripted chain: `--phase all` (still: recon first, loot last).

## What changed in HX

- No post-alert cooldown cap at 0.49 (that made HX lose a 2-window IPS race).
- If the binary head is saturated (~always 1.0), fused P is **catalog shift vs warmup + relative anomaly**, not `1.0 − offset`.
- Anomaly z-threshold is a bit more sensitive.
- `StreamingHXC.rollout(K)` is the forecast cone.

The orange line is **max(world-model rollout, current HX P)** so a live recon does not wait for the state head to invent the next 30s. Alert fires when HX **now ≥ 0.5** and relative anomaly ≥ 0.30 — that is meant to be the scan/enum stage, **before** payroll download.
