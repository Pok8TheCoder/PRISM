# Dual IPS live demo (Shaun RAMX v3 vs HX-C)

## What is running

| URL | What |
|-----|------|
| http://127.0.0.1:8787 | Labs live telemetry (both models, first blocker, 2-window streak) |
| http://127.0.0.1:8080 | Vulnerable lab site (same container the bots hit) |

Docker: `target-server` + ~**1000** simulated users (`250` workers × 4 benign containers). Both models score **5s** PCAPs in **IDS+IPS** mode.

Need **two consecutive 5s windows** with fused P≥0.5. First model to hold 2-in-a-row issues the iptables DROP on the **host-browser NAT IP** (`172.18.0.1`) and the **red-team** IP. Benign Docker users stay up. Same-window tie: higher P is named first blocker.

Start (use `--no-up` if the lab containers are already up):

```powershell
cd D:\Cursor\PRISM
.\venv\Scripts\python.exe scripts\demo_dual_ips.py --no-up --warmup-windows 20
```

Only **one** scorer. A second copy will refuse to start (pid lock).

## Video layout

1. Browser tab A: **http://127.0.0.1:8787** (Labs graph).
2. Browser tab B: **http://127.0.0.1:8080** (the site).
3. Wait until Labs says **LIVE** (20 × 5s warmup windows, about 2–4 minutes). Do not attack during warmup.

## Manual attack (tab B)

Quiet clicks look like the 1000 users and usually will **not** IPS-block. You want a **loud burst that lasts at least 10–15 seconds** (two capture windows). Watch the **streak** cards go `1 / 2` then `2 / 2`.

1. On the site, open **search**.
2. Paste this and spam Submit / refresh for 10–20 seconds:

```
' UNION SELECT username, password_hash FROM users--
```

3. If P stays low, from a terminal **during LIVE** (keeps firing for 15s so it spans two windows):

```powershell
.\venv\Scripts\python.exe scripts\demo_attack_burst.py --n 400 --concurrency 40 --seconds 15
```

4. Watch tab A: **HX-C** (blue) vs **Shaun RAMX v3** (gray). Banner shows **who** blocked and **t+seconds**. Streaks show who got to 2/2 first.
5. Tab B should then **hang / fail** (host IP dropped). Labs stays up on :8787.

## After the block

```powershell
docker exec target-server iptables -L INPUT -n
Get-Content D:\Cursor\PRISM\data\lab_events\demo_live.json
```

Ctrl+C the scorer does **not** stop Docker. To reset for a second take (do **not** `iptables -F` from the host):

```powershell
Stop-Process -Name python -ErrorAction SilentlyContinue   # only if this scorer is the python you mean
docker compose -f docker/docker-compose.yml -f docker/docker-compose.demo.yml -p prism up -d --force-recreate target-server
docker restart benign-client benign-users-b benign-users-c benign-users-d
.\venv\Scripts\python.exe scripts\demo_dual_ips.py --no-up --warmup-windows 20
```

Or from Python: `from src.adversarial.ips_controller import clear_blocks; clear_blocks()`.
