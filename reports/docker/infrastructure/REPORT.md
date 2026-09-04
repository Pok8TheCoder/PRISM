# Docker Adversarial Lab — Infrastructure

**Compose file:** `docker/docker-compose.yml`  
**Control script:** `scripts/lab_ctl.py`  

## Containers

| Service | Role | Host ports |
|---------|------|------------|
| `target-server` | Vulnerable webapp + DB | None (internal) |
| `benign-client` | Background HTTP/SSH/DNS | None |
| `redteam` | Cred-theft / slow-rise agents | None |
| `attacker-bot` | Objective rotation bots | None |

Network: `prism-lab` (internal bridge, no host exposure).

## Event pipeline (IPS)

- Red-team JSONL: `/events/redteam.jsonl` in container
- Host bind mount: `data/lab_events/redteam.jsonl`
- Live tail stream: `docker exec redteam tail -F` (low-latency IPS events)

## Agents

| Script | Purpose |
|--------|---------|
| `redteam_agent.py` | Fast SQLi cred theft (~1s kill chain) |
| `redteam_agent_slow.py` | Slow-rising poison test (noise→recon→probe→strike) |

## Common commands

```powershell
cd D:\Cursor\PRISM
python scripts/lab_ctl.py up
python scripts/lab_ctl.py verify
python scripts/lab_ctl.py reset
python scripts/lab_ctl.py down
```

## Scaled traffic defaults (IPS runs)

- `BENIGN_WORKERS=10`, `BENIGN_INTERVAL=0.5`
- PCAP scale 10×, replicate 5× via `live_ips_multi.py`

## Related reports

- [Theft prevention](../../lab/theft-prevention/REPORT.md)
- [Poison slow-rise](../../lab/poison-slow-rise/SUMMARY.md)
