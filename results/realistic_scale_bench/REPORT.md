# Realistic Scale Benchmark: Does RAMX matter at CIC-like density?

Shaun V2 was trained on **millions of total flows** across CIC/IoT captures,
but each **15s state window** aggregates to **~450 flows** (not millions per window).
This bench sweeps synthetic scale tiers on lab PCAPs plus a **native CIC holdout** replay.

## Scale tiers (lab PCAPs)

| Tier | Scale | Replicate | Note |
|---|---|---|---|
| `quiet_lab` | 1× | 1× | Docker lab default (~20–100 flows/window) |
| `cic_like` | 10× | 5× | CIC training density (~500 flows/window) |
| `enterprise` | 50× | 10× | High-volume enterprise (~5k flows/window) |
| `extreme` | 100× | 20× | Stress tier (~10k+ flows/window) |

## Lab PCAP aggregates (attack captures only)

### quiet_lab

| Model | Mean F1 | Det rate | Attack max P | Warmup P |
|---|---|---|---|---|
| ARY-5s base | 0.000 | 0.0% | 0.008 | 0.008 |
| ARY-5s + RAMX | 0.878 | 95.0% | 0.545 | 0.008 |
| Shaun base | 0.096 | 10.0% | 0.196 | 0.156 |
| Shaun + RAMX | 1.000 | 100.0% | 0.678 | 0.156 |

**Shaun RAMX delta:** F1 +0.904, det +90.0%, attack max P +0.483

### cic_like

| Model | Mean F1 | Det rate | Attack max P | Warmup P |
|---|---|---|---|---|
| ARY-5s base | 0.000 | 0.0% | 0.008 | 0.008 |
| ARY-5s + RAMX | 0.878 | 95.0% | 0.545 | 0.008 |
| Shaun base | 0.600 | 60.0% | 0.558 | 0.156 |
| Shaun + RAMX | 1.000 | 100.0% | 0.823 | 0.156 |

**Shaun RAMX delta:** F1 +0.400, det +40.0%, attack max P +0.265

### enterprise

| Model | Mean F1 | Det rate | Attack max P | Warmup P |
|---|---|---|---|---|
| ARY-5s base | 0.000 | 0.0% | 0.008 | 0.008 |
| ARY-5s + RAMX | 0.932 | 100.0% | 0.573 | 0.008 |
| Shaun base | 0.600 | 60.0% | 0.607 | 0.156 |
| Shaun + RAMX | 1.000 | 100.0% | 0.843 | 0.156 |

**Shaun RAMX delta:** F1 +0.400, det +40.0%, attack max P +0.236

## Native CIC holdout (no PCAP scaling)

Attack windows evaluated: **200** | Median flows/window: **450**

| Model | Mean F1 | Det rate | Attack max P | Warmup P |
|---|---|---|---|---|
| Shaun base | 1.000 | 100.0% | 0.910 | 0.156 |
| Shaun + RAMX | 1.000 | 100.0% | 0.964 | 0.156 |

**Shaun RAMX delta on CIC native:** F1 +0.000, det +0.0%, attack max P +0.054

## Interpretation

- If your friend means **training-corpus scale**, the CIC native row is the honest test.
- If they mean **millions of flows per 15s window**, that is **not** what CIC windows contain;
  the `extreme` tier stress-tests beyond CIC density using replicated lab PCAPs.
- **RAMX delta → 0** would support "base is enough at realistic scale"; a large delta means
  adaptation still helps (warmup calibration / relative anomaly), even when absolute P is high.