# ARY.01 (242-d) vs YMT.01 (64-d) — Feature Comparison

Quick reference for the ablation sweep in `scripts/ablate_ary242_blocks.py`.

## ARY.01 — what it was trained on

**242 dimensions** from CIC-IDS-2018, 30-second wall-clock windows (`StateBuilder`):

| Block | Dims | FEATURES.md §7 equivalent |
|-------|------|---------------------------|
| A_meta | 5 | num_flows, uniq src/dst IPs, uniq dst ports, port_entropy |
| B_mean | 109 | mean of every numeric flow column (log-scaled in preprocessor) |
| B_std | 109 | std of every numeric flow column |
| C_flags | 6 | SYN/ACK/FIN/RST/PSH/URG fractions |
| D_proto | 3 | TCP / UDP / ICMP fractions |
| E_top_ports | 10 | hit counts for ports 21,22,23,25,53,80,110,135,139,143 |

**Dead on CIC CSV (always zero):** dims 1–2 (src/dst IP counts), URG flag fraction.

Data: `data/aryan_splits/{train,val,test}.npz` · Lookback L=20 · Heads: dynamics + binary + 7 MITRE stages.

## YMT.01 — your original Y model

**64 dimensions** from **8-flow trailing windows** (`src/pipeline/features_v2.py`):

| Block | Dims | Role |
|-------|------|------|
| A_composition | 8 | flow count, port entropy, fan-out, scan ratios |
| B_protocol | 4 | TCP/UDP/ICMP/other mix |
| C_portclass | 6 | well-known / web / admin / db port fractions |
| D_dispersion | 26 | log-compressed mean+std of 13 dispersion bases |
| E_flags | 8 | window flag ratios + syn_ack + rst_rate |
| F_packet | 8 | TTL, TCP window, retrans, frag, payload |
| G_current | 4 | current flow passthrough |

Key difference: YMT log-compresses heavy tails **before** aggregation; ARY aggregates raw/log-scaled flow stats into 30s windows. YMT has **no** bulk 109×2 flow-column mirror.

## AMT / extract_aryan (82-d)

PRISM's named reimplementation of Aryan's documented schema on 8-flow windows — **not** the same tensor as the 242-d checkpoint (different windowing + padding).

## Ablation variants

See `VARIANTS` in `scripts/ablate_ary242_blocks.py`. Early signal (4-variant smoke, ~2 min):

- **`ymt_like_lean` (14 dims)** — meta + flags + proto only → best composite so far
- **`core_std_ACD` (123 dims)** — meta + std + flags + proto (drops means, top ports)
- **`full_242`** — baseline

Run locally: `.\scripts\run_ary242_ablation_local.ps1`  
Run on Colab: `notebooks/ARY242_Feature_Ablation_Colab.ipynb` (split PART 0/1 across two runtimes)

Results: `results/ary242_ablation/ablation.json`
