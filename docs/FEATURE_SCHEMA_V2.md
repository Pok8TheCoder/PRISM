# PRISM Feature Schema — v1 (ZMT.01) vs v2 (YMT.01)

Model naming: **Z**/**Y** = model generation, **M** = model, **T** = transformer.

- **ZMT.01** — temporal transformer on the v1 schema, 27 features per flow.
- **YMT.01** — same transformer on the v2 schema, 64 features per window state.

Both consume a length-10 sequence. In v1 each timestep is one raw flow. In v2
each timestep is an aggregate over the trailing window of up to 8 flows ending
at that flow, so the two models see the **same ten flows** and differ only in
how those flows are described.

> A third schema, **AMT.01** (82 columns, ported from the `origin/aryan`
> `StateBuilder`), was later benchmarked on the same traffic and wins on
> accuracy, macro F1 and false-alarm rate, while v2 keeps the lead on
> lab-capture accuracy and the scan/flood family. See
> [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) §3 and §5 — that document, not
> this one, is the source of truth for which schema to deploy.

---

## 1. Before / after at a glance

| | v1 — ZMT.01 | v2 — YMT.01 |
|---|---|---|
| Columns per timestep | 27 | 64 |
| Unit of a timestep | one 5-tuple flow | trailing window of ≤ 8 flows |
| Fan-out / dispersion signal | none | 18 columns (blocks A, B, C) |
| Heavy-tail handling | raw values | signed-log before aggregation |
| Columns unavailable from CIC CSV | 7 of 27 (26%) | 5 of 64 (8%) |
| Identifier columns (IP, hostname, timestamp) | none | none |
| Label-derived columns | none | none |

The v1 vector answers *"what did this one connection look like?"*. The v2 vector
answers *"what is this connection doing relative to the traffic around it?"* —
which is the question that separates a port scan from a flood from a discovery
sweep, since all three can contain individually unremarkable flows.

---

## 2. Design rules

Three constraints shaped the list, and each one rules out features that a
larger schema would have included.

**Every column must be computable from both sources.** The CIC-IDS-2018 CSVs
have no `Src IP` / `Dst IP` column. Any IP-based fan-out feature would
therefore be zero on every CSV row and non-zero on every PCAP row, which lets
the model recover *which corpus a sample came from* rather than what the
traffic is doing — and because the two corpora have different label
distributions, that shortcut scores well while learning nothing. Destination-IP
entropy and unique-source-IP counts are excluded for exactly this reason,
despite being the most obvious window features to add.

**No identifiers and no label-derived columns.** No IPs, hostnames, usernames,
absolute timestamps, `attack_cat`, or `label`. A model given hostnames
memorises which machine was compromised in one capture; a model given a label
column reports a meaningless number.

**Compress before aggregating.** Rates such as `Flow Byts/s`, `Flow IAT` and
`Flow Duration` span six orders of magnitude. Averaging them raw lets one
10 Gb/s flood flow dominate a window mean, so those quantities go through
signed-log \(\operatorname{sgn}(x)\log(1+|x|)\) first.

---

## 3. The v1 schema (27) — carried forward as ZMT.01

Defined in `src/pipeline/features.py`.

**Flow-level (20)** — `Dst Port`, `Protocol`, `Flow Duration`, `Tot Fwd Pkts`,
`Tot Bwd Pkts`, `Fwd Pkt Len Max/Min/Mean`, `Bwd Pkt Len Max/Min/Std`,
`Flow Byts/s`, `Flow Pkts/s`, `Flow IAT Mean/Std`, and the `SYN`/`FIN`/`RST`/
`PSH`/`ACK Flag Cnt` counters.

**Packet-level (7)** — `TTL Mean`, `TTL Std`, `TCP Win Mean`, `IP Frag Cnt`,
`Payload Len Mean`, `Payload Len Std`, `Retrans Cnt`. All seven are zero for
every CIC CSV row.

### Two v1 defects found while building v2

1. **Flow IAT unit mismatch.** `pcap_to_rows` writes `Flow IAT Mean` in
   *seconds* while the CIC CSVs supply *microseconds*, so v1 mixes units by a
   factor of \(10^6\) across its two data sources on two of its 27 columns.
2. **26% of the vector is structurally absent for CSV rows.** The seven
   packet-level columns are zero-filled with no indicator, so the model cannot
   distinguish "TTL was 0" from "TTL was not observable".

Defect 1 is **fixed in the ZMT.01 baseline used for the comparison** — the
benchmark hands that correction to the baseline rather than crediting it to v2.

---

## 4. The v2 schema (64)

Defined in `src/pipeline/features_v2.py`, computed by
`src/pipeline/extract_v2.py`. Aggregation at timestep \(t\) reads flows
\(t-7 \ldots t\) only — never forward — so a live sensor can compute it with no
lookahead.

### Block A — window composition (8)

The block that did not exist in any form in v1.

| Feature | Meaning |
|---|---|
| `win_num_flows` | flows in the trailing window |
| `win_uniq_dst_ports` | distinct destination ports |
| `win_dst_port_entropy` | Shannon entropy over destination ports |
| `win_port_fanout_ratio` | distinct ports ÷ flows |
| `win_port_seq_ratio` | fraction of adjacent sorted ports differing by exactly 1 |
| `win_port_range_log` | log span between highest and lowest port |
| `win_proto_entropy` | Shannon entropy over protocol numbers |
| `win_flow_rate_log` | flows per second across the window |

`win_port_seq_ratio` separates a sequential scan from a randomised one, and
`win_port_fanout_ratio` separates any scan from a flood: a scan touches many
ports once each, a flood touches one port many times. Both are constant-valued
for a single flow and therefore inexpressible in v1.

### Block B — protocol mix (4)

`win_proto_frac_tcp`, `win_proto_frac_udp`, `win_proto_frac_icmp`,
`win_proto_frac_other`.

### Block C — destination port class mix (6)

`win_port_frac_wellknown` (<1024), `win_port_frac_registered`,
`win_port_frac_ephemeral`, `win_port_frac_web` (80/443/8080/8443),
`win_port_frac_remoteadmin` (22/23/3389/5900/5985),
`win_port_frac_db` (1433/3306/5432/1521/27017/6379).

Six semantic buckets replace the twenty one-hot port indicators that a
per-dataset schema typically carries. `Dst Port` is already present as a raw
column and `win_dst_port_entropy` is strictly more informative than a set of
binary flags, so twenty extra dimensions to re-encode one column is a poor
trade.

### Block D — dispersion, mean + std of 13 quantities (26)

`duration_log`, `fwd_pkts`, `bwd_pkts`, `fwd_bwd_ratio`, `byts_s_log`,
`pkts_s_log`, `iat_mean_log`, `iat_std_log`, `fwd_len_mean`, `bwd_len_max`,
`bytes_per_pkt`, `payload_mean`, `asymmetry`.

The **std** half is the point. A beacon has near-zero variance in duration and
byte count across the window; interactive traffic does not. v1 could represent
one flow's mean but never the spread across neighbouring flows.

### Block E — TCP flag profile (8)

`win_syn_frac`, `win_ack_frac`, `win_fin_frac`, `win_rst_frac`,
`win_psh_frac`, `win_syn_ack_ratio`, `win_rst_rate`, `win_flagless_frac`.

Normalised by packet count, so these are rates rather than v1's raw counters.
`win_syn_ack_ratio` is the classic half-open-probe signature.

### Block F — packet-level anomaly (8)

`win_ttl_mean`, `win_ttl_std`, `win_ttl_spread`, `win_tcpwin_mean_log`,
`win_tcpwin_std_log`, `win_retrans_rate`, `win_frag_rate`,
`win_payload_zero_frac`.

`win_ttl_spread` (max − min across the window) flags mixed-origin traffic that
per-flow TTL cannot. `win_payload_zero_frac` is a strong scan signature: probes
complete handshakes and carry nothing.

### Block G — current-flow passthrough (4)

`cur_dst_port_log`, `cur_protocol`, `cur_duration_log`, `cur_payload_mean`.
Retains per-flow resolution so aggregation cannot blur away the identity of the
flow actually being scored.

---

## 5. Source coverage

The v2 base record has 29 columns; both schemas are projected from it, so the
two models are guaranteed identical flow records in identical order.

| Base column group | PCAP | CIC CSV |
|---|---|---|
| Ports, protocol, duration, packet/byte counts, rates, IAT | yes | yes |
| TCP flag counters | yes | yes |
| Packet/payload size statistics | yes | yes (`Pkt Size Avg`, `Pkt Len Std`) |
| TCP window | yes | yes (`Init Fwd Win Byts`) |
| TTL mean / std | yes | **no** |
| Retransmissions | yes | **no** |
| IP fragmentation | yes | **no** |

Recovering TCP window from `Init Fwd Win Byts` and payload size from
`Pkt Size Avg` shrinks the CSV-unavailable share from **7 of 27 columns (26%)**
in v1 to **5 of 64 (8%)** in v2, which is the single largest reduction in
train/serve skew between the two schemas.

---

## 6. What was deliberately left out

| Not included | Reason |
|---|---|
| Source / destination IP entropy, unique-IP counts | absent from CIC CSVs; would encode corpus identity |
| 20 one-hot `port_<n>` indicators | 23 dimensions to re-encode a column that is already present, and weaker than port entropy |
| Host telemetry (`root_shell`, `num_shells`, `logged_in`, `su_attempted`) | not observable from network capture; guarantees train/serve skew in a PCAP-fed sensor |
| `attack_cat`, `label` | ground-truth leakage |
| Absolute timestamps, usernames, hostnames | memorise the capture instead of the behaviour |
| Constant-valued flow columns synthesised from auth logs | zero variance carries no information and gives the scaler a zero-variance column |

---

## 7. Files

| Path | Role |
|---|---|
| `src/pipeline/features.py` | v1 schema (27) |
| `src/pipeline/features_v2.py` | v2 schema (64) + 29-column base record |
| `src/pipeline/extract_v2.py` | PCAP/CSV readers, window aggregation, `base_to_v1` |
| `scripts/build_comparison_dataset.py` | shared capture-split dataset |
| `scripts/train_zmt_ymt.py` | trains both models identically |
| `scripts/ablate_ymt_blocks.py` | leave-one-block-out ablation |
| `scripts/analyze_zmt_ymt.py` | per-class and false-positive breakdown |
