# Feature-Schema Head-to-Head — ZMT.01 vs YMT.01 vs AMT.01

Naming: first letter = schema generation · **M** = model · **T** = transformer.

| | ZMT.01 | YMT.01 | AMT.01 |
|---|---|---|---|
| Feature schema | v1 — per-flow | v2 — window state | `origin/aryan` StateBuilder |
| Columns per timestep | 27 | 64 | 82 |
| Origin | shipped PRISM schema | this redesign | ported from Aryan's branch |
| Architecture | transformer, d_model 256, 16 heads, 3 layers | identical | identical |
| Parameters | 2,394,428 | 2,413,409 | 2,422,643 |
| Checkpoint | `models/checkpoints/zmt_01.pth` | `ymt_01.pth` | `amt_01.pth` |

All three are trained on the same captures, the same splits, the same
architecture and the same seeds. The feature schema is the only variable.

Schema definitions and the v1→v2 feature list are in
[`FEATURE_SCHEMA_V2.md`](FEATURE_SCHEMA_V2.md).

---

## 1. Summary

Both window-based schemas beat the shipped per-flow schema by a wide margin,
and **Aryan's StateBuilder schema wins the overall benchmark**:

| | ZMT.01 (27f) | YMT.01 (64f) | AMT.01 (82f) |
|---|---|---|---|
| Accuracy (33-class) | 0.8091 | 0.8493 | **0.8572** |
| Macro F1 | 0.8716 | 0.8818 | **0.8911** |
| Binary F1 | 0.9209 | 0.9172 | **0.9281** |
| False alarms into a real attack class | 60 | 53 | **0** |
| Lab-capture accuracy | 0.9167 | **0.9880** | 0.9658 |
| Scan/flood family F1 | 0.8453 | **0.9839** | 0.9521 |

The two window schemas are **complementary, not ranked**. AMT.01 is better at
deciding *whether* traffic is benign — it produces zero spurious alarms into a
real attack class, against 53 for YMT.01 and 60 for ZMT.01. YMT.01 is better at
deciding *which* attack, winning lab-capture accuracy by 2.2 points and the
scan/flood family by 3.2 points.

The single clearest result is that **windowing is what matters**. Both window
schemas beat per-flow features by 4–5 accuracy points, and an ablation shows
the gain comes from aggregating statistics over neighbouring flows rather than
from any particular clever column.

![Feature schema comparison](../results/zmt_ymt/comparison.png)

---

## 2. Experimental design

| Control | How |
|---|---|
| Identical flow records | All three schemas project from one shared 29-column base record. No model can see a flow another cannot. |
| Identical information budget | Every timestep aggregates the same trailing window of ≤ 8 flows from the same 10-flow block. No lookahead. |
| Identical architecture | Same transformer depth/width/dropout. Only `input_proj` width differs (27/64/82 → 256), a ≤1.2% parameter spread. |
| Identical training | 100 epochs, cosine schedule, AdamW lr 1e-3, batch 256, same class weights, same loss, same grad clipping, seeds {0,1,2}. Best epoch on validation macro-F1. |
| No split leakage | Split by **capture**, not by window. |
| No scaler leakage | `StandardScaler` fit on training flows only. |
| Baseline given the benefit | v1's Flow-IAT unit bug (seconds from PCAP, microseconds from CSV) is fixed for ZMT.01, so the newer schemas are not credited for it. |

**Data** (unchanged sources): 263 labeled lab PCAPs plus
`thursday_01_03_2018.csv` and `wednesday_28_02_2018.csv`. Both CSVs contain only
Benign and Infilteration, so the shipped label mapping (non-benign →
`T1021_remote_services`) is reproduced exactly.

| Split | Windows | Classes |
|---|---|---|
| Train | 7,973 | 33 |
| Validation | 2,697 | 33 |
| Test | 2,728 | 33 |

### What AMT.01 is, precisely

`src/pipeline/extract_aryan.py` ports
`origin/aryan:src/data/state_builder.py::_aggregate_window_vector` onto our base
flow records: `num_flows`, unique source/destination IP counts, unique
destination ports, port entropy, **mean and std of all 29 numeric columns**, six
TCP flag fractions, three protocol fractions, and ten top-attacked-port counts.

Its fallbacks are preserved rather than repaired, so the port is faithful:

- `num_unique_src_ips` and `num_unique_dst_ips` resolve to **0** — neither our
  base records nor the CIC CSVs carry IP columns, which is exactly what his
  `for … else: features.append(0)` branches do. On a corpus with IPs his schema
  would have two more live features than measured here.
- The URG flag fraction resolves to 0 for the same reason.
- No log compression; raw values are aggregated, as in the original.
- Flag fractions divide by flow count, matching `total_flag_packets = max(len(window), 1)`.

Two things are **not** being tested: the 374-feature multi-dataset union
documented in `FEATURES.md` (never extracted from real traffic — see §7), and
his 30-second wall-clock windows. Window geometry is held at our 8-flow trailing
window for all three models so the comparison isolates the feature schema.

---

## 3. Headline results

Mean ± std over 3 seeds, held-out test split. Best per row in bold.

| Metric | ZMT.01 | YMT.01 | AMT.01 |
|---|---|---|---|
| Accuracy (33-class) | 0.8091 ± 0.0023 | 0.8493 ± 0.0050 | **0.8572 ± 0.0045** |
| Macro F1 | 0.8716 ± 0.0088 | 0.8818 ± 0.0053 | **0.8911 ± 0.0037** |
| Weighted F1 | 0.8038 ± 0.0031 | 0.8449 ± 0.0067 | **0.8571 ± 0.0046** |
| Binary F1 | 0.9209 ± 0.0004 | 0.9172 ± 0.0016 | **0.9281 ± 0.0021** |
| Binary recall | **0.9545 ± 0.0026** | 0.9451 ± 0.0043 | 0.9479 ± 0.0020 |
| Binary precision | 0.8896 ± 0.0022 | 0.8910 ± 0.0026 | **0.9091 ± 0.0023** |
| Binary FPR | 0.5437 ± 0.0135 | 0.5307 ± 0.0160 | **0.4351 ± 0.0111** |
| Binary ROC-AUC | 0.8992 ± 0.0104 | 0.8956 ± 0.0089 | **0.9060 ± 0.0131** |
| Train time / seed | 71.5 s | 71.5 s | 74.4 s |

**Lab captures only** (1,752 PCAP windows, no CIC CSV) — the traffic the Docker
lab and dashboard actually produce:

| | ZMT.01 | YMT.01 | AMT.01 |
|---|---|---|---|
| Accuracy | 0.9167 | **0.9880** | 0.9658 |
| Macro F1 | 0.9012 | **0.9209** | 0.9022 |

Accuracy ranges do not overlap across seeds for any pair, so these orderings are
larger than seed noise. Macro F1 moves less than accuracy because it averages
over 33 classes, ten of which hold fewer than 10 test windows; a single flipped
sample in those swings macro F1 by ~0.03.

---

## 4. Scan / flood family

The six classes the 20-minute adversarial loop kept collapsing into one another.

| Class | Support | ZMT.01 | YMT.01 | AMT.01 |
|---|---|---|---|---|
| `T1018_remote_discovery` | 42 | 0.681 | **1.000** | 0.963 |
| `T1046_service_scan` | 43 | 0.688 | **0.913** | 0.804 |
| `T1498_network_dos` | 233 | 0.796 | **1.000** | **1.000** |
| `T1595_active_scan` | 474 | 0.919 | **1.000** | 0.985 |
| `T1499_http_flood` | 204 | 0.998 | **1.000** | **1.000** |
| `T1049_connections_discovery` | 106 | **0.990** | **0.990** | 0.961 |
| **Family mean** | | **0.845** | **0.984** | **0.952** |

Both window schemas essentially solve this family; the per-flow schema does not.
YMT.01 edges AMT.01 here, which is where its log-compressed rate features and
explicit `port_seq_ratio` earn their place.

---

## 5. False positives — the clearest separation

The raw FPR near 0.5 for ZMT.01 and YMT.01 is not 50% of benign traffic raising
alarms. Almost all of it is one class pair, and decomposing it changes the
ranking completely.

| | ZMT.01 | YMT.01 | AMT.01 |
|---|---|---|---|
| Benign test windows | 488 | 488 | 488 |
| Correctly benign | 232 | 218 | **280** |
| → `T1021_remote_services` (CIC Infilteration) | 196 | 217 | 208 |
| → any other attack class | 60 | 53 | **0** |
| FPR, all attacks | 0.525 | 0.553 | **0.426** |
| **FPR excluding Infiltration** | 0.123 | 0.109 | **0.000** |

`T1021_remote_services` is where CIC-IDS-2018 **Infilteration** maps. That label
is close to inseparable from benign at flow level — it is internal host activity
following a malicious download, and its network signature largely *is* normal
traffic. All three models sit near 0.70 F1 on it.

The operationally meaningful number is the last row. **AMT.01 produced zero
spurious alarms into a real attack class across 488 benign windows.** For a
dashboard whose job is to not cry wolf, that is the most valuable single result
in this benchmark, and it is the strongest argument for adopting his schema.

The likely cause: AMT.01 keeps mean *and* std of all 29 base columns raw (58 of
its 82 features), giving broad, uncompressed statistical coverage. YMT.01 spends
its budget on derived ratios, entropies and log-compressed rates, which sharpen
attack typing but discard magnitude detail that benign discrimination needs.

---

## 6. Ablation: which parts of v2 earned their place

Leave-one-block-out plus two keep-only controls, 2 seeds each. Accuracy std
across seeds is ~0.005, so treat anything under ±0.010 as noise.

| Variant | Features | Accuracy | Δ | Scan-family F1 | PCAP accuracy |
|---|---|---|---|---|---|
| **full_v2** | 64 | 0.8471 | — | 0.9776 | 0.9826 |
| − A composition (entropy, fan-out) | 56 | **0.8561** | +0.0090 | 0.9821 | 0.9869 |
| − B protocol mix | 60 | 0.8453 | −0.0018 | 0.9777 | 0.9832 |
| − C port-class mix | 58 | 0.8405 | −0.0066 | 0.9646 | 0.9786 |
| − D dispersion (mean+std) | 38 | 0.8431 | −0.0040 | 0.9918 | 0.9749 |
| − E flag profile | 56 | 0.8495 | +0.0024 | 0.9771 | 0.9860 |
| − F packet anomaly | 56 | 0.8484 | +0.0013 | 0.9813 | 0.9854 |
| − G current-flow passthrough | 60 | 0.8532 | +0.0060 | 0.9839 | 0.9869 |
| **window_only** (A+B+C) | 18 | 0.7271 | −0.1201 | 0.8536 | 0.8165 |
| **flowlike_only** (D+G) | 30 | 0.8361 | −0.0110 | 0.9633 | 0.9749 |
| *ZMT.01, from the 3-seed run* | 27 | 0.8091 | −0.0380 | 0.8453 | 0.9167 |

**No single block is load-bearing.** Every leave-one-out result lands within
about two seed-standard-deviations of the full model, and four of seven are
nominally *better* without the block. The blocks are substitutable routes to the
same window-level information.

**The gain is windowing itself, not the clever columns.** `flowlike_only` has no
entropy, no fan-out ratio, no port-class mix — just window mean and std of
thirteen ordinary flow quantities plus the raw current flow. At 30 columns, a
width comparable to ZMT.01's 27, it scores 0.8361 against ZMT.01's 0.8091. Two
thirds of the v2 improvement comes from simply computing statistics over
neighbouring flows, and the *std* half does most of the work — it measures
regularity, which is what separates a beacon from interactive traffic.

This also explains AMT.01's win: its 58 mean/std columns are the same mechanism
applied more broadly.

**Block A, which motivated the redesign, is redundant.** Explicit port entropy
and fan-out were the features I most expected to matter. Removing all eight
slightly *improves* every metric.

**Window composition alone is not enough.** `window_only` at 18 columns drops to
0.7271, below even ZMT.01. Fan-out without per-flow magnitude cannot tell a slow
scan from a fast one.

---

## 7. On Aryan's branch: what held up and what did not

### The schema held up. The benchmark did not.

His committed `results/benchmark/benchmark_results.json` was produced from
`demo_states.npz`, generated by `scripts/download_data.py::generate_demo_data`:

```python
T, D = 500, 50
states[:] = np.random.randn(T, D) * 0.3
recon_pattern[0] = 3.0; recon_pattern[4] = 2.5; recon_pattern[8] = -1.5
simulate_attack(200, 250, "Reconnaissance", recon_pattern)
```

That is 500 windows of 50-dimensional Gaussian noise at σ = 0.3, with attacks
injected as fixed additive constants of magnitude 2–4 on hand-picked dimensions.
Every extractor in `src/data/` also falls back to
`_generate_synthetic_frame(num_records=500)` when its real dataset file is
absent.

Three details confirm it: support is exactly 200 benign / 300 attack with
exactly 50 per MITRE stage; `per_feature_mse` has exactly 50 entries; and
`configs/model.yaml` sets `d_state: 110`, which does not match the 50 scored.

| Model | Binary F1 | Precision | Recall | FPR | MITRE macro F1 |
|---|---|---|---|---|---|
| World Model | 0.879 | 0.996 | 0.787 | 0.006 | 0.542 |
| LR baseline | **1.000** | 1.000 | 1.000 | 0.000 | 0.214 |
| RF baseline | **1.000** | 1.000 | 1.000 | 0.000 | 0.214 |

The perfect baselines are the tell. Adding a signal of magnitude 3.0 to a fixed
dimension of σ = 0.3 noise makes the classes linearly separable by construction,
so logistic regression *must* score 1.0. As published, the benchmark is evidence
against his transformer, not for it — it lost to a linear model on data a linear
model solves exactly.

**Those numbers should not be reported anywhere.** Re-run against AMT.01's
figures in §3, which come from real traffic.

### The StateBuilder is the best schema in this benchmark

`src/data/state_builder.py::_aggregate_window_vector` is well built, and on real
data it wins on accuracy, macro F1, binary F1, precision, FPR and AUC, with zero
spurious alarms into a real attack class. The window-aggregation idea in v2 came
from reading that file. He deserves the credit for the concept, and the port
shows the concept is worth more than my elaboration of it.

### The 374-feature list in `FEATURES.md` is a separate matter

`FEATURES.md` and `state_builder.py` are not connected — the 374 columns were
never extracted from real traffic, and the StateBuilder does not consume them.
The issues in that list stand independently of the benchmark result above.

| Issue | Detail |
|---|---|
| Label columns listed as features | `attack_cat` (169), `label` (170), `label` (347) are ground truth. If they reach a model, the metrics are meaningless. |
| Identifiers and absolute timestamps | `srcip`/`dstip` (122–125), `Timestamp` (3), `stime`/`ltime` (150–151), `src_comp`/`dst_comp` (274–275). These memorise the capture, not the behaviour. |
| Dataset identity leaks through missingness | Six corpora merged into one 374-wide vector; the pattern of zero columns identifies the source corpus, and each corpus has its own label distribution. One `has_packet_features` flag does not cover six sources. |
| LANL block is constants | Features 280–289 are fixed (`Flow Duration` = 1.0, `TotLen` = 64, `Flow Byts/s` = 128, flags zero-filled). Zero variance carries zero information, and it discards the real signal — the authentication graph. |
| DARPA block is host telemetry | `root_shell` (320), `su_attempted` (321), `num_shells` (324), `logged_in` (318). Not observable from PCAP, so all-zero at inference in our lab. DARPA/KDD is also a 1998–99 benchmark with documented redundancy problems. |
| ~40% duplication | The block of `fwd_bwd_ratio` + 3 `port_cat_*` + 6 `flag_*_frac` + 20 `port_*` appears five times (79–109, 171–197, 244–270, 289–306, 352–374). About 150 of the 374 collapse to 30 unique definitions. |
| Numbering not audited | UNSW's header says "27 engineered" but lists 30; the CIC block has `Active Min` but no `Idle Min`. |

Also worth noting: the head-to-head does **not** show that more columns win.
`window_only` at 18 columns scored 0.727 and full v2 at 64 scored 0.847, and
AMT.01's edge comes from *which* 82 columns it spends its budget on, not from
having more of them.

---

## 8. Recommendation

Neither schema dominates, and their strengths are complementary in a way that
suggests a merge rather than a choice:

1. **Take AMT.01's broad raw mean/std coverage** over all base columns. It is
   what produces the zero-false-alarm benign discrimination, and it is a
   mechanical change rather than a feature-engineering one.
2. **Keep YMT.01's log compression and `port_seq_ratio`/`port_fanout_ratio`.**
   They are why v2 leads on the scan/flood family and lab-capture accuracy —
   the classes the adversarial loop actually stresses.
3. **Drop v2 blocks A and G.** The ablation shows removing them matches or
   slightly beats full v2, giving a 52-column core to build on.
4. **Do not merge in the 374-feature list** without first removing the label
   columns, identifiers, host-only telemetry and five-times-duplicated port
   indicators.

A merged schema is untested. Until it is run, **AMT.01 is the better default for
the dashboard** on false-alarm grounds, and YMT.01 the better default if
attack-typing accuracy in the lab matters more.

---

## 9. Reproducing

```bash
python scripts/build_comparison_dataset.py   # ~20 s
python scripts/train_zmt_ymt.py              # ~7 min, CUDA (6 runs)
python scripts/train_amt.py                  # ~4 min, CUDA (3 runs)
python scripts/analyze_zmt_ymt.py            # three-way breakdown
python scripts/ablate_ymt_blocks.py          # ~26 min (20 runs)
python scripts/plot_zmt_ymt.py               # -> results/zmt_ymt/comparison.png
python scripts/smoke_test_ymt.py             # loads checkpoints, scores real PCAPs
```

Outputs land in `results/zmt_ymt/`: `metrics.json`, `analysis.json`,
`ablation.json`, `comparison.png`.

---

## 10. Limitations

- **AMT.01 is handicapped by two dead features.** Its unique-source-IP and
  unique-destination-IP columns are constant zero because our corpus has no IP
  fields. On data with IPs his schema would likely do better still.
- **The ablation covers v2 only.** AMT.01's 82 columns were not ablated, so its
  win is not attributed to specific blocks.
- **Thin per-class capture counts.** Most attack classes have 6–15 captures, so
  the capture-level split leaves 1–3 test captures per class. Ten classes hold
  fewer than 10 test windows; their per-class F1 is not meaningful.
- **Lab traffic is synthetic in origin.** The 263 PCAPs come from our own bots
  against our own container, so high PCAP-only accuracy partly reflects each
  bot having a consistent signature.
- **One real-world corpus, two days, one attack type.** Both CIC CSVs contain
  only Benign and Infilteration, so the only non-lab attack class is the one
  hardest to separate from benign. The `T1021` confusion dominating §5 is a
  property of that label, not of the models.
- **Window geometry was held fixed** at 8 flows for all three schemas. Aryan's
  design intended 30-second wall-clock windows, which was not tested.
