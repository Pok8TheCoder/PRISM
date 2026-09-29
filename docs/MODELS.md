# Model checkpoints — lineage and how to reproduce

Canonical **published** weights are in `models/checkpoints/` (Git LFS). Training **scripts and NPZ mixes** live in the sibling **Automode** repo under `train/`.

Machine-readable index: `models/checkpoints/MANIFEST.json`.

## HX-C family (causal 33-class, 292-d Shaun ingest, 5 s windows)

| File | Role | How it is made |
|------|------|----------------|
| `hx_c_v4_w5s.pt` | **Production HX-C** (lab console, IPS demos, v4 bench) | `Automode/train/build_hx_bulk_mix.py` → `python train/train_hx_c_v4.py` |
| `hx_c_v3_w5s.pt` | Prior causal head (v3 bench) | `python train/train_hx_c_v3.py` (Automode) |
| `hx_c_v2_w5s.pt` | Architecture / loss experiments | `python train/train_hx_c_v2.py` |
| `hx_c_w5s.pt` | HX-C v1 causal on `hx_mix.npz` | `python train/train_hx_c.py` |
| `hx_w5s.pt` | HX v1 — Shaun w5s backbone + technique head | `python train/train_hx.py` (needs PRISM-shaun w5s) |

Sidecar metrics: matching `*.json` next to each checkpoint (val FP rate, recall, epoch).

**Architecture code (PRISM):** `src/hx/causal.py` (`CausalHXClassifier`), `src/hx/heads.py`, optional `src/hx/benign_guard.py` at inference.

**Lab finetune (optional):** `src/hx/lab_adapt.py` writes `Automode/train/checkpoints/hx_c_w5s_lab.pt` from Harborline replay — not always shipped; run locally after lab capture.

### v4 training summary

1. Build curated mix: `train/data/hx_mix.npz` (Automode ingest pipelines).  
2. Build bulk internet corpus: `train/build_hx_bulk_mix.py` → `hx_bulk_mix.npz`.  
3. Train: `train/train_hx_c_v4.py` — PCAP-grouped validation, focal + FP penalty, 30-window lookback (`LOOKBACK` in `causal.py`).  
4. Colab path: `train/COLAB_HX_V4.md`, `train/colab_train_hx_c_v4.ipynb`.

## FlowTrack / Shaun (lab baselines)

| File | Role | How it is made |
|------|------|----------------|
| `shaun_branch_fairfit.pt` | Fair-fit Shaun branch for lab benches | Automode `train/finetune_shaun_quiet_lab.py` / fair-fit pipeline |
| `shaun_quietlab_e6.pt` | Quiet-lab Shaun epoch-6 | Automode quiet-lab finetune scripts |

Backbone weights for full Shaun v3 often live under **`PRISM-shaun/weights/`** (separate repo; not duplicated here).

## ARY / XMT (PRISM world models)

| File | Role |
|------|------|
| `ary_5sv01.pt`, `ary5s_v01.pt` | 5 s ARY checkpoints for IPS red-team scripts |
| `aryan_world_model_best.pt` | Full ARY world model (longer horizon) |
| `xmt_world_model_best.pt` | XMT variant |

Training entrypoints are under `scripts/` and historical Automode train jobs; see `results/` reports for the eval that produced each filename.

## Inference pointers

| Use case | Checkpoint | Entry script |
|----------|------------|--------------|
| Multi-model IPS lab | `hx_c_v4_w5s.pt` | `scripts/live_ips_multi.py --backend hx_c` |
| HX streaming bundle | `hx_w5s.pt` or `hx_c_*.pt` | `src/hx/streaming.py` |
| SHAP at block | same as live HX-C | `src/explain/forecast_block_explain.py` |
| Lab Console stub paths | `docs` + `apps/lab-console/api/stub_data.py` | UI metadata only |

## Re-publish weights to GitHub after training

```powershell
cd D:\Cursor\PRISM
python scripts/sync_model_checkpoints.py
git add models/checkpoints/
git add models/checkpoints/MANIFEST.json
git commit -m "Update published model checkpoints and manifest."
git push origin HEAD
```

Ensure Git LFS is installed (`git lfs install`) before adding new `.pt` files.
