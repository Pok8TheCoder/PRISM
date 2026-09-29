# PRISM — setup and install

PRISM is the core repo (inference, lab UI, HX-C code). **Automode** (sibling folder `../Automode`) holds training scripts, mix datasets, and the same checkpoints under `train/checkpoints/` when you train locally.

## 1. Prerequisites

| Tool | Notes |
|------|--------|
| **Python 3.11+** | Recommended for PyTorch + lab scripts |
| **Git** + **Git LFS** | Required to pull model weights (`git lfs install` once per machine) |
| **Node.js 20+** | Lab Console frontend (`apps/lab-console/frontend`) |
| **CUDA GPU** | Optional but required for training and live scoring |
| **Docker Desktop** | Optional — adversarial lab (`scripts/lab_ctl.py`) |
| **Playwright** | Optional — demo video capture (`python -m playwright install chromium ffmpeg`) |

### Clone with LFS

```powershell
git lfs install
git clone https://github.com/Pok8TheCoder/PRISM.git
cd PRISM
git lfs pull
```

If you already cloned without LFS, run `git lfs pull` inside the repo after installing Git LFS.

## 2. Python environment

```powershell
cd PRISM
python -m venv venv
.\venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Use the [PyTorch install selector](https://pytorch.org/get-started/locally/) if your CUDA version differs. CPU-only wheels work for UI development without GPU scoring.

## 3. Sibling repos (training & baselines)

Layout on disk:

```
D:\Cursor\
  PRISM\          ← this repo (code + published checkpoints in models/checkpoints/)
  Automode\       ← training, benchmarks, hx_mix.npz builders
  PRISM-shaun\    ← Shaun w5s backbone weights (HX v1 finetune path)
```

Clone Automode next to PRISM if you plan to retrain HX-C:

```powershell
cd D:\Cursor
git clone https://github.com/Pok8TheCoder/Automode.git
cd Automode
git lfs install
git lfs pull
pip install -r requirements.txt  # if present; else use PRISM venv
```

**PRISM-shaun** is only needed for `train/train_hx.py` (HX v1 Shaun-wrap). Path expected: `PRISM-shaun/weights/w5s/world_model.pt`.

## 4. Model checkpoints

Published weights live in **`models/checkpoints/`** (Git LFS). See **[MODELS.md](./MODELS.md)** for lineage, metrics JSON, and training commands.

Quick verify:

```powershell
python -c "from pathlib import Path; p=Path('models/checkpoints/hx_c_v4_w5s.pt'); print(p.exists(), p.stat().st_size)"
```

Re-sync from a local Automode train tree (optional):

```powershell
python scripts/sync_model_checkpoints.py
```

## 5. Lab Console (primary UI)

```powershell
.\venv\Scripts\activate
cd apps\lab-console\frontend
npm install
cd ..\..\..
python run_lab_console.py --api
```

- UI: http://127.0.0.1:5173  
- API: http://127.0.0.1:8790  

Recorded IPS demo session: `python scripts/build_lab_demo_session.py` then open `/session?demo=1`.

## 6. Live / IPS scoring

```powershell
python scripts/live_ips_multi.py --backend hx_c --help
```

Checkpoint resolution (HX-C v4): `Automode/train/checkpoints/hx_c_v4_w5s.pt` if present, else `PRISM/models/checkpoints/hx_c_v4_w5s.pt`. See `src/hx/streaming.py` and `scripts/live_ips_multi.py`.

## 7. Legacy Streamlit dashboard

```powershell
python run_dashboard.py
```

## 8. Tests and smoke checks

```powershell
python apps/lab-console/scripts/smoke_api.py   # if available on your branch
pytest tests/ -q                               # when test suite is configured
```

## Troubleshooting

- **Missing `.pt` after clone** — run `git lfs pull` and ensure Git LFS is installed.  
- **CUDA OOM** — reduce batch in Automode train scripts or use CPU for smoke tests only.  
- **Automode remote 404** — repo may be private; use checkpoints bundled in PRISM `models/checkpoints/`.
