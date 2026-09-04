# Model SHAP / Attribution Analysis

Per-model outputs live in subfolders with:
- `analysis.json` — full numeric results
- `REPORT.md` — human summary
- `blocks_by_bucket.png` — block-level |attribution| by outcome bucket
- `fn_vs_tp_delta.png` — FN vs TP root-cause candidates (if both buckets exist)

Re-run:
```bash
python scripts/shap_model_analysis.py --models ary5,ary30,shaun
```