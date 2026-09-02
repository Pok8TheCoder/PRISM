# Run ARY242 ablation batches on local RTX 3060 (12 GB).
# Usage: .\scripts\run_ary242_ablation_local.ps1
# Or single variant: python scripts/ablate_ary242_blocks.py lean_no_mean

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Step 1: resource estimate + smoke test ===" -ForegroundColor Cyan
.\venv\Scripts\python.exe scripts\estimate_ary242_resources.py --smoke-train

Write-Host "`n=== Step 2: quick variants (under 20 min target) ===" -ForegroundColor Cyan
$quick = @(
    "full_242",
    "minus_dead_meta",
    "minus_B_mean",
    "minus_B_std",
    "minus_C_flags",
    "minus_E_top_ports",
    "core_std_ACD",
    "lean_no_mean",
    "lean_no_ports",
    "ymt_like_lean"
)
foreach ($v in $quick) {
    Write-Host "`n--- $v ---" -ForegroundColor Yellow
    .\venv\Scripts\python.exe scripts\ablate_ary242_blocks.py $v
}

Write-Host "`n=== Done. See results/ary242_ablation/ablation.json ===" -ForegroundColor Green
