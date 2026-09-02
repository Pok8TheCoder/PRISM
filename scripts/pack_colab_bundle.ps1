# Create PRISM_colab.zip for Colab upload (code + aryan splits).

# Uses Python zipfile so paths use forward slashes (Linux/Colab safe).

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$Out = Join-Path $Root "PRISM_colab.zip"



$items = @(

    "requirements.txt",

    "scripts\estimate_ary242_resources.py",

    "scripts\train_aryan_wm.py",

    "scripts\ablate_ary242_blocks.py",

    "src\aryan",

    "data\aryan_splits"

)



$staging = Join-Path $env:TEMP "PRISM_colab_staging"

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }

New-Item -ItemType Directory -Path $staging | Out-Null



foreach ($rel in $items) {

    $src = Join-Path $Root $rel

    if (-not (Test-Path $src)) { Write-Warning "Skip missing: $rel"; continue }

    $dest = Join-Path $staging $rel

    $parent = Split-Path $dest -Parent

    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }

    Copy-Item $src $dest -Recurse -Force

}



if (Test-Path $Out) { Remove-Item $Out -Force }

python -c "import zipfile; from pathlib import Path; import sys; staging=Path(sys.argv[1]); out=Path(sys.argv[2]); z=zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED); [z.write(p, p.relative_to(staging).as_posix()) for p in staging.rglob('*') if p.is_file()]; z.close()" $staging $Out

Remove-Item $staging -Recurse -Force

$mb = [math]::Round((Get-Item $Out).Length / 1MB, 2)

Write-Host "Created $Out ($mb MB). Upload in Colab Setup cell."

