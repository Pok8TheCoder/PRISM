# Launch PRISM results viewer
param(
    [ValidateSet("falloff", "lab", "live_lab", "ary_compare")]
    [string]$Mode = "falloff"
)
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
& $Py -m pip install -q PySide6 pyqtgraph 2>$null
& $Py "$Root\scripts\falloff_viewer.py" --mode $Mode @args
