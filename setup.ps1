# Phosphor - dev environment bootstrap.
# Creates .venv, installs CUDA torch + deps, then fetches models (~19 GB).
# Safe to re-run: venv creation and model downloads both skip completed work.
#
# ASCII-only on purpose. Windows PowerShell 5.1 reads .ps1 as ANSI unless a BOM is
# present, so non-ASCII characters here become mojibake and break the parser.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"

function Step($n, $text) {
    Write-Host ""
    Write-Host ("=" * 74) -ForegroundColor DarkCyan
    Write-Host "  [$n/5]  $text" -ForegroundColor Cyan
    Write-Host ("=" * 74) -ForegroundColor DarkCyan
    Write-Host ""
}

Write-Host ""
Write-Host "  P H O S P H O R  -  dev setup" -ForegroundColor Yellow
Write-Host "  RTX 4090 / CUDA 13.0 / Python 3.12" -ForegroundColor DarkGray

# ---- 1. venv -----------------------------------------------------------------
Step 1 "Creating virtual environment"
if (Test-Path $Py) {
    Write-Host "  .venv already exists - skipping." -ForegroundColor DarkGray
}
else {
    py -3.12 -m venv (Join-Path $Root ".venv")
    Write-Host "  Created .venv (Python 3.12)" -ForegroundColor Green
}

& $Py -m pip install --upgrade pip --quiet

# ---- 2. torch ----------------------------------------------------------------
Step 2 "Installing PyTorch 2.13.0 + CUDA 13.0  (~3 GB)"
& $Py -m pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { Write-Host "`n  torch install failed." -ForegroundColor Red; exit 1 }

# ---- 3. rest of deps ---------------------------------------------------------
Step 3 "Installing diffusers / transformers / gguf"
& $Py -m pip install -r (Join-Path $Root "sidecar\requirements.txt")
if ($LASTEXITCODE -ne 0) { Write-Host "`n  dependency install failed." -ForegroundColor Red; exit 1 }

# ---- 4. CUDA sanity ----------------------------------------------------------
Step 4 "Verifying CUDA"
& $Py (Join-Path $Root "tools\check_cuda.py")
if ($LASTEXITCODE -ne 0) { Write-Host "`n  CUDA check failed." -ForegroundColor Red; exit 1 }
Write-Host "  CUDA OK" -ForegroundColor Green

# ---- 5. models ---------------------------------------------------------------
Step 5 "Downloading models  (~19 GB, resumable)"
& $Py (Join-Path $Root "tools\download_models.py")

Write-Host ""
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host "  This window can be closed." -ForegroundColor DarkGray
Write-Host ""
