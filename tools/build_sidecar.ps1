# Build the frozen Python sidecar.
#
# Use this rather than calling pyinstaller directly. PyInstaller's default output is
# `dist/`, which belongs to Vite — `npm run build` runs with emptyOutDir and deletes
# everything in it. Putting the freeze there means the Tauri bundle step then fails with
# "resource path ..\dist\phosphor-sidecar doesn't exist", which points at the wrong thing
# entirely. Output goes to `sidecar-dist/` instead.
#
#   ./tools/build_sidecar.ps1            build
#   ./tools/build_sidecar.ps1 -Clean     rebuild from scratch
param([switch]$Clean)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$pyinstaller = Join-Path $repo ".venv\Scripts\pyinstaller.exe"
if (-not (Test-Path $pyinstaller)) {
    throw "pyinstaller not found at $pyinstaller. Run ./setup.ps1, then: .venv\Scripts\pip install pyinstaller"
}

$out = Join-Path $repo "sidecar-dist"
$work = Join-Path $repo "build\pyinstaller"

if ($Clean) {
    foreach ($d in @($out, $work)) {
        if (Test-Path $d) { Write-Host "removing $d"; Remove-Item -Recurse -Force $d }
    }
}

Write-Host "building frozen sidecar -> $out"
& $pyinstaller --noconfirm --distpath $out --workpath $work (Join-Path $repo "sidecar\phosphor-sidecar.spec")
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed with exit code $LASTEXITCODE" }

$exe = Join-Path $out "phosphor-sidecar\phosphor-sidecar.exe"
if (-not (Test-Path $exe)) { throw "build reported success but $exe is missing" }

$size = (Get-ChildItem (Join-Path $out "phosphor-sidecar") -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("done: {0} ({1:N2} GB)" -f $exe, ($size / 1GB))
Write-Host "tauri.conf.json bundles this from ../sidecar-dist/phosphor-sidecar/"
