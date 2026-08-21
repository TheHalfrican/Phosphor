# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Phosphor sidecar.

Build from the repo root:

    .venv/Scripts/pyinstaller.exe --noconfirm sidecar/phosphor-sidecar.spec

Output is `dist/phosphor-sidecar/` (a directory, not a single file) containing
`phosphor-sidecar.exe` plus `_internal/`.

WHY ONEDIR AND NOT ONEFILE
--------------------------
The payload is torch plus CUDA, several gigabytes of it. `--onefile` would extract that
whole payload into a temp directory on *every* launch, which means a multi-second-to-
minute startup and gigabytes of churn each time the user opens the app. Onedir starts
instantly and costs nothing at runtime.

The consequence is that this cannot ship through Tauri's `externalBin`, which expects a
single file. It is bundled as a resource directory instead, and `bundled_binary()` in
`src-tauri/src/lib.rs` looks for it under the resource dir. That also avoids the trap in
`src-tauri/binaries/README.md`, where a missing `externalBin` entry fails every build.

WHAT IS COLLECTED AND WHY
-------------------------
diffusers and transformers are both `_LazyModule` packages: their `__init__` declares
submodules in a dict and imports them on attribute access, so PyInstaller's static
analysis sees almost nothing. `collect_submodules` is what actually pulls them in.

`copy_metadata` matters just as much. Both libraries call `importlib.metadata.version(...)`
on themselves and their optional integrations at import time; without the `.dist-info`
present, that raises `PackageNotFoundError` and the import fails at startup rather than
degrading. This is the single most common way a frozen diffusers app dies.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, copy_metadata

# Copy the .dist-info for EVERY installed distribution, not a hand-picked list.
#
# diffusers and transformers call importlib.metadata.version() on packages they merely
# *probe* for, through require_version() and the is_x_available() helpers, and the set is
# not knowable by reading our own imports. A hand-maintained list failed the first build
# on 'requests', which nothing here imports directly.
#
# The cost is trivial: dist-info directories are metadata text, a few hundred KB total
# against a 2.9 GB bundle. The benefit is that this entire class of
# PackageNotFoundError-at-startup simply cannot happen.
from importlib.metadata import distributions

METADATA = sorted({
    name for name in (d.metadata["Name"] for d in distributions()) if name
})

datas = []
for dist in METADATA:
    try:
        datas += copy_metadata(dist)
    except Exception:
        # A missing optional package is not fatal; its availability check will simply
        # report False at runtime, which is the behaviour we want anyway.
        pass

# Non-Python files these ship (json configs, .pyi, vocab data).
datas += collect_data_files("diffusers")
datas += collect_data_files("transformers")
datas += collect_data_files("gguf")

hiddenimports = []
hiddenimports += collect_submodules("diffusers")
hiddenimports += collect_submodules("transformers")
hiddenimports += collect_submodules("gguf")
# Our own sibling modules. `pipeline` is imported inside main() and `vendor.craft` inside
# load_craft(), both late enough that they are easy for the analysis to miss.
hiddenimports += ["pipeline", "vendor", "vendor.craft", "vendor.craft.craft",
                  "vendor.craft.vgg16_bn"]
# torchvision's VGG16-BN backs CRAFT.
hiddenimports += ["torchvision", "torchvision.models", "torchvision.models.vgg"]

# Dev-only or plainly unreachable. Kept deliberately short: excluding something diffusers
# probes for turns a working availability check into an ImportError, and the size win is
# marginal next to torch.
excludes = [
    "tkinter", "matplotlib", "IPython", "notebook", "jupyter",
    "pytest", "_pytest", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "torch.utils.tensorboard", "tensorboard",
]

a = Analysis(
    ["inference_server.py"],
    pathex=["sidecar"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="phosphor-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX on multi-GB CUDA DLLs is slow and gains little
    # console=True keeps stdin/stdout as ordinary handles, which the JSONL protocol
    # depends on. No window appears regardless: sidecar.rs spawns with CREATE_NO_WINDOW.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="phosphor-sidecar",
)
