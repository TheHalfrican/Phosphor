"""Package the frozen sidecar for first-run download.

The freeze is ~2.9 GB, which no Windows installer format can carry: NSIS dies at the
1.96 GB signed-32-bit boundary, and WiX packs into CAB with the same ceiling. So it is
downloaded and unpacked on first run, exactly like the model weights (CLAUDE.md 3).

    .venv/Scripts/python.exe tools/package_sidecar.py

Writes `sidecar-dist/phosphor-sidecar.zip` and prints the models.json entry to paste in,
with the real size and SHA256. Archive members are stored at the archive root (no
`phosphor-sidecar/` prefix), so `unpack_to: "sidecar"` lands the exe at
`<appdata>/sidecar/phosphor-sidecar.exe`.
"""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "sidecar-dist" / "phosphor-sidecar"
OUT = REPO / "sidecar-dist" / "phosphor-sidecar.zip"


def sha256(path, chunk=4 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    if not SRC.is_dir():
        sys.exit(f"not built: {SRC}\nRun ./tools/build_sidecar.ps1 first.")

    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    raw = sum(p.stat().st_size for p in files)
    print(f"packing {len(files)} files, {raw / 1e9:.2f} GB raw")

    t0 = time.time()
    # Deflate rather than stored: the Python and metadata files compress well, and even the
    # CUDA DLLs give some back. compresslevel=6 is the knee — level 9 costs far more time
    # for a fraction of a percent on binaries that are already dense.
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, p in enumerate(files, 1):
            z.write(p, p.relative_to(SRC).as_posix())
            if i % 250 == 0 or i == len(files):
                print(f"  {i}/{len(files)}", flush=True)

    size = OUT.stat().st_size
    print(f"\nwrote {OUT}")
    print(f"  {size / 1e9:.2f} GB compressed from {raw / 1e9:.2f} GB "
          f"({size / raw * 100:.0f}%) in {time.time() - t0:.0f}s")

    print("  hashing...", flush=True)
    digest = sha256(OUT)

    entry = {
        "key": "sidecar",
        "path": "sidecar/phosphor-sidecar.zip",
        "url": "REPLACE_ME",
        "bytes": size,
        "sha256": digest,
        "unpack_to": "sidecar",
        "note": ("Frozen Python inference sidecar (torch + CUDA). Too large for any Windows "
                 "installer format, so it is downloaded and unpacked on first run."),
    }
    print("\nmodels.json entry:\n")
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
