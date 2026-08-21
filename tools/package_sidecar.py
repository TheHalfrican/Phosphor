"""Package the frozen sidecar for first-run download, split across N archives.

The freeze is ~3 GB, which no Windows installer format can carry: NSIS dies at the 1.96 GB
signed-32-bit boundary, and WiX packs into CAB with the same ceiling. So it is downloaded
and unpacked on first run, exactly like the model weights (CLAUDE.md 3).

    .venv/Scripts/python.exe tools/package_sidecar.py [--parts N] [--tag v0.1.0]

WHY SPLIT
---------
The host is GitHub Releases, which caps a single asset at 2 GiB. In one piece the archive
lands at 1.92 GiB — under the cap, but with about 83 MiB to spare, so one torch update
would break it. Two parts sit near 1 GB each and the cliff goes away.

Each part is a COMPLETE, INDEPENDENT zip, not a byte-split of one archive. That matters:
the downloader verifies and extracts each entry on its own, so a part can fail, resume and
retry without touching the other. Byte-splitting would force both to land before anything
could be checked. Both parts declare `unpack_to: "sidecar"` and extract into the same
directory, which needs no code beyond what the manifest already does.

Files are distributed largest-first into whichever part is currently smallest, which keeps
the two within a few percent of each other without trying to model the compression ratio.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "sidecar-dist" / "phosphor-sidecar"
OUTDIR = REPO / "sidecar-dist"

GITHUB_RELEASE = "https://github.com/TheHalfrican/Phosphor/releases/download"


def sha256(path, chunk=4 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def split_evenly(files, n):
    """Greedy largest-first bin packing on raw size."""
    bins = [[] for _ in range(n)]
    totals = [0] * n
    for p, size in sorted(files, key=lambda t: -t[1]):
        i = totals.index(min(totals))
        bins[i].append(p)
        totals[i] += size
    return bins, totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=2)
    ap.add_argument("--tag", default="v0.1.0", help="GitHub release tag for the URLs")
    args = ap.parse_args()

    if not SRC.is_dir():
        raise SystemExit(f"not built: {SRC}\nRun ./tools/build_sidecar.ps1 first.")

    files = [(p, p.stat().st_size) for p in sorted(SRC.rglob("*")) if p.is_file()]
    raw = sum(s for _, s in files)
    print(f"packing {len(files)} files, {raw / 1e9:.2f} GB raw, into {args.parts} parts")

    # A stale single-part archive from an earlier run would be silently uploaded otherwise.
    for old in OUTDIR.glob("phosphor-sidecar*.zip"):
        old.unlink()

    bins, totals = split_evenly(files, args.parts)
    entries = []

    for i, (members, total) in enumerate(zip(bins, totals), 1):
        out = OUTDIR / f"phosphor-sidecar-{i}of{args.parts}.zip"
        print(f"\npart {i}/{args.parts}: {len(members)} files, {total / 1e9:.2f} GB raw")
        t0 = time.time()
        # Deflate at level 6: the knee. Level 9 costs far more time for a fraction of a
        # percent on binaries this dense.
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for j, p in enumerate(sorted(members), 1):
                z.write(p, p.relative_to(SRC).as_posix())
                if j % 500 == 0 or j == len(members):
                    print(f"  {j}/{len(members)}", flush=True)

        size = out.stat().st_size
        gib = size / (1 << 30)
        print(f"  {out.name}: {size / 1e9:.2f} GB ({gib:.2f} GiB) in {time.time() - t0:.0f}s")
        if size >= (2 << 30):
            print("  !! OVER GitHub's 2 GiB per-asset cap - increase --parts")
        print("  hashing...", flush=True)

        entries.append({
            "key": f"sidecar-part{i}",
            "path": f"sidecar/{out.name}",
            "url": f"{GITHUB_RELEASE}/{args.tag}/{out.name}",
            "bytes": size,
            "sha256": sha256(out),
            "unpack_to": "sidecar",
            "note": (f"Frozen Python inference sidecar, part {i} of {args.parts} "
                     f"(torch + CUDA). Too large for any Windows installer format, so it is "
                     f"downloaded and unpacked on first run. Each part is an independent "
                     f"zip extracting into the same directory."),
        })

    total_zip = sum(e["bytes"] for e in entries)
    print(f"\ntotal compressed: {total_zip / 1e9:.2f} GB from {raw / 1e9:.2f} GB "
          f"({total_zip / raw * 100:.0f}%)")
    print(f"largest part: {max(e['bytes'] for e in entries) / (1 << 30):.2f} GiB "
          f"(GitHub cap is 2.00 GiB)")

    out_json = OUTDIR / "manifest-entries.json"
    out_json.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest entries written to {out_json}")
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
