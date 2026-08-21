"""
Phosphor — Spike 3: ping-pong loop assembly + upscale + encode.

Takes a directory of generated PNG frames at generation size and produces the shipping
artifact: a seamlessly looping animated WebP at output resolution (CLAUDE.md §5/§6/§7).

Three things this is here to prove:
  1. The ping-pong sequence is genuinely seamless (no duplicate frame at either seam).
  2. The Lanczos upscale to 1350x1800 / 1200x1800 holds up.
  3. What the resulting file actually WEIGHS — the open question flagged in §7.

Usage
-----
  python tools/build_loop.py --frames out/ember_glow_stock_20s_cfg2_seed0
  python tools/build_loop.py --frames <dir> --gif           # also emit the GIF export
  python tools/build_loop.py --frames <dir> --quality-sweep # size vs -q:v curve
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(ROOT, "bin", "ffmpeg.exe")

OUT_SIZES = {"3:4": (1350, 1800), "2:3": (1200, 1800)}
FPS = 24


def pingpong_indices(n):
    """CLAUDE.md §6: [0,1,...,N-1,N-2,...,2,1] -> length 2N-2.

    Both endpoints are dropped on the reverse pass. Keeping frame N-1 duplicates the
    turnaround frame; keeping frame 0 duplicates the loop-point frame. Either reads as a
    visible hitch.
    """
    return list(range(n)) + list(range(n - 2, 0, -1))


def verify_seams(paths, idx):
    """Confirm the loop has no duplicate frames and no discontinuity at either seam.

    Compares every adjacent pair *including the wrap* from last back to first. A seam is
    only smooth if its delta sits inside the distribution of ordinary frame-to-frame
    deltas — an outlier there is exactly the visible hitch §6 warns about.
    """
    arrs = [np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) for p in paths]
    n = len(arrs)
    deltas = [np.abs(arrs[(i + 1) % n] - arrs[i]).mean() for i in range(n)]
    d = np.array(deltas)

    turn = idx.index(max(idx))          # frame N-1, the turnaround
    wrap = n - 1                        # last -> first, the loop point
    interior = np.delete(d, [turn, wrap])

    print(f"\n  seam check ({n} frames)")
    print(f"    typical frame delta : mean {interior.mean():.3f}  max {interior.max():.3f}")
    print(f"    turnaround seam     : {d[turn]:.3f}", end="")
    print("   OK" if d[turn] <= interior.max() else "   <-- HITCH")
    print(f"    loop-point seam     : {d[wrap]:.3f}", end="")
    print("   OK" if d[wrap] <= interior.max() else "   <-- HITCH")

    dupes = int((d < 1e-6).sum())
    print(f"    duplicate frames    : {dupes}" + ("   <-- would stutter" if dupes else "   none"))
    return d[turn] <= interior.max() and d[wrap] <= interior.max() and dupes == 0


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("\n  ffmpeg failed:\n" + (r.stderr or "")[-1500:])
        sys.exit(1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="directory of frame_%04d.png")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--quality-sweep", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(FFMPEG):
        raise SystemExit(f"  ffmpeg not found at {FFMPEG}")

    src = sorted(
        os.path.join(args.frames, f) for f in os.listdir(args.frames) if f.endswith(".png")
    )
    if not src:
        raise SystemExit(f"  no PNG frames in {args.frames}")

    gw, gh = Image.open(src[0]).size
    aspect = "3:4" if abs(gw / gh - 0.75) < abs(gw / gh - 2 / 3) else "2:3"
    ow, oh = OUT_SIZES[aspect]

    idx = pingpong_indices(len(src))
    name = os.path.basename(os.path.normpath(args.frames))
    outdir = args.out or os.path.join(ROOT, "out", "_loops")
    os.makedirs(outdir, exist_ok=True)

    print(f"\n  Phosphor — loop assembly")
    print(f"  source frames : {len(src)} @ {gw}x{gh} ({aspect})")
    print(f"  ping-pong     : {len(src)} -> {len(idx)} frames  (2N-2)")
    print(f"  upscale       : {gw}x{gh} -> {ow}x{oh}  ({ow/gw:.2f}x, lanczos)")
    print(f"  fps           : {FPS}   duration {len(idx)/FPS:.2f}s")

    ordered = [src[i] for i in idx]
    ok = verify_seams(ordered, idx)

    # Materialise the ping-pong order as a numbered sequence. Hardlinks where possible so
    # the duplicated half costs no disk.
    tmp = tempfile.mkdtemp(prefix="phosphor_pp_")
    try:
        for i, p in enumerate(ordered):
            dst = os.path.join(tmp, f"pp_%04d.png" % i)
            try:
                os.link(p, dst)
            except OSError:
                shutil.copy2(p, dst)

        pattern = os.path.join(tmp, "pp_%04d.png")
        scale = f"scale={ow}:{oh}:flags=lanczos"
        results = []

        qualities = [50, 65, 75, 85, 95] if args.quality_sweep else [args.quality]
        for q in qualities:
            webp = os.path.join(outdir, f"{name}_q{q}.webp")
            run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-framerate", str(FPS), "-i", pattern,
                 "-vf", scale,
                 "-c:v", "libwebp_anim", "-lossless", "0", "-q:v", str(q),
                 "-loop", "0", "-preset", "picture", webp])
            sz = os.path.getsize(webp)
            results.append((f"webp q{q}", sz, webp))

        if args.gif:
            pal = os.path.join(tmp, "palette.png")
            # Scale BEFORE palettegen, and apply the identical scale in both passes —
            # the palette must be built from the same pixels it is applied to.
            run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", pattern,
                 "-vf", f"{scale},palettegen=stats_mode=diff", pal])
            gif = os.path.join(outdir, f"{name}.gif")
            run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                 "-framerate", str(FPS), "-i", pattern, "-i", pal,
                 "-lavfi", f"{scale}[s];[s][1:v]paletteuse=dither=bayer:bayer_scale=5",
                 "-loop", "0", gif])
            results.append(("gif", os.path.getsize(gif), gif))

        print(f"\n  {'='*62}")
        print(f"  {'artifact':<14}{'size':>12}   {'per frame':>10}")
        print(f"  {'-'*62}")
        for label, sz, path in results:
            print(f"  {label:<14}{sz/1024/1024:>9.2f} MB   {sz/len(idx)/1024:>7.1f} KB")
        print(f"  {'='*62}")
        print(f"  written to {outdir}")
        print(f"  seams: {'SEAMLESS' if ok else 'CHECK ABOVE'}\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
