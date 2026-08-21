"""
Phosphor — generic motion/degradation metrics for a generated frame sequence.

Replaces the hardcoded y-band heuristic used in the first BO3 tests, which only worked
because that cover happens to put its title along the bottom. The region of interest is
derived from the source image, so this works on any layout.

!! KNOWN BROKEN FOR CROSS-COVER COMPARISON — read before trusting a number from this !!
---------------------------------------------------------------------------------------
`stability` was calibrated on ONE cover (BO3) varying only guidance, and it reproduces that
axis well. It does NOT transfer across different artwork. Measured 2026-08-20:

    Assassin's Creed  stability 0.1658 "OK"  -> "CREED" actually rendered as "NMEAV"
    The Desolate Hope stability 0.4369 "OK"  -> vertical title collapsed to garbage

Both scored at or above BO3, which was genuinely fine. The metric ranks destroyed type as
healthy, so it cannot gate anything on its own.

Why it fails: the top-decile edge mask is dominated by whatever is busiest in a given cover
— fire, armour trim, foliage — and legitimate motion there swamps the comparatively few
pixels belonging to type. On BO3 the title is large and heavy enough to carry the mask; on
covers with thin or stylised lettering it does not.

Use it only for A/B within a single cover, where the mask is held constant. **Cross-cover
verdicts require looking at the frames.** Fixing this properly needs actual text detection
(locate glyphs, then measure only those pixels), which is a real piece of work rather than
a tweak to the threshold.

Primary metric: detail_stability
--------------------------------
Text and logos are high-edge-density regions. We take the top-decile edge pixels of the
SOURCE as a detail mask, then measure the Pearson correlation between each frame's edge map
and frame 0's, inside that mask. Type that deforms or dissolves decorrelates; type that
holds stays correlated. Higher is better.

Calibrated against the guidance sweep, where the visual verdict is known:

    guidance 1.5  -> 0.152   crisp
    guidance 2.0  -> 0.128   crisp
    guidance 3.0  -> 0.044   letters visibly melting
    guidance 5.0  -> 0.026   destroyed

    >= 0.10  OK        0.05-0.10  marginal        < 0.05  degraded

Two earlier candidates were tried and rejected: absolute edge drift and edge-loss magnitude
separated the known-good from known-bad cases by only 13% and 18% respectively, versus 75%
for correlation. A metric that cannot reproduce a verdict you can see with your eyes is not
worth reporting.

flat_motion is the counterweight: temporal variation in the LOW-edge regions — fog, glow,
flat colour. That is the motion we actually want. A run can score perfectly on stability by
simply not moving, so both numbers have to be read together.
"""

import os

import numpy as np
from PIL import Image

OK, MARGINAL = 0.10, 0.05


def _edges(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def _z(v):
    return (v - v.mean()) / (v.std() + 1e-6)


def analyze(frame_dir, source_path):
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    if not files:
        raise ValueError(f"no frames in {frame_dir}")

    arr = np.stack(
        [np.asarray(Image.open(os.path.join(frame_dir, f)).convert("RGB"), dtype=np.float32)
         for f in files]
    )
    grey = arr.mean(axis=3)
    h, w = grey.shape[1:]

    src = np.asarray(
        Image.open(source_path).convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32
    )
    se = _edges(src.mean(axis=2))
    detail = se >= np.percentile(se, 90)
    flat = se <= np.percentile(se, 50)

    e0 = _z(_edges(grey[0])[detail])
    stability = float(np.mean([(e0 * _z(_edges(g)[detail])).mean() for g in grey[1:]]))

    return dict(
        frames=len(files),
        size=f"{w}x{h}",
        f0_mae=float(np.abs(arr[0] - src).mean()),
        stability=stability,
        flat_motion=float(arr.std(axis=0).mean(axis=2)[flat].mean()),
        overall=float(arr.std(axis=0).mean()),
        verdict="OK" if stability >= OK else ("marginal" if stability >= MARGINAL else "DEGRADED"),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    for k, v in analyze(a.frames, a.source).items():
        print(f"  {k:<14} {v}")
