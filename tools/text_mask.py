"""
Phosphor — text/logo detection and protection.

WHY THIS EXISTS
---------------
Measured 2026-08-20: TI2V-5B cannot preserve fine glyph structure through
encode -> denoise -> decode. On "Assassin's Creed" the thin letterspaced "CREED" renders as
"NMEAV" at guidance 1.0, 1.5 AND 2.0 — identical damage with classifier-free guidance fully
disabled. It is not a guidance artifact and no parameter tuning fixes it. Heavy display type
("ASSASSIN'S" directly above it) survives everywhere; thin or small type never does.

The fix is not to make the model preserve text, but to stop asking it to: detect the type in
the source, then composite the original pixels back over every generated frame. That is what
a viewer expects anyway — a cover's title lockup is a static overlay, not something that
should drift.

DETECTION — CRAFT
-----------------
Two classical-CV attempts failed first, and it is worth recording why so nobody retries them:

  * edge-density + morphology OVER-detects — 20.4% of the BO3 frame, grabbing characters and
    fire, because dense edges from artwork are indistinguishable from dense edges from glyphs.
  * adding a colour-consistency filter OVER-corrects to zero, because morphological closing
    merges glyphs together with the background gaps between them, so the variance measured is
    the background's rather than the strokes'.

CRAFT (Character Region Awareness For Text detection, clovaai, **MIT**) is trained on scene
text rather than documents, which is the right category for stylised cover titles. It is
torch-native, so it adds no new runtime — only ~83 MB of weights.

Note we use CRAFT's raw character-region heatmap directly as a soft mask and skip its
box/polygon postprocessing entirely. We want "which pixels are type", not "where are the
word boxes", so the heatmap is a better fit than the boxes and far less code.
"""

import os

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "craft", "craft_mlt_25k.pth")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None


def _load(device="cuda"):
    global _model
    if _model is not None:
        return _model
    import sys
    sys.path.insert(0, os.path.join(ROOT, "sidecar"))
    from vendor.craft import CRAFT

    if not os.path.exists(WEIGHTS):
        raise SystemExit(
            f"\n  CRAFT weights missing: {WEIGHTS}\n"
            f"  huggingface-cli download Manbehindthemadness/craft_mlt_25k craft_mlt_25k.pth\n"
        )
    net = CRAFT()
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    # Saved from a DataParallel wrapper, so every key carries a "module." prefix.
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    _model = net.to(device).eval()
    return _model


@torch.no_grad()
def detect_text_mask(img, device="cuda", long_side=1280, thresh=0.30,
                     use_affinity=True, dilate=3, feather=5):
    """Soft mask in [0,1] marking probable text pixels.

    img         PIL RGB at the resolution the mask will be applied at
    long_side   CRAFT input scale. It is resolution sensitive; ~1280 matches its training.
    thresh      heatmap cutoff. Lower catches more (and more false positives).
    use_affinity  include the character-affinity channel, which fills the gaps between
                  glyphs so a word becomes one region rather than isolated letters.
    dilate      grow the mask so anti-aliased glyph edges are covered too
    feather     gaussian blur radius on the mask edge, avoids a visible seam
    """
    net = _load(device)
    W, H = img.size

    scale = long_side / max(W, H)
    # CRAFT's decoder needs dimensions divisible by 32.
    tw, th = (max(32, int(round(W * scale / 32)) * 32), max(32, int(round(H * scale / 32)) * 32))
    x = np.asarray(img.convert("RGB").resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    y, _ = net(t)                       # [1, th/2, tw/2, 2]
    region = y[0, :, :, 0].cpu().numpy()
    affinity = y[0, :, :, 1].cpu().numpy()
    score = np.maximum(region, affinity) if use_affinity else region

    m = (score >= thresh).astype(np.uint8) * 255
    m = Image.fromarray(m).resize((W, H), Image.BILINEAR)
    if dilate:
        m = m.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m, dtype=np.float32) / 255.0


def composite(frame, source, mask):
    """Paste source pixels back wherever mask is high. Both PIL RGB, same size."""
    a = np.asarray(frame, dtype=np.float32)
    b = np.asarray(source, dtype=np.float32)
    w = mask[..., None]
    return Image.fromarray((a * (1 - w) + b * w).astype(np.uint8))


def protect_dir(frame_dir, source_path, out_dir=None, **kw):
    """Apply text protection to a whole generated sequence."""
    files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".png"))
    size = Image.open(os.path.join(frame_dir, files[0])).size
    src = Image.open(source_path).convert("RGB").resize(size, Image.LANCZOS)
    mask = detect_text_mask(src, **kw)

    out_dir = out_dir or frame_dir.rstrip("/\\") + "_protected"
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        fr = Image.open(os.path.join(frame_dir, f)).convert("RGB")
        composite(fr, src, mask).save(os.path.join(out_dir, f))
    return out_dir, mask


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--frames")
    ap.add_argument("--thresh", type=float, default=0.30)
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()

    src = Image.open(a.source).convert("RGB")
    if a.frames:
        src = src.resize(Image.open(
            os.path.join(a.frames, sorted(os.listdir(a.frames))[0])).size, Image.LANCZOS)
    mask = detect_text_mask(src, thresh=a.thresh)
    print(f"  mask covers {mask.mean()*100:.2f}% of the frame")

    if a.preview:
        ov = np.asarray(src, dtype=np.float32).copy()
        ov[..., 0] = np.minimum(255, ov[..., 0] + mask * 150)
        ov[..., 2] = np.maximum(0, ov[..., 2] - mask * 90)
        Image.fromarray(ov.astype(np.uint8)).save(a.preview)
        print(f"  wrote {a.preview}")

    if a.frames:
        out, _ = protect_dir(a.frames, a.source, thresh=a.thresh)
        print(f"  protected frames -> {out}")
