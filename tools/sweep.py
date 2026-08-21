"""
Phosphor — cover x preset sweep.

Two questions, deliberately kept separate so each varies one thing:

  --covers   all covers, one preset. Does guidance 2.0 hold across different artwork, or was
             it calibrated on a single lucky sample? Also exercises the 2:3 / 768x1152 path,
             which nothing has touched yet.
  --presets  all presets, one cover. Seven of the eight presets are entirely unvalidated.
             Cloth Sway is the one most likely to disturb faces.

Loads the pipeline ONCE and reuses it across runs — model load is ~5s but repeated 13 times
is a minute of pure waste, and reusing it also removes load-order as a variable.
"""

import argparse
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_run import analyze  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = r"C:\Users\NoahM\Documents\Custom Steam Artwork\Game RetroVoid Artwork"
BASE_COVER = "COD - BO3Z - Mod Tools[RV].png"
BASE_PRESET = "ember_glow"
GUIDANCE = 2.0
STEPS = 20


def build_pipe():
    from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig,
                           UniPCMultistepScheduler, WanImageToVideoPipeline,
                           WanTransformer3DModel)
    MODEL_DIR = os.path.join(ROOT, "models", "wan-ti2v-5b-diffusers")
    GGUF = os.path.join(ROOT, "models", "gguf", "Wan2.2-TI2V-5B-Q6_K.gguf")
    dt = torch.bfloat16

    print("  loading transformer (GGUF)...", flush=True)
    tr = WanTransformer3DModel.from_single_file(
        GGUF, quantization_config=GGUFQuantizationConfig(compute_dtype=dt),
        dtype=dt, config=MODEL_DIR, subfolder="transformer")
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae", dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR, transformer=tr, vae=vae, text_encoder=None, tokenizer=None,
        image_encoder=None, image_processor=None, dtype=dt)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    return pipe


def load_embeds(preset_id):
    from safetensors import safe_open
    p = os.path.join(ROOT, "assets", "embeddings.safetensors")
    with safe_open(p, framework="pt", device="cpu") as f:
        max_len = int((f.metadata() or {})["max_sequence_length"])
        pos, neg = f.get_tensor(preset_id), f.get_tensor("__negative__")

    def pad(t):
        o = torch.zeros(max_len, t.shape[-1], dtype=t.dtype)
        o[: t.shape[0]] = t
        return o.unsqueeze(0).to("cuda", torch.bfloat16)

    return pad(pos), pad(neg)


GEN = {"3:4": (768, 1024), "2:3": (768, 1152)}


def run_one(pipe, image_path, preset, seed=0):
    src = Image.open(image_path).convert("RGB")
    r = src.size[0] / src.size[1]
    aspect = "3:4" if abs(r - 0.75) < abs(r - 2 / 3) else "2:3"
    gw, gh = GEN[aspect]

    stem = "".join(c if c.isalnum() else "-" for c in
                   os.path.splitext(os.path.basename(image_path))[0]).strip("-")[:28]
    outdir = os.path.join(ROOT, "out", f"{stem}__{preset}_stock_{STEPS}s_cfg{GUIDANCE:g}_seed{seed}")

    if os.path.isdir(outdir) and len([f for f in os.listdir(outdir) if f.endswith(".png")]) >= 33:
        print(f"    (cached) {os.path.basename(outdir)}")
        return outdir, aspect, 0.0

    os.makedirs(outdir, exist_ok=True)
    pos, neg = load_embeds(preset)
    t0 = time.time()
    res = pipe(image=src.resize((gw, gh), Image.LANCZOS), prompt=None, negative_prompt=None,
               prompt_embeds=pos, negative_prompt_embeds=neg, height=gh, width=gw,
               num_frames=33, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
               generator=torch.Generator(device="cuda").manual_seed(seed), output_type="pil")
    el = time.time() - t0
    for i, fr in enumerate(res.frames[0]):
        fr.save(os.path.join(outdir, f"frame_{i:04d}.png"))
    return outdir, aspect, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covers", action="store_true")
    ap.add_argument("--presets", action="store_true")
    args = ap.parse_args()
    if not (args.covers or args.presets):
        args.covers = args.presets = True

    import json
    with open(os.path.join(ROOT, "assets", "presets.json"), encoding="utf-8") as f:
        presets = [p["id"] for p in json.load(f)["presets"]]

    jobs = []
    if args.covers:
        for f in sorted(os.listdir(ART)):
            jobs.append(("cover", os.path.join(ART, f), BASE_PRESET))
    if args.presets:
        for p in presets:
            jobs.append(("preset", os.path.join(ART, BASE_COVER), p))

    # de-dup (BO3 x ember_glow appears in both halves)
    seen, uniq = set(), []
    for kind, img, pre in jobs:
        k = (img, pre)
        if k not in seen:
            seen.add(k)
            uniq.append((kind, img, pre))

    print(f"\n  Phosphor — sweep: {len(uniq)} runs @ guidance {GUIDANCE}, {STEPS} steps\n")
    pipe = build_pipe()

    rows = []
    for i, (kind, img, pre) in enumerate(uniq, 1):
        label = os.path.splitext(os.path.basename(img))[0].replace("[RV]", "")
        print(f"\n  [{i}/{len(uniq)}] {label}  x  {pre}", flush=True)
        outdir, aspect, el = run_one(pipe, img, pre)
        m = analyze(outdir, img)
        rows.append((kind, label, pre, aspect, m, el))
        print(f"    {m['size']}  stability {m['stability']:.4f} ({m['verdict']})  "
              f"flat_motion {m['flat_motion']:.2f}  f0 {m['f0_mae']:.2f}  {el:.0f}s")

    print(f"\n\n  {'='*94}")
    print(f"  {'cover':<30}{'preset':<19}{'asp':<6}{'stab':>8}{'flat':>8}{'f0':>7}  verdict")
    print(f"  {'-'*94}")
    for kind, label, pre, aspect, m, el in sorted(rows, key=lambda r: -r[4]["stability"]):
        print(f"  {label[:29]:<30}{pre:<19}{aspect:<6}{m['stability']:>8.4f}"
              f"{m['flat_motion']:>8.2f}{m['f0_mae']:>7.2f}  {m['verdict']}")
    print(f"  {'='*94}\n")


if __name__ == "__main__":
    main()
