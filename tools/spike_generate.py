"""
Phosphor — end-to-end generation spike.  Image in, PNG frames out.

Per CLAUDE.md §9: prove the pipeline before any Tauri code exists. This script exists to
answer four questions and then be thrown away or folded into sidecar/:

  1. Does image conditioning work at all on TI2V-5B via WanImageToVideoPipeline?
  2. Does 768x1024 / 768x1152 fit in 24 GB?
  3. Is the Turbo 4-step distill good enough on real cover art vs stock 20-step?
  4. Do baked embeddings match live-encoded ones?  (--live-encode)

Examples
--------
  # stock, 20 steps, baked embeddings (the shipping config)
  python tools/spike_generate.py --image "cover.png" --preset ember_glow

  # turbo LoRA, 4 steps, CFG 1
  python tools/spike_generate.py --image "cover.png" --preset ember_glow --lora

  # A/B the embedding path: loads the 11.4 GB encoder and encodes live
  python tools/spike_generate.py --image "cover.png" --preset ember_glow --live-encode
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers import GGUFQuantizationConfig, WanTransformer3DModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "wan-ti2v-5b-diffusers")
GGUF = os.path.join(ROOT, "models", "gguf", "Wan2.2-TI2V-5B-Q6_K.gguf")
LORA = os.path.join(ROOT, "models", "lora", "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors")
EMBEDS = os.path.join(ROOT, "assets", "embeddings.safetensors")
PRESETS = os.path.join(ROOT, "assets", "presets.json")

DTYPE = torch.bfloat16

# CLAUDE.md §5 — generation sizes. NOT the output size; export upscales to
# 1350x1800 / 1200x1800. Both must be divisible by 32 (VAE 16 x patch 2).
GEN_SIZES = {"3:4": (768, 1024), "2:3": (768, 1152)}
OUT_SIZES = {"3:4": (1350, 1800), "2:3": (1200, 1800)}


def pick_aspect(w, h):
    """Classify the source cover. 3:4 = 0.7500, 2:3 = 0.6667."""
    r = w / h
    return "3:4" if abs(r - 0.75) < abs(r - 2 / 3) else "2:3"


def load_embeds(preset_id, device):
    """Load trimmed embeddings and zero-pad back to max_sequence_length.

    The pad MUST match what the pipeline would have produced — see the contract note in
    tools/bake_embeddings.py. Mismatch here is a silent quality bug.
    """
    from safetensors import safe_open

    if not os.path.exists(EMBEDS):
        raise SystemExit(
            f"\n  {EMBEDS} not found.\n"
            f"  Run: python tools/bake_embeddings.py   (or pass --live-encode)\n"
        )

    with safe_open(EMBEDS, framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        max_len = int(meta.get("max_sequence_length", 512))
        if meta.get("storage") != "trimmed":
            raise SystemExit("  embeddings.safetensors is not in 'trimmed' storage form.")
        keys = list(f.keys())
        if preset_id not in keys:
            raise SystemExit(f"  preset '{preset_id}' not baked. Available: "
                             f"{[k for k in keys if k != '__negative__']}")
        pos = f.get_tensor(preset_id)
        neg = f.get_tensor("__negative__")

    def pad(t):
        out = torch.zeros(max_len, t.shape[-1], dtype=t.dtype)
        out[: t.shape[0]] = t
        return out.unsqueeze(0).to(device=device, dtype=DTYPE)

    print(f"  embeddings   : baked  (pos {tuple(pos.shape)} -> padded to {max_len})")
    return pad(pos), pad(neg)


def live_encode(preset_id, device):
    """A/B path: load the real encoder and encode on the fly."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    with open(PRESETS, encoding="utf-8") as f:
        presets = {p["id"]: p for p in json.load(f)["presets"]}
    neg_cache = os.path.join(ROOT, "assets", ".negative_prompt.txt")
    negative = open(neg_cache, encoding="utf-8").read() if os.path.exists(neg_cache) else ""

    print("  embeddings   : LIVE (loading encoder as bf16, ~11.2 GB)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
    # bf16, matching both the bake and the stock pipeline (which builds its text_encoder at
    # the pipeline dtype). float32 here would UPCAST the bf16 checkpoint to 22.7 GB, which
    # does not fit in 24 GB — and Windows WDDM pages VRAM to host RAM instead of raising
    # OOM, so it presents as 100% GPU at near-zero throughput. It would also invalidate the
    # A/B by comparing two different precisions.
    enc = UMT5EncoderModel.from_pretrained(
        MODEL_DIR, subfolder="text_encoder", dtype=DTYPE
    ).to(device).eval()

    def go(text):
        ti = tok([prompt_clean(text)], padding="max_length", max_length=512, truncation=True,
                 add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        ids, mask = ti.input_ids.to(device), ti.attention_mask.to(device)
        n = int(mask.gt(0).sum())
        with torch.no_grad():
            out = enc(ids, mask).last_hidden_state
        padded = torch.zeros_like(out)
        padded[:, :n] = out[:, :n]
        return padded.to(DTYPE)

    pos, neg = go(presets[preset_id]["prompt"]), go(negative)
    del enc
    torch.cuda.empty_cache()
    return pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--preset", default="ember_glow")
    ap.add_argument("--frames", type=int, default=33, help="must be 4n+1")
    ap.add_argument("--steps", type=int, default=None, help="default 4 with --lora, else 20")
    ap.add_argument("--guidance", type=float, default=None, help="default 1.0 with --lora, else 5.0")
    ap.add_argument("--shift", type=float, default=5.0, help="flow-match shift")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", action="store_true", help="apply the Turbo 4-step distill")
    ap.add_argument("--live-encode", action="store_true", help="use the real text encoder")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if (args.frames - 1) % 4 != 0:
        raise SystemExit(f"  --frames must be 4n+1 (VAE temporal compression). Got {args.frames}.")

    steps = args.steps if args.steps is not None else (4 if args.lora else 20)
    guidance = args.guidance if args.guidance is not None else (1.0 if args.lora else 5.0)
    device = "cuda"

    # Cover stem must be in the tag or two covers sharing a preset collide into one dir.
    stem = "".join(c if c.isalnum() else "-" for c in
                   os.path.splitext(os.path.basename(args.image))[0]).strip("-")[:28]
    tag = (f"{stem}__{args.preset}_{'turbo' if args.lora else 'stock'}"
           f"_{steps}s_cfg{guidance:g}_seed{args.seed}")
    outdir = args.out or os.path.join(ROOT, "out", tag)
    os.makedirs(outdir, exist_ok=True)

    src = Image.open(args.image).convert("RGB")
    aspect = pick_aspect(*src.size)
    gw, gh = GEN_SIZES[aspect]
    ow, oh = OUT_SIZES[aspect]

    print(f"\n  Phosphor — generation spike")
    print(f"  source       : {os.path.basename(args.image)}  {src.size[0]}x{src.size[1]}")
    print(f"  aspect       : {aspect}")
    print(f"  generate at  : {gw}x{gh}   (export will upscale to {ow}x{oh})")
    print(f"  preset       : {args.preset}")
    print(f"  frames       : {args.frames}  -> ping-pong {2*args.frames-2}")
    print(f"  steps / cfg  : {steps} / {guidance}")
    print(f"  lora         : {'Turbo rank-64' if args.lora else 'none'}")
    print(f"  out          : {outdir}")

    cover = src.resize((gw, gh), Image.LANCZOS)

    # --- pipeline ------------------------------------------------------------
    print("\n  Loading transformer from GGUF...", flush=True)
    t_load = time.time()
    transformer = WanTransformer3DModel.from_single_file(
        GGUF,
        quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
        dtype=DTYPE,
        config=MODEL_DIR,
        subfolder="transformer",
    )

    print("  Loading VAE...", flush=True)
    vae = AutoencoderKLWan.from_pretrained(MODEL_DIR, subfolder="vae", dtype=torch.float32)

    print("  Assembling pipeline (text_encoder=None)...", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_DIR,
        transformer=transformer,
        vae=vae,
        text_encoder=None,      # verified: dropped by load_module, never downloaded
        tokenizer=None,
        image_encoder=None,     # image_dim is null -> encode_image unreachable
        image_processor=None,
        dtype=DTYPE,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.shift
    )

    if args.lora:
        print("  Applying Turbo LoRA...", flush=True)
        try:
            pipe.load_lora_weights(LORA)
            pipe.fuse_lora()
        except Exception as e:
            raise SystemExit(
                f"\n  LoRA load failed: {e}\n"
                f"  If this mentions PEFT: pip install peft (it is a hard requirement of\n"
                f"  load_lora_weights, not an optional extra).\n"
                f"  If it mentions unexpected keys: this is a ComfyUI-format LoRA and\n"
                f"  diffusers needs a key remap.\n"
                f"  Either way, re-run without --lora to confirm the base pipeline works.\n"
            )

    pipe.to(device)
    # VAE decode is the OOM risk here, not the transformer: 33 frames at 768x1024 through a
    # 16x-spatial-compression decoder in fp32. WanImageToVideoPipeline does not proxy
    # enable_vae_tiling() in diffusers 0.40 — the methods live on the VAE itself.
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print(f"  Loaded in {time.time()-t_load:.1f}s", flush=True)

    # --- conditioning --------------------------------------------------------
    pos, neg = (live_encode if args.live_encode else load_embeds)(args.preset, device)

    # --- generate ------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    print(f"\n  Generating {steps} steps...\n", flush=True)
    t0 = time.time()

    def cb(pipe_, step, timestep, kw):
        el = time.time() - t0
        print(f"    step {step+1:>2}/{steps}   {el:5.1f}s   "
              f"({el/(step+1):4.1f}s/step)", flush=True)
        return kw

    result = pipe(
        image=cover,
        prompt=None,              # check_inputs rejects prompt + prompt_embeds together
        negative_prompt=None,
        prompt_embeds=pos,
        negative_prompt_embeds=neg,
        height=gh,
        width=gw,
        num_frames=args.frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        output_type="pil",
        callback_on_step_end=cb,
    )
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3

    frames = result.frames[0]
    for i, fr in enumerate(frames):
        fr.save(os.path.join(outdir, f"frame_{i:04d}.png"))

    print(f"\n  {'='*66}")
    print(f"  frames       : {len(frames)} written to {outdir}")
    print(f"  wall clock   : {elapsed:.1f}s  ({elapsed/steps:.1f}s/step)")
    print(f"  peak VRAM    : {peak:.2f} GiB / 24.0 GiB")
    print(f"  {'='*66}")
    print(f"\n  Inspect frame_0000.png — it should be near-identical to the source.")
    print(f"  Then check the faces across the sequence. That is the real test.\n")


if __name__ == "__main__":
    main()
