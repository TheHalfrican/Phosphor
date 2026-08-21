"""
Phosphor — inference pipeline.

Everything here is ported from the verified spike (tools/spike_generate.py,
tools/text_mask.py) rather than written fresh. The numbers and API choices below were all
measured on 2026-08-20 against an RTX 4090; see docs/pipeline-verification.md.

Load order matters and is not arbitrary:
  * transformer comes from the Q6_K GGUF, not the repo's fp32 shards (saves 20 GB)
  * text_encoder=None and tokenizer=None — verified to prevent the 11.4 GB download
    entirely, because `from_pretrained` drops explicitly-None components before fetching
  * image_encoder=None — the transformer's `image_dim` is null, so `encode_image` is
    unreachable and the CLIP vision tower is dead weight
"""

import os
import tempfile

import numpy as np
import torch
from PIL import Image, ImageFilter

DTYPE = torch.bfloat16

# §5 generation sizes. NOT the output size — export upscales to 1350x1800 / 1200x1800.
# Both must be divisible by 32: VAE scale_factor_spatial 16 x transformer patch 2. The
# pipeline's own check_inputs only enforces %16, which is too lax.
GEN_SIZES = {"3:4": (768, 1024), "2:3": (768, 1152)}

# §5, measured. The model card says 5.0; at 5.0 title treatments disintegrate within ~8
# frames. The cliff sits between 2.0 and 3.0.
DEFAULT_GUIDANCE = 2.0
DEFAULT_STEPS = 20
DEFAULT_FRAMES = 33          # must be 4n+1 (VAE temporal compression = 4)
FLOW_SHIFT = 5.0


def pick_aspect(w, h):
    r = w / h
    return "3:4" if abs(r - 0.75) < abs(r - 2 / 3) else "2:3"


class PhosphorPipeline:
    """Lazily-loaded diffusion pipeline plus the CRAFT text detector.

    Both models stay resident once loaded. Model load is ~5s and generation is ~63s, so
    reloading per request would be a large fraction of total time for no benefit.
    """

    def __init__(self, models_root, assets_root):
        # Two roots, not one: once installed, models sit in the user's app data (7 GB,
        # downloaded on first run, must survive an app update) while the baked embeddings
        # ship inside the app's resource directory. See inference_server._resolve_roots.
        self.models_root = models_root
        self.assets_root = assets_root
        self.model_dir = os.path.join(models_root, "wan-ti2v-5b-diffusers")
        self.gguf = os.path.join(models_root, "gguf", "Wan2.2-TI2V-5B-Q6_K.gguf")
        self.craft_weights = os.path.join(models_root, "craft", "craft_mlt_25k.pth")
        self.embeddings = os.path.join(assets_root, "embeddings.safetensors")
        self._pipe = None
        self._craft = None

    # -- diffusion ----------------------------------------------------------------------

    def load(self, log=print):
        if self._pipe is not None:
            return self._pipe

        from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig,
                               UniPCMultistepScheduler, WanImageToVideoPipeline,
                               WanTransformer3DModel)

        log("loading transformer (Q6_K GGUF)")
        transformer = WanTransformer3DModel.from_single_file(
            self.gguf,
            quantization_config=GGUFQuantizationConfig(compute_dtype=DTYPE),
            dtype=DTYPE,
            config=self.model_dir,
            subfolder="transformer",
        )

        log("loading VAE (fp32 - do not quantise)")
        vae = AutoencoderKLWan.from_pretrained(
            self.model_dir, subfolder="vae", dtype=torch.float32
        )

        log("assembling pipeline (text_encoder=None)")
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_dir,
            transformer=transformer,
            vae=vae,
            text_encoder=None,     # never downloaded; embeddings are baked (§4)
            tokenizer=None,
            image_encoder=None,    # image_dim is null -> encode_image unreachable
            image_processor=None,
            dtype=DTYPE,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=FLOW_SHIFT
        )
        pipe.to("cuda")
        # VAE decode is the memory spike, not the transformer: 33 frames at 768x1024
        # through a 16x-spatial-compression decoder in fp32.
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        self._pipe = pipe
        return pipe

    def _embeds(self, preset_id):
        """Load baked embeddings and zero-pad to max_sequence_length.

        The pad must reproduce what the pipeline would have produced. Verified
        bit-identical to live UMT5 output for all 8 presets plus the negative
        (tools/verify_embeddings.py) — a mismatch here is a silent quality bug, not a
        crash.
        """
        from safetensors import safe_open

        with safe_open(self.embeddings, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            max_len = int(meta.get("max_sequence_length", 512))
            if meta.get("storage") != "trimmed":
                raise RuntimeError("embeddings.safetensors is not in 'trimmed' storage form")
            if preset_id not in f.keys():
                raise RuntimeError(f"preset '{preset_id}' is not baked into embeddings.safetensors")
            pos, neg = f.get_tensor(preset_id), f.get_tensor("__negative__")

        def pad(t):
            out = torch.zeros(max_len, t.shape[-1], dtype=t.dtype)
            out[: t.shape[0]] = t
            return out.unsqueeze(0).to("cuda", DTYPE)

        return pad(pos), pad(neg)

    def generate(self, image_path, preset, guidance=DEFAULT_GUIDANCE, steps=DEFAULT_STEPS,
                 frames=DEFAULT_FRAMES, seed=0, progress=None, log=print):
        if (frames - 1) % 4 != 0:
            raise ValueError(f"frames must be 4n+1 (VAE temporal compression); got {frames}")

        pipe = self.load(log=log)
        src = Image.open(image_path).convert("RGB")
        aspect = pick_aspect(*src.size)
        gw, gh = GEN_SIZES[aspect]
        pos, neg = self._embeds(preset)

        def cb(_pipe, step, _t, kw):
            if progress:
                progress(step + 1, steps)
            return kw

        out = pipe(
            image=src.resize((gw, gh), Image.LANCZOS),
            prompt=None,               # check_inputs rejects prompt + prompt_embeds together
            negative_prompt=None,
            prompt_embeds=pos,
            negative_prompt_embeds=neg,
            height=gh,
            width=gw,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator(device="cuda").manual_seed(int(seed)),
            output_type="pil",
            callback_on_step_end=cb,
        )

        d = tempfile.mkdtemp(prefix="phosphor_frames_")
        for i, fr in enumerate(out.frames[0]):
            fr.save(os.path.join(d, f"frame_{i:04d}.png"))
        return {"frames_dir": d, "frame_count": frames, "width": gw, "height": gh}

    # -- text protection (§5a) ----------------------------------------------------------

    def load_craft(self, log=print):
        if self._craft is not None:
            return self._craft
        # sys.path is prepared at startup (inference_server.HERE) so this works both from
        # source and frozen, and so PyInstaller can see the import during analysis.
        from vendor.craft import CRAFT

        log("loading CRAFT text detector")
        net = CRAFT()
        sd = torch.load(self.craft_weights, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd)
        self._craft = net.to("cuda").eval()
        return self._craft

    @torch.no_grad()
    def detect_text(self, image_path, width, height, threshold=0.30, log=print):
        """Return a PNG mask path marking probable text pixels.

        CRAFT's raw character-region heatmap is used directly as a soft mask; its
        box/polygon postprocessing is skipped entirely because we want "which pixels are
        type", not "where are the word boxes".
        """
        net = self.load_craft(log=log)
        img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)

        long_side = 1280
        scale = long_side / max(width, height)
        tw = max(32, int(round(width * scale / 32)) * 32)
        th = max(32, int(round(height * scale / 32)) * 32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = np.asarray(img.resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0
        x = (x - mean) / std
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to("cuda")

        y, _ = net(t)
        region = y[0, :, :, 0].cpu().numpy()
        affinity = y[0, :, :, 1].cpu().numpy()
        # affinity fills the gaps between glyphs so a word becomes one region
        score = np.maximum(region, affinity)

        m = Image.fromarray(((score >= threshold) * 255).astype(np.uint8))
        m = m.resize((width, height), Image.BILINEAR)
        m = m.filter(ImageFilter.MaxFilter(7))            # cover anti-aliased glyph edges
        m = m.filter(ImageFilter.GaussianBlur(5))         # feather, avoids a visible seam

        path = os.path.join(tempfile.mkdtemp(prefix="phosphor_mask_"), "mask.png")
        m.save(path)
        coverage = float(np.asarray(m, dtype=np.float32).mean() / 255.0)
        return {"mask_path": path, "coverage": coverage}

    def protect(self, frames_dir, source_path, mask_path, log=print):
        """Composite source pixels back wherever the mask is high.

        `mask_path` is the USER-CORRECTED mask, not CRAFT's raw output. Detection failures
        are silent — a missed region ships mangled type rather than erroring — so the UI
        owns the final say.
        """
        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        if not files:
            raise RuntimeError(f"no frames in {frames_dir}")
        size = Image.open(os.path.join(frames_dir, files[0])).size

        src = np.asarray(
            Image.open(source_path).convert("RGB").resize(size, Image.LANCZOS), dtype=np.float32
        )
        mask = np.asarray(
            Image.open(mask_path).convert("L").resize(size, Image.LANCZOS), dtype=np.float32
        )[..., None] / 255.0

        out = tempfile.mkdtemp(prefix="phosphor_protected_")
        for f in files:
            a = np.asarray(Image.open(os.path.join(frames_dir, f)).convert("RGB"), dtype=np.float32)
            Image.fromarray((a * (1 - mask) + src * mask).astype(np.uint8)).save(
                os.path.join(out, f)
            )
        return {"frames_dir": out, "frame_count": len(files)}
