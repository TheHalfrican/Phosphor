# §9 Verification Pass — Results

**Date:** 2026-08-20
**Method:** diffusers `main` source read directly from GitHub raw; model and quant repo
metadata via the Hugging Face API. No local install — nothing here is from memory or from a
doc summary. Every claim cites the file and symbol it came from.

**Verdict: the §3 storage design holds. No budget revision needed. Cleared to build.**

---

## Q1 — Which pipeline class? → `WanImageToVideoPipeline`, not `WanPipeline`

`Wan-AI/Wan2.2-TI2V-5B-Diffusers/model_index.json` declares:

```json
{ "_class_name": "WanPipeline", "boundary_ratio": null, "expand_timesteps": true,
  "transformer_2": [null, null] }
```

**Do not follow that.** `WanPipeline.__call__` has **no `image` parameter at all** — it is
text-to-video only. Loading the repo as declared leaves no way to condition on the cover.

Image conditioning for TI2V-5B lives in `WanImageToVideoPipeline`
(`src/diffusers/pipelines/wan/pipeline_wan_i2v.py`), gated on the `expand_timesteps` config
flag. The source says so explicitly:

```python
if self.config.expand_timesteps:
    # wan 2.2 5b i2v use firt_frame_mask to mask timesteps   [sic - typo is upstream]
    latents, condition, first_frame_mask = latents_outputs
```

Construct the class explicitly rather than via `DiffusionPipeline.from_pretrained`.
`expand_timesteps: true` and `boundary_ratio: null` are read from `model_index.json` into
`__init__`, so those carry across correctly.

### No CLIP image encoder required

`transformer/config.json` has `"image_dim": null`, and the call site is gated on it:

```python
# only wan 2.1 i2v transformer accepts image_embeds
if self.transformer is not None and self.transformer.config.image_dim is not None:
    image_embeds = self.encode_image(image, device)
```

`encode_image` is therefore never reached. `image_encoder` and `image_processor` are both in
`_optional_components`. Pass neither. That is a CLIP ViT-H download avoided that the brief
did not account for.

### How the conditioning actually works

`prepare_latents` takes the single source frame as the entire video condition
(`video_condition = image`), VAE-encodes it, and builds a mask with frame 0 zeroed. The
denoise loop pins frame 0 to that condition and drives a **per-token timestep**:

```python
latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
```

Consistency check: `in_channels: 48 == out_channels: 48 == vae z_dim: 48`. There is no
channel-concat of mask+condition — that is the Wan 2.1 path, which needs 36ch. Blend, not
concat. Useful signal that the wiring is correct.

---

## Q2 — `prompt_embeds` with no text encoder → **CONFIRMED** (load-bearing)

The one the whole storage budget rested on. It holds, in three independent parts.

**1. `encode_prompt` short-circuits.** `self.text_encoder` is referenced in exactly one
method, `_get_t5_prompt_embeds`, called only under these guards:

```python
if prompt_embeds is None:                                           # -> _get_t5_prompt_embeds
if do_classifier_free_guidance and negative_prompt_embeds is None:  # -> _get_t5_prompt_embeds
```

Supply both tensors and the text encoder is never touched.

**2. `from_pretrained` never downloads it.** In `pipeline_utils.py`, an explicitly-passed
`None` drops the component before any fetch:

```python
def load_module(name, value):
    if value[0] is None: return False
    if name in passed_class_obj and passed_class_obj[name] is None: return False
    return True
init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}
```

**3. Construction still succeeds.** `text_encoder` has no default in `__init__` and is *not*
in `_optional_components`, so step 2 alone would break the constructor. Step 9 rescues it:

```python
missing_modules = set(expected_modules) - set(init_kwargs.keys())
if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
    for module in missing_modules:
        init_kwargs[module] = passed_class_obj.get(module, None)
```

Because the key was passed (with value `None`), it is in `passed_modules`, the subset test
passes, and `text_encoder=None` reaches `__init__`. Type-checking at step 10 skips `None`.
Confirmed viable.

**Caveat — `check_inputs` rejects the obvious call.** Passing `prompt` and `prompt_embeds`
together raises. Pass `prompt=None` and `negative_prompt=None`. Also pass `tokenizer=None`
explicitly or it will be downloaded — harmless at ~5MB, but pointless.

---

## Q3 — GGUF loading → confirmed, sizes match the budget exactly

`WanTransformer3DModel.from_single_file(path, quantization_config=GGUFQuantizationConfig(
compute_dtype=torch.bfloat16), torch_dtype=torch.bfloat16)`.

Measured from `QuantStack/Wan2.2-TI2V-5B-GGUF` (`X-Linked-Size`, actual bytes):

| File | Bytes | | §3 claimed |
|---|---|---|---|
| `Wan2.2-TI2V-5B-Q6_K.gguf` | 4,211,683,680 | 4.21 GB | ~4.2 GB ✓ |
| `VAE/Wan2.2_VAE.safetensors` | 1,409,400,960 | 1.41 GB | ~1.4 GB ✓ |

Q6_K is present. The full ladder Q2_K→Q8_0 exists if the quant choice is ever revisited. The
VAE ships from the same repo, unquantized, as §3 requires.

---

## Q4 — 4-step distill for TI2V-5B → the brief's suspicion was right, but there is a path

**Confirmed as suspected:** `lightx2v/Wan2.2-Lightning` contains **only A14B** variants
(`Wan2.2-I2V-A14B-*`, `Wan2.2-T2V-A14B-*`). Same for `lightx2v/Wan2.2-Distill-Loras` and
`Wan2.2-Distill-Models` — zero 5B/TI2V files. The §5 warning was accurate and remains so.

**But community distills of TI2V-5B specifically do exist, Apache-2.0:**

- `quanhaol/Wan2.2-TI2V-5B-Turbo` — full distilled model
- `hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF` — Turbo as Q6_K/Q8_0, **plus
  `Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors`**, i.e. the distill as a LoRA
  applicable to the stock 5B GGUF already being shipped. States that these "work fine with
  most LoRAs made for the regular 5B."
- `FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers` (Apache-2.0), with
  `Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors`

Reported settings: **4 steps, CFG 1**, Euler / UniPC / SA_Solver.

**Consequence to settle before baking embeddings:** CFG 1 means classifier-free guidance is
off, so `do_classifier_free_guidance` is False and the **negative prompt is never used**. If
this ships on a Turbo LoRA, the baked negative embedding is dead weight and §4's
negative-prompt handling becomes moot. Decide the LoRA question *before* finalizing
`bake_embeddings.py`, not after.

These are community repos and need quality evaluation on real cover art before being
committed to — but "20 steps vs 4" is exactly the gap §5 flagged as make-or-break, and the
option is real rather than hypothetical.

### Update 2026-08-20 — tested, and it does not work as planned

**A LoRA cannot be fused into a GGUF-quantized transformer.** `pipe.load_lora_weights()` +
`fuse_lora()` on the Q6_K transformer fails with:

    The size of tensor a (2520) must match the size of tensor b (3072) at dim 1

That is not a key-remap problem. Q6_K packs 256 weights into 210 bytes, so a logical
3072-wide row is stored as `3072/256 * 210 = 2520` bytes. The fuse is trying to add a
logical-shaped delta to block-packed bytes. Any LoRA-on-GGUF plan has the same issue.

Two viable routes if the distill is ever wanted:
1. **Ship the pre-distilled GGUF instead** — `hum-ma/Wan2.2-TI2V-5B-Turbo-Q6_K.gguf`. The
   distill is already in the weights, so there is no LoRA machinery, no PEFT dependency,
   and no size change. This is the clean option.
2. Load the LoRA *without* fusing, so PEFT applies it as a runtime side-path around the
   dequantized output. Untested here.

Also note `load_lora_weights()` requires **`peft`** installed — it is a hard dependency, not
an optional extra, and the error message says only "PEFT backend is required".

**Priority note:** this is now low-urgency. Stock 20-step measured **63.3s** at 768x1024 on a
4090, so the distill is a convenience, not an enabler. See the revised §5 in CLAUDE.md.

---

## Findings beyond the four questions

### 1. §4's embedding size math is wrong — but the conclusion survives

`max_sequence_length` defaults to **512** in `__call__`. A padded embedding is
`512 × 4096 × 2 bytes (fp16) = 4.19 MB` **each**. Eight presets plus the negative is
**~37.7 MB**, not the "well under 10MB" §4 claims. (§4 also says "sixteen presets plus the
negative comes to well under 10MB" while separately stating 4MB each — those two numbers
contradict each other.)

**The fix is free.** `_get_t5_prompt_embeds` trims to real token length before zero-padding:

```python
prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
prompt_embeds = torch.stack(
    [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
)
```

The padding is pure zeros. So **bake trimmed, zero-pad at load**. A ~60-token prompt is
~0.5 MB; eight presets plus a longer negative lands around 4–5 MB, and the <10MB line in §3
holds as written.

`bake_embeddings.py` must reproduce that trim-then-pad exactly, and the runtime pad width
must match whatever `max_sequence_length` is passed at call time. A mismatch here is a
silent quality bug, not a crash.

### 2. §5's "divisible by 32" is right — and the pipeline's own guard is too lax

`check_inputs` only enforces `height % 16 != 0 or width % 16 != 0`. That guard is
**insufficient**. The real constraint is VAE `scale_factor_spatial: 16` × transformer
`patch_size: [1, 2, 2]` → **32**. A dimension like 496 passes the pipeline's check and then
produces a fractional patch grid. Keep §5's rule; do not relax it to match the library.

Both defaults are safe: 480×640 → latent 30×40 → patch grid 15×20. 512×768 likewise.

### 3. 4n+1 frame rule confirmed

VAE `scale_factor_temporal: 4`. 33 frames → 9 latent frames. §5 stands.

### 4. `last_image` exists but is unusable here — §6 and the v2 InP plan both stand

`WanImageToVideoPipeline.__call__` accepts `last_image`, which looks like it would deliver a
genuine loop for free and make the ping-pong machinery unnecessary. **It would not.** In
`prepare_latents`, the `expand_timesteps` branch is evaluated first and discards it:

```python
if self.config.expand_timesteps:
    video_condition = image          # <- last_image never consulted
elif last_image is None:
    ...
else:
    last_image = last_image.unsqueeze(2)   # <- unreachable for TI2V-5B
```

The `last_image` path is reachable only for non-`expand_timesteps` (Wan 2.1 FLF2V) models.
Ping-pong (§6) remains necessary, the safe/unsafe preset split still governs v1, and
`Wan2.2-Fun-5B-InP` remains the v2 answer for directional motion. This confirms the existing
design rather than overturning it.

### 5. Official negative prompt located

`Wan-Video/Wan2.2` → `wan/configs/shared_config.py` → `wan_shared_cfg.sample_neg_prompt`.
Pull it verbatim from source at bake time rather than transcribing it — it is a long CJK
string, and a silent copy or encoding error would degrade output in a way that is hard to
trace. The same file confirms upstream default `frame_num = 81`; ours is 33 per §5,
deliberately.

---

## Recommended construction

```python
transformer = WanTransformer3DModel.from_single_file(
    "Wan2.2-TI2V-5B-Q6_K.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    transformer=transformer,
    text_encoder=None,      # never downloaded (load_module), never instantiated
    tokenizer=None,         # not needed once embeds are baked
    image_encoder=None,     # image_dim is null -> encode_image unreachable
    image_processor=None,
    torch_dtype=torch.bfloat16,
)
pipe(
    image=cover,
    prompt=None, negative_prompt=None,        # check_inputs rejects prompt + embeds together
    prompt_embeds=baked, negative_prompt_embeds=baked_neg,
    height=1024, width=768, num_frames=33,   # 3:4; use 768x1152 for 2:3 (see CLAUDE.md 5)
)
```

Note that `from_pretrained` still pulls the `vae/` and `scheduler/` folders from the Wan-AI
repo.

**Correction (2026-08-20):** an earlier draft of this doc suggested pointing `vae=` at the
`QuantStack` copy to avoid a duplicate download. That does not work as written — the two are
different *formats*, not just different copies. `QuantStack/.../VAE/Wan2.2_VAE.safetensors`
(1.41 GB) is the original Wan/ComfyUI layout; `AutoencoderKLWan.from_pretrained` expects the
diffusers layout in `Wan-AI/.../vae/` (2.82 GB, fp32). Use the Wan-AI one.

This has a **§3 consequence**: the budget line says "VAE ~1.4 GB", which is the ComfyUI-format
figure. Downloading the diffusers-format VAE costs 2.82 GB on disk, +1.4 GB against the stated
budget. Options: re-save it as bf16 at install time (gets back to ~1.41 GB, one-time local
conversion), or teach the loader to consume the ComfyUI-format file. Not urgent for the spike,
but §3's total is understated by ~1.4 GB until it is resolved.

Similarly, §4's "~6.7 GB text encoder" is the ComfyUI fp8 figure. Wan-AI ships UMT5-XXL as
**11.36 GB of BF16 shards** (5.68B params x 2 bytes; verified from the safetensors header).
That does not affect the shipping budget — it is dev-only and baked away — but the one-time
developer download is larger than §4 implies.

**Trap, hit for real on 2026-08-20:** passing `dtype=torch.float32` to
`UMT5EncoderModel.from_pretrained` *upcasts* that BF16 checkpoint to **22.72 GB**, which does
not fit in 24 GB alongside activations. Windows WDDM does not raise OOM in this situation —
it silently pages VRAM to host RAM, which presents as 100% GPU utilisation at near-zero
throughput for many minutes. Load the encoder as **bfloat16**. That is also the more faithful
choice: the stock pipeline is built with `torch_dtype=bfloat16`, so its text encoder is bf16
and `_get_t5_prompt_embeds` resolves its output dtype to bf16 as well. Baking in fp32 would
produce embeddings the live pipeline would never generate.

The VAE is the opposite case — genuinely **fp32** (705M params, 2.82 GB). Loading it as bf16
gives exactly 1.41 GB, matching §3's budget line.

## Open items — all closed 2026-08-20

1. ~~Evaluate the Turbo / FastWan LoRA~~ — **moot.** Stock 20-step measured 63.3s, so the
   distill is a convenience, not an enabler. And it cannot be applied as a LoRA anyway (see
   the GGUF block-packing note in Q4). If ever wanted, ship the pre-distilled GGUF.
2. ~~Confirm baked embeddings match live-encoded~~ — **PASS, bit-identical.**
   `tools/verify_embeddings.py` compares all 8 presets plus the negative at the tensor level:
   `torch.equal` true for every entry, max|diff| exactly 0.0. The trim-then-pad contract is
   correct and a **3.57 MB** asset validly replaces the 11.4 GB encoder. §4 is proven, not
   assumed. Re-run this after any preset edit — it is the regression test for §4.
3. ~~Confirm peak VRAM~~ — **8.48 GiB of 24** at 768×1024 x 33 frames with the Q6_K
   transformer. ~15 GiB of headroom; the predicted VAE-decode OOM never materialised
   (`vae.enable_tiling()` + `enable_slicing()` are wired in regardless).

## Measured end-to-end (RTX 4090, 768×1024, 33 frames, 20 steps, guidance 2.0)

| stage | result |
|---|---|
| Denoise | 45.3s (2.3s/step) |
| VAE decode | ~18s |
| Total generation | **63.3s**, peak 8.48 GiB |
| Frame 0 vs source | MAE 1.73/255 — conditioning near-exact |
| Ping-pong | 33 → 64 frames, both seams inside normal delta distribution |
| WebP q75 @ 1350×1800 | **6.18 MB**, 38.3 dB |
| GIF @ 1350×1800 | 44.41 MB (7.2×) |

The pipeline is proven end to end: cover in, seamless 1350×1800 animated WebP out.
   These are the real generation sizes (CLAUDE.md §5, revised 2026-08-20) — ~6.6–8.3× the
   attention compute of the 480×640 figure this doc was originally written against. VRAM
   headroom, not just wall-clock, is the thing that could force a rethink here.
