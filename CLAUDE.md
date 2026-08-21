# Phosphor — Animated Game Cover Art Generator

> Single-purpose Windows desktop app: drop in a static game cover, get a seamlessly
> looping animated cover out. Runs entirely locally on the user's GPU.
>
> Published under **The Halfrican Software**. Tauri `productName: "Phosphor"`,
> identifier `software.halfrican.phosphor`.
>
> The name refers to CRT phosphor afterglow — the decay trail that makes an old display
> feel alive rather than merely lit. Useful for tone when writing UI copy: understated,
> warm, analog. Not neon-cyberpunk maximalism; that's RetroVoid's register, and these
> are siblings rather than twins.

---

## 1. What this is (and isn't)

**Is:** A narrow tool that takes one static image, applies one of a fixed set of subtle
motion presets, and exports a seamless loop as animated WebP (primary) or GIF (compat).

**Is not:**

- A general video generation frontend. No freeform prompting in v1. No workflow graph.
  No model browser. If the user wants ComfyUI, they know where to find it.
- A video editor. No timeline, no trimming UI, no keyframes.
- A batch processing pipeline. One image at a time is fine for the use case.

Scope discipline matters more than usual here, because the underlying model can do a
hundred things and every one of them is a tempting feature. The value of this tool is
that it does exactly one thing without making the user learn a diffusion pipeline.

**Target user:** someone curating a game library in a launcher (RetroVoid, Playnite,
LaunchBox, ES-DE) who wants their grid to feel alive. They have a 3:4 box scan. They
want it to breathe. They do not want to learn what a sampler is.

---

## 2. Architecture

```
┌─────────────────────────────┐
│  Tauri app (Rust + React)   │   UI, file I/O, ffmpeg invocation,
│                             │   loop assembly, preset selection
└──────────┬──────────────────┘
           │ JSON Lines over stdin/stdout
┌──────────▼──────────────────┐
│  Python sidecar (frozen)    │   diffusers + torch, GGUF model load,
│  inference_server.py        │   denoise loop, returns PNG frames
└─────────────────────────────┘
```

**Why a Python sidecar rather than native Rust:** there is no Rust inference stack that
can run a 5B video DiT. candle/burn don't cover this architecture. This is not a
decision to revisit.

**Why a custom diffusers server rather than bundling ComfyUI:**

1. ComfyUI is GPL-3.0. Sidecar-over-IPC is *probably* aggregation rather than
   derivation, but "probably" is a bad foundation for a distribution decision.
   **Note:** this originally read "intended for commercial release ... a bad foundation
   for a paid product". Phosphor is now free and Apache-2.0 (§10), which does *not*
   invalidate the reasoning: refusing an **imposed** copyleft is exactly what left the
   licence free to be chosen deliberately later. RetroVoid is the paid product; Phosphor
   is a free tool in the same ecosystem.
2. We bypass the text encoder entirely (see §4), which was most of what ComfyUI's
   ecosystem was buying us.
3. Installer size. ComfyUI drags in a large dependency tree we don't use.

**IPC protocol:** newline-delimited JSON over the sidecar's stdin/stdout. Not HTTP —
avoids port conflicts, avoids binding a socket on the user's machine, avoids firewall
prompts. Requests are `{"op": "...", "id": "...", ...}`. The sidecar emits progress
events `{"id": ..., "type": "progress", "step": 3, "total": 8}` and a terminal
`{"id": ..., "type": "result" | "error", ...}`.

Frames come back as a path to a temp directory of PNGs, not as base64 over the pipe.
Serializing ~33 frames of 480x640 through stdout is needless overhead.

---

## 3. Storage budget

This is a hard design constraint. The user should not need 25GB free to install this.

| Component | Size | Notes |
|---|---|---|
| Python runtime + torch + CUDA wheels | ~5–7 GB | The floor. Unavoidable. |
| `Wan2.2-TI2V-5B` Q6_K GGUF | ~4.2 GB | From `QuantStack/Wan2.2-TI2V-5B-GGUF` |
| `wan2.2_vae.safetensors` | ~1.4 GB | Do **not** quantize — see below |
| Precomputed prompt embeddings | <10 MB | Ships with the app |
| ffmpeg (LGPL build, webp+gif only) | ~80 MB | |
| **Total** | **~11–13 GB** | |

Roughly half of that is torch, which reframes the problem: there is no point
micro-optimizing the model files further. Q4 would save 800MB against a 12GB install
while measurably degrading temporal coherence. Not worth it.

**Quant choice: Q6_K.** Q8_0 (5.4GB) buys almost nothing on a 24GB card. Below Q5_K_M
temporal artifacts appear, and unlike still-image quantization damage, these *move* —
which is far more visible.

**The VAE stays at full precision.** TI2V-5B uses the high-compression Wan 2.2 VAE. At
those compression ratios the decoder is doing a lot of load-bearing reconstruction work
and quantizing it shows up immediately as mush.

**Model acquisition:** do not ship weights in the installer. First-run downloads them
from Hugging Face with a progress UI and SHA256 verification. Installer stays small,
and it keeps model licensing (Apache 2.0) cleanly separate from app distribution.

---

## 4. The text encoder trick

**This is the central design decision. Read this section carefully before changing
anything about prompt handling.**

The stock setup needs `umt5_xxl_fp8_e4m3fn_scaled.safetensors` at ~6.7 GB — as large as
the quantized model itself — purely to turn prompt strings into conditioning tensors.

We don't ship it. Because the app only offers a fixed set of motion presets, every
prompt is known at build time. So:

1. A **build-time script** (`tools/bake_embeddings.py`, developer machine only) loads
   UMT5-XXL once, encodes each preset prompt plus the shared negative prompt, and
   writes the tensors to `assets/embeddings.safetensors`.
2. At runtime the sidecar loads that file and passes tensors straight into the pipeline
   as `prompt_embeds` / `negative_prompt_embeds`. The text encoder is never
   instantiated.

UMT5-XXL has a 4096-dim hidden state. A ~60-token prompt is roughly 500KB in fp16; even
padded to Wan's full 512-token context it's 4MB. Sixteen presets plus the negative
comes to well under 10MB, replacing 6.7GB.

**Implications to respect:**

- Adding a preset requires re-running the bake script and shipping a new asset file.
  That's the trade. Document it in the contributor notes.
- Do not add a "custom prompt" text field. It cannot work without the encoder, and a
  disabled-looking field is worse than no field.
- If freeform prompting is ever wanted, it's an **optional download** ("Custom Prompt
  Pack") that fetches a GGUF'd UMT5 (~3–4 GB) and enables a separate code path. Most
  users never trigger it. This is v3 at the earliest.
- The negative prompt should be Wan's official default (a long CJK string in the
  upstream repo). Pull it verbatim from `Wan-AI/Wan2.2-TI2V-5B` rather than inventing
  one — it's tuned for this model family.

---

## 5. Generation parameters

- **Resolution — generation size and output size are different numbers.** Keep them
  distinct everywhere in the code; conflating them is the easiest mistake to make here.

  **Output targets (what ships):** **1350×1800** for 3:4 sources, **1200×1800** for 2:3.
  These are SteamGridDB-standard grid dimensions. The target display is a large-format
  TV running Steam and similar managers, *not* a ~350px launcher thumbnail — output is
  viewed near full size, so upscaling softness is visible and matters.

  **Generation sizes (what the pipeline actually renders):** **768×1024** for 3:4,
  **768×1152** for 2:3. Lanczos-upscale to the output target on export (1.76× and 1.56×
  respectively).

  **Why not generate at the output size directly:** you can't. Generation dimensions must
  be divisible by **32** — VAE `scale_factor_spatial: 16` × transformer `patch_size` 2 —
  and none of `1200`, `1350`, `1800` satisfy that (`1800 % 32 = 8`, `1350 % 32 = 6`,
  `1200 % 32 = 16`). A resize on export is mandatory, not a shortcut. Note that the
  pipeline's own `check_inputs` only enforces `% 16`; that guard is too lax — a size like
  496 passes it and then yields a fractional patch grid. Enforce 32 ourselves.

  The nearest valid native sizes (1344×1792 / 1216×1824) are **~2.5–2.7× the model's
  trained pixel budget** (704×1280 = 901K px) and ~55–60× the attention compute of the
  old 480×640 baseline. That is both artifact-prone — Wan models duplicate/"twin" content
  when pushed well past training resolution — and far too slow to be interactive. The
  768×1024 / 768×1152 pair sits at 0.87× and 0.98× of the native budget: inside the
  training distribution, with only a mild upscale left to cover.

  This supersedes the earlier "do not offer 720p, it's pure waste" rule, which assumed a
  ~300–400px display size that does not apply here.
- **Frame count:** must be `4n+1` (VAE temporal compression). **33 frames** is the
  default — 1.4 seconds at 24fps, which ping-pongs to 64 frames. 49 is already longer
  than a cover loop needs.
- **fps:** 24 (TI2V-5B native).
- **Steps: 20 works fine. Start there.**

  **Measured 2026-08-20** on the target hardware (RTX 4090, Q6_K transformer, 768×1024,
  33 frames, 20 steps): **63.3s wall clock** (45.3s denoise at 2.3s/step, ~18s VAE
  decode) and **8.48 GiB peak VRAM of 24**. That is comfortably interactive for a
  one-cover-at-a-time tool, and leaves ~15 GiB of headroom.

  An earlier revision of this section claimed the 4-step distill was "load-bearing"
  because the resolution increase costs ~7× the attention compute. **That was wrong** —
  it reasoned from a compute ratio without measuring the absolute number. 7× a small
  number is still a small number. The distill is a nice-to-have that would take a minute
  down to ~15s; it is not what makes the tool viable. Do not let it block anything.

  The real constraint turned out to be **motion quality, not speed** — see the drift note
  under Guidance below.

  Verified 2026-08-20 (see `docs/pipeline-verification.md`): the lightx2v Lightning /
  Distill repos are **A14B-only** — the caution in the original draft was correct. But
  Apache-2.0 community distills of TI2V-5B specifically *do* exist, including
  `Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors`, a rank-64 LoRA applicable to the
  stock 5B GGUF we already ship. Reported at 4 steps / CFG 1.

  **CFG 1 disables classifier-free guidance**, which means the negative prompt is never
  evaluated — so this decision reaches into §4. Settle it before finalizing
  `bake_embeddings.py`, or a baked negative embedding becomes dead weight.

  Fallback if the distill fails quality evaluation on real cover art: 20 steps at the
  full generation size, accepting the slower generation.
- **Guidance: this is the single most important quality knob. Do not ship the model-card
  default.**

  The original note here — "lower guidance generally produces gentler motion, which is
  what we want" — was right, and badly understated. Measured 2026-08-20 on a real cover
  (BO3 Mod Tools, 3:4, ember_glow, 20 steps, 33 frames, seed 0), where every number is
  mean absolute difference on a 0–255 scale:

  | guidance | drift avg | overall motion | **logo band** | face band | title legible? |
  |---|---|---|---|---|---|
  | 1.5 | 17.65 | 11.78 | **19.43** | 7.55 | yes, crisp |
  | 2.0 | 19.53 | 13.46 | **22.65** | 8.07 | yes, crisp |
  | 3.0 | 24.96 | 19.37 | **49.25** | 8.99 | no — letters melting |
  | 5.0 (model card) | 44.11 | 34.49 | **72.31** | 17.12 | no — destroyed |

  **There is a cliff between 2.0 and 3.0.** The logo band more than doubles across that
  step while the face band barely moves. Below it the title lockup is pristine across all
  33 frames; above it letters visibly deform ("CALL OF DUTY" -> "CALL OF DL/TY" by frame
  32). Treat **2.5 as a hard ceiling** and default to **2.0**.

  **Small text is lost even in the safe range**, and guidance cannot save it — see §5a.

  **The failure mode is text, not faces.** §5 originally warned about faces melting;
  faces were the *most* stable region in both runs (17.12 / 8.07, roughly half the global
  average). It is the logo band that blows up — 72.31, more than double the global mean.
  This matters more than the face concern ever did, because essentially every game cover
  has a title treatment and a viewer reads garbled type instantly.

  Practical consequences:
  - Default guidance should be **~2.0**, not the model-card 5.0.
  - Any future preset or parameter change must be regression-tested against **text
    legibility on a title-heavy cover**, not just "does it look like it moves".
  - Motion strength is essentially a guidance dial, which resolves the §11 backlog
    question of whether a motion-strength slider needs its own mechanism. It does not.

Prompts should be written for **restraint**. The failure mode that makes this whole
category look cheap is a character's face melting. Drifting fog, parallax on a
background layer, flickering neon, rippling water, subtle rim-light shifts. Short clips,
minimal motion. Both faster and safer.

---

## 5a. Text destruction — and the fix

**This changed the v1 plan.** Measured 2026-08-20 across 7 covers. The problem is real and
unfixable by tuning; the fix is compositing, and detection is solved via CRAFT.

TI2V-5B cannot preserve fine glyph structure through encode -> denoise -> decode. On
"Assassin's Creed" the thin letterspaced **CREED** renders as **NMEAV** — *identically at
guidance 1.0, 1.5 and 2.0*. Guidance 1.0 means classifier-free guidance is fully disabled,
so this is not a guidance artifact and **no parameter tuning fixes it.** The Desolate Hope's
thin vertical title collapses the same way at every setting.

What survives vs what does not:

| survives | destroyed |
|---|---|
| Large heavy display type (BO3 "CALL OF DUTY", "ASSASSIN'S", Halo "HALO") | Thin letterspaced type ("CREED") |
| High-contrast solid lettering | Small subtitles ("MOD TOOLS", "The Full Compilation!") |
| | Thin vertical/rotated titles |

Glyph **weight and size** decide survival, not guidance, not preset, not steps.

### The fix that works: composite, don't generate

Verified working. Detect the type in the source, then paste those original pixels back over
every generated frame with a feathered mask. A crude rectangle plus a 12px feather restored
"CREED" perfectly with no visible seam. This is also what a viewer expects — a cover's title
lockup is a static overlay, not something that should drift.

`tools/text_mask.py` implements the compositing half. It works.

### Detection: CRAFT (solved 2026-08-20)

**Two classical-CV attempts failed first.** Recorded so nobody retries them:

- Edge-density + morphology **over-detects** — 20.4% of the BO3 frame, grabbing characters
  and fire, because dense edges from artwork are indistinguishable from dense edges from
  glyphs.
- Adding a colour-consistency filter (type is near-monochrome) **over-corrects to zero**,
  because morphological closing merges glyphs together with the background gaps between
  them, so the variance measured is the background's rather than the strokes'.

**CRAFT** (Character Region Awareness For Text detection, clovaai) works. It is **MIT**,
**torch-native** — no new runtime, which PP-OCRv5 would have needed (Paddle or ONNX) — and
trained on *scene* text rather than documents, which is the right category for stylised
cover titles. ~83 MB of weights, 20.8M params.

Only the model definition is vendored (`sidecar/vendor/craft/`, MIT, patched for modern
torchvision which removed `model_urls`). **We use CRAFT's raw character-region heatmap
directly as a soft mask and skip its box/polygon postprocessing entirely** — we want "which
pixels are type", not "where are the word boxes". Far less code and a better fit.

Measured coverage, and it correctly ignored artwork in every case:

| cover | mask | caught |
|---|---|---|
| Assassin's Creed | 8.5% | both title lines |
| BO3 | 14.4% | both lockups; **not** the fire or characters |
| Desolate Hope | 3.7% | rotated *vertical* title |
| Halo | 6.3% | title + subtitle |
| TUNG | 11.4% | title |
| Jumpscare Sim | 12.6% | title + small subtitle |

End-to-end result: "NMEAV" -> "CREED", Desolate Hope's vertical title fully restored, BO3's
"MOD TOOLS" recovered from an empty bar. Loop seams stay seamless (1.24 / 1.94 against a
typical max of 3.14) and **the file gets smaller** — 5.70 MB vs 6.18 MB, because static text
regions compress better across frames.

### Still to do

Per the chosen hybrid approach, the mask must be **user-correctable in the UI**. Detection
failures are silent — a missed region does not error, it ships mangled type — so the mask
has to be visible and editable regardless of how good CRAFT is. `thresh` (default 0.30) is
the sensitivity dial.

Also unresolved: what happens when a cover's *artwork* is largely lettering (a logo-only
cover). Protecting 40%+ of the frame may leave too little moving for the loop to read as
animated. Untested.

---

## 6. Loop assembly

Generated video is not a loop. Ping-pong makes it one.

**Verified 2026-08-20** on a real 33-frame generation. The 2N-2 sequence produced 64
frames with zero duplicates, and neither seam is detectable: against a typical
frame-to-frame delta of mean 2.45 / max 4.22, the turnaround seam measured **1.71** and
the loop point **2.38** — both inside the ordinary distribution rather than outliers. The
endpoint-dropping rule below is what makes that work; `tools/build_loop.py` asserts it on
every build.

For `N` generated frames, the output sequence is:

```
[0, 1, 2, ..., N-1, N-2, N-3, ..., 2, 1]     → length 2N - 2
```

Both endpoints are dropped on the reverse pass. Keeping frame `N-1` produces a duplicate
frame at the turnaround; keeping frame `0` produces one at the loop point. Either reads
as a visible hitch.

### The preset constraint (important)

Ping-pong only works for **non-causal** motion. Anything the viewer reads as having a
direction looks like a rewind, and it is immediately obvious.

**Safe to ping-pong:** parallax drift, water ripple, glow/neon pulse, cloth or hair
sway, swirling fog, lens flare drift, subtle camera push, rim-light shift, shimmer.

**Not safe:** rain, rising smoke, falling leaves, snow, a character walking, anything
falling or flowing one way, particle emission.

**v1 ships only from the safe list.** This is a design constraint, not a limitation to
work around. Directional motion needs `Wan2.2-Fun-5B-InP` (same 5B base, same GGUF
sizes) which accepts both a start and end frame — set both to the source cover and you
get a genuine loop. That's a +4GB optional model download and a v2 feature.

### Starter preset list

1. Slow Drift — gentle parallax push on the background
2. Neon Flicker — pulsing sign / emissive glow
3. Water Ripple — reflective surface disturbance
4. Fog Roll — low ambient mist swirl
5. Ember Glow — warm light breathing
6. Cloth Sway — cape/flag/hair micro-motion
7. Starfield Shimmer — subtle twinkle, good for sci-fi covers
8. Rain Sheen — light glinting on a wet surface (*sheen*, not falling rain)

---

## 7. Export

ffmpeg is invoked from Rust as a bundled sidecar binary. **Use an LGPL build with only
webp and gif support** — we need none of the GPL-encumbered codecs (no x264, no x265),
which keeps commercial distribution clean and the binary small.

**The export stage also owns the scale to the output target** (§5). Frames arrive from
the sidecar at generation size and must be scaled here. Do the scale in the same ffmpeg
pass as the encode; a separate PNG resize pass would write a second full set of frames to
disk for nothing.

Use `lanczos`. `bicubic` is visibly softer at these ratios and `neighbor`/`bilinear` are
not worth considering for artwork.

### Two output scales, added 2026-08-21

The user picks **Full** or **Half**, and the aspect ratio still comes from the source, never
from the user. So the four sizes are two ratios × two scales:

| | 3:4 | 2:3 |
|---|---|---|
| Full | 1350×1800 | 1200×1800 |
| Half | 675×900 | 600×900 |

**Why.** RetroVoid hitches while scrolling a grid of full-size animated covers. Decoding is
per-pixel, and halving each axis is a **quarter** of the pixels: 2.43 Mpx → 0.61 Mpx per
frame at 3:4.

**Half is closer to native than Full is.** Generation is 768×1024 / 768×1152, so Full
*upscales* 1.76× / 1.56× while Half *downscales* to 0.88× / 0.78×. Half discards resolution
the model never produced rather than detail. Checked on Halo's title band at equal display
size: Half is indistinguishable from Full, and the thin "CAMPAIGN EVOLVED" subtitle is
marginally cleaner.

**The decode saving is 4×; the file-size saving is only about 2×.** These are different
numbers and it is easy to conflate them. Measured 2026-08-21, 64 frames, WebP q75:

| cover | Full | Half | half as % of full |
|---|---|---|---|
| BO3 3:4, protected | 5.97 MB | 2.81 MB | 47% |
| Halo 2:3 | 3.78 MB | 1.38 MB | 36% |
| Halo 2:3, protected, in-app | 3.49 MB | 1.24 MB | 36% |

Half does not land at a quarter of the file size, because downscaling raises per-pixel
entropy and the frame count is unchanged at 64. Do not "correct" the UI's estimates to
full/4. The spread between covers (36% vs 47%) is content, so treat any single figure as an
order of magnitude.

GIF halves harder: 40.36 → 11.18 MB on BO3 (28%), 34.24 → 8.52 MB on Halo (25%).

**Half is the default** (set 2026-08-21). An earlier revision of this section defaulted to
Full and argued the §5 large-format-TV point against it. That was a misread: §5 is about the
*generation* size, which has not changed. What Half drops is upscale, not detail, and at
equal display size the two are indistinguishable. Covers are seen in a grid far more often
than they are seen large, so the common case should be the smooth one. Full is one click
away.

The default lives in **exactly one place**, `App.tsx`'s `half` state. `OutScale` has no
`Default` impl on purpose, so there is no second answer to go stale. The Rust command's
`half.unwrap_or(false)` is a *fallback* rather than a default: a caller that omits the field
gets Full, because failing toward the larger export is the recoverable direction.

### Animated WebP (primary)

```
ffmpeg -framerate 24 -i frame_%04d.png \
  -vf "scale=1350:1800:flags=lanczos" \
  -c:v libwebp_anim -lossless 0 -q:v 75 -loop 0 \
  -preset picture out.webp
```

(`scale=1200:1800` for 2:3 sources.)

Full color, alpha support, good compression. Renders in an `<img>` tag with zero effort —
trivial for Tauri/React launchers.

**File size — measured 2026-08-20, 64 frames at 1350×1800 (2.67s loop):**

| `-q:v` | size | per frame | PSNR |
|---|---|---|---|
| 50 | 4.52 MB | 72 KB | 36.5 dB |
| 65 | 5.49 MB | 88 KB | 37.5 dB |
| **75 (default)** | **6.18 MB** | 99 KB | **38.3 dB** |
| 85 | 9.20 MB | 147 KB | 40.3 dB |
| 95 | 17.89 MB | 286 KB | 42.9 dB |

An earlier revision of this section predicted "tens of MB" at q75 and flagged size as a
likely UX problem. **That was too pessimistic** — 6.18 MB is unremarkable for a local
library asset. The curve is gentle from 50→75 then steepens sharply; q95 costs 2.9× the
size of q75 for 4.6 dB. **q75 is the right default and does not need a UI slider in v1.**

The size concern that *does* survive is aggregate, not per-file: a 500-game library at
6 MB each is ~3 GB. Worth a passing thought if bulk export ever leaves §11.

GIF at the same resolution is **44.41 MB — 7.2× the WebP.** That is severe enough to
warrant an explicit size warning in the UI before a GIF export runs.

### GIF (compatibility export)

Two-pass, and the flags matter:

```
ffmpeg -i frame_%04d.png \
  -vf "scale=1350:1800:flags=lanczos,palettegen=stats_mode=diff" palette.png

ffmpeg -framerate 24 -i frame_%04d.png -i palette.png \
  -lavfi "scale=1350:1800:flags=lanczos[s];[s][1:v]paletteuse=dither=bayer:bayer_scale=5" \
  -loop 0 out.gif
```

Scale **before** `palettegen`, and apply the identical scale in both passes — the palette
must be built from the same pixels it will be applied to, or the color selection is subtly
wrong.

`stats_mode=diff` allocates the 256-color budget toward the *changing* regions rather
than the static majority of the frame — which is exactly the content profile of an
animated cover, where most pixels never move. Large quality win over the default.

`dither=bayer` is both smaller than error-diffusion and aesthetically correct for retro
cover art. `bayer_scale=5` is a reasonable middle; expose it if users complain.

Position GIF in the UI as the compatibility option, not the default. 256 colors will
destroy gradient-heavy cover art and the files are much larger — and at 1350×1800 that
"much larger" is now severe enough to be worth a size warning in the UI before export.

---

## 7a. UI — implemented from Claude Design

The interface is implemented from a Claude Design canvas, not designed ad hoc. See §12 for
how to reopen it.

**Nocturne** is the design system (`src/nocturne.css`, ported from the project's
`_ds/nocturne-*/styles.css`). Its rules are not decoration and should be respected when
adding anything:

- Every colour, space, radius and shadow comes from a token. Never hard-code a hex.
- **Primary buttons are an accent outline, never a fill.** Do not flood the accent.
- Dark ground `#161826`, one accent `#9184d9` (blurple), Inter at weight 500.
- Rules fade to transparent over 48px at each end.
- Compact 0.7x density — the system is dense on purpose.

**One deliberate deviation:** the upstream sheet `@import`s Inter from Google Fonts.
Phosphor runs entirely locally and must look right offline, so the remote import is dropped
and the stack falls back to system-ui. **Bundle Inter as a local font before release.**

### Screens

| Design | Implemented as | Notes |
|---|---|---|
| 1a Main (cover-first) | `App.tsx` `screen === "main"` | Chosen over 1b — puts the artwork first |
| 1c Empty / drop | `DropZone` | Native drop via `onDragDropEvent`; the browser's DataTransfer carries no real path |
| 1d Mask editor | `MaskEditor.tsx` | The load-bearing screen — see §5a |
| 1e Generating | `screen === "generating"` | Scan sweep is tied to real step progress, not a decorative loop |
| 1f Export | `ExportDialog` | WebP default; GIF carries its 7x size warning |
| 1g First-run download | `screen === "setup"` | UI only — the downloader itself is not wired yet |
| 1h Settings | `screen === "settings"` | Deliberately small, per the design |

Design 1b (preset list beside a smaller cover) was not implemented; 1a covers the same job.

### Motion strength maps to guidance, with a measured ceiling

`Gentle 1.5 · Standard 2.0 · Bold 2.5`. **Bold is 2.5, not 3.0, deliberately** — §5 measured
the cliff between 2.0 and 3.0, and the UI must not offer a setting known to wreck the
artwork. If that mapping is ever changed, re-read §5 first.

### Wide window, added 2026-08-21

The canvas artboards are all 460px, and maximised that composition became a narrow centred
column: measured at 1292x732 CSS px the cover rendered 208x310, about **7% of the screen**,
with 542px of dead space each side. Because the cover is height-driven and the controls sit
below it, widening the window barely grew the artwork.

In a landscape window the non-artwork controls now sit **beside** the cover, which hands the
full column height to the artwork: **208x310 -> 393x590**, 7% to 24% of the window. This is
screen 1a's own premise (artwork first) holding at a size the artboards never covered.

**The trigger is window shape, not width.** Which layout gives the bigger cover depends on
aspect, because the portrait layout spends ~250px of *height* on controls while the wide one
spends ~380px of *width*. `@media (min-width: 720px) and (min-aspect-ratio: 5 / 4)`. A width
breakpoint alone was tried first and is wrong: it forces the split onto tall narrow windows
where it makes the cover smaller.

The grid's `max-width` tracks the viewport height rather than being a fixed number, so the
artwork column is exactly as wide as a full-height 3:4 cover needs. A flat cap starves the
cover on a tall landscape window and opens dead space on a short wide one.

Applies to 1a and 1e. 1d was already two-column; the dialogs are centred and unaffected.

### Window

460x720, `decorations: false`, custom title bar. Tauri v2 denies permissions by default, so
everything the UI touches is listed explicitly in `src-tauri/capabilities/default.json`.

---

## 8. Layout (as built)

```
phosphor/
├── src-tauri/
│   ├── src/
│   │   ├── main.rs
│   │   ├── lib.rs            # tauri commands + the pipeline order
│   │   ├── sidecar.rs        # spawn + JSONL protocol (§2)
│   │   ├── loop_build.rs     # ping-pong sequencing (§6) + its unit tests
│   │   ├── encode.rs         # ffmpeg: upscale + webp/gif (§7)
│   │   └── models.rs         # first-run download: resume, SHA256, status (§3)
│   ├── capabilities/         # Tauri v2 permissions — denied by default
│   └── binaries/             # ffmpeg (LGPL); frozen sidecar NOT yet built
├── src/                      # React UI (§7a)
│   ├── App.tsx               # screens 1a/1c/1e/1f/1g/1h
│   ├── MaskEditor.tsx        # screen 1d — text protection
│   ├── nocturne.css          # design system, ported from Claude Design
│   └── App.css               # app layout
├── sidecar/
│   ├── inference_server.py   # JSONL protocol; stdout is protocol ONLY
│   ├── pipeline.py           # diffusion + CRAFT
│   ├── vendor/craft/         # CRAFT model def (MIT, vendored + patched)
│   └── requirements.txt
├── tools/                    # DEV ONLY — see §12 for the full list
├── assets/
│   ├── embeddings.safetensors  # 3.57 MB, committed — replaces 11.4 GB
│   ├── presets.json
│   └── models.json           # download manifest + real SHA256s
├── docs/pipeline-verification.md
├── bin/ffmpeg.exe            # dev copy (gitignored)
├── models/                   # downloaded, ~21 GB (gitignored)
├── out/                      # generated frames + loops (gitignored)
├── setup.ps1
└── CLAUDE.md
```

---

## 9. Verify before implementing

These are assumptions from a design conversation, not confirmed APIs. **Check current
diffusers docs before writing the pipeline code** — this library moves fast and the
exact class and argument names have changed across releases:

- Which pipeline class loads Wan 2.2 TI2V-5B for image-conditioned generation. It's a
  hybrid T2V/I2V model, so the naming may not be obvious.
- That the pipeline accepts `prompt_embeds` / `negative_prompt_embeds` and can be
  constructed without a text encoder. **This is load-bearing for the whole storage
  design.** If it turns out the text encoder can't be omitted cleanly, fall back to a
  GGUF'd UMT5 and revise §3's budget upward by ~3–4GB before building anything else.
- GGUF loading support (`GGUFQuantizationConfig` or equivalent) for this architecture.
- Whether a 4-step distill LoRA exists for TI2V-5B specifically.

Get a bare end-to-end generation working from a script — image in, PNG frames out —
before any Tauri code exists. Everything downstream is straightforward; the pipeline
assumptions are the only real risk in this project.

---

## 10. Decision log

| Decision | Rationale |
|---|---|
| TI2V-5B over I2V-A14B | A14B is MoE — two 14GB checkpoints with a mid-generation expert swap. Storage and latency both unacceptable for an interactive desktop tool. |
| Q6_K over Q8_0 / Q4_K_M | Q8 buys nothing on 24GB; below Q5 temporal artifacts appear and they *move*, which is far more visible than still-image quant damage. |
| Full-precision VAE | High-compression VAE does load-bearing reconstruction; quantizing it visibly degrades output. |
| Precomputed embeddings | Removes 6.7GB. Viable because presets are fixed at build time. |
| Custom diffusers server over ComfyUI | GPL-3.0 vs. commercial release; and bypassing the text encoder removes most of ComfyUI's advantage. |
| JSONL over stdio, not HTTP | No port conflicts, no socket binding, no firewall prompts. |
| Animated WebP default, GIF as export | GIF's 256-color palette destroys gradient-heavy cover art. |
| Ping-pong over true-loop model | InP variant is +4GB for a v1 that only needs non-causal motion anyway. |
| Weights downloaded on first run | Small installer; keeps Apache-2.0 model licensing separate from app distribution. |
| Output 1350×1800 / 1200×1800 | SteamGridDB-standard grid sizes. Target display is a large-format TV, not a ~350px thumbnail, so output is viewed near full size. |
| Generate at 768×1024 / 768×1152, upscale on export | Output dims aren't divisible by 32, so a resize is mandatory regardless. These sit at 0.87×/0.98× of the model's trained pixel budget — in-distribution, and only a 1.56–1.76× Lanczos upscale to cover. Generating natively at 1344×1792 is ~2.7× the trained budget and ~60× the attention compute. |
| Guidance 2.0, not model-card 5.0 | Measured: logo band motion 22.65 vs 72.31. At 5.0 the title treatment disintegrates within ~8 frames. Cliff sits between 2.0 and 3.0. |
| 20 steps is fine; distill is optional | Measured 63.3s / 8.48 GiB peak at 768×1024 on a 4090. An earlier entry called the distill load-bearing — that reasoned from a compute *ratio* without measuring the absolute number, and was wrong. |
| Text protected by compositing, not generation | TI2V-5B cannot preserve thin glyphs through the VAE at ANY guidance (identical damage at 1.0/1.5/2.0). Pasting source pixels back fixes it completely, keeps seams, and shrinks the file. |
| CRAFT for text detection | MIT, torch-native (no new runtime, unlike PP-OCRv5's Paddle/ONNX), scene-text-trained. ~83MB. Raw heatmap used as mask; box postprocessing skipped. |
| Mask must stay user-editable | Detection failures are silent — a miss ships mangled type rather than erroring. |
| No LoRA-on-GGUF | Q6_K stores a 3072-wide row as 2520 packed bytes; fusing a logical-shaped delta fails. Ship a pre-distilled GGUF instead if the distill is ever wanted. |
| Download to `.part`, rename only after the hash passes | The final path then only ever exists with verified contents, which is what lets startup stay a cheap size check rather than rehashing 7 GB every launch. |
| Phosphor is free and Apache-2.0; RetroVoid is paid | Free, self-contained tools are the draw into The Halfrican Software's ecosystem, and Phosphor suits that unusually well because its output is *displayed* — every cover made keeps advertising. Apache rather than MIT for its §6: it grants the code and explicitly withholds the trademarks, so a fork cannot present itself as Phosphor. Weights are never redistributed (§3), so repo licensing stays independent of the models'. |
| Aspect-ratio boxes fit against both axes, via `cqw`/`cqh` | Pinning one axis and clamping the other with `max-*` silently puts the box off-ratio: the clamp does not shrink the definite dimension. It squashed 2:3 covers 33% in the mask editor. Percentages cannot cross axes, so `min(100cqh, 100cqw / var(--ar))` on a size container is the actual fix. |
| Export offers Full and Half, **Half by default**, ratio still automatic | A launcher grid hitches on full-size animated covers, and decode cost is per-pixel: half each axis is a quarter of the pixels. Half is also *closer to native* than Full, which upscales 1.76x from the generation size, so the default costs no visible quality at grid size. Exposing four raw sizes instead of a scale would let a 3:4 size be picked for a 2:3 cover. |
| `native-tls` for the downloader, not rustls | **reqwest 0.13 renamed its TLS features** — `rustls-tls` no longer exists and `default-tls` now *means* rustls, whose provider is aws-lc-rs (wants cmake + nasm on windows-msvc). `native-tls` is schannel: no crypto toolchain, no vendored roots, and it trusts what Windows already trusts. Do not "fix" this back to the 0.11/0.12 spelling. |

---

## 11. Backlog (do not build in v1)

- `Wan2.2-Fun-5B-InP` optional model for directional/causal motion presets. **Now has
  evidence behind it beyond directional motion:** `docs/animation-timing.md` measured
  SteamGridDB's animated covers as one-way loops of 6.6 s and 13.6 s, against our 2.67 s
  ping-pong that reverses every 1.33 s. The reversal rate, not the speed, is what reads
  as restless. InP removes it.
- Custom Prompt Pack (optional UMT5 download)
- Batch processing across a library folder
- Direct export into launcher metadata formats (Playnite, ES-DE)
- Motion strength slider (needs evaluation — may just expose guidance scale)

---

## 12. Where things stand — 2026-08-20

Everything below was measured or built on this date. Start here in a new session.

### Proven end to end

Cover in -> seamless 1350x1800 animated WebP out, on an RTX 4090:

| | |
|---|---|
| Generation | 63.3s, peak **8.48 GiB / 24** |
| Conditioning | frame 0 vs source MAE **1.73/255** |
| Baked embeddings | **bit-identical** to live UMT5 — 3.57 MB replaces 11.4 GB |
| Loop | 33 -> 64 frames, both seams inside the normal delta distribution |
| Export | **6.18 MB** WebP q75 (5.70 MB with text protection) |
| Model download | resumed a 36 MB `.part` → 83,152,330 B, SHA256 matches the manifest |

Read `docs/pipeline-verification.md` for the §9 verification pass and every measurement.

### Working scripts (dev)

```
setup.ps1                      venv + CUDA torch + deps + ~19 GB of models
tools/download_models.py       model fetch, resumable
tools/bake_embeddings.py       one-time UMT5 bake -> assets/embeddings.safetensors
tools/verify_embeddings.py     REGRESSION TEST for §4 — re-run after any preset edit
tools/spike_generate.py        single generation, --lora / --live-encode
tools/build_loop.py            ping-pong + upscale + encode, asserts seam quality
tools/text_mask.py             CRAFT detect + composite
tools/sweep.py                 cover x preset sweep
tools/analyze_run.py           metrics — READ ITS HEADER, it is broken cross-cover
tools/webp_timing.py           frame delays / fps / duration from the WebP container
```

### Sidecar freeze — done 2026-08-20

`sidecar/phosphor-sidecar.spec` freezes the sidecar with PyInstaller. Verified against the
real JSONL protocol, not by importing it in-process: ping, generate, detect_text, protect,
clean shutdown.

**Parity is exact where it matters.** `detect_text` on BO3 returned **14.4 %** coverage,
matching §5a's recorded figure for that cover. A full 33-frame / 20-step run took **58.5 s
warm** (vs §5's 63.3 s baseline) with an 8.5 s model load, so freezing costs nothing.

**Onedir, not onefile — and therefore not an `externalBin`.** The payload is ~2.9 GB of
torch and CUDA. Onefile would re-extract all of it to temp on every launch; onedir starts in
**1.2 s**. Tauri's `externalBin` takes single files only, so the sidecar ships as a bundle
*resource* (`"../sidecar-dist/phosphor-sidecar/": "sidecar/"`), built by
`tools/build_sidecar.ps1`, and `bundled_binary()` resolves it.
Only ffmpeg remains an `externalBin`.

**Two roots, not one.** `inference_server.py` used to derive `ROOT` from `__file__`, which
frozen points into PyInstaller's temp extraction dir. It now takes `--models` and `--assets`,
which are genuinely different places once installed: models live in app data (7 GB, must
survive an update), embeddings ship in the resource dir. Running from source with no
arguments still resolves the repo layout, so dev is unchanged.

**The failure worth remembering:** the first build died at runtime with
`PackageNotFoundError: No package metadata was found for ... 'requests'`. diffusers and
transformers call `importlib.metadata.version()` on packages they merely *probe* for, so the
set is not knowable from our own imports. The spec now copies `.dist-info` for **every**
installed distribution — a few hundred KB against 2.9 GB, and the whole class of error
disappears. Do not replace that with a hand-picked list.

### Model downloader — done 2026-08-20

Runtime download is **7.11 GB** (vs 21 GB dev; the gap is §4's text-encoder trick), driven
from screen 1g. `download_models` / `cancel_download` / `verify_models` in `lib.rs`, the
machinery in `models.rs`.

- **Resumable.** Bytes land in `<name>.part` and continue with an HTTP `Range` request.
  `status()` reports `partial` so the setup screen opens an interrupted download at the
  percentage it actually reached — without that, resume works but looks like it restarted.
- **Verified before publish.** The hash runs on the `.part`; the rename into place is the
  last step. A file that fails its hash is deleted rather than left to poison the retry.
- **Retried.** Five attempts per file with 2/4/8/16s backoff. Cheap, because each attempt
  resumes rather than restarting. A short read (connection dropped without an error) is
  treated as retryable; a checksum failure is not.
- **Cancellable** mid-chunk and mid-hash. Cancelling is not an error — `.part` files stay.

Progress goes through a `Sink` closure rather than an `AppHandle`, so the whole path runs
headless. That is what the tests in `models.rs` use: two of them really fetch from Hugging
Face (~2 KB), and `a_prior_partial_is_resumed_not_restarted` seeds a *poisoned* prefix so
the run can only fail its checksum if the remainder was genuinely appended — a restart
would silently pass, which is why the obvious version of that test proves nothing.

Verified in the app, not just in tests: a staged 36 MB `.part` resumed to a byte-correct
83 MB file, then the run completed, started the sidecar and dropped to the cover screen.

**One packaging bug fixed in passing:** `model_root()` joined `"models"` onto the app data
dir, but every manifest path already begins with `models/`. Release builds would have
written to `<appdata>/models/models/…` while dev — rooted at the repo — resolved correctly,
so it could not have surfaced until packaging. Now `data_root()`, returning the app data dir
itself. `assets/models.json` was also missing from `tauri.conf.json`'s `resources`, which
would have made `model_status` fail in a packaged build.

### Export scale, added 2026-08-21

Full or Half, chosen in the export dialog and reflected on the main screen and status bar.
**Half is the default.** See §7 for the sizes, the measurements and the reasoning. Verified
in the app on one Halo generation with only the toggle flipped: 1200×1800 / 3.49 MB and
600×900 / 1.24 MB, both 64 frames.

### Animation timing, measured and shelved 2026-08-21

`docs/animation-timing.md`. Phosphor's covers read as faster than SteamGridDB's; the cause
is **not** frame rate. Theirs run at 30 fps against our 24, and by motion per second ours is
already a third as busy. What reads as restless is the ping-pong reversing every 1.33 s
where theirs are one-way loops of 6.6 s and 13.6 s.

Shelved without code changes: the RetroVoid hitch that prompted it is a decode-cost problem
that the half-size export addresses. The doc carries the reference measurements, the fps and
frame-count options with their costs, and three traps in analysing animated WebP, starting
with PIL reporting every frame duration as 0.

### Built but not finished

- **Inter font** is not bundled (see §7a).
- **ffmpeg is 111 MB**, not §7's ~80 MB estimate. A webp+gif-only build would be far smaller.

### Open bugs — found installing 0.2.0, 2026-08-21

Three things seen in the installed app, none of them blocking export. **Bug 1 is fixed**
(2026-08-21); 2 and 3 are open.

**1. A 2:3 cover is squashed horizontally in the mask editor. FIXED 2026-08-21.**

The diagnosis in the original write-up was right. `.ph-maskcanvas` took `height: 100%` from
the stage and derived its width from `aspect-ratio`, with `max-width: 100%` as a backstop.
Clamping the width does not shrink the definite height, so the box went off-ratio and
`object-fit: fill` stretched the artwork into it. A 2:3 box is narrower for a given height,
so it clamped sooner and distorted more.

Measured on Halo in the mask editor at the shipped 460x720 window: **804x1804, ratio 0.446**,
where 2:3 wants 0.667. A **33% horizontal squash**.

The fix sizes the box against *both* of the stage's axes at once. Percentages cannot cross
axes, so this needs container query units: `.ph-maskstage` becomes `container-type: size`
and the box takes `height: min(100cqh, 100cqw / var(--ar))`, with `--ar` set inline by
`MaskEditor` as a bare number so CSS can divide by it. `.ph-mask` gained an explicit
`grid-template-rows: minmax(0, 1fr)` so the stage's height is definite rather than
content-derived, which is what a size container requires.

Verified in the running app, not only by reasoning:

| | box | ratio | vs target |
|---|---|---|---|
| Halo 2:3, before | 804x1804 | 0.4457 | -33.1% |
| Halo 2:3, after | 804x1203 | 0.6683 | +0.2% |
| Halo 2:3, after, maximized 3876x2196 | 1204x1804 | 0.6674 | +0.1% |
| BO3 3:4, after | 804x1070 | 0.7514 | +0.2% |

The two Halo "after" rows matter more than the numbers suggest: the first is **width**
limited and the second **height** limited, so `min()` is demonstrably picking the correct
axis in both regimes rather than happening to agree in one.

**`object-fit: fill` stays, and is not a workaround.** `pipeline.py` feeds the model a plain
`resize((width, height))`, a non-uniform stretch to the generation size, so filling the box
shows exactly what the model is handed. It also guarantees the image and the mask canvas stay
co-registered. Do not "fix" it to `contain` (brings back the grey band) or `cover` (crops the
mask out of alignment).

**The same latent bug exists on `.ph-cover`** (main and generating screens), which uses the
same height-driven pattern. It is not reachable at any window shape tested, because the
controls below always leave the stage short enough that the width never clamps, and its
`object-fit: cover` would crop rather than distort. Left alone deliberately; if it ever
surfaces, apply the same `min()` treatment.

**2. The icon does not show on a Start Menu pin.** Taskbar and the exe are correct now, so
this is a third icon path, distinct from the two already fixed (window icon via
`entries()[0]`, and exe resource). Start pins take their icon from the shortcut's
`IconResource`, and Windows caches Start tiles aggressively and separately from the taskbar.
Worth checking, in order: what the NSIS/MSI shortcut actually points at, whether it names an
icon explicitly, and whether it survives clearing the Start tile cache. The `Square*Logo.png`
assets are MSIX/Store tiles and are almost certainly *not* what a classic Start pin reads.

**3. First-run setup errored on Download, then worked after a restart.** Error text was not
captured, so this needs reproducing before it can be diagnosed. The installed app writes no
log, which is itself part of the problem — `sidecar://log` and errors only go to the UI.

Suspects worth eliminating first: `model_status` was read at boot and the Download click acts
on that snapshot, the `downloading` guard in `download_models`, and a partially-populated app
data directory left by the previous install. When reproducing, capture the exact banner text;
without it this is guesswork.

### Known limitations, not bugs

- **Small text is destroyed and cannot be recovered by tuning** (§5a). Protection via CRAFT
  + compositing is the fix. Detection failures are *silent*.
- **Untested:** a cover whose artwork is mostly lettering. Protecting 40%+ of the frame may
  leave too little moving for the loop to read as animated.
- `tools/analyze_run.py`'s `stability` metric **does not generalise across covers** — it
  scored two visibly destroyed covers as "OK". Use it only for A/B within one cover.

### The icon

Design **2a, "Ghost sweep"**, from the same Claude Design canvas as the UI: a portrait card
holding a bright leading bar with five echo bars decaying behind it, raked -14°. The name
made literal, since phosphor afterglow is a decay trail rather than a glow.

`tools/make_icon.py` draws it with Pillow rather than rasterising the SVG, so the build
needs no extra dependency, supersampled 4× because the mark is all diagonal edges.

```powershell
.venv/Scripts/python.exe tools/make_icon.py    # writes every icon, icon.ico included
```

**Do not run `npm run tauri icon` after it.** That overwrites these with downscales of a
single master and re-creates `android/` and `ios/` trees this app has no use for.

**The canvas draws 2a three times and all three are used.** Six bars at full size, three at
32px, two at 16px, with the stroke and corner radius thickening at each step. Every ICO entry
is rendered from the variant intended for its size rather than resized from one master; five
thin bars downscaled to 32px turn into grey mush, which is what a low-fidelity taskbar icon
looks like.

`icon.ico` carries **256/128/96/64/48/40/32/24/20/16, largest first**, and both the sizes
and the order matter.

The sizes: `tauri icon` stopped at 64 and then jumped to 256, so a display at 300% scaling —
which wants about 96px — upscaled the 64px entry.

**The order is the one that actually caused the muddy taskbar icon.** `tauri-codegen` builds
the *window* icon by taking `icon_dir.entries()[0]` verbatim, the first entry in the file
rather than the best match (`tauri-codegen/src/image.rs`, `CachedIcon::new_ico`). Ascending
sizes therefore handed it the **16×16**, which Windows upscaled to fill the taskbar button.
Windows itself scans the whole directory and picks a best match regardless of order, so
leading with 256 costs nothing and fixes Tauri.

The tell that located it: the taskbar button looked muddy while the *context menu* icon at
the same size looked sharp. Those come from different places — the context menu reads the
exe's icon resource, where Windows chooses the size.

Checked against Nocturne, Windows dark, Windows light and white grounds.

### The UI source

The interface was designed on a Claude Design canvas and ported here rather than authored ad
hoc: eight screens (1a main, 1b alt main, 1c empty, 1d mask editor, 1e generating, 1f export,
1g first-run download, 1h settings). **Nocturne** is the design system, ported to
`src/nocturne.css`; §7a has its rules, and they are not decoration.

The canvas is a private project, so its link is kept outside this repo. If you have access:
`support.js` and `_ds_bundle.js` in it are the canvas editor's own runtime (`dc-runtime`,
which renders `<x-dc>` documents), not application code. Do not port them.

### Before this is ready to share

Public at <https://github.com/TheHalfrican/Phosphor> under Apache-2.0. **0.2.0** builds both
installers, which install, download the runtime on first run, generate, protect text and
export:

| target | size | |
|---|---|---|
| `Phosphor_0.2.0_x64-setup.exe` (NSIS) | 41.0 MB | per-user, no elevation |
| `Phosphor_0.2.0_x64_en-US.msi` (WiX) | 53.4 MB | per-machine, wants admin, deployable by policy |

MSI is only viable because the sidecar left the bundle. It packs payload into CAB, which has
the same 2 GB ceiling that killed NSIS at 2.88 GB, so it would have failed identically had
the freeze still been bundled.

Still missing before it is worth pointing anyone at:

- **Screenshots in the README.** They belong near the top, under the opening paragraphs and
  above "What it does". A tool whose whole value is visual currently shows none of it.
- ~~**A wide-window composition.**~~ **Done 2026-08-21**, see §7a. The wrapper element this
  entry predicted is `.ph-controls`. It was prototyped directly in the app rather than on the
  canvas, since the canvas has no artboard wider than 460px to extend.
- ~~**A 2:3 cover generated end to end.**~~ **Done 2026-08-21.** Halo ran the whole path in
  the app: 768x1152 generation, **113.4 s** on a debug build including model load, CRAFT
  found the title at 7.7% of frame, and the export offered 1200x1800. BO3 3:4 in the same
  session took 69.3 s warm. Nothing in the 2:3 path needed changing.

**The sidecar release tag is independent of the app version.** `assets/models.json` points
at the `v0.1.0` release because that is where the archives live and the frozen sidecar has
not changed since. Only re-upload and re-tag when `sidecar/` actually changes.

### How the sidecar reaches the user — settled, do not re-litigate

**Measured 2026-08-20, not predicted.** `npm run tauri build` compiles and links the app
fine, then dies in the bundler:

```
Running makensis to produce Phosphor_0.1.0_x64-setup.exe
Internal compiler error #12345: error mmapping file (2102812255, 33554432) is out of range.
```

2,102,812,255 bytes is **1.96 GB** — the signed 32-bit boundary. NSIS cannot build an
installer carrying the 2.88 GB freeze, full stop.

**Switching to MSI does not help.** Tauri's `msi` target is WiX v3, which packs payload into
CAB files, and CAB has the same 2 GB ceiling. Both Windows installer formats are bound by
32-bit offsets. This is a payload problem, not a format problem.

**Trimming under 2 GB is not the answer either.** The bulk is CUDA: cublasLt 478 MB,
torch_cuda 410 MB, torch_cpu 305 MB, cufft 284 MB, cudnn_engines_precompiled 207 MB,
cusparse 150 MB, cusolver 126 MB, cudnn_adv 107 MB. Dropping the ones this pipeline probably
never calls (cufft, cusolver, cusparse, cudnn_adv) might save ~650 MB and land around
2.2 GB — still over, and pruning CUDA libraries by guesswork is a classic source of
failures that appear only on someone else's machine.

**So: download the sidecar on first run, like the models.** `models.rs` already does
resumable, checksum-verified, cancellable downloads, and `models.json` is already a
manifest. The sidecar becomes another entry: zip `sidecar-dist/phosphor-sidecar/`, host it,
add its SHA256. The installer drops to ~20 MB and first-run goes from 7.11 GB to ~10 GB.

This is what §3 already says to do about big things, so it is a return to the design rather
than a departure from it.

**Built 2026-08-20.** `models.rs` now unpacks archive entries, and the installer builds:

| | |
|---|---|
| Installer (NSIS) | **40 MB** — builds cleanly with the sidecar out of the bundle |
| Sidecar archive | **2.06 GB** zipped from 3.09 GB (67%), `tools/package_sidecar.py` |
| First-run download | **9.17 GB** total (was 7.11 GB) |

A manifest entry with `unpack_to` is extracted after its checksum passes, the archive is
then deleted, and completeness is tracked by a marker file holding the verified sha256 —
storing the hash rather than a bare flag means a manifest bump invalidates an old install
for free. Extraction refuses any entry whose path escapes the destination (zip-slip); that
is a test, not a comment, because this archive arrives over the network and is unpacked
into the user's app data.

The frozen sidecar therefore lives at `<appdata>/sidecar/phosphor-sidecar.exe`, not in the
install directory, and `bundled_binary()` looks there.

### Sidecar hosting — published and verified 2026-08-21

Released at <https://github.com/TheHalfrican/Phosphor/releases/tag/v0.1.0>, marked
**prerelease**: there is no distributed installer, so it must not read as "latest".

| asset | size |
|---|---|
| `phosphor-sidecar-1of2.zip` | 1.06 GB (0.99 GiB) |
| `phosphor-sidecar-2of2.zip` | 1.00 GB (0.93 GiB) |

**Verified against the live release**, not only in unit tests:

```text
cargo test --lib -- --ignored --nocapture downloads_and_unpacks_the_real_sidecar
```

pulls both parts from GitHub, checks both SHA256s, unpacks, writes markers, deletes the
archives and reports the manifest complete. **119.8 s** for ~2 GB. Re-run it after any
sidecar rebuild or re-upload.

GitHub's asset CDN answers **206 Partial Content**, so the downloader's resume works against
it. Worth confirming rather than assuming — resume is the only reason a 9.17 GB first run is
tolerable.

**Not Hugging Face**, despite the downloader already pointing there for models. HF's free
public storage is "best-effort" and explicitly conditioned on uploads being "as useful to
the community as possible": it is for models and datasets. A PyInstaller bundle of torch and
CUDA DLLs is a Windows application payload, so hosting it there is outside what that storage
is for.

**Why two parts.** GitHub caps a release asset at 2 GiB; one archive was 1.92 GiB, leaving
~83 MiB that a torch update would erase. Each part is a *complete, independent* zip rather
than a byte-split, so the downloader verifies, resumes and retries each on its own. Both
extract into the same directory, which needed no new code — marker files are named per key,
and a test pins that one part landing does not mark the whole sidecar installed.

**`unpack_to` is `sidecar-runtime`, not `sidecar`.** In dev the download root is the repo,
and `sidecar/` there is the Python *source* directory; unpacking 5838 frozen files over it
would bury `inference_server.py` in its own build output.

To republish after a rebuild:

```powershell
./tools/build_sidecar.ps1                                     # -> sidecar-dist/phosphor-sidecar/
.venv/Scripts/python.exe tools/package_sidecar.py --parts 2   # -> zips + manifest entries
gh release upload v0.1.0 sidecar-dist/phosphor-sidecar-*of2.zip --clobber
```

**The freeze is not bit-reproducible**, so the hashes change even when the code does not.
Paste the new entries from `sidecar-dist/manifest-entries.json` into `assets/models.json`,
or first run fails its checksum for everyone.

### Next step

Install the built installer and run the app from a clean machine state. Everything upstream
of that is verified; what has *not* been proven is the installed layout — that
`bundled_binary()` finds ffmpeg where NSIS actually puts it, that the resource dir resolves,
and that the first-run download lands the sidecar somewhere the app then launches it from.
Those paths have only ever run in dev.
