# Phosphor

**Drop in a static game cover, get a seamlessly looping animated one out.** Runs entirely
on your own GPU. Nothing is uploaded, and there is no account or API key anywhere in it.

The name refers to CRT phosphor afterglow: the decay trail that makes an old display feel
alive rather than merely lit. That is the register the whole tool aims for, a cover that
breathes rather than a cover that performs.

Built for someone curating a game library in a launcher (Steam, RetroVoid, Playnite,
LaunchBox, ES-DE) who has a 3:4 or 2:3 box scan and wants their grid to feel alive, without
first learning what a sampler is.

---

## What it does

1. You drop in a cover, either 3:4 or 2:3.
2. You pick one of eight motion presets: slow drift, neon flicker, water ripple, fog roll,
   ember glow, cloth sway, starfield shimmer, rain sheen.
3. It generates ~1.4 seconds of subtle motion, ping-pongs it into a seamless loop, and
   exports an animated WebP at SteamGridDB grid dimensions. **3:4 sources come out at
   1350×1800, 2:3 sources at 1200×1800.** GIF is available as a compatibility export.

Both ratios are generated below their output size (768×1024 and 768×1152 respectively) and
Lanczos-upscaled on export. Generation dimensions have to be divisible by 32, which none of
1350, 1200 or 1800 are, so a resize is mandatory rather than a shortcut. Those two sizes
also sit just inside the model's trained pixel budget, where it is least prone to
duplicating content.

On an RTX 4090 a cover takes about **63 seconds** and peaks around **8.5 GB of VRAM**.

### What it deliberately isn't

Not a general video-generation frontend, not a video editor, not a batch pipeline. There is
no freeform prompt box and no workflow graph. The underlying model can do a hundred things
and every one of them is a tempting feature; the value here is that it does exactly one
thing without making you learn a diffusion pipeline.

---

## How it works

```
┌─────────────────────────────┐
│  Tauri app (Rust + React)   │  UI, file I/O, ffmpeg, loop assembly
└──────────┬──────────────────┘
           │ JSON Lines over stdin/stdout
┌──────────▼──────────────────┐
│  Python sidecar             │  diffusers + torch, GGUF model, denoise loop
└─────────────────────────────┘
```

Three decisions are worth calling out, because they are what keep this a ~12 GB install
instead of a ~25 GB one:

**No text encoder.** Wan's UMT5-XXL encoder is ~6.7 GB, as large as the quantised model
itself, and exists only to turn prompt strings into tensors. Since every preset prompt is
known at build time, they are encoded once on a developer machine and shipped as a 3.6 MB
`embeddings.safetensors`. The encoder is never instantiated at runtime. This is also why
there is no custom-prompt field: it genuinely cannot work without the encoder, and a
disabled-looking input is worse than none.

**Text is composited, not generated.** The model cannot preserve fine glyph structure
through encode → denoise → decode. "CREED" comes out as "NMEAV", identically at every
guidance value, including 1.0 where classifier-free guidance is fully off, so no amount of
tuning fixes it. Instead [CRAFT](https://github.com/clovaai/CRAFT-pytorch) detects the type
and the original pixels are composited back over every frame with a feathered mask. The
loop stays seamless and the file actually gets *smaller*, because static regions compress
well across frames. Detection failures are silent, so the mask is user-editable.

**Loops are ping-pong.** Frames run forward then backward with both endpoints dropped
(`2N-2`), which means presets must be non-causal: drift, ripple, glow, sway. Anything with
a direction (rain, rising smoke, falling leaves) reads as an obvious rewind and is out of
scope for v1.

Design notes, measurements and the reasoning behind each choice live in
[`CLAUDE.md`](CLAUDE.md); the verification pass is in
[`docs/pipeline-verification.md`](docs/pipeline-verification.md).

---

## Requirements

- **Windows** with an NVIDIA GPU, 10 GB+ VRAM recommended (measured 8.5 GB peak)
- Python 3.11+ and Rust, for building from source
- **~7.1 GB** of model weights, downloaded on first run and checksum-verified

Weights are not shipped in the installer. That keeps the installer small and keeps the
models' Apache-2.0 licensing cleanly separate from app distribution.

## Building from source

```powershell
./setup.ps1          # venv, CUDA torch, deps, and the model download
npm install
npm run tauri dev
```

`setup.ps1` fetches a larger dev set (~19 GB) than the runtime download, because the tools
in `tools/` include the UMT5 encoder used to bake embeddings.

## Status

The pipeline is proven end to end, cover in and seamless animated WebP out, measured at
1350×1800 on a 3:4 cover. The first-run model downloader is resumable, checksum-verified
and cancellable.

The Python sidecar is frozen, its runtime components are published, and the installer builds
at 40 MB. **There is still no download link**, because the installed app has not been tested
end to end yet. Building from source works today.

A note on scope, so nobody wastes an evening: Phosphor targets **Windows with an NVIDIA GPU
of roughly 10 GB or more**. AMD, Intel Arc, Apple silicon and smaller cards are not
supported and are not planned. Bug reports against the supported configuration are very
welcome.

---

## Part of The Halfrican Software

Phosphor is free, and it is one of a set of small tools built around **RetroVoid**, a paid
game library manager from The Halfrican Software. Phosphor works with any launcher that
takes custom grid art, and RetroVoid is simply where it fits most naturally.

More of these are planned. If you find this one useful, that is the whole idea.

---

## Third-party components

- [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B): Apache-2.0, downloaded
  at runtime ([Q6_K GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF))
- [CRAFT](https://github.com/clovaai/CRAFT-pytorch): MIT. The model definition is vendored
  in `sidecar/vendor/craft/` (patched for modern torchvision, which removed `model_urls`);
  its licence is included there.
- ffmpeg: an LGPL build with only webp and gif enabled, deliberately no GPL codecs.

## Licence

[Apache-2.0](LICENSE). Use it, fork it, learn from it.

Apache rather than MIT for one reason worth naming: it grants the code and explicitly does
not grant the trademarks. The names *Phosphor*, *RetroVoid* and *The Halfrican Software*
stay put, so a fork cannot present itself as this project. Everything else is yours.

Note that Phosphor never redistributes model weights. They are downloaded from Hugging Face
on first run under their own licences, which keeps this repository's licensing independent
of theirs.
