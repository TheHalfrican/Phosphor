# Animation timing: how Phosphor's loops compare to SteamGridDB

Measured **2026-08-21**. Investigated because Phosphor's covers read as moving *faster* than
the animated covers on SteamGridDB, and the working assumption was that their frame rate
was lower. It is not. This file records what the reference assets actually contain, what
that means, and why the follow-up work was shelved.

**Status: shelved.** No code changed as a result of this. The RetroVoid grid hitch that
started the conversation was addressed by the half-size export instead (CLAUDE.md §7).

---

## The reference assets

Two genuine SteamGridDB animated assets, already applied in Steam and sitting in

```
C:\Program Files (x86)\Steam\userdata\123838231\config\grid\
```

Both are animated WebP carrying a `.png` extension, which is the same trick Phosphor's
Steam export uses: Steam checks the extension while its Chromium UI reads the header.

| asset | file | dimensions | frames | fps | loop | size |
|---|---|---|---|---|---|---|
| portrait grid | `3017860p.png` | **600×900** | **409** | **30.08** | **13.60 s** | 21.58 MB |
| hero banner | `2806050_hero.png` | 1920×620 | 198 | 30.00 | 6.60 s | 44.95 MB |

Frame delays alternate 33/34 ms in both, which is how an encoder hits exactly 30 fps with
integer milliseconds. Both declare infinite loop.

## Phosphor, for comparison

| | frames | fps | loop | structure | size |
|---|---|---|---|---|---|
| Phosphor (Halo, half size) | 64 | 24.01 | 2.67 s | ping-pong | 1.24 MB |

Delays alternate 41/42 ms, so our declared rate is correct and matches `FPS = 24` in
`encode.rs`.

---

## What the numbers actually say

**1. Their frame rate is higher, not lower.** 30 fps against our 24. The original premise
was backwards, so no amount of fps tuning was ever going to explain the difference.

**2. Their loops are much longer.** 13.60 s and 6.60 s against our 2.67 s.

**3. Their loops are one-way; ours mirrors.** Checked by comparing frame `i` against frame
`n - i`: for a ping-pong those are the same picture and collapse to near zero. Both
SteamGridDB assets fail that test decisively, so they are true one-way loops, presumably
authored rather than generated.

**4. By raw motion, ours is already the calmer one.** Mean absolute frame-to-frame
difference, scaled to per-second of playback, everything compared at the same 240×360
downscale so the figures are commensurate:

| | motion per frame | motion per second |
|---|---|---|
| SteamGridDB portrait grid | 0.96 | **28.79** |
| SteamGridDB hero | 0.91 | **27.37** |
| Phosphor (encoded) | 0.44 | **10.66** |
| Phosphor (uncompressed source frames) | 0.418 | **10.04** |

Ours moves roughly **a third as much per second** as theirs.

### The conclusion: it is the reversal rate, not the speed

Ping-pong (§6) means the motion changes direction every **half** the loop, so a 2.67 s loop
reverses every **1.33 s**, about 45 times a minute. The references run one direction for
13.60 s and 6.60 s.

A gentle drift that reverses twice a second reads as a wobble regardless of how small the
displacement is. That is what makes Phosphor's covers feel busy while measuring as three
times calmer than the reference.

**This is the first concrete evidence for the `Wan2.2-Fun-5B-InP` item in §11.** That model
accepts both a start and an end frame, which yields a genuine one-way loop and removes the
reversal entirely. It was previously filed as a nice-to-have for directional presets; the
reference assets show it also fixes the thing that actually reads as wrong.

---

## Options, and what they cost

Neither was implemented.

**Lower the playback fps.** Free: the same frames, the same file size, only the declared
delay changes. Samples were built at 600×900 from one Halo generation:

| fps | loop | reverses every | motion/s | size |
|---|---|---|---|---|
| 24 (current) | 2.67 s | 1.33 s | 10.03 | 1.38 MB |
| 16 | 4.00 s | 2.00 s | 6.69 | 1.38 MB |
| 12 | 5.33 s | 2.67 s | 5.02 | 1.38 MB |
| 10 | 6.40 s | 3.20 s | 4.18 | 1.38 MB |
| 8 | 8.00 s | 4.00 s | 3.34 | 1.38 MB |

The cost is judder. 12 fps is animating on twos, which subtle drift usually hides but a
flicker preset may strobe against.

**Generate more frames.** 81 frames instead of 33 gives 160 after ping-pong, a 6.67 s loop
at a smooth 24 fps, cutting the reversal rate 2.5× without touching the frame rate. Costs
about 2.5× the generation time. Note this contradicts §5's "49 is already longer than a
cover loop needs", a judgment that predates this comparison and does not survive it.

---

## Why it was shelved

The hitch that prompted all of this was RetroVoid stuttering while scrolling a grid of
full-size animated covers, and the likelier cause is per-frame decode cost, which the
half-size export already cuts 4× (CLAUDE.md §7). Timing is a separate, aesthetic question.

**One loose end worth remembering.** The SteamGridDB portrait grid is **600×900 with 409
frames at 21.58 MB**, against our half-size **600×900 with 64 frames at 1.24 MB**. Same
per-frame pixel count, roughly 17× the file and 6× the frames. If that asset scrolls
smoothly in RetroVoid, per-frame resolution really was the constraint and the matter is
closed. If it also hitches, the bottleneck is decode throughput or frame count and the
half-size export only bought headroom. **Untested.**

Also worth noting independently: SteamGridDB's own animated portrait grid is 600×900, the
same size Phosphor now exports by default.

---

## Measurement notes, and three traps

`tools/webp_timing.py` reads frame delays straight out of the WebP container. Re-run it
rather than trusting a library.

**PIL reports every frame duration as 0** for these files. `Image.info["duration"]` after
`seek(i)` returns 0 for all frames in both the SteamGridDB assets and our own exports, which
silently produces a total duration of 0 s and an undefined frame rate. The durations are
present in the container; parse it. An `ANMF` payload is `X(3) Y(3) W-1(3) H-1(3)
Duration(3) flags(1)`, all little endian, so the delay in milliseconds sits at payload
offset 12..15, and chunks are padded to even length.

**Do not `seek(0)` after walking to the last frame.** Animated WebP frames are composited
with blending and disposal, so seeking backwards does not reliably re-composite frame 0. An
early version of this analysis measured the loop seam that way and got a number 3.7× the
per-frame delta, which was an artifact. Decode once, forward, into a list.

**Mirror-pair difference in an encoded file is prediction drift, not per-frame noise.** In a
ping-pong export, frame `i` and frame `n - i` come from the same source frame, so the
difference between them looks like it should measure codec noise. It measured 1.13 against a
0.44 per-frame delta, which briefly suggested that most of our apparent motion was
compression fizz. It is not: those two frames sit ~54 frames apart in the bitstream and the
figure is accumulated inter-frame prediction drift, which varies slowly and is invisible in
playback. The honest check is that our encoded per-frame delta (0.444) tracks the
uncompressed source (0.418) closely, so playback is faithful.

The same drift defeats a naive palindrome detector on our own files: the check reported our
export as a one-way loop when it is a ping-pong by construction (`loop_build.rs`). Trust the
construction, not the measurement, for our own output. The detector is reliable on the
reference assets only because their real mirrored difference is far larger than any drift.
