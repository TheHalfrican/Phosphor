"""Read animated-WebP frame timing straight out of the RIFF container.

DEV TOOL. Use it on our own exports and on reference assets alike:

    .venv/Scripts/python.exe tools/webp_timing.py out/cover_animated.webp

**PIL cannot do this.** `Image.info["duration"]` after `seek(i)` returns 0 for every frame
of these files, which silently yields a total duration of 0 s and an undefined frame rate.
The delays are present in the container, so parse it.

Per the WebP container spec an `ANMF` payload begins

    X(3) Y(3) Width-1(3) Height-1(3) Duration(3) flags(1)

all little endian, so the per-frame delay in milliseconds sits at payload offset 12..15.
Chunks are padded to an even length.

See docs/animation-timing.md for the measurements this produced and the two other traps in
analysing animated WebP.
"""
import os
import struct
import sys
from collections import Counter


def read(path):
    """Return (delays_ms, loop_count, canvas). `loop_count` 0 means infinite."""
    b = open(path, "rb").read()
    if b[:4] != b"RIFF" or b[8:12] != b"WEBP":
        raise ValueError(f"{path} is not a WebP file")

    i, delays, loop, canvas = 12, [], None, None
    while i + 8 <= len(b):
        fourcc = b[i:i + 4]
        size = struct.unpack("<I", b[i + 4:i + 8])[0]
        payload = b[i + 8:i + 8 + size]

        if fourcc == b"ANMF":
            delays.append(int.from_bytes(payload[12:15], "little"))
        elif fourcc == b"ANIM":
            loop = struct.unpack("<H", payload[4:6])[0]
        elif fourcc == b"VP8X":
            canvas = (int.from_bytes(payload[4:7], "little") + 1,
                      int.from_bytes(payload[7:10], "little") + 1)

        i += 8 + size + (size & 1)

    return delays, loop, canvas


def report(path):
    delays, loop, canvas = read(path)
    if not delays:
        print(f"{os.path.basename(path)}: still image, no animation chunks")
        return

    total = sum(delays) / 1000
    counts = Counter(delays)
    modal = counts.most_common(1)[0][0]

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  canvas        : {canvas[0]}x{canvas[1]}" if canvas else "  canvas        : ?")
    print(f"  size          : {os.path.getsize(path) / 1e6:.2f} MB")
    print(f"  frames        : {len(delays)}")
    print(f"  loop          : {'infinite' if loop == 0 else loop}")
    print(f"  delays (ms)   : {counts.most_common(4)}")
    if modal:
        print(f"  modal delay   : {modal} ms -> {1000 / modal:.2f} fps")
    print(f"  duration      : {total:.2f} s")
    if total:
        print(f"  average fps   : {len(delays) / total:.2f}")
    # Ping-pong (CLAUDE.md 6) mirrors, so the motion reverses at half the loop length.
    print(f"  if ping-pong, reverses every {total / 2:.2f} s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        report(p)
