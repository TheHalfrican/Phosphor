"""Render the Phosphor app icon from design 2a, "Ghost sweep".

Source of truth is the Claude Design canvas, option 2a: a portrait card holding a bright
leading bar with echo bars decaying behind it, raked -14 degrees. That is the name made
literal - phosphor afterglow is a decay trail, not a glow.

    .venv/Scripts/python.exe tools/make_icon.py

Writes every icon the Windows build actually uses, including `icon.ico` directly. Do NOT
run `npm run tauri icon` afterwards: it would overwrite these with single-master downscales
and re-create android/ and ios/ trees this app has no use for.

THREE LEVELS OF DETAIL, BECAUSE THE CANVAS SPECIFIES THREE
----------------------------------------------------------
2a is drawn three times: six bars at full size, three at 32px, two at 16px, with the stroke
and corner radius thickening each step. That is not decoration, it is the mark surviving
scale. Downscaling the full drawing instead turns five thin bars into grey mush, which is
what "low fidelity in the taskbar" looks like.

Each ICO entry is therefore rendered from the variant intended for its size, not resized
from one master.

The other half of the taskbar problem is size coverage. `tauri icon` emits 16/24/32/48/64
and 256, so a display at 300% scaling - which wants roughly 96px - takes the 64 and scales
it up. The sizes below include 96 and 128 so Windows has something close to land on.

THE ICO ENTRIES ARE ORDERED LARGEST FIRST ON PURPOSE
-----------------------------------------------------
tauri-codegen builds the *window* icon by taking `icon_dir.entries()[0]` verbatim - the
first entry in the file, not the best match. Sizes ascending would hand it the 16x16, which
Windows then upscales to fill the taskbar button. Do not "tidy" this back into ascending
order. See the comment at ico_sizes.
"""
import struct
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ICONS = Path(__file__).resolve().parent.parent / "src-tauri" / "icons"

VB_W, VB_H = 96, 128
CARD = (3, 3, 90, 122)
CARD_FILL = (24, 27, 48, 255)      # #181b30
CARD_STROKE = (51, 53, 90, 255)    # #33355a
ACCENT = (145, 132, 217)           # #9184d9
HEAD = (207, 199, 242, 255)        # #cfc7f2
ROTATE_DEG = 14                    # SVG rotate(-14) is 14 degrees anticlockwise
PIVOT = (48, 64)
BAR_Y, BAR_H = -20, 168

# Straight off the three drawings in the 2a artboard. (x, width, opacity); opacity None is
# the bright head.
VARIANTS = {
    "full": dict(
        bars=[(10, 5, 0.06), (20, 5, 0.12), (30, 5, 0.22),
              (40, 5, 0.38), (50, 5, 0.62), (60, 6, None)],
        stroke=2.0, radius=14, bar_r=2.5,
    ),
    "mid": dict(
        bars=[(24, 8, 0.15), (38, 8, 0.40), (54, 9, None)],
        stroke=3.0, radius=14, bar_r=4.0,
    ),
    "small": dict(
        bars=[(30, 13, 0.35), (52, 14, None)],
        stroke=5.0, radius=18, bar_r=6.0,
    ),
}


def variant_for(size):
    if size <= 24:
        return "small"
    if size <= 48:
        return "mid"
    return "full"


def render(size, variant=None, ss=4):
    """Draw the mark at `size` px square, supersampled then reduced."""
    v = VARIANTS[variant or variant_for(size)]
    w = size * ss
    scale = (size / VB_H) * 0.92 * ss
    ox = (w - VB_W * scale) / 2
    oy = (w - VB_H * scale) / 2

    def rrect(draw, x, y, bw, bh, r, **kw):
        draw.rounded_rectangle(
            [ox + x * scale, oy + y * scale, ox + (x + bw) * scale, oy + (y + bh) * scale],
            radius=r * scale, **kw)

    icon = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    card = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    rrect(ImageDraw.Draw(card), *CARD, v["radius"], fill=CARD_FILL)

    mask = Image.new("L", (w, w), 0)
    rrect(ImageDraw.Draw(mask), *CARD, v["radius"], fill=255)

    bars = Image.new("RGBA", (w, w), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bars)
    for x, bw, opacity in v["bars"]:
        fill = HEAD if opacity is None else (*ACCENT, round(255 * opacity))
        rrect(bd, x, BAR_Y, bw, BAR_H, v["bar_r"], fill=fill)
    bars = bars.rotate(ROTATE_DEG, resample=Image.BICUBIC,
                       center=(ox + PIVOT[0] * scale, oy + PIVOT[1] * scale))

    card.paste(bars, (0, 0), Image.composite(
        bars.getchannel("A"), Image.new("L", (w, w), 0), mask))
    icon.alpha_composite(card)
    rrect(ImageDraw.Draw(icon), *CARD, v["radius"],
          outline=CARD_STROKE, width=max(1, round(v["stroke"] * scale)))

    return icon.resize((size, size), Image.LANCZOS)


def write_ico(path, sizes):
    """Write a multi-resolution ICO, each entry rendered at its own size.

    Entries are PNG-compressed, which Windows has accepted for every size since Vista and
    which keeps the file small. Pillow's own ICO writer takes a single image and resizes
    it, so it cannot express per-size artwork; hence writing the container by hand.
    """
    blobs = []
    for s in sizes:
        buf = BytesIO()
        render(s).save(buf, format="PNG")
        blobs.append(buf.getvalue())

    out = BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(sizes)))       # reserved, type=icon, count
    offset = 6 + 16 * len(sizes)
    for s, blob in zip(sizes, blobs):
        dim = 0 if s >= 256 else s                          # 0 means 256 in an ICO entry
        out.write(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    for blob in blobs:
        out.write(blob)
    path.write_bytes(out.getvalue())
    return len(out.getvalue())


def main():
    ICONS.mkdir(parents=True, exist_ok=True)

    # 1024 master, kept for stores and for regenerating anything by hand.
    render(1024, "full").save(ICONS / "source.png")
    print("source.png        1024  full")

    # The PNGs tauri.conf references, each from the variant meant for its size.
    for name, size in [("32x32.png", 32), ("64x64.png", 64),
                       ("128x128.png", 128), ("128x128@2x.png", 256),
                       ("icon.png", 512)]:
        render(size).save(ICONS / name)
        print(f"{name:18s}{size:5d}  {variant_for(size)}")

    # LARGEST FIRST, and that order is load-bearing.
    #
    # tauri-codegen builds the *window* icon by reading icon.ico and taking
    # `icon_dir.entries()[0]` verbatim - the first entry, not the best match
    # (tauri-codegen/src/image.rs, CachedIcon::new_ico). With sizes ascending, entry 0 is
    # 16x16, so the taskbar button gets a 16px image upscaled to ~96px on a 300% display.
    # That is the "muddy taskbar icon, crisp context-menu icon" split: the context menu
    # reads the exe's icon resource, where Windows picks the size itself.
    #
    # Windows scans the whole directory for a best match regardless of order, so leading
    # with 256 costs nothing there and hands Tauri something worth downscaling.
    ico_sizes = [256, 128, 96, 64, 48, 40, 32, 24, 20, 16]
    n = write_ico(ICONS / "icon.ico", ico_sizes)
    print(f"icon.ico          {n:5d} bytes  sizes {ico_sizes}")


if __name__ == "__main__":
    main()
