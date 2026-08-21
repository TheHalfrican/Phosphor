"""Render the Phosphor app icon from design 2a, "Ghost sweep".

Source of truth is the Claude Design canvas, option 2a: a portrait card holding a bright
leading bar with five echo bars decaying behind it, raked -14 degrees. That is the name
made literal - phosphor afterglow is a decay trail, not a glow.

Drawn with Pillow rather than rasterising the SVG so the build needs no extra dependency,
and supersampled 4x because the whole mark is diagonal edges, which alias badly.

    .venv/Scripts/python.exe tools/make_icon.py
    npm run tauri icon src-tauri/icons/source.png

The faint bars fall away first when the icon is scaled down, leaving the bright head and
the card silhouette. That is roughly what the canvas' own 32px and 16px variants do by
hand, so the mark degrades the way it was designed to.
"""
from PIL import Image, ImageDraw
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "src-tauri" / "icons" / "source.png"

SIZE = 1024
SS = 4                     # supersample factor
W = SIZE * SS

# Geometry straight from the 2a artboard's 96x128 viewBox.
VB_W, VB_H = 96, 128
CARD = (3, 3, 90, 122)     # x, y, w, h
CARD_R = 14
CARD_FILL = (24, 27, 48, 255)      # #181b30
CARD_STROKE = (51, 53, 90, 255)    # #33355a
STROKE_W = 2.4                     # a touch heavier than the artboard's 2, to survive
                                   # the downscale to 32px without breaking up
ACCENT = (145, 132, 217)           # #9184d9
HEAD = (207, 199, 242, 255)        # #cfc7f2
ROTATE_DEG = 14                    # SVG rotate(-14) is 14 degrees anticlockwise
PIVOT = (48, 64)

# x, width, opacity. The last is the bright head; the rest are its decaying echo.
BARS = [
    (10, 5, 0.06),
    (20, 5, 0.12),
    (30, 5, 0.22),
    (40, 5, 0.38),
    (50, 5, 0.62),
    (60, 6, None),      # None = the head colour, fully opaque
]
BAR_Y, BAR_H, BAR_R = -20, 168, 2.5

# Fit the 96x128 artboard into the square with a little air around it.
SCALE = (SIZE / VB_H) * 0.92 * SS
OX = (W - VB_W * SCALE) / 2
OY = (W - VB_H * SCALE) / 2


def px(x, y):
    return (OX + x * SCALE, OY + y * SCALE)


def rrect(draw, x, y, w, h, r, **kw):
    x0, y0 = px(x, y)
    x1, y1 = px(x + w, y + h)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r * SCALE, **kw)


def main():
    icon = Image.new("RGBA", (W, W), (0, 0, 0, 0))

    # Card body.
    card = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    rrect(ImageDraw.Draw(card), *CARD, CARD_R, fill=CARD_FILL)

    # Clip mask for the sweep: the same rounded rect, solid.
    mask = Image.new("L", (W, W), 0)
    rrect(ImageDraw.Draw(mask), *CARD, CARD_R, fill=255)

    # The sweep, drawn upright then rotated about the card's centre.
    bars = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    bd = ImageDraw.Draw(bars)
    for x, w, opacity in BARS:
        fill = HEAD if opacity is None else (*ACCENT, round(255 * opacity))
        rrect(bd, x, BAR_Y, w, BAR_H, BAR_R, fill=fill)
    bars = bars.rotate(ROTATE_DEG, resample=Image.BICUBIC, center=px(*PIVOT))

    card.paste(bars, (0, 0), Image.composite(bars.getchannel("A"), Image.new("L", (W, W), 0), mask))
    icon.alpha_composite(card)

    # Stroke last, so the sweep cannot bleed over the card edge.
    rrect(ImageDraw.Draw(icon), *CARD, CARD_R,
          outline=CARD_STROKE, width=max(1, round(STROKE_W * SCALE)))

    icon = icon.resize((SIZE, SIZE), Image.LANCZOS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    icon.save(OUT)
    print(f"wrote {OUT}  ({SIZE}x{SIZE})")

    # Preview strip at the sizes that actually ship, to check it survives the downscale.
    strip_sizes = [256, 128, 64, 48, 32, 16]
    pad = 16
    strip = Image.new("RGBA", (sum(strip_sizes) + pad * (len(strip_sizes) + 1), 256 + pad * 2),
                      (22, 24, 38, 255))
    x = pad
    for s in strip_sizes:
        strip.alpha_composite(icon.resize((s, s), Image.LANCZOS), (x, pad + (256 - s) // 2))
        x += s + pad
    strip.save(OUT.parent / "_preview.png")
    print(f"wrote {OUT.parent / '_preview.png'}  (256/128/64/48/32/16)")


if __name__ == "__main__":
    main()
