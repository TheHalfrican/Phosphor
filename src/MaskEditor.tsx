import { useEffect, useRef, useState } from "react";

/**
 * Text-mask correction screen — design 1d, "Text protection".
 *
 * CRAFT proposes; the user disposes. This screen exists because detection failures are
 * *silent* — a missed region does not raise an error, it ships mangled type — which is
 * why the design's own footer says "missed text will ship garbled — check subtitles".
 *
 * Add paints protection, Erase removes it. Protected pixels are pasted back from the
 * source in every frame, so over-painting is the safe direction to err: the cost is a
 * region that does not animate, versus type that visibly deforms.
 *
 * HOW THE MASK IS STORED, AND WHY IT MATTERS
 * ------------------------------------------
 * The working canvas holds the accent colour in RGB and the mask strength in ALPHA. That
 * is a deliberate change from painting flat white: a white mask blended over artwork was
 * nearly impossible to read, and a mask you cannot see is a safety net you cannot check.
 * Accent-on-artwork reads instantly on both bright type and dark backgrounds.
 *
 * The consequence is that alpha, not luminance, is the source of truth here, and the
 * export has to convert: `buildMaskPng` writes alpha back out as greyscale, because
 * `pipeline.protect` reads the mask with PIL's `convert("L")`, which uses RGB and ignores
 * alpha entirely. Getting that backwards produces a mask that is uniformly white, which
 * protects the whole frame and yields a completely static "animation".
 */

type Props = {
  coverUrl: string;
  maskUrl: string;
  width: number;
  height: number;
  threshold: number;
  redetecting: boolean;
  onThresholdChange: (t: number) => void;
  onRedetect: () => void;
  onDone: (maskDataUrl: string) => void;
  onBack: () => void;
};

type Shape = "brush" | "box";
type Mode = "add" | "erase";

/** Read the accent from the Nocturne token rather than hard-coding a hex (§7a). */
function accentRgb(): [number, number, number] {
  const css = getComputedStyle(document.documentElement)
    .getPropertyValue("--color-accent")
    .trim();
  const c = document.createElement("canvas");
  c.width = c.height = 1;
  const ctx = c.getContext("2d");
  if (!ctx) return [145, 132, 217];
  ctx.fillStyle = css || "#9184d9";
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return [r, g, b];
}

export function MaskEditor({
  coverUrl, maskUrl, width, height,
  threshold, redetecting, onThresholdChange, onRedetect, onDone, onBack,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  /** Canvas contents at the start of a box drag, so the preview can be redrawn live. */
  const snapshot = useRef<ImageData | null>(null);
  const origin = useRef<{ x: number; y: number } | null>(null);

  const [shape, setShape] = useState<Shape>("brush");
  const [mode, setMode] = useState<Mode>("add");
  const [brush, setBrush] = useState(44);
  const [coverage, setCoverage] = useState(0);
  const [regions, setRegions] = useState(0);
  const accent = useRef<[number, number, number]>([145, 132, 217]);

  useEffect(() => { accent.current = accentRgb(); }, []);

  // Load the detector's proposal as the starting point. Re-runs whenever a re-detect
  // produces a new mask, which deliberately discards manual edits — the design's
  // "Reset to detected" is the same action.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !maskUrl) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const img = new Image();
    // REQUIRED, and its absence fails silently. The mask is served over the asset
    // protocol, a different origin from the app, so without this the canvas is tainted
    // the moment it is drawn to. Drawing still works - the overlay looks perfect - but
    // getImageData() and toDataURL() both throw SecurityError, which killed measure()
    // before it set any state and made Done unable to export the mask at all. Tauri does
    // send Access-Control-Allow-Origin on this protocol (tauri/src/protocol/asset.rs), so
    // requesting CORS is all that was missing.
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // CRAFT hands back an opaque greyscale PNG. Re-encode it as accent + alpha so it is
      // visible, and so brush, box and erase all operate on one representation.
      const tmp = document.createElement("canvas");
      tmp.width = width;
      tmp.height = height;
      const tctx = tmp.getContext("2d", { willReadFrequently: true });
      if (!tctx) return;
      tctx.drawImage(img, 0, 0, width, height);
      const data = tctx.getImageData(0, 0, width, height);
      const px = data.data;
      const [ar, ag, ab] = accent.current;
      for (let i = 0; i < px.length; i += 4) {
        const lum = px[i];
        px[i] = ar; px[i + 1] = ag; px[i + 2] = ab; px[i + 3] = lum;
      }
      ctx.clearRect(0, 0, width, height);
      ctx.putImageData(data, 0, 0);
      measure();
    };
    img.src = maskUrl;
  }, [maskUrl, width, height]);

  function measure() {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    // Sampled rather than per-pixel — this runs at the end of every stroke.
    const step = 6;
    const d = ctx.getImageData(0, 0, width, height).data;
    let on = 0, n = 0;
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        if (d[(y * width + x) * 4 + 3] > 40) on++;
        n++;
      }
    }
    setCoverage(n ? on / n : 0);

    // Cheap region count: horizontal bands that contain any mask. Enough to say
    // "3 regions" as the design does, without a full connected-components pass.
    let bands = 0, inBand = false;
    for (let y = 0; y < height; y += step) {
      let any = false;
      for (let x = 0; x < width; x += step) {
        if (d[(y * width + x) * 4 + 3] > 40) { any = true; break; }
      }
      if (any && !inBand) bands++;
      inBand = any;
    }
    setRegions(bands);
  }

  /** Pointer position in canvas pixels. */
  function at(e: React.PointerEvent) {
    const c = canvasRef.current!;
    const r = c.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * width,
      y: ((e.clientY - r.top) / r.height) * height,
      scale: width / r.width,
    };
  }

  function fill(ctx: CanvasRenderingContext2D, draw: () => void) {
    const [r, g, b] = accent.current;
    ctx.globalCompositeOperation = mode === "add" ? "source-over" : "destination-out";
    ctx.fillStyle = `rgb(${r} ${g} ${b})`;
    draw();
    ctx.globalCompositeOperation = "source-over";
  }

  function down(e: React.PointerEvent) {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    drawing.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);

    const p = at(e);
    if (shape === "box") {
      // Keep the pre-drag pixels so each preview frame starts clean rather than
      // accumulating every intermediate rectangle.
      snapshot.current = ctx.getImageData(0, 0, width, height);
      origin.current = { x: p.x, y: p.y };
      return;
    }
    paintBrush(e);
  }

  function move(e: React.PointerEvent) {
    if (!drawing.current) return;
    if (shape === "box") return previewBox(e);
    paintBrush(e);
  }

  function paintBrush(e: React.PointerEvent) {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const p = at(e);
    // Scale the brush from screen px to canvas px so it feels the same size regardless
    // of how the preview maps onto a 768px-wide mask.
    const rad = (brush / 2) * p.scale;
    fill(ctx, () => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, rad, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function previewBox(e: React.PointerEvent) {
    const c = canvasRef.current;
    if (!c || !snapshot.current || !origin.current) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const p = at(e);
    const o = origin.current;
    ctx.putImageData(snapshot.current, 0, 0);
    fill(ctx, () => {
      ctx.fillRect(Math.min(o.x, p.x), Math.min(o.y, p.y),
                   Math.abs(p.x - o.x), Math.abs(p.y - o.y));
    });
  }

  function end() {
    if (!drawing.current) return;
    drawing.current = false;
    snapshot.current = null;
    origin.current = null;
    measure();
  }

  /**
   * Flatten the accent+alpha working canvas into the greyscale PNG the sidecar expects.
   * `pipeline.protect` opens it with `convert("L")`, which reads RGB and discards alpha,
   * so alpha has to become luminance here or the mask arrives meaningless.
   */
  function buildMaskPng(): string {
    const c = canvasRef.current;
    if (!c) return "";
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return "";
    const src = ctx.getImageData(0, 0, width, height);
    const out = new ImageData(width, height);
    for (let i = 0; i < src.data.length; i += 4) {
      const a = src.data[i + 3];
      out.data[i] = a; out.data[i + 1] = a; out.data[i + 2] = a; out.data[i + 3] = 255;
    }
    const flat = document.createElement("canvas");
    flat.width = width;
    flat.height = height;
    flat.getContext("2d")!.putImageData(out, 0, 0);
    return flat.toDataURL("image/png");
  }

  return (
    <>
      <div className="ph-backbar">
        <button onClick={onBack}>‹&nbsp;&nbsp;Text protection</button>
        <span className="note">these pixels stay perfectly still</span>
      </div>

      <div className="ph-body">
        <div className="ph-mask">
          <div className="ph-maskstage">
            {/* `--ar` rather than `aspectRatio`: App.css needs the ratio as a bare
                number so it can divide by it, to fit the box against both of the
                stage's axes at once. It sets `aspect-ratio: var(--ar)` from this. */}
            <div
              className="ph-maskcanvas"
              style={{ "--ar": width / height } as React.CSSProperties}
            >
              <img src={coverUrl} alt="" draggable={false} />
              <canvas
                ref={canvasRef}
                width={width}
                height={height}
                style={{ cursor: shape === "box" ? "crosshair" : "cell" }}
                onPointerDown={down}
                onPointerMove={move}
                onPointerUp={end}
                onPointerLeave={end}
              />
            </div>
          </div>

          <div className="ph-masktools">
            <div className="ph-toolgroup">
              <span className="ph-toollabel">Tool</span>
              <div className="seg" style={{ display: "flex", fontSize: 11.5 }}>
                <button
                  className="seg-opt"
                  style={{ flex: 1, padding: "5px 0" }}
                  aria-pressed={shape === "brush"}
                  onClick={() => setShape("brush")}
                >
                  Brush
                </button>
                <button
                  className="seg-opt"
                  style={{ flex: 1, padding: "5px 0" }}
                  aria-pressed={shape === "box"}
                  onClick={() => setShape("box")}
                >
                  Box
                </button>
              </div>
              {/* Box drags out a rectangle, which suits title lockups far better than
                  tracing them by hand — most cover type sits in a band. */}
              <div className="seg" style={{ display: "flex", fontSize: 11.5 }}>
                <button
                  className="seg-opt"
                  style={{ flex: 1, padding: "5px 0" }}
                  aria-pressed={mode === "add"}
                  onClick={() => setMode("add")}
                >
                  Add
                </button>
                <button
                  className="seg-opt"
                  style={{ flex: 1, padding: "5px 0" }}
                  aria-pressed={mode === "erase"}
                  onClick={() => setMode("erase")}
                >
                  Erase
                </button>
              </div>
              {shape === "brush" && (
                <label className="ph-slider">
                  <span>Size</span>
                  <input
                    type="range" min={8} max={90} value={brush}
                    onChange={(e) => setBrush(+e.target.value)}
                  />
                </label>
              )}
            </div>

            <div className="ph-toolgroup">
              <span className="ph-toollabel">Detection</span>
              <label className="ph-slider">
                <span>Sensitivity</span>
                <input
                  type="range" min={10} max={70} value={Math.round(threshold * 100)}
                  onChange={(e) => onThresholdChange(+e.target.value / 100)}
                />
              </label>
              <button
                className="btn btn-ghost"
                style={{ fontSize: 11, padding: "3px 4px", alignSelf: "flex-start" }}
                onClick={onRedetect}
                disabled={redetecting}
              >
                {redetecting ? "Detecting…" : "Re-detect text"}
              </button>
            </div>

            <div className="ph-maskstats">
              <span>
                {regions} region{regions === 1 ? "" : "s"} · {(coverage * 100).toFixed(1)}% of frame
              </span>
              <span>Feathered — no visible seam</span>
            </div>
          </div>
        </div>
      </div>

      <div className="ph-maskfoot">
        <button
          className="btn btn-primary"
          style={{ padding: "7px 18px", fontSize: 13 }}
          onClick={() => onDone(buildMaskPng())}
        >
          Done
        </button>
        <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={onRedetect} disabled={redetecting}>
          Reset to detected
        </button>
        <span className="ph-maskwarn">missed text will ship garbled — check subtitles</span>
      </div>
    </>
  );
}
