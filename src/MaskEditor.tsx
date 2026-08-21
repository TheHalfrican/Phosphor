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

export function MaskEditor({
  coverUrl, maskUrl, width, height,
  threshold, redetecting, onThresholdChange, onRedetect, onDone, onBack,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [tool, setTool] = useState<"add" | "erase">("add");
  const [brush, setBrush] = useState(44);
  const [coverage, setCoverage] = useState(0);
  const [regions, setRegions] = useState(0);

  // Load the detector's proposal as the starting point. Re-runs whenever a re-detect
  // produces a new mask, which deliberately discards manual edits — the design's
  // "Reset to detected" is the same action.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !maskUrl) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0, width, height);
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

  function paint(e: React.PointerEvent) {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    const r = c.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * width;
    const y = ((e.clientY - r.top) / r.height) * height;
    // Scale the brush from screen px to canvas px so it feels the same size regardless
    // of how the 252px preview maps onto a 768px-wide mask.
    const rad = (brush / 2) * (width / r.width);

    ctx.globalCompositeOperation = tool === "add" ? "source-over" : "destination-out";
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, rad, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = "source-over";
  }

  function end() {
    if (!drawing.current) return;
    drawing.current = false;
    measure();
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
            <div className="ph-maskcanvas" style={{ aspectRatio: `${width} / ${height}` }}>
              <img src={coverUrl} alt="" draggable={false} />
              <canvas
                ref={canvasRef}
                width={width}
                height={height}
                onPointerDown={(e) => {
                  drawing.current = true;
                  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                  paint(e);
                }}
                onPointerMove={(e) => drawing.current && paint(e)}
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
                  aria-pressed={tool === "add"}
                  onClick={() => setTool("add")}
                >
                  Add
                </button>
                <button
                  className="seg-opt"
                  style={{ flex: 1, padding: "5px 0" }}
                  aria-pressed={tool === "erase"}
                  onClick={() => setTool("erase")}
                >
                  Erase
                </button>
              </div>
              <label className="ph-slider">
                <span>Brush</span>
                <input
                  type="range" min={8} max={90} value={brush}
                  onChange={(e) => setBrush(+e.target.value)}
                />
              </label>
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
          onClick={() => onDone(canvasRef.current?.toDataURL("image/png") ?? "")}
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
