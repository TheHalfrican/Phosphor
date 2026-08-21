import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open, save } from "@tauri-apps/plugin-dialog";

import { MaskEditor } from "./MaskEditor";
import "./nocturne.css";
import "./App.css";

/* ═══════════════════════════════════════════════════════════════════════════════
   Phosphor — implements the Claude Design canvas:
     1a main · 1c empty · 1d mask editor · 1e generating · 1f export
     1g first-run download · 1h settings

   The design offered two main layouts, 1a (cover-first, preset chips) and 1b
   (preset list beside a smaller cover). 1a is implemented: it puts the artwork
   first, which is what the user is actually judging, and the chips stay readable
   at the compact window width the design specifies.
   ═══════════════════════════════════════════════════════════════════════════ */

type Preset = { id: string; name: string; blurb: string; prompt: string };
type Format = "webp" | "steam" | "gif";
/* The file extension each format writes. "steam" is the whole trick: WebP bytes, .png
   name. Keep this the single source of truth so the save dialog, the filter and the
   button label cannot drift apart. */
const EXT: Record<Format, string> = { webp: "webp", steam: "png", gif: "gif" };
type Screen = "setup" | "empty" | "main" | "generating" | "mask" | "settings";
type Strength = "gentle" | "standard" | "bold";

/* Motion strength maps onto guidance, and the ceiling is not arbitrary.
   Measured 2026-08-20: the cliff sits between 2.0 and 3.0 — at 3.0 title lettering
   visibly deforms. So "Bold" is 2.5, not 3.0. The UI must not offer a setting that
   is known to wreck the artwork. */
const GUIDANCE: Record<Strength, number> = { gentle: 1.5, standard: 2.0, bold: 2.5 };

type ModelFile = {
  key: string; present: boolean; incomplete: boolean;
  bytes: number; partial: number; note: string;
};
type ModelStatus = { complete: boolean; missing_bytes: number; files: ModelFile[] };

/* Mirrors models::DownloadProgress. `file_received` is download bytes only — during the
   hash it stays pinned at the file's full size and `verify_frac` carries the position,
   so neither bar ever rewinds. */
type DlProgress = {
  key: string; index: number; count: number;
  /* "unpack" only occurs for archive entries (the frozen sidecar). Like "verify" it
     reports through verify_frac, and the byte counters stay pinned so nothing rewinds. */
  stage: "download" | "verify" | "unpack";
  file_received: number; file_bytes: number;
  received: number; total: number;
  bytes_per_sec: number; verify_frac: number;
};
type DownloadOutcome = { cancelled: boolean; status: ModelStatus };

const gb = (n: number) => `${(n / 1e9).toFixed(1)} GB`;

const rate = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB/s` : `${Math.round(n / 1e3)} KB/s`);

/* Deliberately coarse. A 7 GB download's ETA is a guess, and a to-the-second readout
   claims a precision it does not have. */
const eta = (secs: number) => {
  if (!isFinite(secs) || secs <= 0) return "";
  if (secs > 3600) return `about ${Math.round(secs / 3600)}h left`;
  if (secs > 90) return `about ${Math.round(secs / 60)} min left`;
  return `under a minute left`;
};

export default function App() {
  const [screen, setScreen] = useState<Screen>("empty");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [preset, setPreset] = useState("ember_glow");
  const [strength, setStrength] = useState<Strength>("standard");
  const [models, setModels] = useState<ModelStatus | null>(null);
  const [error, setError] = useState("");

  const [dl, setDl] = useState<DlProgress | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  /* Files finished during THIS run. `models` is only refreshed when the run ends, so
     without this a completed file's row would sit at 0% until the very last byte of the
     very last file landed. */
  const [dlDone, setDlDone] = useState<Set<string>>(new Set());
  const [verifying, setVerifying] = useState(false);
  const lastDlKey = useRef("");

  const [cover, setCover] = useState("");           // absolute path
  const [coverSize, setCoverSize] = useState<[number, number] | null>(null);
  const [gen, setGen] = useState<{ frames_dir: string; width: number; height: number; seconds: number } | null>(null);

  const [step, setStep] = useState(0);
  const [total, setTotal] = useState(20);
  const [stage, setStage] = useState("generate");

  const [maskUrl, setMaskUrl] = useState("");
  const [maskData, setMaskData] = useState("");
  const [threshold, setThreshold] = useState(0.3);
  const [redetecting, setRedetecting] = useState(false);

  const [showExport, setShowExport] = useState(false);
  /* "steam" is animated WebP written to a .png filename. Steam accepts custom grid art
     by extension, but its UI is Chromium, which picks a decoder from the header bytes and
     ignores the name, so the file passes the check and still animates. */
  const [format, setFormat] = useState<Format>("webp");
  const [exporting, setExporting] = useState(false);
  const [autoProtect, setAutoProtect] = useState(true);

  const startedAt = useRef(0);


  /* ── boot ─────────────────────────────────────────────────────────────────── */
  useEffect(() => {
    (async () => {
      try {
        const [p, m] = await Promise.all([
          invoke<{ presets: Preset[] }>("get_presets"),
          invoke<ModelStatus>("model_status"),
        ]);
        setPresets(p.presets);
        setModels(m);
        if (!m.complete) setScreen("setup");
        else await invoke("start_sidecar");
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  useEffect(() => {
    const un = listen<DlProgress>("models://progress", (e) => {
      const p = e.payload;
      if (lastDlKey.current && lastDlKey.current !== p.key) {
        const finished = lastDlKey.current;
        setDlDone((d) => (d.has(finished) ? d : new Set(d).add(finished)));
      }
      lastDlKey.current = p.key;
      setDl(p);
    });
    return () => { un.then((f) => f()); };
  }, []);

  useEffect(() => {
    const un = listen<{ stage: string; step: number; total: number }>("sidecar://progress", (e) => {
      setStage(e.payload.stage);
      if (e.payload.total > 0) {
        setStep(e.payload.step);
        setTotal(e.payload.total);
      }
    });
    return () => { un.then((f) => f()); };
  }, []);

  /* ── model download (§3) ──────────────────────────────────────────────────── */
  async function startDownload() {
    setError("");
    // Re-read before starting. The boot-time snapshot can be stale by now (a file may have
    // arrived by other means), and rows for files that are already present would otherwise
    // sit at "queued" for the whole run while only the missing ones move, which reads as a
    // stall rather than as a skip.
    try {
      setModels(await invoke<ModelStatus>("model_status"));
    } catch {
      /* fall through to the download, which re-checks the filesystem itself */
    }
    setDlDone(new Set());
    setCancelling(false);
    lastDlKey.current = "";
    setDownloading(true);
    try {
      const out = await invoke<DownloadOutcome>("download_models");
      setModels(out.status);
      if (out.status.complete) {
        await invoke("start_sidecar");
        setScreen("empty");
      }
      // Cancelled, or a file is still missing: stay put. Every byte already fetched is
      // kept in a .part file, so pressing Download again resumes rather than restarts.
    } catch (e) {
      setError(String(e));
      // The failure may have been the last of several files — re-read so the rows show
      // what actually made it to disk rather than freezing mid-run.
      invoke<ModelStatus>("model_status").then(setModels).catch(() => {});
    } finally {
      setDownloading(false);
      setCancelling(false);
      setDl(null);
      lastDlKey.current = "";
    }
  }

  async function verifyModels() {
    setError("");
    setVerifying(true);
    try {
      const bad = await invoke<string[]>("verify_models");
      const m = await invoke<ModelStatus>("model_status");
      setModels(m);
      if (bad.length) setError(`Failed checksum: ${bad.join(", ")}. Re-download to repair.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setVerifying(false);
    }
  }

  const active = useMemo(() => presets.find((p) => p.id === preset), [presets, preset]);
  const aspect = coverSize ? (Math.abs(coverSize[0] / coverSize[1] - 0.75) < Math.abs(coverSize[0] / coverSize[1] - 2 / 3) ? "3:4" : "2:3") : "3:4";
  const outSize = aspect === "3:4" ? "1350×1800" : "1200×1800";

  /* ── cover intake ─────────────────────────────────────────────────────────── */
  const loadCover = useCallback((path: string) => {
    setCover(path);
    setGen(null);
    setMaskUrl("");
    setMaskData("");
    setError("");
    const img = new Image();
    img.onload = () => setCoverSize([img.naturalWidth, img.naturalHeight]);
    img.src = convertFileSrc(path);
    setScreen("main");
  }, []);


  async function pickCover() {
    const f = await open({
      multiple: false,
      filters: [{ name: "Cover art", extensions: ["png", "jpg", "jpeg", "webp"] }],
    });
    if (typeof f === "string") loadCover(f);
  }

  /* ── generate ─────────────────────────────────────────────────────────────── */
  async function animate() {
    if (!cover) return;
    setError("");
    setStep(0);
    setStage("load");
    setScreen("generating");
    startedAt.current = Date.now();
    try {
      const out = await invoke<{ frames_dir: string; width: number; height: number; seconds: number }>(
        "generate",
        { image: cover, preset, guidance: GUIDANCE[strength] },
      );
      setGen(out);
      if (autoProtect) {
        await detect(out.width, out.height, threshold);
        setScreen("mask");
      } else {
        setScreen("main");
        setShowExport(true);
      }
    } catch (e) {
      setError(String(e));
      setScreen("main");
    }
  }

  async function detect(w: number, h: number, t: number) {
    setRedetecting(true);
    try {
      const p = await invoke<string>("detect_text", { image: cover, width: w, height: h, threshold: t });
      // Cache-bust: a re-detect writes to a new temp path, but be explicit.
      setMaskUrl(`${convertFileSrc(p)}?v=${Date.now()}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setRedetecting(false);
    }
  }

  /* ── export ───────────────────────────────────────────────────────────────── */
  async function runExport() {
    if (!gen) return;
    const base = cover.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "") ?? "cover";
    const ext = EXT[format];
    const dest = await save({
      defaultPath: `${base}_animated.${ext}`,
      filters: [{ name: ext.toUpperCase(), extensions: [ext] }],
    });
    if (!dest) return;

    setExporting(true);
    setError("");
    try {
      await invoke("export", {
        framesDir: gen.frames_dir,
        source: cover,
        mask: maskData || null,
        outPath: dest,
        // "steam" takes the WebP path; only the filename differs.
        gif: format === "gif",
      });
      setShowExport(false);
    } catch (e) {
      setError(String(e));
    } finally {
      setExporting(false);
    }
  }

  /* ── chrome ───────────────────────────────────────────────────────────────── */
  const busy = screen === "generating" || exporting || redetecting || downloading || verifying;
  const win = getCurrentWindow();

  const TitleBar = () => (
    <div className="ph-titlebar" data-tauri-drag-region>
      <div className="ph-brand" data-tauri-drag-region>
        <span className={`ph-dot${busy ? " busy" : ""}`} />
        PHOSPHOR
      </div>
      <div className="ph-winctl">
        <button onClick={() => win.minimize()} title="Minimise">–</button>
        <button onClick={() => win.toggleMaximize()} title="Maximise" style={{ fontSize: 10 }}>▢</button>
        <button className="close" onClick={() => win.close()} title="Close">✕</button>
      </div>
    </div>
  );

  const StatusBar = () => (
    <div className="ph-status">
      <span className={`led${busy ? " busy" : ""}`} />
      <span>
        {screen === "generating"
          ? "Generating · 8.5 GB VRAM"
          : downloading
            ? dl
              ? `Downloading ${dl.index} of ${dl.count} · ${gb(dl.received)} of ${gb(dl.total)}`
              : "Starting download"
            : verifying
              ? "Verifying model files"
              : models?.complete
                ? "Ready"
                : "Models incomplete"}
      </span>
      <span className="right">
        {gen ? `WebP · 24 fps · ${gen ? 64 : 0} frames` : `${outSize} out`}
      </span>
    </div>
  );

  const Err = () =>
    error ? (
      <div className="ph-error">
        {error}
        <div style={{ marginTop: 6 }}>
          <button className="btn btn-ghost" style={{ fontSize: 11 }} onClick={() => setError("")}>
            Dismiss
          </button>
        </div>
      </div>
    ) : null;

  /* ═══════════════════════════════════════════════════════════════════════════
     1g — first-run download
     ═════════════════════════════════════════════════════════════════════════ */
  if (screen === "setup" && models) {
    const big = models.files.filter((f) => f.bytes > 1e6);
    const overall = dl && dl.total ? dl.received / dl.total : 0;
    const remaining = dl ? Math.max(0, dl.total - dl.received) : 0;
    const anyPartial = big.some((f) => !f.present && f.partial > 0);

    return (
      <div className="ph-app">
        <TitleBar />
        <div className="ph-body">
          <div className="ph-setup">
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              <div className="ph-setuptitle">One-time setup</div>
              <div className="ph-setupbody">
                Phosphor runs entirely on your GPU. It needs to download its models and
                inference runtime once, about{" "}
                {gb(models.files.reduce((s, f) => s + f.bytes, 0))} in total. Nothing you
                make ever leaves your machine.
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {big.map((f) => {
                /* Rows are matched by key, not by index: the backend only walks the files
                   that are actually missing, so its index would not line up with this
                   list once anything is already on disk. */
                const live = dl && dl.key === f.key ? dl : null;
                const done = f.present || dlDone.has(f.key);
                const pct = done
                  ? 1
                  : live && live.file_bytes
                    ? live.file_received / live.file_bytes
                    : f.bytes
                      ? f.partial / f.bytes
                      : 0;

                const state = done
                  ? "verified"
                  : live
                    ? live.stage === "verify"
                      ? `verifying · ${Math.round(live.verify_frac * 100)}%`
                      : live.stage === "unpack"
                        ? `unpacking · ${Math.round(live.verify_frac * 100)}%`
                        : `${Math.round(pct * 100)}%`
                    : f.partial > 0
                      ? `${Math.round(pct * 100)}% · resumable`
                      : downloading
                        ? "queued"
                        : f.incomplete
                          ? "incomplete"
                          : "queued";

                return (
                  <div className="ph-dlrow" key={f.key}>
                    <div className="ph-dlhead">
                      <span>
                        {f.key} <span className="muted">· {gb(f.bytes)}</span>
                      </span>
                      <span className="muted">{state}</span>
                    </div>
                    <div className="ph-track">
                      <div
                        className={`ph-fill${live && live.stage !== "download" ? " verifying" : ""}`}
                        style={{ width: `${Math.min(100, pct * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
              <div className="ph-dlok">
                <span className="check">✓</span>
                <span>Motion presets · verified</span>
              </div>
            </div>

            {downloading && (
              <div className="ph-dlrow">
                <div className="ph-dlhead">
                  <span>{cancelling ? "Stopping…" : "Total"}</span>
                  <span className="muted">
                    {dl ? `${gb(dl.received)} of ${gb(dl.total)}` : "starting…"}
                  </span>
                </div>
                <div className="ph-track">
                  <div className="ph-fill" style={{ width: `${Math.min(100, overall * 100)}%` }} />
                </div>
                <div className="ph-dlmeta">
                  {dl && dl.bytes_per_sec > 0 && dl.stage === "download"
                    ? `${rate(dl.bytes_per_sec)} · ${eta(remaining / dl.bytes_per_sec)}`
                    : dl?.stage === "verify"
                      ? "Checking the file against its checksum"
                      : dl?.stage === "unpack"
                        ? "Unpacking — this one is large and takes a minute"
                        : "Connecting to Hugging Face"}
                </div>
              </div>
            )}

            <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 2 }}>
              {downloading ? (
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 12, padding: "6px 14px" }}
                  disabled={cancelling}
                  onClick={() => { setCancelling(true); invoke("cancel_download"); }}>
                  {cancelling ? "Stopping…" : "Cancel"}
                </button>
              ) : (
                <button
                  className="btn btn-secondary"
                  style={{ fontSize: 12, padding: "6px 14px" }}
                  onClick={startDownload}>
                  {anyPartial ? "Resume download" : "Download"}
                </button>
              )}
              <span style={{ fontSize: 10.5, color: "color-mix(in srgb, var(--color-text) 40%, transparent)" }}>
                {downloading
                  ? "You can close this and pick up where you left off"
                  : "every file is checksum-verified"}
              </span>
            </div>
            <Err />
          </div>
        </div>
        <StatusBar />
      </div>
    );
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     1d — mask editor
     ═════════════════════════════════════════════════════════════════════════ */
  if (screen === "mask" && gen) {
    return (
      <div className="ph-app">
        <MaskEditor
          coverUrl={convertFileSrc(cover)}
          maskUrl={maskUrl}
          width={gen.width}
          height={gen.height}
          threshold={threshold}
          redetecting={redetecting}
          onThresholdChange={setThreshold}
          onRedetect={() => detect(gen.width, gen.height, threshold)}
          onBack={() => setScreen("main")}
          onDone={(data) => {
            setMaskData(data);
            setScreen("main");
            setShowExport(true);
          }}
        />
      </div>
    );
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     1h — settings
     ═════════════════════════════════════════════════════════════════════════ */
  if (screen === "settings") {
    return (
      <div className="ph-app">
        <div className="ph-backbar">
          <button onClick={() => setScreen(cover ? "main" : "empty")}>‹&nbsp;&nbsp;Settings</button>
        </div>
        <div className="ph-body">
          <div className="ph-settings">
            <div className="ph-setgroup">
              <span className="ph-toollabel">Generation</span>
              <div className="ph-setrow">
                <div className="label">
                  <span className="t">Default motion strength</span>
                  <span className="s">Bolder motion risks softening title art</span>
                </div>
                <div className="seg" style={{ fontSize: 11.5 }}>
                  {(["gentle", "standard", "bold"] as Strength[]).map((s) => (
                    <button key={s} className="seg-opt" style={{ padding: "4px 11px" }}
                      aria-pressed={strength === s} onClick={() => setStrength(s)}>
                      {s[0].toUpperCase() + s.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
              <div className="ph-setrow">
                <div className="label">
                  <span className="t">Protect text automatically</span>
                  <span className="s">Detect titles and keep them pixel-perfect</span>
                </div>
                <button className="ph-toggle" aria-pressed={autoProtect} onClick={() => setAutoProtect(!autoProtect)}>
                  <span className="knob" />
                </button>
              </div>
            </div>

            <div className="ph-setgroup">
              <span className="ph-toollabel">Models</span>
              <div className="ph-setrow">
                <div className="label">
                  <span className="t">Model files</span>
                  <span className="s">
                    {models ? gb(models.files.reduce((s, f) => s + f.bytes, 0)) : "—"} ·{" "}
                    {models?.complete ? "verified" : "incomplete"}
                  </span>
                </div>
                <button className="btn btn-ghost" style={{ fontSize: 11.5 }}
                  disabled={verifying} onClick={verifyModels}>
                  {verifying ? "Verifying…" : "Verify"}
                </button>
              </div>
            </div>

            <div className="ph-version">Phosphor 0.1.0 · The Halfrican Software</div>
          </div>
        </div>
        <StatusBar />
      </div>
    );
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     1c / 1a / 1e — empty, main, generating
     ═════════════════════════════════════════════════════════════════════════ */
  return (
    <div className="ph-app">
      <TitleBar />

      <div className="ph-body">
        <Err />

        {screen === "empty" && (
          <div className="ph-empty">
            <DropZone onPick={pickCover} onFile={loadCover} />
          </div>
        )}

        {screen === "generating" && (
          <div className="ph-gen">
            <Cover path={cover} aspect={aspect} busy scanAt={total ? step / total : 0} />
            <div className="ph-progwrap">
              <div className="ph-progtop">
                <span>
                  {stage === "load" ? "Loading model" : `Denoising · step ${step} of ${total}`}
                </span>
                <span>{etaText(step, total, startedAt.current)}</span>
              </div>
              <div className="ph-track">
                <div className="ph-fill" style={{ width: `${total ? (step / total) * 100 : 0}%` }} />
              </div>
              <div className="ph-progbot">
                <span>then frame decode, ~18 s</span>
                <span>{active?.name} · 33 frames</span>
              </div>
            </div>
          </div>
        )}

        {screen === "main" && (
          <div className="ph-main">
            <div className="ph-stage">
              <Cover path={cover} aspect={aspect} />
              <div className="ph-covermeta">
                <span className="tag tag-neutral" style={{ fontSize: 9 }}>{aspect}</span>
                <span>{cover.split(/[\\/]/).pop()}</span>
                <span className="dot-sep">·</span>
                <span>exports {outSize}</span>
                <button className="btn btn-ghost" style={{ fontSize: 11, padding: "2px 6px" }} onClick={pickCover}>
                  Replace
                </button>
              </div>
            </div>

            <div className="ph-section">
              <div className="ph-sechead">
                <span className="ph-seclabel">Motion</span>
                <span className="ph-sechint">seamless 2.7s loop · pick one</span>
              </div>
              <div className="ph-chips">
                {presets.map((p) => (
                  <button key={p.id} className="ph-chip" aria-pressed={p.id === preset} onClick={() => setPreset(p.id)}>
                    {p.name}
                  </button>
                ))}
              </div>
              <div className="ph-chipnote">{active ? `${active.name} — ${active.blurb}.` : ""}</div>
            </div>

            <div className="ph-row">
              <span style={{ fontSize: 11, color: "color-mix(in srgb, var(--color-text) 60%, transparent)" }}>
                Motion strength
              </span>
              <div className="seg" style={{ fontSize: 12 }}>
                {(["gentle", "standard", "bold"] as Strength[]).map((s) => (
                  <button key={s} className="seg-opt" style={{ padding: "5px 12px" }}
                    aria-pressed={strength === s} onClick={() => setStrength(s)}>
                    {s[0].toUpperCase() + s.slice(1)}
                  </button>
                ))}
              </div>
              <button className="btn btn-ghost grow" style={{ fontSize: 11 }} onClick={() => setScreen("settings")}>
                Settings
              </button>
            </div>

            <div className="ph-actions">
              <button className="btn btn-primary" style={{ padding: "9px 22px" }} onClick={animate} disabled={busy}>
                {gen ? "Re-animate" : "Animate"}
              </button>
              <span className="ph-eta">~1 min on this GPU</span>
              {gen && (
                <span className="ph-protnote">
                  <button className="btn btn-ghost" style={{ fontSize: 10.5 }} onClick={() => setScreen("mask")}>
                    edit text protection
                  </button>
                </span>
              )}
            </div>

            {gen && (
              <div className="ph-row">
                <button className="btn btn-secondary" style={{ fontSize: 12.5 }} onClick={() => setShowExport(true)}>
                  Export…
                </button>
                <span className="ph-eta">generated in {gen.seconds}s</span>
              </div>
            )}
          </div>
        )}
      </div>

      <StatusBar />

      {showExport && gen && (
        <ExportDialog
          name={cover.split(/[\\/]/).pop() ?? ""}
          outSize={outSize}
          format={format}
          setFormat={setFormat}
          exporting={exporting}
          onCancel={() => setShowExport(false)}
          onExport={runExport}
        />
      )}
    </div>
  );
}

/* ── pieces ─────────────────────────────────────────────────────────────────── */

function Cover({ path, aspect, busy, scanAt }: { path: string; aspect: string; busy?: boolean; scanAt?: number }) {
  const ratio = aspect === "3:4" ? "3 / 4" : "2 / 3";
  return (
    <div className={`ph-cover${busy ? " busy" : ""}`} style={{ aspectRatio: ratio }}>
      {path ? <img src={convertFileSrc(path)} alt="" draggable={false} /> : null}
      {busy && (
        <>
          <div className="ph-scan" style={{ top: `${Math.max(0, (scanAt ?? 0) * 100 - 34)}%` }} />
          <div className="ph-scanline" style={{ top: `${(scanAt ?? 0) * 100}%` }} />
        </>
      )}
    </div>
  );
}

function DropZone({ onPick, onFile }: { onPick: () => void; onFile: (p: string) => void }) {
  const [over, setOver] = useState(false);

  // Tauri v2 delivers native file drops as a window event rather than through the
  // browser's DataTransfer, which carries no real path.
  useEffect(() => {
    const un = getCurrentWindow().onDragDropEvent((e) => {
      if (e.payload.type === "over") setOver(true);
      else if (e.payload.type === "leave") setOver(false);
      else if (e.payload.type === "drop") {
        setOver(false);
        const f = e.payload.paths?.[0];
        if (f && /\.(png|jpe?g|webp)$/i.test(f)) onFile(f);
      }
    });
    return () => { un.then((f) => f()); };
  }, [onFile]);

  return (
    <div className={`ph-drop${over ? " over" : ""}`}>
      <div className="ph-dropinner">
        <div className="ph-dropicon" />
        <div className="ph-droptitle">Drop a cover to bring it to life</div>
        <div className="ph-dropsub">
          PNG or JPG, 3:4 or 2:3. You'll get a seamlessly looping animated cover for your library.
        </div>
        <button className="btn btn-secondary" style={{ marginTop: 6, fontSize: 12.5, padding: "7px 16px" }} onClick={onPick}>
          Browse files…
        </button>
      </div>
    </div>
  );
}

function ExportDialog(props: {
  name: string; outSize: string; format: Format;
  setFormat: (f: Format) => void; exporting: boolean;
  onCancel: () => void; onExport: () => void;
}) {
  const { name, outSize, format, setFormat, exporting, onCancel, onExport } = props;
  return (
    <div className="dialog-backdrop" onClick={onCancel}>
      <div className="dialog" style={{ width: 372 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <div style={{ fontFamily: "var(--font-heading)", fontSize: 15 }}>Export loop</div>
          <div style={{ fontSize: 11, color: "color-mix(in srgb, var(--color-text) 50%, transparent)" }}>
            {name} · 2.7 s · 64 frames · {outSize}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <button className="ph-fmt" aria-pressed={format === "webp"} onClick={() => setFormat("webp")}>
            <span className="mark">{format === "webp" ? "◉" : "○"}</span>
            <span className="col">
              <span style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span className="name">Animated WebP</span>
                <span className="tag tag-accent" style={{ fontSize: 9 }}>recommended</span>
              </span>
              <span className="desc">Full color, smallest file. Works in RetroVoid, Playnite, ES-DE.</span>
              <span className="size">≈ 6 MB</span>
            </span>
          </button>

          <button className="ph-fmt" aria-pressed={format === "steam"} onClick={() => setFormat("steam")}>
            <span className="mark">{format === "steam" ? "◉" : "○"}</span>
            <span className="col">
              <span style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                <span className="name">Steam (.png)</span>
                <span className="tag" style={{ fontSize: 9 }}>for Steam</span>
              </span>
              <span className="desc">
                The same WebP, named .png. Steam checks the extension; its Chromium UI reads
                the header and animates it anyway.
              </span>
              <span className="size">≈ 6 MB · identical to WebP</span>
            </span>
          </button>

          <button className="ph-fmt" aria-pressed={format === "gif"} onClick={() => setFormat("gif")}>
            <span className="mark">{format === "gif" ? "◉" : "○"}</span>
            <span className="col">
              <span className="name">GIF</span>
              <span className="desc">For launchers that can't play WebP. 256 colors — gradients will band.</span>
              <span className="size warn">≈ 44 MB — 7× larger than WebP</span>
            </span>
          </button>
        </div>

        <div className="dialog-actions">
          <button className="btn btn-secondary" style={{ fontSize: 12.5 }} onClick={onCancel} disabled={exporting}>
            Cancel
          </button>
          <button className="btn btn-primary" style={{ fontSize: 12.5 }} onClick={onExport} disabled={exporting}>
            {exporting ? "Exporting…" : `Export ${EXT[format].toUpperCase()}`}
          </button>
        </div>
      </div>
    </div>
  );
}

function etaText(step: number, total: number, startedAt: number) {
  if (!step || !total || !startedAt) return "estimating…";
  const elapsed = (Date.now() - startedAt) / 1000;
  const per = elapsed / step;
  const left = Math.max(0, Math.round(per * (total - step) + 18)); // + VAE decode
  return `~${left} s left`;
}
