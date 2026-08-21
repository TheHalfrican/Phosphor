//! Phosphor — Tauri backend.
//!
//! Pipeline, in order (see CLAUDE.md):
//!
//! ```text
//!   cover in
//!     -> §5  generate at 768x1024 / 768x1152, guidance 2.0
//!     -> §5a detect text  (CRAFT)
//!     -> §5a USER CORRECTS THE MASK   <- not optional; detection failures are silent
//!     -> §5a composite source text back over every frame
//!     -> §6  ping-pong to 2N-2 frames
//!     -> §7  lanczos upscale + encode to animated WebP
//!   loop out
//! ```
//!
//! Rust owns file I/O, the sidecar lifecycle, loop assembly and ffmpeg. All GPU work
//! happens in the Python sidecar, because there is no Rust inference stack that can run a
//! 5B video DiT — candle and burn do not cover this architecture. That is not a decision
//! to revisit (§2).

mod encode;
mod loop_build;
mod models;
mod sidecar;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::Serialize;
use serde_json::json;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::RwLock;

use encode::Aspect;
use sidecar::Sidecar;

/// Long-lived app state.
#[derive(Default)]
pub struct AppState {
    sidecar: RwLock<Option<Arc<Sidecar>>>,
    /// Set by `cancel_download`, cleared at the start of each run. Shared with the
    /// download task rather than owned by it, so cancelling does not need a handle to
    /// the task itself.
    cancel_download: Arc<AtomicBool>,
    /// Guards against a second download starting while one is in flight — two writers
    /// appending to the same `.part` file would interleave garbage.
    downloading: Arc<AtomicBool>,
}

#[derive(Debug, Serialize)]
pub struct GenerateOutput {
    pub frames_dir: String,
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub seconds: f32,
}

type CmdResult<T> = Result<T, String>;

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

// ---------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------

/// Locate a binary bundled via `externalBin`.
///
/// Tauri strips the target triple and places these next to the app executable, but the
/// exact directory has moved between versions and differs by bundler, so this tries the
/// plausible locations and reports every one it looked at rather than failing with a
/// bare "not found". Cheap, and it turns a packaging mistake into a legible error
/// instead of a silent "sidecar not running".
fn bundled_binary(app: &AppHandle, stem: &str) -> CmdResult<PathBuf> {
    let name = format!("{stem}.exe");
    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(&name));
            candidates.push(dir.join("binaries").join(&name));
        }
    }
    if let Ok(res) = app.path().resource_dir() {
        candidates.push(res.join(&name));
        candidates.push(res.join("binaries").join(&name));
    }
    // The frozen sidecar is not shipped at all: at 2.88 GB it exceeds what NSIS (and WiX)
    // can package, so it is downloaded and unpacked into app data on first run, next to
    // the models. See models.rs and CLAUDE.md 12.
    if let Ok(data) = models::data_root(app) {
        // "sidecar-runtime", not "sidecar": in dev the root is the repo, and `sidecar/`
        // there is the Python SOURCE directory. Unpacking 5838 frozen files over it would
        // bury inference_server.py in its own build output.
        candidates.push(data.join("sidecar-runtime").join(&name));
    }

    candidates
        .iter()
        .find(|p| p.exists())
        .cloned()
        .ok_or_else(|| {
            let tried: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
            format!("bundled binary '{name}' not found. Looked in:\n  {}", tried.join("\n  "))
        })
}

/// Where the sidecar should look for models and for the baked embeddings.
///
/// These are two different places once installed: models are downloaded to app data
/// (7 GB, and they must survive an app update), while `embeddings.safetensors` ships in
/// the app's resource directory. In dev both live in the repo.
fn sidecar_roots(app: &AppHandle) -> CmdResult<(PathBuf, PathBuf)> {
    if cfg!(debug_assertions) {
        let root = project_root();
        Ok((root.join("models"), root.join("assets")))
    } else {
        let models = models::data_root(app).map_err(err)?.join("models");
        let assets = app.path().resource_dir().map_err(err)?.join("assets");
        Ok((models, assets))
    }
}

/// In dev we run the sidecar from source through the project venv; in release it is the
/// frozen binary bundled as an externalBin.
///
/// Either way the roots are passed explicitly. The sidecar cannot derive them from
/// `__file__` once frozen, because that points into PyInstaller's temp extraction dir.
fn sidecar_command(app: &AppHandle) -> CmdResult<(PathBuf, Vec<String>)> {
    let (models, assets) = sidecar_roots(app)?;
    let roots = vec![
        "--models".to_string(),
        models.to_string_lossy().into_owned(),
        "--assets".to_string(),
        assets.to_string_lossy().into_owned(),
    ];

    if cfg!(debug_assertions) {
        let root = project_root();
        let py = root.join(".venv").join("Scripts").join("python.exe");
        let script = root.join("sidecar").join("inference_server.py");
        if !py.exists() {
            return Err(format!("dev sidecar interpreter missing: {}", py.display()));
        }
        let mut args = vec![script.to_string_lossy().into_owned()];
        args.extend(roots);
        Ok((py, args))
    } else {
        Ok((bundled_binary(app, "phosphor-sidecar")?, roots))
    }
}

fn ffmpeg_path(app: &AppHandle) -> PathBuf {
    if cfg!(debug_assertions) {
        project_root().join("bin").join("ffmpeg.exe")
    } else {
        bundled_binary(app, "ffmpeg").unwrap_or_else(|_| PathBuf::from("ffmpeg.exe"))
    }
}

/// Repo root during development. `CARGO_MANIFEST_DIR` is `<root>/src-tauri`.
fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------

#[tauri::command]
async fn get_presets(app: AppHandle) -> CmdResult<serde_json::Value> {
    let p = if cfg!(debug_assertions) {
        project_root().join("assets").join("presets.json")
    } else {
        app.path().resource_dir().map_err(err)?.join("assets").join("presets.json")
    };
    let raw = std::fs::read_to_string(&p)
        .map_err(|e| format!("presets.json ({}): {e}", p.display()))?;
    serde_json::from_str(&raw).map_err(err)
}

fn manifest_and_root(app: &AppHandle) -> CmdResult<(models::Manifest, PathBuf)> {
    let manifest_path = if cfg!(debug_assertions) {
        project_root().join("assets").join("models.json")
    } else {
        app.path().resource_dir().map_err(err)?.join("assets").join("models.json")
    };
    let mut manifest = models::Manifest::load(&manifest_path).map_err(err)?;

    // The frozen sidecar is a release-only artifact: in dev `sidecar_command` runs
    // inference_server.py through the project venv, so downloading and unpacking a 2 GB
    // freeze into the repo would be pure waste. Without this, first run in dev sits on the
    // setup screen asking for a runtime it is never going to use.
    if cfg!(debug_assertions) {
        manifest.files.retain(|f| f.unpack_to.is_none());
    }

    // In dev the models live in the repo, not app-data — that is where setup.ps1 and the
    // tools/ scripts already put them, and re-downloading 7 GB to a second location just
    // to run the app from source would be absurd.
    let root = if cfg!(debug_assertions) {
        project_root()
    } else {
        models::data_root(app).map_err(err)?
    };
    Ok((manifest, root))
}

#[tauri::command]
async fn model_status(app: AppHandle) -> CmdResult<models::ModelStatus> {
    let (manifest, root) = manifest_and_root(&app)?;
    Ok(manifest.status(&root))
}

/// First-run download (§3). Streams `models://progress` throughout.
///
/// Resolves only when every missing file has landed and been verified, or when the user
/// cancels. Long-running by design — the whole point of the progress events is that the
/// UI does not have to guess what is happening for the ~7 GB in between.
#[tauri::command]
async fn download_models(
    app: AppHandle,
    state: State<'_, AppState>,
) -> CmdResult<models::DownloadOutcome> {
    if state
        .downloading
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("a download is already running".into());
    }
    state.cancel_download.store(false, Ordering::SeqCst);

    let (manifest, root) = manifest_and_root(&app)?;
    let cancel = state.cancel_download.clone();
    let downloading = state.downloading.clone();

    let emitter = app.clone();
    let sink: models::Sink = Arc::new(move |p| {
        let _ = emitter.emit("models://progress", p);
    });

    let result = models::download_all(sink, &manifest, &root, cancel).await;

    // Release the guard on every path, including the error one, or a single failed
    // download would wedge the button for the rest of the session.
    downloading.store(false, Ordering::SeqCst);
    result.map_err(err)
}

/// Ask the running download to stop. Returns immediately; the download command resolves
/// with `cancelled: true` once the in-flight chunk or hash notices.
#[tauri::command]
fn cancel_download(state: State<'_, AppState>) {
    state.cancel_download.store(true, Ordering::SeqCst);
}

/// Full hash check of everything already on disk — what the Settings "Verify" button
/// should mean. `status()` only compares sizes, which catches a truncated file but not a
/// corrupted one.
///
/// Returns the keys that failed, so an empty list means everything checks out.
#[tauri::command]
async fn verify_models(app: AppHandle) -> CmdResult<Vec<String>> {
    let (manifest, root) = manifest_and_root(&app)?;
    tokio::task::spawn_blocking(move || {
        let mut bad = Vec::new();
        for f in &manifest.files {
            let p = root.join(&f.path);
            if !p.exists() {
                bad.push(f.key.clone());
            } else if models::verify(&p, &f.sha256, &f.key).is_err() {
                bad.push(f.key.clone());
            }
        }
        bad
    })
    .await
    .map_err(err)
}

/// Boot the sidecar. Idempotent — safe to call from the UI on mount.
#[tauri::command]
async fn start_sidecar(app: AppHandle, state: State<'_, AppState>) -> CmdResult<bool> {
    if state.sidecar.read().await.is_some() {
        return Ok(true);
    }
    let (exe, args) = sidecar_command(&app)?;
    let sc = Sidecar::spawn(app.clone(), exe, args).await.map_err(err)?;
    *state.sidecar.write().await = Some(Arc::new(sc));
    Ok(true)
}

async fn sc(state: &State<'_, AppState>) -> CmdResult<Arc<Sidecar>> {
    state
        .sidecar
        .read()
        .await
        .clone()
        .ok_or_else(|| "sidecar not started".to_string())
}

/// Generate frames from a cover. Progress streams as `sidecar://progress`.
#[tauri::command]
async fn generate(
    state: State<'_, AppState>,
    image: String,
    preset: String,
    guidance: Option<f32>,
    steps: Option<u32>,
    frames: Option<u32>,
    seed: Option<u64>,
) -> CmdResult<GenerateOutput> {
    let sc = sc(&state).await?;
    let v = sc
        .request(
            "generate",
            json!({
                "image": image,
                "preset": preset,
                // §5: 2.0, not the model card's 5.0. At 5.0 title treatments
                // disintegrate within ~8 frames.
                "guidance": guidance.unwrap_or(2.0),
                "steps": steps.unwrap_or(20),
                "frames": frames.unwrap_or(33),   // must be 4n+1
                "seed": seed.unwrap_or(0),
            }),
        )
        .await
        .map_err(err)?;

    Ok(GenerateOutput {
        frames_dir: v["frames_dir"].as_str().unwrap_or_default().into(),
        frame_count: v["frame_count"].as_u64().unwrap_or(0) as u32,
        width: v["width"].as_u64().unwrap_or(0) as u32,
        height: v["height"].as_u64().unwrap_or(0) as u32,
        seconds: v["seconds"].as_f64().unwrap_or(0.0) as f32,
    })
}

/// Run CRAFT over the source and return a PNG mask for the UI to display and edit.
#[tauri::command]
async fn detect_text(
    state: State<'_, AppState>,
    image: String,
    width: u32,
    height: u32,
    threshold: Option<f32>,
) -> CmdResult<String> {
    let sc = sc(&state).await?;
    let v = sc
        .request(
            "detect_text",
            json!({
                "image": image,
                "width": width,
                "height": height,
                "threshold": threshold.unwrap_or(0.30),
            }),
        )
        .await
        .map_err(err)?;
    Ok(v["mask_path"].as_str().unwrap_or_default().into())
}

/// Composite protected text back over the frames, then ping-pong and encode.
///
/// `mask` is the *user-corrected* mask from the UI, not CRAFT's raw output.
#[tauri::command]
async fn export(
    app: AppHandle,
    state: State<'_, AppState>,
    frames_dir: String,
    source: String,
    mask: Option<String>,
    out_path: String,
    quality: Option<u8>,
    gif: Option<bool>,
) -> CmdResult<String> {
    // 1. Text protection (§5a). Skipped only if the user cleared the mask entirely.
    let mut mask_tmp: Option<PathBuf> = None;
    let frames_dir = if let Some(mask) = mask.filter(|m| !m.is_empty()) {
        // The mask is edited on a canvas inside the webview, so it can only come across as
        // a data URL. The sidecar's contract is a *path* (§2) - `pipeline.protect` opens it
        // with PIL - so materialise it here rather than teaching the sidecar about data
        // URLs. Handing the URL straight through fails as
        // `OSError: [Errno 22] Invalid argument: 'data:image/png;base64,...'`.
        let path = write_data_url_png(&mask)?;
        let path_str = path.to_string_lossy().into_owned();
        mask_tmp = Some(path);

        let sc = sc(&state).await?;
        let v = sc
            .request(
                "protect",
                json!({ "frames_dir": frames_dir, "source": source, "mask": path_str }),
            )
            .await
            .map_err(err)?;
        v["frames_dir"].as_str().unwrap_or(&frames_dir).to_string()
    } else {
        frames_dir
    };

    // 2. Ping-pong (§6).
    let mut frames: Vec<PathBuf> = std::fs::read_dir(&frames_dir)
        .map_err(err)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .collect();
    frames.sort();
    if frames.is_empty() {
        return Err(format!("no frames in {frames_dir}"));
    }

    let (w, h) = image_size(&frames[0])?;
    let aspect = Aspect::classify(w, h);

    let staging = std::env::temp_dir().join(format!("phosphor_pp_{}", uuid::Uuid::new_v4()));
    loop_build::materialise(&frames, &staging).map_err(err)?;

    // 3. Upscale + encode (§7).
    let ffmpeg = ffmpeg_path(&app);
    let pattern = staging.join("pp_%04d.png");
    let out = PathBuf::from(&out_path);

    let result = if gif.unwrap_or(false) {
        let palette = staging.join("palette.png");
        encode::gif(&ffmpeg, &pattern, &palette, &out, aspect).await.map_err(err)
    } else {
        encode::webp(&ffmpeg, &pattern, &out, aspect, quality.unwrap_or(75))
            .await
            .map_err(err)
    };

    let _ = std::fs::remove_dir_all(&staging);
    if let Some(p) = mask_tmp {
        let _ = std::fs::remove_file(p);
    }
    result.map(|p| p.to_string_lossy().into_owned())
}

/// Decode a `data:image/png;base64,...` URL into a temp PNG and return its path.
fn write_data_url_png(data_url: &str) -> CmdResult<PathBuf> {
    use base64::Engine;

    let b64 = data_url
        .split_once(",")
        .map(|(_, rest)| rest)
        .ok_or_else(|| "mask is not a data URL".to_string())?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| format!("mask is not valid base64: {e}"))?;

    let path = std::env::temp_dir().join(format!("phosphor_mask_{}.png", uuid::Uuid::new_v4()));
    std::fs::write(&path, bytes).map_err(err)?;
    Ok(path)
}

/// Minimal PNG header read — avoids pulling an image crate into the Rust side just to
/// learn the dimensions of a file the sidecar already produced.
fn image_size(path: &PathBuf) -> CmdResult<(u32, u32)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(err)?;
    let mut head = [0u8; 24];
    f.read_exact(&mut head).map_err(err)?;
    if &head[1..4] != b"PNG" {
        return Err(format!("not a PNG: {}", path.display()));
    }
    let w = u32::from_be_bytes([head[16], head[17], head[18], head[19]]);
    let h = u32::from_be_bytes([head[20], head[21], head[22], head[23]]);
    Ok((w, h))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_presets,
            model_status,
            download_models,
            cancel_download,
            verify_models,
            start_sidecar,
            generate,
            detect_text,
            export,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask crosses from the webview as a data URL, but the sidecar opens it with PIL
    /// and needs a path. Passing the URL through produced
    /// `OSError: [Errno 22] Invalid argument: 'data:image/png;base64,...'` at export time.
    ///
    /// That bug was unreachable until the canvas stopped being tainted: `toDataURL()` threw,
    /// so the mask was always empty and protection was silently skipped. Worth a test
    /// precisely because nothing else fails loudly when it breaks.
    #[test]
    fn a_mask_data_url_becomes_a_real_png_file() {
        // 1x1 PNG.
        const PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
        let url = format!("data:image/png;base64,{PNG_B64}");

        let path = write_data_url_png(&url).expect("decode");
        let bytes = std::fs::read(&path).expect("written");

        assert_eq!(&bytes[1..4], b"PNG", "should be a real PNG on disk");
        assert!(path.extension().is_some_and(|e| e == "png"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_mask_that_is_not_a_data_url_is_refused() {
        assert!(write_data_url_png(r"C:\somewhere\mask.png").is_err());
        assert!(write_data_url_png("data:image/png;base64,not valid base64!!").is_err());
    }
}
