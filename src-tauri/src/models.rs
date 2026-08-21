//! First-run model download + SHA256 verification (CLAUDE.md §3).
//!
//! Weights are deliberately **not** shipped in the installer. That keeps the installer
//! small and keeps the models' Apache-2.0 licensing cleanly separate from app
//! distribution.
//!
//! The manifest lives in `assets/models.json` rather than being hardcoded here, so
//! swapping a quant or bumping a revision does not require a Rust change. Hashes in it
//! were computed from local copies that were verified working end-to-end, not copied from
//! a model card — a matching hash therefore means "byte-identical to what we actually
//! tested against".
//!
//! Verification is not optional. A truncated 4.2 GB download does not announce itself; it
//! surfaces later as a confusing load failure or, worse, as silently degraded output.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub key: String,
    /// Destination path, relative to the app data directory.
    pub path: String,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
    /// When set, `path` is a zip archive: extract it into this directory (relative to the
    /// root) once the checksum passes, then delete the archive. Used for the frozen
    /// sidecar, which is 2.88 GB and cannot ship inside a Windows installer.
    #[serde(default)]
    pub unpack_to: Option<String>,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    #[serde(default)]
    pub total_bytes: u64,
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileStatus {
    pub key: String,
    pub present: bool,
    /// Present but the wrong size — a partial or interrupted download.
    pub incomplete: bool,
    pub bytes: u64,
    /// Bytes already sitting in a `.part` file. Non-zero means an interrupted download
    /// that will resume rather than restart — the UI shows this so a user who closed the
    /// app at 60%% does not think the progress was thrown away.
    pub partial: u64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelStatus {
    pub complete: bool,
    pub missing_bytes: u64,
    pub files: Vec<FileStatus>,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("manifest: {0}")]
    Manifest(String),
    #[error("i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{key}: sha256 mismatch (expected {expected}, got {actual}) - the download is \
             corrupt or truncated; delete it and retry")]
    Checksum { key: String, expected: String, actual: String },
    #[error("network: {0}")]
    Net(#[from] reqwest::Error),
    #[error("{key}: server said {status} for {url}")]
    Http { key: String, status: u16, url: String },
    #[error("{key}: download ended early - got {got} bytes of {want}")]
    Short { key: String, got: u64, want: u64 },
}

/// Records the sha256 an unpacked archive was verified against.
///
/// An unpacked entry has no single file to size-check, and the archive is deleted after
/// extraction rather than kept as 1.2 GB of dead weight, so completeness is tracked by a
/// marker written only after a successful verify-and-extract. Storing the hash rather than
/// a bare flag means a manifest bump invalidates the old install for free.
fn marker_path(root: &Path, f: &ModelFile) -> Option<PathBuf> {
    f.unpack_to
        .as_ref()
        .map(|d| root.join(d).join(format!(".phosphor-{}.sha256", f.key)))
}

fn unpacked_ok(root: &Path, f: &ModelFile) -> bool {
    match marker_path(root, f) {
        Some(m) => std::fs::read_to_string(m)
            .map(|s| s.trim().eq_ignore_ascii_case(&f.sha256))
            .unwrap_or(false),
        None => false,
    }
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Self, ModelError> {
        let raw = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&raw)?)
    }

    /// Cheap check used on startup: existence and size only.
    ///
    /// Deliberately does NOT hash — rehashing ~8 GB on every launch would add tens of
    /// seconds to startup for no benefit. Hashing happens once, immediately after a file
    /// is downloaded.
    pub fn status(&self, root: &Path) -> ModelStatus {
        let mut files = Vec::new();
        let mut missing_bytes = 0u64;

        for f in &self.files {
            let p = root.join(&f.path);
            let meta = std::fs::metadata(&p).ok();
            let actual = meta.as_ref().map(|m| m.len()).unwrap_or(0);
            // An unpacked entry is judged by its marker, not by the archive, which is
            // deleted once extracted.
            let present = if f.unpack_to.is_some() {
                unpacked_ok(root, f)
            } else {
                meta.is_some() && actual == f.bytes
            };
            let incomplete = f.unpack_to.is_none() && meta.is_some() && actual != f.bytes;
            if !present {
                missing_bytes += f.bytes.saturating_sub(if incomplete { actual } else { 0 });
            }
            let partial = if present {
                0
            } else {
                std::fs::metadata(part_path(&p))
                    .map(|m| m.len().min(f.bytes))
                    .unwrap_or(0)
            };
            files.push(FileStatus {
                key: f.key.clone(),
                present,
                incomplete,
                bytes: f.bytes,
                partial,
                note: f.note.clone(),
            });
        }

        ModelStatus {
            complete: files.iter().all(|f| f.present),
            missing_bytes,
            files,
        }
    }
}

/// Verify one file against its manifest hash.
pub fn verify(path: &Path, expected: &str, key: &str) -> Result<(), ModelError> {
    match sha256_file(path, |_| true)? {
        Some(actual) if actual.eq_ignore_ascii_case(expected) => Ok(()),
        Some(actual) => Err(ModelError::Checksum {
            key: key.to_string(),
            expected: expected.to_string(),
            actual,
        }),
        // The closure above never asks to stop, so `None` is unreachable here.
        None => Ok(()),
    }
}

/// Hash a file, reporting progress and letting the caller bail out.
///
/// `tick` receives the running byte count and returns `false` to abandon the hash, which
/// surfaces as `Ok(None)`. Hashing 4.2 GB is not instant — tens of seconds on a cold file
/// — so a cancel that only took effect *after* a hash finished would feel broken.
fn sha256_file(
    path: &Path,
    mut tick: impl FnMut(u64) -> bool,
) -> Result<Option<String>, ModelError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    // 4 MiB chunks: large enough that syscall overhead is irrelevant on a multi-GB file,
    // small enough not to spike memory.
    let mut buf = vec![0u8; 4 << 20];
    let mut read_so_far = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        read_so_far += n as u64;
        if !tick(read_so_far) {
            return Ok(None);
        }
    }
    Ok(Some(format!("{:x}", hasher.finalize())))
}

/// The directory the manifest's relative paths hang off.
///
/// Not inside the install directory — that is often read-only for non-admin users, and
/// models must survive an app update.
///
/// Returns the app data dir *itself*, not a `models` subdirectory: every `path` in
/// `models.json` already begins with `models/`. An earlier version joined `"models"` here
/// too, which would have written to `<appdata>/models/models/gguf/…` in a release build
/// while dev — rooted at the repo — resolved correctly. That bug could not have surfaced
/// until packaging.
pub fn data_root(app: &tauri::AppHandle) -> Result<PathBuf, ModelError> {
    use tauri::Manager;
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| ModelError::Manifest(format!("no app data dir: {e}")))?;
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

// =======================================================================================
// Download
// =======================================================================================
//
// Streamed, resumable and verified. Each of those earns its place against a 7.11 GB first
// run:
//
//   * streamed  — a 4.2 GB response buffered before it touches disk is a 4.2 GB
//                 allocation.
//   * resumable — a partial file is kept as `<name>.part` and continued with an HTTP
//                 Range request. Losing 3 GB of progress to a dropped Wi-Fi frame is the
//                 difference between "retry" and "uninstall".
//   * verified  — a truncated download does not announce itself. It surfaces much later
//                 as a confusing load failure, or worse as silently degraded output.
//
// Files are fetched one at a time. Parallel connections would not beat a single stream to
// a Hugging Face CDN edge by enough to matter here, and sequential keeps both the progress
// model and the failure model simple.

/// How many times one file is re-attempted before the run gives up.
///
/// Retrying is nearly free because each attempt resumes from the `.part` file instead of
/// starting over, so this is generous on purpose.
const ATTEMPTS: u32 = 5;

/// Emitted to the frontend as `models://progress`.
#[derive(Debug, Clone, Serialize)]
pub struct DownloadProgress {
    pub key: String,
    /// 1-based, for "file 2 of 3".
    pub index: usize,
    pub count: usize,
    /// `"download"` or `"verify"`.
    pub stage: &'static str,
    pub file_received: u64,
    pub file_bytes: u64,
    /// Across the whole run. Monotonic — see the note on `Reporter::emit`.
    pub received: u64,
    pub total: u64,
    pub bytes_per_sec: u64,
    /// 0.0 unless `stage == "verify"`, where it is how far the hash has read.
    pub verify_frac: f64,
}

/// Where progress goes. A closure rather than an `AppHandle` so the download path has
/// no dependency on a running Tauri app — `lib.rs` passes one that emits
/// `models://progress`, and the tests pass one that collects into a Vec.
///
/// `Arc<dyn Fn>` rather than a generic parameter because the hash runs on a
/// `spawn_blocking` task and so needs an owned, `Send + Sync + 'static` handle.
pub type Sink = Arc<dyn Fn(DownloadProgress) + Send + Sync>;

#[derive(Debug, Clone, Serialize)]
pub struct DownloadOutcome {
    pub cancelled: bool,
    pub status: ModelStatus,
}

/// Rate-limits progress events and keeps the byte counters monotonic.
struct Reporter {
    sink: Sink,
    count: usize,
    total: u64,
    /// Bytes belonging to files already finished this run.
    completed: u64,
    last_emit: Instant,
    window_at: Instant,
    window_bytes: u64,
    rate: f64,
}

impl Reporter {
    fn new(sink: Sink, count: usize, total: u64) -> Self {
        let now = Instant::now();
        Self {
            sink,
            count,
            total,
            completed: 0,
            // Back-dated so the first real update is not swallowed by the throttle.
            last_emit: now - Duration::from_secs(1),
            window_at: now,
            window_bytes: 0,
            rate: 0.0,
        }
    }

    /// `file_received` counts download bytes only. During verification it stays pinned at
    /// the file's full size and `verify_frac` carries the hash position instead —
    /// otherwise both bars would rewind to zero at the moment the download completed,
    /// which reads as a failure rather than as progress.
    fn emit(
        &mut self,
        key: &str,
        index: usize,
        stage: &'static str,
        file_received: u64,
        file_bytes: u64,
        verify_frac: f64,
        force: bool,
    ) {
        let now = Instant::now();
        // ~7 updates/sec. Emitting per chunk would be hundreds of events per second across
        // the IPC bridge, and the webview cannot paint that fast anyway.
        if !force && now.duration_since(self.last_emit) < Duration::from_millis(140) {
            return;
        }
        self.last_emit = now;

        let received = self.completed + file_received;

        let dt = now.duration_since(self.window_at).as_secs_f64();
        if dt >= 0.5 {
            let instant = received.saturating_sub(self.window_bytes) as f64 / dt;
            // Exponential smoothing: the raw figure swings enough between windows to make
            // the ETA jump around distractingly.
            self.rate = if self.rate == 0.0 {
                instant
            } else {
                self.rate * 0.7 + instant * 0.3
            };
            self.window_at = now;
            self.window_bytes = received;
        }

        (self.sink)(DownloadProgress {
            key: key.to_string(),
            index,
            count: self.count,
            stage,
            file_received,
            file_bytes,
            received,
            total: self.total,
            bytes_per_sec: self.rate.max(0.0) as u64,
            verify_frac,
        });
    }
}

fn part_path(dest: &Path) -> PathBuf {
    // Append rather than replace the extension. `Path::set_extension` would turn
    // `Wan2.2-TI2V-5B-Q6_K.gguf` into `Wan2.2-TI2V-5B-Q6_K.part`, and the `2.2` in the
    // name makes that kind of mangling easy to get wrong.
    let mut s = dest.to_path_buf().into_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// Download every missing file in the manifest.
///
/// `cancelled: true` is not an error — `.part` files are left in place and the next run
/// picks up where this one stopped.
pub async fn download_all(
    sink: Sink,
    manifest: &Manifest,
    root: &Path,
    cancel: Arc<AtomicBool>,
) -> Result<DownloadOutcome, ModelError> {
    let todo: Vec<&ModelFile> = manifest
        .files
        .iter()
        .filter(|f| {
            if f.unpack_to.is_some() {
                return !unpacked_ok(root, f);
            }
            let p = root.join(&f.path);
            std::fs::metadata(&p).map(|m| m.len()).ok() != Some(f.bytes)
        })
        .collect();

    let total: u64 = todo.iter().map(|f| f.bytes).sum();

    let client = reqwest::Client::builder()
        .user_agent(concat!("phosphor/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        // Per-read, NOT a whole-request timeout: a 4.2 GB body legitimately takes many
        // minutes, but a connection that goes quiet for 60s is dead and should become a
        // retryable error rather than hanging the setup screen forever.
        .read_timeout(Duration::from_secs(60))
        .build()?;

    let mut rep = Reporter::new(sink, todo.len(), total);
    let mut completed = 0u64;

    for (i, f) in todo.iter().enumerate() {
        let index = i + 1;
        rep.completed = completed;

        if !fetch_one(&client, f, root, index, &mut rep, &cancel).await? {
            return Ok(DownloadOutcome {
                cancelled: true,
                status: manifest.status(root),
            });
        }

        completed += f.bytes;
        rep.completed = completed;
        rep.emit(&f.key, index, "download", 0, f.bytes, 0.0, true);
    }

    Ok(DownloadOutcome {
        cancelled: false,
        status: manifest.status(root),
    })
}

/// Fetch one file: resume, stream, verify, then move into place.
///
/// `Ok(false)` means the user cancelled.
async fn fetch_one(
    client: &reqwest::Client,
    f: &ModelFile,
    root: &Path,
    index: usize,
    rep: &mut Reporter,
    cancel: &Arc<AtomicBool>,
) -> Result<bool, ModelError> {
    let dest = root.join(&f.path);
    let part = part_path(&dest);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut last_err: Option<ModelError> = None;

    for attempt in 1..=ATTEMPTS {
        if cancel.load(Ordering::Relaxed) {
            return Ok(false);
        }

        match stream_to_part(client, f, &part, index, rep, cancel).await {
            Ok(None) => return Ok(false),
            Ok(Some(())) => {
                last_err = None;
                break;
            }
            Err(e) => {
                last_err = Some(e);
                if attempt < ATTEMPTS {
                    // 2s, 4s, 8s, 16s.
                    tokio::time::sleep(Duration::from_secs(2u64.pow(attempt))).await;
                }
            }
        }
    }

    if let Some(e) = last_err {
        return Err(e);
    }

    // ---- verify, then publish ---------------------------------------------------------
    //
    // The hash runs against the `.part` file and the rename is the last step, so the final
    // path only ever exists with verified contents. That is what lets `status()` stay a
    // cheap existence-and-size check on startup instead of rehashing 7 GB every launch.
    //
    // Hashing is CPU-bound and blocking, so it must not sit on a tokio worker thread.
    // `AppHandle` is Send + Sync, so progress is emitted straight from the blocking task.
    let sink = rep.sink.clone();
    let stop = cancel.clone();
    let (key, expected, bytes) = (f.key.clone(), f.sha256.clone(), f.bytes);
    let (count, total, completed) = (rep.count, rep.total, rep.completed);
    let part_for_hash = part.clone();

    let hashed = tokio::task::spawn_blocking(move || {
        let mut last = Instant::now() - Duration::from_secs(1);
        sha256_file(&part_for_hash, |read| {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            let now = Instant::now();
            if now.duration_since(last) >= Duration::from_millis(140) {
                last = now;
                sink(DownloadProgress {
                    key: key.clone(),
                    index,
                    count,
                    stage: "verify",
                    file_received: bytes,
                    file_bytes: bytes,
                    received: completed + bytes,
                    total,
                    bytes_per_sec: 0,
                    verify_frac: if bytes > 0 { read as f64 / bytes as f64 } else { 1.0 },
                });
            }
            true
        })
    })
    .await
    .map_err(|e| ModelError::Manifest(format!("hash task panicked: {e}")))??;

    let actual = match hashed {
        Some(h) => h,
        None => return Ok(false), // cancelled mid-hash
    };

    if !actual.eq_ignore_ascii_case(&expected) {
        // Keeping a file that failed its hash only guarantees the next run resumes from
        // corrupt bytes and fails identically.
        let _ = std::fs::remove_file(&part);
        return Err(ModelError::Checksum {
            key: f.key.clone(),
            expected,
            actual,
        });
    }

    // ---- publish ----------------------------------------------------------------------
    if let Some(dir) = &f.unpack_to {
        let dest_dir = root.join(dir);
        let sink = rep.sink.clone();
        let stop = cancel.clone();
        let (key, sha) = (f.key.clone(), f.sha256.clone());
        let (count, total, completed) = (rep.count, rep.total, rep.completed);
        let archive = part.clone();
        let target = dest_dir.clone();

        // Extraction is blocking I/O over gigabytes; it does not belong on a tokio worker.
        let done = tokio::task::spawn_blocking(move || {
            unpack_zip(&archive, &target, &stop, |read, of| {
                sink(DownloadProgress {
                    key: key.clone(),
                    index,
                    count,
                    stage: "unpack",
                    file_received: bytes,
                    file_bytes: bytes,
                    received: completed + bytes,
                    total,
                    bytes_per_sec: 0,
                    verify_frac: if of > 0 { read as f64 / of as f64 } else { 1.0 },
                });
            })
        })
        .await
        .map_err(|e| ModelError::Manifest(format!("unpack task panicked: {e}")))??;

        if !done {
            return Ok(false); // cancelled mid-extract
        }

        // Marker last, so an interrupted extraction is simply not complete and the next
        // run redoes it rather than trusting a half-populated directory.
        std::fs::write(
            marker_path(root, f).expect("unpack_to is Some"),
            format!("{sha}\n"),
        )?;
        // The archive has served its purpose; keeping it would cost another 1.2 GB
        // permanently.
        let _ = std::fs::remove_file(&part);
        return Ok(true);
    }

    // Windows will not rename onto an existing file.
    let _ = std::fs::remove_file(&dest);
    std::fs::rename(&part, &dest)?;
    Ok(true)
}

/// Extract `archive` into `dest`. `Ok(false)` means cancelled.
///
/// Rejects entries whose path escapes `dest` (zip-slip). `ZipFile::enclosed_name` returns
/// `None` for absolute paths and for anything containing `..`, which is exactly the check
/// wanted here — a malicious or corrupt archive must not be able to write outside the
/// directory we chose.
fn unpack_zip(
    archive: &Path,
    dest: &Path,
    cancel: &AtomicBool,
    mut on_progress: impl FnMut(u64, u64),
) -> Result<bool, ModelError> {
    use std::io::Read;

    let file = std::fs::File::open(archive)?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))
        .map_err(|e| ModelError::Manifest(format!("not a readable zip: {e}")))?;

    let total: u64 = (0..zip.len())
        .filter_map(|i| zip.by_index_raw(i).ok().map(|e| e.size()))
        .sum();
    let mut written = 0u64;
    let mut buf = vec![0u8; 1 << 20];

    std::fs::create_dir_all(dest)?;

    for i in 0..zip.len() {
        if cancel.load(Ordering::Relaxed) {
            return Ok(false);
        }
        let mut entry = zip
            .by_index(i)
            .map_err(|e| ModelError::Manifest(format!("corrupt zip entry {i}: {e}")))?;

        let Some(rel) = entry.enclosed_name() else {
            return Err(ModelError::Manifest(format!(
                "zip entry {i} has an unsafe path and was refused",
                )));
        };
        let out = dest.join(rel);

        if entry.is_dir() {
            std::fs::create_dir_all(&out)?;
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut w = std::io::BufWriter::new(std::fs::File::create(&out)?);
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            let n = entry.read(&mut buf)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut w, &buf[..n])?;
            written += n as u64;
            on_progress(written, total);
        }
        std::io::Write::flush(&mut w)?;
    }

    on_progress(total, total);
    Ok(true)
}

/// One HTTP attempt. `Ok(None)` means cancelled.
async fn stream_to_part(
    client: &reqwest::Client,
    f: &ModelFile,
    part: &Path,
    index: usize,
    rep: &mut Reporter,
    cancel: &Arc<AtomicBool>,
) -> Result<Option<()>, ModelError> {
    use reqwest::header::RANGE;
    use reqwest::StatusCode;

    let have = tokio::fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);

    let mut req = client.get(&f.url);
    if have > 0 && have < f.bytes {
        req = req.header(RANGE, format!("bytes={have}-"));
    }
    let resp = req.send().await?;
    let status = resp.status();

    let (mut file, mut received) = if status == StatusCode::PARTIAL_CONTENT {
        (
            tokio::fs::OpenOptions::new().append(true).open(part).await?,
            have,
        )
    } else if status == StatusCode::RANGE_NOT_SATISFIABLE {
        // The `.part` is at or past the full length but was never verified, so it cannot
        // be trusted. Drop it and let the retry start clean rather than guessing.
        let _ = tokio::fs::remove_file(part).await;
        return Err(ModelError::Http {
            key: f.key.clone(),
            status: status.as_u16(),
            url: f.url.clone(),
        });
    } else if status.is_success() {
        // Range ignored, or none was asked for. Whatever is on disk is a prefix of
        // nothing, so start the file over.
        (tokio::fs::File::create(part).await?, 0)
    } else {
        return Err(ModelError::Http {
            key: f.key.clone(),
            status: status.as_u16(),
            url: f.url.clone(),
        });
    };

    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            file.flush().await?;
            return Ok(None);
        }
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        received += chunk.len() as u64;
        rep.emit(&f.key, index, "download", received, f.bytes, 0.0, false);
    }
    file.flush().await?;
    drop(file);

    if received != f.bytes {
        // A short read means the connection dropped mid-body without raising an error.
        // Retryable, and the `.part` keeps everything that did arrive.
        return Err(ModelError::Short {
            key: f.key.clone(),
            got: received,
            want: f.bytes,
        });
    }

    rep.emit(&f.key, index, "download", received, f.bytes, 0.0, true);
    Ok(Some(()))
}

// =======================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Several of these hit the network. They are deliberately not `#[ignore]`d: the whole
    // point of this module is talking to Hugging Face, and a downloader whose tests never
    // download is not telling you much. They pull the two smallest files in the real
    // manifest — about 2 KB total — so the cost is a round trip, not bandwidth.

    fn tmp_root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("phosphor_t_{tag}_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn null_sink() -> Sink {
        Arc::new(|_| {})
    }

    fn collecting() -> (Sink, Arc<std::sync::Mutex<Vec<DownloadProgress>>>) {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let out = seen.clone();
        (Arc::new(move |p| out.lock().unwrap().push(p)), seen)
    }

    /// The smallest real file in the manifest: 499 bytes, hash from `assets/models.json`.
    fn small_file() -> ModelFile {
        ModelFile {
            key: "model_index".into(),
            path: "models/wan-ti2v-5b-diffusers/model_index.json".into(),
            url: "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers/resolve/main/model_index.json".into(),
            bytes: 499,
            sha256: "6a72faeb564b0e894aea8fc4ef27241106eb739e8584e869b61589a65473add7".into(),
            unpack_to: None,
            note: String::new(),
        }
    }

    /// Build a small zip in memory so the unpack tests need no fixture on disk.
    fn make_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let opts: zip::write::FileOptions<()> = zip::write::FileOptions::default();
            for (name, body) in entries {
                w.start_file(*name, opts).unwrap();
                w.write_all(body).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    fn manifest_of(files: Vec<ModelFile>) -> Manifest {
        Manifest { total_bytes: files.iter().map(|f| f.bytes).sum(), files }
    }

    #[test]
    fn part_path_appends_rather_than_replacing_the_extension() {
        // `Path::set_extension` would turn this into `Wan2.2-TI2V-5B-Q6_K.part` — the
        // `2.2` in the model name makes that mistake easy and its damage silent.
        let p = part_path(Path::new(r"C:\m\gguf\Wan2.2-TI2V-5B-Q6_K.gguf"));
        assert_eq!(
            p.file_name().unwrap().to_str().unwrap(),
            "Wan2.2-TI2V-5B-Q6_K.gguf.part"
        );
    }

    #[test]
    fn status_reports_resumable_bytes() {
        let root = tmp_root("status");
        let f = small_file();
        let dest = root.join(&f.path);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(part_path(&dest), vec![0u8; 200]).unwrap();

        let st = manifest_of(vec![f]).status(&root);
        assert!(!st.complete);
        // Without this the setup screen opens an interrupted 7 GB download at 0% and the
        // user has no way to know the bytes were kept.
        assert_eq!(st.files[0].partial, 200);
        assert!(!st.files[0].present);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_partial_larger_than_the_file_is_clamped() {
        let root = tmp_root("clamp");
        let f = small_file();
        let dest = root.join(&f.path);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(part_path(&dest), vec![0u8; 9_000]).unwrap();

        // Otherwise a stale oversized .part renders as a >100% progress bar.
        assert_eq!(manifest_of(vec![f]).status(&root).files[0].partial, 499);
        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn downloads_verifies_and_publishes() {
        let root = tmp_root("fresh");
        let f = small_file();
        let (sink, seen) = collecting();

        let out = download_all(
            sink,
            &manifest_of(vec![f.clone()]),
            &root,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .expect("download failed");

        assert!(!out.cancelled);
        assert!(out.status.complete);

        let dest = root.join(&f.path);
        assert_eq!(std::fs::metadata(&dest).unwrap().len(), f.bytes);
        // The .part is renamed, not copied — leaving one behind would mean every install
        // carries a duplicate of every model file.
        assert!(!part_path(&dest).exists());
        // Hash it independently of the download path that just wrote it.
        verify(&dest, &f.sha256, &f.key).expect("published file does not match its hash");

        let events = seen.lock().unwrap();
        assert!(events.iter().any(|e| e.stage == "download"));
        assert!(
            events.iter().any(|e| e.stage == "verify"),
            "verification should report progress; hashing 4.2 GB in silence looks like a hang"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn a_prior_partial_is_resumed_not_restarted() {
        // Proving resume by pre-seeding a *correct* prefix cannot distinguish resuming
        // from restarting — both end with the right file. So seed a prefix of the right
        // length but the wrong contents: the run can only fail its checksum if the
        // remainder was appended to those bytes. A restart would silently pass.
        let root = tmp_root("resume");
        let f = small_file();
        let dest = root.join(&f.path);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        let part = part_path(&dest);
        std::fs::write(&part, vec![b'x'; 200]).unwrap();

        let err = download_all(
            null_sink(),
            &manifest_of(vec![f.clone()]),
            &root,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .expect_err("a poisoned .part must not verify");

        match err {
            ModelError::Checksum { key, .. } => assert_eq!(key, f.key),
            other => panic!("expected a checksum failure, got {other}"),
        }
        // A file that failed its hash must not survive, or the next attempt resumes from
        // the same corrupt bytes and fails identically forever.
        assert!(!part.exists(), ".part should be removed after a checksum failure");
        assert!(!dest.exists(), "a failed download must not be published");

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn an_already_present_file_is_skipped() {
        let root = tmp_root("skip");
        let f = small_file();
        let dest = root.join(&f.path);
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        // Right size, wrong contents. If this is re-fetched the bytes would change; if it
        // is skipped (the intended cheap check) they stay as written.
        std::fs::write(&dest, vec![b'z'; f.bytes as usize]).unwrap();

        let (sink, seen) = collecting();
        let out = download_all(
            sink,
            &manifest_of(vec![f.clone()]),
            &root,
            Arc::new(AtomicBool::new(false)),
        )
        .await
        .unwrap();

        assert!(out.status.complete);
        assert_eq!(std::fs::read(&dest).unwrap(), vec![b'z'; f.bytes as usize]);
        assert!(seen.lock().unwrap().is_empty(), "nothing to download, nothing to report");

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn unpack_extracts_nested_paths() {
        let root = tmp_root("unpack");
        let archive = root.join("payload.zip");
        std::fs::write(
            &archive,
            make_zip(&[
                ("phosphor-sidecar.exe", b"MZ fake exe"),
                ("_internal/torch/lib/torch_cuda.dll", b"fake dll"),
            ]),
        )
        .unwrap();

        let dest = root.join("sidecar");
        let ok = unpack_zip(&archive, &dest, &AtomicBool::new(false), |_, _| {}).unwrap();

        assert!(ok);
        assert_eq!(std::fs::read(dest.join("phosphor-sidecar.exe")).unwrap(), b"MZ fake exe");
        // Nested directories must be created, not silently skipped: the freeze is almost
        // entirely _internal/.
        assert_eq!(
            std::fs::read(dest.join("_internal/torch/lib/torch_cuda.dll")).unwrap(),
            b"fake dll"
        );

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn unpack_refuses_paths_that_escape_the_destination() {
        // Zip-slip. A crafted archive must not be able to write outside the directory we
        // chose, and this is worth pinning because the sidecar archive is fetched over the
        // network and extracted into the user's app data.
        let root = tmp_root("zipslip");
        let archive = root.join("evil.zip");
        std::fs::write(&archive, make_zip(&[("../../escaped.txt", b"pwned")])).unwrap();

        let dest = root.join("sidecar");
        let err = unpack_zip(&archive, &dest, &AtomicBool::new(false), |_, _| {})
            .expect_err("an escaping entry must be refused");
        assert!(format!("{err}").contains("unsafe path"), "got: {err}");
        assert!(!root.join("escaped.txt").exists());
        assert!(!root.parent().unwrap().join("escaped.txt").exists());

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn an_unpacked_entry_is_judged_by_its_marker_not_the_archive() {
        let root = tmp_root("marker");
        let mut f = small_file();
        f.key = "sidecar".into();
        f.path = "sidecar/phosphor-sidecar.zip".into();
        f.unpack_to = Some("sidecar".into());

        let m = manifest_of(vec![f.clone()]);

        // No marker: not installed, even though nothing is obviously missing.
        assert!(!m.status(&root).complete);

        // Marker with the wrong hash: a manifest bump must invalidate the old install.
        let marker = root.join("sidecar").join(".phosphor-sidecar.sha256");
        std::fs::create_dir_all(marker.parent().unwrap()).unwrap();
        std::fs::write(&marker, "0000000000000000000000000000000000000000000000000000000000000000\n").unwrap();
        assert!(!m.status(&root).complete);

        // Marker matching the manifest: installed, and the archive is long gone.
        std::fs::write(&marker, format!("{}\n", f.sha256)).unwrap();
        let st = m.status(&root);
        assert!(st.complete);
        assert!(st.files[0].present);
        assert!(!root.join(&f.path).exists());

        std::fs::remove_dir_all(&root).ok();
    }

    #[tokio::test]
    async fn cancelling_before_the_first_byte_leaves_nothing_published() {
        let root = tmp_root("cancel");
        let f = small_file();

        let out = download_all(
            null_sink(),
            &manifest_of(vec![f.clone()]),
            &root,
            Arc::new(AtomicBool::new(true)), // already cancelled
        )
        .await
        .unwrap();

        assert!(out.cancelled);
        assert!(!out.status.complete);
        assert!(!root.join(&f.path).exists());

        std::fs::remove_dir_all(&root).ok();
    }
}
