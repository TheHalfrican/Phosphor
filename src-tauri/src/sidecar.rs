//! Python sidecar: spawn + newline-delimited JSON over stdin/stdout.
//!
//! Protocol (CLAUDE.md §2). Not HTTP — that would mean a port to conflict over, a socket
//! bound on the user's machine, and a firewall prompt on first run.
//!
//! We write one JSON object per line to the child's stdin:
//!
//! ```text
//! {"op":"generate","id":"<uuid>","image":"...","preset":"ember_glow", ...}
//! ```
//!
//! The child writes one JSON object per line back. Every line carries the originating
//! `id`, so replies can be routed even though requests may overlap:
//!
//! ```text
//! {"id":"...","type":"progress","step":3,"total":20}
//! {"id":"...","type":"result","frames_dir":"C:\\...\\phosphor_ab12"}
//! {"id":"...","type":"error","message":"CUDA out of memory"}
//! ```
//!
//! `progress` may arrive any number of times; `result` and `error` are terminal and
//! complete the request. Frames come back as a path to a temp directory of PNGs, never as
//! base64 over the pipe — serialising ~33 frames of 768x1024 through stdout is needless
//! overhead.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{AppHandle, Emitter};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex};

#[derive(Debug, thiserror::Error)]
pub enum SidecarError {
    #[error("sidecar not running")]
    NotRunning,
    #[error("sidecar i/o: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad json from sidecar: {0}")]
    Json(#[from] serde_json::Error),
    #[error("sidecar died before replying")]
    Died,
    #[error("{0}")]
    Remote(String),
}

/// A terminal reply. Progress frames never reach here — they are emitted to the UI as
/// they stream in.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Terminal {
    Result(Value),
    Error { message: String },
}

/// Progress event forwarded to the frontend as `sidecar://progress`.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub id: String,
    pub stage: String,
    pub step: u32,
    pub total: u32,
}

type Pending = Arc<Mutex<HashMap<String, oneshot::Sender<Terminal>>>>;

pub struct Sidecar {
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    _child: Mutex<Child>,
}

impl Sidecar {
    /// Spawn the sidecar and start the reader task.
    ///
    /// `exe` is the frozen Python sidecar in release, or `python inference_server.py`
    /// during development.
    pub async fn spawn(app: AppHandle, exe: PathBuf, args: Vec<String>) -> Result<Self, SidecarError> {
        let mut cmd = Command::new(&exe);
        cmd.args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        #[cfg(windows)]
        {
            // CREATE_NO_WINDOW — without this the frozen sidecar flashes a console window
            // on every launch, which looks broken in a desktop app.
                cmd.creation_flags(0x0800_0000);
        }

        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().ok_or(SidecarError::NotRunning)?;
        let stdout = child.stdout.take().ok_or(SidecarError::NotRunning)?;
        let stderr = child.stderr.take().ok_or(SidecarError::NotRunning)?;

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));

        // Reader: route each line by id.
        {
            let pending = pending.clone();
            let app = app.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let msg: Value = match serde_json::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => {
                            // Anything non-JSON on stdout is a bug in the sidecar (a stray
                            // print, a warning). Surface it rather than swallowing it.
                            let _ = app.emit("sidecar://stray", line);
                            continue;
                        }
                    };
                    let id = msg.get("id").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                    let kind = msg.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    match kind {
                        "progress" => {
                            let _ = app.emit(
                                "sidecar://progress",
                                Progress {
                                    id,
                                    stage: msg.get("stage").and_then(|v| v.as_str())
                                        .unwrap_or("generate").to_string(),
                                    step: msg.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                    total: msg.get("total").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                                },
                            );
                        }
                        "result" | "error" => {
                            if let Some(tx) = pending.lock().await.remove(&id) {
                                let t = if kind == "error" {
                                    Terminal::Error {
                                        message: msg.get("message").and_then(|v| v.as_str())
                                            .unwrap_or("unknown sidecar error").to_string(),
                                    }
                                } else {
                                    Terminal::Result(msg)
                                };
                                let _ = tx.send(t);
                            }
                        }
                        _ => {}
                    }
                }
                // stdout closed: the child exited. Fail every outstanding request rather
                // than leaving the UI spinning forever.
                let mut p = pending.lock().await;
                for (_, tx) in p.drain() {
                    let _ = tx.send(Terminal::Error {
                        message: "sidecar exited unexpectedly".into(),
                    });
                }
            });
        }

        // stderr is for human-readable logging only; it never carries protocol.
        {
            let app = app.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let _ = app.emit("sidecar://log", line);
                }
            });
        }

        Ok(Self {
            stdin: Mutex::new(stdin),
            pending,
            _child: Mutex::new(child),
        })
    }

    /// Send a request and await its terminal reply.
    pub async fn request(&self, op: &str, mut params: Value) -> Result<Value, SidecarError> {
        let id = uuid::Uuid::new_v4().to_string();
        if let Some(obj) = params.as_object_mut() {
            obj.insert("op".into(), Value::String(op.into()));
            obj.insert("id".into(), Value::String(id.clone()));
        }

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id.clone(), tx);

        let mut line = serde_json::to_string(&params)?;
        line.push('\n');
        {
            let mut w = self.stdin.lock().await;
            w.write_all(line.as_bytes()).await?;
            w.flush().await?;
        }

        match rx.await {
            Ok(Terminal::Result(v)) => Ok(v),
            Ok(Terminal::Error { message }) => Err(SidecarError::Remote(message)),
            Err(_) => Err(SidecarError::Died),
        }
    }
}
