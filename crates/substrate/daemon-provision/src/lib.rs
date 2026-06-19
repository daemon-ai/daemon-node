//! `daemon-provision` — workspace + placement provisioning (the first cut).
//!
//! Creates the execution environment a unit runs in (working dirs, process or container sandboxes).
//! Placement is a *cut* in the unit tree (host-spec §7, §9): a boundary where the management
//! protocol and the §17 stream are serialized over the wire instead of called in-process. This
//! crate owns only the OS-level mechanics of opening that cut — spawning the child process and
//! handing back a raw, length-framed byte [`CutChannel`]. The protocol that rides the channel
//! (management commands, events, and the brokered [`daemon_store::SessionStore`] calls) lives in
//! `daemon-host`, keeping this crate protocol-agnostic and dependent only on `daemon-common`.
//!
//! `process` (default) and `container` features select isolation backends. Phase 5 ships the
//! `process` backend ([`ProcessProvisioner`]); `container` is a deferred feature gate.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::SessionId;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Provisioning specs + the Provisioner seam
// ---------------------------------------------------------------------------

/// A request to materialize a per-session working directory / sandbox (host-spec §7).
#[derive(Clone, Debug)]
pub struct WorkspaceSpec {
    /// The root under which the session's workspace is created.
    pub root: PathBuf,
}

/// The resolved root of a provisioned workspace.
#[derive(Clone, Debug)]
pub struct WorkspaceRoot(pub PathBuf);

/// A request to place a unit's execution environment (host-spec §7). The `process` backend spawns
/// `program` with the given args/env and wires its stdio into the [`CutChannel`].
#[derive(Clone, Debug)]
pub struct PlacementSpec {
    /// The program to exec for the placed child (e.g. the node binary in placed-child mode).
    pub program: PathBuf,
    /// Arguments passed to the child.
    pub args: Vec<String>,
    /// Environment variables set for the child (added to the inherited environment).
    pub env: Vec<(String, String)>,
}

/// A live placement: the realized cut. Owns the child process handle and the byte channel the
/// management/§17/store traffic is framed over.
pub struct Placement {
    /// The framed byte duplex to the placed child.
    pub channel: CutChannel,
    /// The child process; killed on drop (best-effort) or via [`ChildGuard::shutdown`].
    pub child: ChildGuard,
}

/// Errors surfaced by a [`Provisioner`].
#[derive(Debug, thiserror::Error)]
pub enum ProvErr {
    /// Spawning the placed child failed.
    #[error("placement spawn failed: {0}")]
    Spawn(String),
    /// Creating the workspace failed.
    #[error("workspace provisioning failed: {0}")]
    Workspace(String),
    /// The requested backend is not available in this build.
    #[error("placement backend unavailable: {0}")]
    Unavailable(String),
}

/// Provisions workspaces and opens placement cuts (host-spec §7). Composed by `daemon-host` when it
/// fulfils a delegation that must run in an isolated process.
#[async_trait]
pub trait Provisioner: Send + Sync {
    /// Create (or resolve) the working directory / sandbox for `id`.
    async fn workspace(
        &self,
        id: &SessionId,
        spec: WorkspaceSpec,
    ) -> Result<WorkspaceRoot, ProvErr>;

    /// Open a placement cut for `id`, returning the live [`Placement`].
    async fn place(&self, id: &SessionId, spec: PlacementSpec) -> Result<Placement, ProvErr>;

    /// Tear down any host-owned resources for `id` (workspaces, sockets). The child process itself
    /// is owned by the returned [`Placement`]/[`ChildGuard`].
    async fn reclaim(&self, id: &SessionId);
}

// ---------------------------------------------------------------------------
// The cut channel — a length-framed byte duplex
// ---------------------------------------------------------------------------

/// A length-prefixed (`u32` little-endian) byte-frame duplex over a child's stdio.
///
/// Protocol-agnostic: it carries opaque frames. `daemon-host` serializes its `CutFrame` onto it.
/// Split into a shareable [`CutWriter`] and an owned [`CutReader`] so a single reader task can
/// demultiplex inbound frames while multiple producers send concurrently.
pub struct CutChannel {
    reader: CutReader,
    writer: CutWriter,
}

impl CutChannel {
    /// Build a channel from an async reader + writer pair.
    pub fn from_parts(
        reader: Box<dyn AsyncRead + Send + Unpin>,
        writer: Box<dyn AsyncWrite + Send + Unpin>,
    ) -> Self {
        Self {
            reader: CutReader { inner: reader },
            writer: CutWriter {
                inner: Arc::new(Mutex::new(writer)),
            },
        }
    }

    /// The child end of a cut: frames are read from this process's stdin and written to its stdout.
    pub fn from_stdio() -> Self {
        Self::from_parts(Box::new(tokio::io::stdin()), Box::new(tokio::io::stdout()))
    }

    /// Split into the shareable writer and the owned reader.
    pub fn split(self) -> (CutWriter, CutReader) {
        (self.writer, self.reader)
    }
}

/// The read half of a [`CutChannel`].
pub struct CutReader {
    inner: Box<dyn AsyncRead + Send + Unpin>,
}

impl CutReader {
    /// Read the next length-framed message, or `None` on EOF / a broken channel.
    pub async fn recv(&mut self) -> Option<Vec<u8>> {
        let mut len_buf = [0u8; 4];
        self.inner.read_exact(&mut len_buf).await.ok()?;
        let len = u32::from_le_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        self.inner.read_exact(&mut buf).await.ok()?;
        Some(buf)
    }
}

/// The write half of a [`CutChannel`], cheaply cloneable and safe to share across tasks.
#[derive(Clone)]
pub struct CutWriter {
    inner: Arc<Mutex<Box<dyn AsyncWrite + Send + Unpin>>>,
}

impl CutWriter {
    /// Send one length-framed message. Each frame's length prefix + body + flush are written under
    /// a single lock, so concurrent senders never interleave a frame.
    pub async fn send(&self, frame: &[u8]) -> std::io::Result<()> {
        let len = (frame.len() as u32).to_le_bytes();
        let mut guard = self.inner.lock().await;
        guard.write_all(&len).await?;
        guard.write_all(frame).await?;
        guard.flush().await
    }
}

// ---------------------------------------------------------------------------
// Child process guard
// ---------------------------------------------------------------------------

/// Owns a placed child process, killing it on drop so a cut never leaks an OS process.
pub struct ChildGuard(Option<tokio::process::Child>);

impl ChildGuard {
    /// Gracefully stop the child: signal it, then reap it.
    pub async fn shutdown(&mut self) {
        if let Some(mut child) = self.0.take() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if let Some(child) = self.0.as_mut() {
            // Best-effort: signal the child. The reaping `wait` happens via `shutdown` when awaited.
            let _ = child.start_kill();
        }
    }
}

// ---------------------------------------------------------------------------
// The process backend
// ---------------------------------------------------------------------------

/// The OS-process placement backend (`process` feature): each cut is a real child process whose
/// stdin/stdout carry the framed cut traffic (stderr is inherited for logs).
#[cfg(feature = "process")]
#[derive(Clone, Default)]
pub struct ProcessProvisioner;

#[cfg(feature = "process")]
impl ProcessProvisioner {
    /// Construct a process provisioner.
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "process")]
#[async_trait]
impl Provisioner for ProcessProvisioner {
    async fn workspace(
        &self,
        id: &SessionId,
        spec: WorkspaceSpec,
    ) -> Result<WorkspaceRoot, ProvErr> {
        let root = spec.root.join(id.as_str());
        tokio::fs::create_dir_all(&root)
            .await
            .map_err(|e| ProvErr::Workspace(e.to_string()))?;
        Ok(WorkspaceRoot(root))
    }

    async fn place(&self, _id: &SessionId, spec: PlacementSpec) -> Result<Placement, ProvErr> {
        use std::process::Stdio;

        let mut command = tokio::process::Command::new(&spec.program);
        command
            .args(&spec.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        for (key, value) in &spec.env {
            command.env(key, value);
        }

        let mut child = command.spawn().map_err(|e| ProvErr::Spawn(e.to_string()))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| ProvErr::Spawn("child stdin not piped".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| ProvErr::Spawn("child stdout not piped".into()))?;

        // Parent reads the child's stdout and writes the child's stdin.
        let channel = CutChannel::from_parts(Box::new(stdout), Box::new(stdin));
        Ok(Placement {
            channel,
            child: ChildGuard(Some(child)),
        })
    }

    async fn reclaim(&self, _id: &SessionId) {
        // The child process is owned by the returned `Placement`/`ChildGuard`; workspace teardown
        // is a no-op in phase 5 (workspaces are not yet copy-on-write managed here).
    }
}
