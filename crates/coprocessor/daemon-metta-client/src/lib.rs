// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-metta-client` — the supervised client for the `daemon-metta` symbolic coprocessor.
//!
//! [`MettaCoprocessor`] is to the MeTTa worker what `LocalProvider` is to the inference worker: it
//! lazily spawns the `daemon-metta` child over a length-framed [`CutChannel`], speaks the
//! [`daemon_metta::protocol`], serializes requests on the single worker (the runner is
//! single-threaded), respawns the worker after a crash / transport fault, and trips a crash-loop
//! "meltdown" to [`MettaError::Fatal`] when restarts exceed a budget within a sliding window.
//!
//! It depends on `daemon-metta` with `default-features = false`, so it links only the wire types —
//! never `hyperon`. The daemon stays light; the engine lives entirely in the supervised child.
//!
//! [`CutChannel`]: daemon_provision::CutChannel

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use daemon_common::SessionId;
use daemon_metta::protocol::{self, Command, ErrorClass, Event, OpResponse};
use daemon_provision::{
    ChildGuard, CutWriter, Placement, PlacementSpec, ProcessProvisioner, Provisioner,
};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// Construction + tuning for a [`MettaCoprocessor`]'s worker.
#[derive(Clone, Debug)]
pub struct MettaConfig {
    /// Path to the `daemon-metta` worker binary.
    pub worker_bin: PathBuf,
    /// The durable state directory passed to the worker (`None` => an ephemeral in-memory store).
    pub state_dir: Option<PathBuf>,
    /// Extra environment variables set on the worker child.
    pub env: Vec<(String, String)>,
    /// How long to wait for `Event::Ready` after spawning.
    pub spawn_timeout: Duration,
    /// How long to wait for an op reply before declaring a transport fault.
    pub op_timeout: Duration,
    /// Crash-loop meltdown: max restarts allowed within [`MettaConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which [`MettaConfig::max_restarts`] is counted.
    pub restart_window: Duration,
}

impl MettaConfig {
    /// A config with sensible defaults for `worker_bin`.
    pub fn new(worker_bin: impl Into<PathBuf>) -> Self {
        Self {
            worker_bin: worker_bin.into(),
            state_dir: None,
            env: Vec::new(),
            spawn_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(30),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// A classified coprocessor failure (mirrors [`ErrorClass`] + transport faults).
#[derive(Debug, thiserror::Error)]
pub enum MettaError {
    /// A bad request (unparseable atom, unknown id, CAS mismatch).
    #[error("bad request: {0}")]
    BadRequest(String),
    /// The op needs the `hyperon` engine, which this worker build does not provide.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A transient transport/engine fault — a retry on a fresh worker may succeed.
    #[error("transient: {0}")]
    Transient(String),
    /// Unrecoverable (crash-loop meltdown, internal bug).
    #[error("fatal: {0}")]
    Fatal(String),
    /// The op was cancelled / timed out cooperatively.
    #[error("cancelled")]
    Cancelled,
}

impl MettaError {
    fn from_class(class: ErrorClass, message: String) -> Self {
        match class {
            ErrorClass::BadRequest => MettaError::BadRequest(message),
            ErrorClass::Unsupported => MettaError::Unsupported(message),
            ErrorClass::Transient => MettaError::Transient(message),
            ErrorClass::Fatal => MettaError::Fatal(message),
            ErrorClass::Cancelled => MettaError::Cancelled,
        }
    }

    /// Whether the failure warrants tearing down the worker so the next call respawns it.
    fn should_replace_worker(&self) -> bool {
        matches!(self, MettaError::Transient(_) | MettaError::Fatal(_))
    }
}

/// A supervised client over a single `daemon-metta` worker process.
pub struct MettaCoprocessor {
    inner: Arc<Inner>,
}

struct Inner {
    cfg: MettaConfig,
    worker: Mutex<Option<Worker>>,
    restarts: Mutex<Vec<Instant>>,
}

impl MettaCoprocessor {
    /// Build a coprocessor client for `cfg`. The worker is spawned lazily on the first request.
    pub fn new(cfg: MettaConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                cfg,
                worker: Mutex::new(None),
                restarts: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Issue one op against the (lazily spawned) worker. The client assigns the request id; the
    /// caller need not set it. Tears the worker down on a fault that warrants a fresh process.
    pub async fn request(&self, mut cmd: Command) -> Result<OpResponse, MettaError> {
        let mut guard = self.inner.worker.lock().await;
        if guard.is_none() {
            *guard = Some(self.inner.spawn_worker().await?);
        }
        let worker = guard.as_mut().expect("worker present after spawn");

        let request_id = worker.next_request_id;
        worker.next_request_id += 1;
        cmd.set_request_id(request_id);

        let result = worker
            .round_trip(&cmd, request_id, self.inner.cfg.op_timeout)
            .await;

        if let Err(ref failure) = result {
            if failure.should_replace_worker() {
                if let Some(mut dead) = guard.take() {
                    dead.shutdown().await;
                }
            }
        }
        result
    }

    /// Probe the worker (spawning it if needed); returns `Ok(())` if it answers `Pong`.
    pub async fn ping(&self) -> Result<(), MettaError> {
        let mut guard = self.inner.worker.lock().await;
        if guard.is_none() {
            *guard = Some(self.inner.spawn_worker().await?);
        }
        let worker = guard.as_mut().expect("worker present after spawn");
        match worker.send_and_wait_pong(self.inner.cfg.op_timeout).await {
            Ok(()) => Ok(()),
            Err(e) => {
                if let Some(mut dead) = guard.take() {
                    dead.shutdown().await;
                }
                Err(e)
            }
        }
    }

    /// Gracefully stop the worker (if any). Idempotent.
    pub async fn shutdown(&self) {
        let mut guard = self.inner.worker.lock().await;
        if let Some(mut worker) = guard.take() {
            worker.shutdown().await;
        }
    }
}

impl Inner {
    /// Spawn a fresh worker, enforcing the crash-loop meltdown budget.
    async fn spawn_worker(&self) -> Result<Worker, MettaError> {
        {
            let mut restarts = self.restarts.lock().await;
            let now = Instant::now();
            restarts.retain(|t| now.duration_since(*t) < self.cfg.restart_window);
            if restarts.len() as u32 >= self.cfg.max_restarts {
                return Err(MettaError::Fatal(format!(
                    "daemon-metta worker crash-loop: {} restarts within {:?}",
                    restarts.len(),
                    self.cfg.restart_window
                )));
            }
            restarts.push(now);
        }
        Worker::spawn(&self.cfg).await
    }
}

/// A live worker process: the framed writer, an event inbox fed by a reader task, the child guard.
struct Worker {
    writer: CutWriter,
    events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    child: ChildGuard,
    next_request_id: u64,
    reader: JoinHandle<()>,
}

impl Worker {
    /// Spawn the worker and block until it reports `Ready` (or fails / times out).
    async fn spawn(cfg: &MettaConfig) -> Result<Worker, MettaError> {
        let mut args = Vec::new();
        if let Some(dir) = &cfg.state_dir {
            args.push("--state-dir".to_string());
            args.push(dir.to_string_lossy().into_owned());
        }
        let session = SessionId::new("daemon-metta-worker");
        // Crash-reporting correlation env (DSN + consent + session id + parent pid); no-op when the
        // node has no DSN. Mirrors the infer/train worker spawns.
        let mut env = cfg.env.clone();
        env.extend(daemon_telemetry::correlation_env(session.as_str()));
        let spec = PlacementSpec {
            program: cfg.worker_bin.clone(),
            args,
            env,
        };
        let Placement { channel, child } = ProcessProvisioner::new()
            .place(&session, spec)
            .await
            .map_err(|e| MettaError::Transient(format!("spawn daemon-metta worker: {e}")))?;

        let (writer, mut framed_reader) = channel.split();
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let reader = tokio::spawn(async move {
            while let Some(bytes) = framed_reader.recv().await {
                match protocol::decode::<Event>(&bytes) {
                    Ok(event) => {
                        if ev_tx.send(event).is_err() {
                            break;
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "daemon-metta: undecodable event frame"),
                }
            }
        });

        let mut worker = Worker {
            writer,
            events: ev_rx,
            child,
            next_request_id: 1,
            reader,
        };

        match tokio::time::timeout(cfg.spawn_timeout, worker.events.recv()).await {
            Err(_) => {
                worker.shutdown().await;
                Err(MettaError::Transient("worker spawn timed out".into()))
            }
            Ok(None) => {
                worker.shutdown().await;
                Err(MettaError::Transient("worker exited before ready".into()))
            }
            Ok(Some(Event::Ready { .. })) => Ok(worker),
            Ok(Some(Event::Error { class, message, .. })) => {
                worker.shutdown().await;
                Err(MettaError::from_class(class, message))
            }
            Ok(Some(other)) => {
                worker.shutdown().await;
                Err(MettaError::Fatal(format!(
                    "unexpected event during startup: {other:?}"
                )))
            }
        }
    }

    /// Send one command and await its matching reply (or a classified error), bounded by `timeout`.
    async fn round_trip(
        &mut self,
        cmd: &Command,
        request_id: u64,
        timeout: Duration,
    ) -> Result<OpResponse, MettaError> {
        self.send(cmd)
            .await
            .map_err(|e| MettaError::Transient(format!("send command: {e}")))?;
        loop {
            match tokio::time::timeout(timeout, self.events.recv()).await {
                Err(_) => {
                    return Err(MettaError::Transient(format!(
                        "worker watchdog: no reply within {timeout:?}"
                    )))
                }
                Ok(None) => {
                    return Err(MettaError::Transient("worker exited during request".into()))
                }
                Ok(Some(Event::Reply(resp))) if resp.request_id == request_id => return Ok(resp),
                Ok(Some(Event::Error {
                    request_id: rid,
                    class,
                    message,
                })) if rid.is_none() || rid == Some(request_id) => {
                    return Err(MettaError::from_class(class, message))
                }
                // A stale / unrelated frame — ignore and keep waiting.
                Ok(Some(_)) => {}
            }
        }
    }

    /// Send `Ping` and wait for `Pong`.
    async fn send_and_wait_pong(&mut self, timeout: Duration) -> Result<(), MettaError> {
        self.send(&Command::Ping)
            .await
            .map_err(|e| MettaError::Transient(format!("send ping: {e}")))?;
        loop {
            match tokio::time::timeout(timeout, self.events.recv()).await {
                Err(_) => return Err(MettaError::Transient("ping timed out".into())),
                Ok(None) => return Err(MettaError::Transient("worker exited during ping".into())),
                Ok(Some(Event::Pong)) => return Ok(()),
                Ok(Some(_)) => {}
            }
        }
    }

    /// Encode and send one command frame.
    async fn send(&self, cmd: &Command) -> std::io::Result<()> {
        let bytes = protocol::encode(cmd)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.writer.send(&bytes).await
    }

    /// Best-effort graceful stop: ask the worker to exit, kill + reap the child, stop the reader.
    async fn shutdown(&mut self) {
        let _ = self.send(&Command::Shutdown).await;
        self.child.shutdown().await;
        self.reader.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bogus worker binary makes every spawn fail; the supervisor must trip the crash-loop
    /// meltdown to `Fatal` after `max_restarts` attempts within the window.
    #[tokio::test]
    async fn crash_loop_trips_meltdown() {
        let mut cfg = MettaConfig::new("/nonexistent/daemon-metta-worker-binary");
        cfg.max_restarts = 2;
        cfg.restart_window = Duration::from_secs(60);
        let copro = MettaCoprocessor::new(cfg);

        // Each spawn fails (Transient), counting against the restart budget.
        for _ in 0..2 {
            let err = copro
                .request(Command::Inspect { request_id: 0 })
                .await
                .expect_err("spawn of a bogus binary must fail");
            assert!(matches!(err, MettaError::Transient(_)), "got {err:?}");
        }
        // The third attempt exceeds the budget and melts down to Fatal.
        let err = copro
            .request(Command::Inspect { request_id: 0 })
            .await
            .expect_err("meltdown");
        assert!(matches!(err, MettaError::Fatal(_)), "got {err:?}");
    }
}
