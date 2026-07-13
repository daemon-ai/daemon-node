// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-train-client` — the node-side training-worker supervisor.
//!
//! [`TrainSupervisor`] is to the `daemon-train` worker what `LocalProvider` is to the inference
//! worker and `MettaCoprocessor` is to the MeTTa worker: it lazily spawns the child over a
//! length-framed [`CutChannel`], speaks the worker protocol
//! ([`daemon_swarm_run::protocol`], swarm-training-spec.md §10.2), respawns with backoff after a
//! crash / transport fault, and trips a crash-loop "meltdown" to [`TrainClientError::Fatal`] when
//! restarts exceed a budget within a sliding window (§13).
//!
//! It links only the light node-side crates — never wasmtime / Burn — so the daemon stays out of
//! the worker fault domain (§10.1, §10.5). The node keeps only durable *intent* ("be joined to run
//! X under policy Y"); supervision converges the worker back to desired state after any crash.
//!
//! [`CutChannel`]: daemon_provision::CutChannel

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use daemon_common::SessionId;
use daemon_provision::{
    ChildGuard, CutWriter, Placement, PlacementSpec, ProcessProvisioner, Provisioner,
};
use daemon_swarm_run::protocol::{
    self, Command, Eligibility, ErrorClass, Event, Hardware, JoinPolicy, LeaveMode,
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// A live worker→node event pump sink. When set (by [`TrainSupervisor::join_streaming`]) the
/// worker's reader task routes every decoded [`Event`] here (the continuous round stream the node's
/// `SwarmService` consumes) instead of the request/reply inbox; cleared automatically when the
/// receiver is dropped. Shared with each spawned `Worker`'s reader so a respawn keeps pumping.
type PumpSink = Arc<StdMutex<Option<UnboundedSender<Event>>>>;

/// Construction + tuning for a [`TrainSupervisor`]'s worker (mirrors `WorkerConfig` / `MettaConfig`).
#[derive(Clone, Debug)]
pub struct TrainClientConfig {
    /// Path to the `daemon-train` worker binary.
    pub worker_bin: PathBuf,
    /// Arguments passed to the worker (e.g. `--backend cpu`).
    pub args: Vec<String>,
    /// Extra environment variables set on the worker child (e.g. `CUDA_VISIBLE_DEVICES`).
    pub env: Vec<(String, String)>,
    /// How long to wait for `Event::Ready` after spawning.
    pub spawn_timeout: Duration,
    /// How long to wait for a command reply before declaring a transport fault.
    pub op_timeout: Duration,
    /// Crash-loop meltdown: max restarts allowed within [`TrainClientConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which [`TrainClientConfig::max_restarts`] is counted.
    pub restart_window: Duration,
    /// Backoff applied before a *respawn* (never the first spawn).
    pub respawn_backoff: Duration,
}

impl TrainClientConfig {
    /// A config with sensible supervision defaults for `worker_bin`.
    #[must_use]
    pub fn new(worker_bin: impl Into<PathBuf>) -> Self {
        Self {
            worker_bin: worker_bin.into(),
            args: Vec::new(),
            env: Vec::new(),
            spawn_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(30),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
            respawn_backoff: Duration::from_millis(200),
        }
    }
}

/// Errors surfaced by the training-worker supervisor.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TrainClientError {
    /// A transient transport/spawn fault — a retry on a fresh worker may succeed.
    #[error("transient: {0}")]
    Transient(String),
    /// A classified worker [`Event::Error`] (mapped from its [`ErrorClass`]).
    #[error("worker error ({class:?}): {detail}")]
    Worker {
        /// The worker's failure class.
        class: ErrorClass,
        /// The worker's detail message.
        detail: String,
    },
    /// Unrecoverable: the worker crash-looped past its meltdown budget, or an internal bug.
    #[error("fatal: {0}")]
    Fatal(String),
    /// A codec error framing/parsing a worker frame.
    #[error("codec: {0}")]
    Codec(String),
}

impl TrainClientError {
    fn from_worker(class: ErrorClass, detail: String) -> Self {
        Self::Worker { class, detail }
    }

    /// Whether the failure warrants tearing down the worker so the next call respawns a fresh one.
    fn should_replace_worker(&self) -> bool {
        match self {
            TrainClientError::Transient(_) | TrainClientError::Codec(_) => true,
            TrainClientError::Fatal(_) => true,
            // A classified worker error keeps the worker unless it is a hard failure.
            TrainClientError::Worker { class, .. } => {
                matches!(class, ErrorClass::OutOfMemory | ErrorClass::Fatal)
            }
        }
    }
}

/// A supervised client over a single `daemon-train` worker process.
pub struct TrainSupervisor {
    inner: Arc<Inner>,
}

struct Inner {
    cfg: TrainClientConfig,
    worker: Mutex<Option<Worker>>,
    restarts: Mutex<Vec<Instant>>,
    /// Total spawns; respawns are spawns beyond the first (observability + backoff gate).
    spawns: Mutex<u32>,
    /// The live event-pump sink (A3). Shared into every spawned worker's reader so the continuous
    /// stream survives a respawn; `None` outside a streaming join (request/reply routing).
    pump: PumpSink,
}

impl TrainSupervisor {
    /// Build a supervisor for `cfg`. The worker is spawned lazily on the first request.
    #[must_use]
    pub fn new(cfg: TrainClientConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                cfg,
                worker: Mutex::new(None),
                restarts: Mutex::new(Vec::new()),
                spawns: Mutex::new(0),
                pump: Arc::new(StdMutex::new(None)),
            }),
        }
    }

    /// Probe the (lazily spawned) worker for hardware + capabilities (§10.2).
    pub async fn probe(&self) -> Result<Hardware, TrainClientError> {
        self.exchange(Command::Probe, |ev| match ev {
            Event::Probed(hw) => Some(Ok(hw)),
            _ => None,
        })
        .await
    }

    /// Assess a run envelope against this peer's effective resources (§6.5, read-only).
    pub async fn assess(&self, envelope: Vec<u8>) -> Result<Eligibility, TrainClientError> {
        self.exchange(Command::AssessRun { envelope }, |ev| match ev {
            Event::Assessed(elig) => Some(Ok(elig)),
            _ => None,
        })
        .await
    }

    /// Join a run; resolves once the worker acknowledges with its first `RunPhase` (§10.2). The
    /// full event stream is consumed by the round loop in a later wave.
    pub async fn join(
        &self,
        run_id: impl Into<String>,
        coordinator: impl Into<String>,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<(), TrainClientError> {
        let cmd = Command::JoinRun {
            run_id: run_id.into(),
            coordinator: coordinator.into(),
            credentials,
            policy,
        };
        self.exchange(cmd, |ev| match ev {
            Event::RunPhase { .. } => Some(Ok(())),
            _ => None,
        })
        .await
    }

    /// Join a run and return the **continuous** worker event stream (A3 — the event pump).
    ///
    /// Unlike [`join`](Self::join) (which resolves on the first `RunPhase` and drops the rest), this
    /// installs a pump sink so the worker's reader routes **every** subsequent [`Event`]
    /// (`RunPhase`/`Metric`/`RoundOutcome`/`Warning` per round, plus the additive `MicroBatch` /
    /// `OomLadder` telemetry) into the returned receiver. The node's `SwarmService` drains it into
    /// `handle_worker_event`, so `swarm.db` reflects live round progression (§10.3/§10.4). The sink
    /// clears automatically when the receiver is dropped (back to request/reply routing).
    pub async fn join_streaming(
        &self,
        run_id: impl Into<String>,
        coordinator: impl Into<String>,
        credentials: Vec<u8>,
        policy: JoinPolicy,
    ) -> Result<UnboundedReceiver<Event>, TrainClientError> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        *self.inner.pump.lock().expect("pump lock") = Some(tx);
        let cmd = Command::JoinRun {
            run_id: run_id.into(),
            coordinator: coordinator.into(),
            credentials,
            policy,
        };
        // One-way: the worker streams events (incl. the first RunPhase) over the pump, so we do not
        // block on a reply here. A spawn/transport fault clears the pump + surfaces the error.
        if let Err(e) = self.send_oneway(cmd).await {
            *self.inner.pump.lock().expect("pump lock") = None;
            return Err(e);
        }
        Ok(rx)
    }

    /// Send a GPU-governor throttle lever (§10.5). Fire-and-forget (no reply frame).
    pub async fn throttle(
        &self,
        vram_cap_mb: Option<u32>,
        duty_cycle_pct: Option<u8>,
        paused: bool,
    ) -> Result<(), TrainClientError> {
        self.send_oneway(Command::Throttle {
            vram_cap_mb,
            duty_cycle_pct,
            paused,
        })
        .await
    }

    /// Leave a run (§10.2). Fire-and-forget.
    pub async fn leave(
        &self,
        run_id: impl Into<String>,
        mode: LeaveMode,
    ) -> Result<(), TrainClientError> {
        self.send_oneway(Command::Leave {
            run_id: run_id.into(),
            mode,
        })
        .await
    }

    /// Liveness check: spawn if needed, then `Ping`/`Pong`.
    pub async fn ping(&self) -> Result<(), TrainClientError> {
        self.exchange(Command::Ping, |ev| match ev {
            Event::Pong => Some(Ok(())),
            _ => None,
        })
        .await
    }

    /// Total worker respawns so far (spawns beyond the first) — the health `restarts` count.
    pub async fn restarts(&self) -> u32 {
        self.inner.spawns.lock().await.saturating_sub(1)
    }

    /// Gracefully stop the worker (if any). Idempotent.
    pub async fn shutdown(&self) {
        let mut guard = self.inner.worker.lock().await;
        if let Some(mut worker) = guard.take() {
            worker.shutdown().await;
        }
    }

    /// Send a command and await the first event `extract` accepts, mapping `Event::Error` frames to
    /// [`TrainClientError::Worker`] and tearing the worker down on a fault that warrants a respawn.
    async fn exchange<T>(
        &self,
        cmd: Command,
        extract: impl Fn(Event) -> Option<Result<T, TrainClientError>>,
    ) -> Result<T, TrainClientError> {
        let mut guard = self.inner.worker.lock().await;
        if guard.is_none() {
            *guard = Some(self.inner.spawn_worker().await?);
        }
        let worker = guard.as_mut().expect("worker present after spawn");
        let result = worker
            .round_trip(&cmd, self.inner.cfg.op_timeout, &extract)
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

    /// Send a one-way command (no reply expected), spawning the worker if needed.
    async fn send_oneway(&self, cmd: Command) -> Result<(), TrainClientError> {
        let mut guard = self.inner.worker.lock().await;
        if guard.is_none() {
            *guard = Some(self.inner.spawn_worker().await?);
        }
        let worker = guard.as_mut().expect("worker present after spawn");
        let result = worker.send(&cmd).await;
        if result.is_err() {
            if let Some(mut dead) = guard.take() {
                dead.shutdown().await;
            }
        }
        result
    }
}

impl Inner {
    /// Spawn a fresh worker, enforcing the crash-loop meltdown budget + respawn backoff.
    async fn spawn_worker(&self) -> Result<Worker, TrainClientError> {
        {
            let mut restarts = self.restarts.lock().await;
            let now = Instant::now();
            restarts.retain(|t| now.duration_since(*t) < self.cfg.restart_window);
            if restarts.len() as u32 >= self.cfg.max_restarts {
                return Err(TrainClientError::Fatal(format!(
                    "daemon-train worker crash-loop: {} restarts within {:?}",
                    restarts.len(),
                    self.cfg.restart_window
                )));
            }
            restarts.push(now);
        }
        // Backoff before a respawn (the first spawn is immediate).
        {
            let mut spawns = self.spawns.lock().await;
            if *spawns > 0 && !self.cfg.respawn_backoff.is_zero() {
                tokio::time::sleep(self.cfg.respawn_backoff).await;
            }
            *spawns += 1;
        }
        Worker::spawn(&self.cfg, self.pump.clone()).await
    }
}

/// A live worker process: the framed writer, an event inbox fed by a reader task, the child guard.
struct Worker {
    writer: CutWriter,
    events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    child: ChildGuard,
    reader: JoinHandle<()>,
}

impl Worker {
    /// Spawn the worker and block until it reports `Ready` (or fails / times out).
    async fn spawn(cfg: &TrainClientConfig, pump: PumpSink) -> Result<Worker, TrainClientError> {
        let session = SessionId::new("daemon-train-worker");
        // Crash-reporting correlation: forward the node's DSN + current consent and tag the child
        // with this placement's session id + our pid, so a train-worker crash correlates with the
        // node in one Sentry project. A no-op env-wise when no DSN is set.
        let mut env = cfg.env.clone();
        env.extend(daemon_telemetry::correlation_env(session.as_str()));
        let spec = PlacementSpec {
            program: cfg.worker_bin.clone(),
            args: cfg.args.clone(),
            env,
        };
        let Placement { channel, child } = ProcessProvisioner::new()
            .place(&session, spec)
            .await
            .map_err(|e| TrainClientError::Transient(format!("spawn daemon-train worker: {e}")))?;

        let (writer, mut framed_reader) = channel.split();
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let reader = tokio::spawn(async move {
            // The first event (`Ready`) always goes to the request/reply inbox so the spawn
            // handshake completes even during a streaming respawn; subsequent events route to the
            // live pump sink when one is installed (A3 event pump), else to the inbox.
            let mut first = true;
            while let Some(bytes) = framed_reader.recv().await {
                match protocol::decode::<Event>(&bytes) {
                    Ok(event) => {
                        if !first {
                            let sink = pump.lock().expect("pump lock").clone();
                            if let Some(tx) = sink {
                                if tx.send(event).is_err() {
                                    // The node dropped the stream (run left): back to inbox routing.
                                    *pump.lock().expect("pump lock") = None;
                                }
                                continue;
                            }
                        }
                        first = false;
                        if ev_tx.send(event).is_err() {
                            break;
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "daemon-train: undecodable event frame"),
                }
            }
        });

        let mut worker = Worker {
            writer,
            events: ev_rx,
            child,
            reader,
        };

        match tokio::time::timeout(cfg.spawn_timeout, worker.events.recv()).await {
            Err(_) => {
                worker.shutdown().await;
                Err(TrainClientError::Transient("worker spawn timed out".into()))
            }
            Ok(None) => {
                worker.shutdown().await;
                Err(TrainClientError::Transient(
                    "worker exited before ready".into(),
                ))
            }
            Ok(Some(Event::Ready { .. })) => Ok(worker),
            Ok(Some(Event::Error { class, detail })) => {
                worker.shutdown().await;
                Err(TrainClientError::from_worker(class, detail))
            }
            Ok(Some(other)) => {
                worker.shutdown().await;
                Err(TrainClientError::Fatal(format!(
                    "unexpected event during startup: {other:?}"
                )))
            }
        }
    }

    /// Send a command and await the first event `extract` accepts (skipping streaming progress
    /// events), bounded by `timeout`. `Event::Error` frames become [`TrainClientError::Worker`].
    async fn round_trip<T>(
        &mut self,
        cmd: &Command,
        timeout: Duration,
        extract: &impl Fn(Event) -> Option<Result<T, TrainClientError>>,
    ) -> Result<T, TrainClientError> {
        self.send(cmd).await?;
        loop {
            match tokio::time::timeout(timeout, self.events.recv()).await {
                Err(_) => {
                    return Err(TrainClientError::Transient(format!(
                        "worker watchdog: no reply within {timeout:?}"
                    )))
                }
                Ok(None) => {
                    return Err(TrainClientError::Transient(
                        "worker exited during request".into(),
                    ))
                }
                Ok(Some(Event::Error { class, detail })) => {
                    return Err(TrainClientError::from_worker(class, detail))
                }
                Ok(Some(event)) => {
                    if let Some(result) = extract(event) {
                        return result;
                    }
                    // A streaming/progress event unrelated to this request — keep waiting.
                }
            }
        }
    }

    /// Encode and send one command frame.
    async fn send(&self, cmd: &Command) -> Result<(), TrainClientError> {
        let bytes = protocol::encode(cmd).map_err(|e| TrainClientError::Codec(e.to_string()))?;
        self.writer
            .send(&bytes)
            .await
            .map_err(|e| TrainClientError::Transient(format!("send command: {e}")))
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

    #[test]
    fn config_has_defaults() {
        let cfg = TrainClientConfig::new("/usr/bin/daemon-train");
        assert_eq!(cfg.max_restarts, 3);
        assert_eq!(cfg.spawn_timeout, Duration::from_secs(30));
    }

    /// A bogus worker binary makes every spawn fail; the supervisor must trip the crash-loop
    /// meltdown to `Fatal` after `max_restarts` attempts within the window (CLI-3).
    #[tokio::test]
    async fn supervisor_meltdown() {
        let mut cfg = TrainClientConfig::new("/nonexistent/daemon-train-worker-binary");
        cfg.max_restarts = 2;
        cfg.restart_window = Duration::from_secs(60);
        cfg.respawn_backoff = Duration::from_millis(1);
        let sup = TrainSupervisor::new(cfg);

        for _ in 0..2 {
            let err = sup
                .probe()
                .await
                .expect_err("spawn of a bogus binary must fail");
            assert!(matches!(err, TrainClientError::Transient(_)), "got {err:?}");
        }
        let err = sup.probe().await.expect_err("meltdown");
        assert!(matches!(err, TrainClientError::Fatal(_)), "got {err:?}");
    }
}
