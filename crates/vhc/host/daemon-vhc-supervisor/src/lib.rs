// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-supervisor` — the node-side training-worker supervisor.
//!
//! [`TrainSupervisor`] is to the `daemon-vhc-host` worker what `LocalProvider` is to the inference
//! worker and `MettaCoprocessor` is to the MeTTa worker: it lazily spawns the child over a
//! length-framed [`CutChannel`], speaks the worker protocol
//! ([`daemon_vhc_session::protocol`], swarm-training-spec.md §10.2), respawns with backoff after a
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
use daemon_vhc_session::protocol::{
    self, Command, Eligibility, ErrorClass, Event, Hardware, JoinPolicy, LeaveMode,
};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

/// A live worker→node event pump sink. When set (by [`TrainSupervisor::join_streaming`]) the
/// worker's reader task routes every decoded [`Event`] here (the continuous round stream the node's
/// `VhcService` consumes) instead of the request/reply inbox; cleared automatically when the
/// receiver is dropped. Shared with each spawned `Worker`'s reader so a respawn keeps pumping.
type PumpSink = Arc<StdMutex<Option<UnboundedSender<Event>>>>;

/// The graceful-leave terminal waiter: `(run label, fire-once sender)` — see [`Inner::terminal`].
type TerminalSlot = Arc<StdMutex<Option<(String, tokio::sync::oneshot::Sender<()>)>>>;

/// Construction + tuning for a [`TrainSupervisor`]'s worker (mirrors `WorkerConfig` / `MettaConfig`).
#[derive(Clone, Debug)]
pub struct TrainClientConfig {
    /// Path to the `daemon-vhc-host` worker binary.
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

/// The outcome of a [`TrainSupervisor::switch_module`] live upgrade (ABI §10.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchOutcome {
    /// The new module activated locally under the target epoch (§10.3 step 6): the old module
    /// quiesced, its state migrated, and the new module resumed — no process restart.
    Activated {
        /// The epoch now running locally (the already-committed target epoch).
        epoch: u64,
        /// The new module hash now bound to the role-instance.
        module: [u8; 32],
        /// Rollback-and-retry cycles used before activation (`0` on a clean first migration).
        retries: u32,
    },
    /// The local transaction failed closed / exhausted its retries and **left the run** (§10.3
    /// step 7): the chain is NOT rolled back and the old epoch is not resumed; the worker is
    /// unharmed, and the node churns / re-admits. `reason` is the typed `LeaveReason` detail.
    Left {
        /// Why the worker left the run (the `LeaveReason` display).
        reason: String,
    },
    /// The switch was refused BEFORE the transaction touched the running instance (§10.3
    /// pre-transaction refusals: no live instance, unresolvable/mismatched target artifact,
    /// missing or mis-scoped re-issued identity, an admission refusal evaluated ahead of the
    /// fence). The old module keeps running untouched; the node reassesses/reprovisions.
    Refused {
        /// Why the switch was refused.
        reason: String,
    },
}

/// A supervised client over a single `daemon-vhc-host` worker process.
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
    /// A one-shot terminal waiter: `leave()` installs `(run_id, tx)` and the reader fires it on
    /// that run's `RunTerminated`, so a GRACEFUL leave returns only after the drain (quiesce +
    /// snapshot + checkpoint publication) actually finished — the caller's teardown must never
    /// race the drain, and the drain's events must flow through the still-armed pump.
    terminal: TerminalSlot,
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
                terminal: Arc::new(StdMutex::new(None)),
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

    /// Assess a run envelope against this peer's effective resources (§6.5, read-only). `role`
    /// names the envelope role to assess for (node-directed selection); `None` = the default.
    pub async fn assess(
        &self,
        envelope: Vec<u8>,
        role: Option<String>,
    ) -> Result<Eligibility, TrainClientError> {
        self.exchange(Command::AssessRun { envelope, role }, |ev| match ev {
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
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<(), TrainClientError> {
        let cmd = Command::JoinRun {
            run_id: run_id.into(),
            coordinator: coordinator.into(),
            credentials,
            policy,
            // The node-minted admitted tuple (carrying the incarnation this instance runs as);
            // the worker rederives + re-verifies it before running.
            admitted_tuple,
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
    /// (`RunPhase`/`Metric`/`RoundOutcome`/`Warning` per round) into the returned receiver. The
    /// node's `VhcService` drains it into
    /// `handle_worker_event`, so `vhc.db` reflects live round progression (§10.3/§10.4). The sink
    /// clears automatically when the receiver is dropped (back to request/reply routing).
    pub async fn join_streaming(
        &self,
        run_id: impl Into<String>,
        coordinator: impl Into<String>,
        credentials: Vec<u8>,
        policy: JoinPolicy,
        admitted_tuple: Option<protocol::AdmittedTuple>,
    ) -> Result<UnboundedReceiver<Event>, TrainClientError> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        *self.inner.pump.lock().expect("pump lock") = Some(tx);
        let cmd = Command::JoinRun {
            run_id: run_id.into(),
            coordinator: coordinator.into(),
            credentials,
            policy,
            // The node-minted admitted tuple (carrying the incarnation this instance runs as).
            admitted_tuple,
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

    /// Leave a run (§10.2), returning only after the run's terminal `RunTerminated` (bounded).
    ///
    /// The pump sink stays ARMED while the drain runs, so a graceful leave's drain events —
    /// above all `CheckpointPublished`, which the node turns into the registry checkpoint
    /// pointer — flow to the node's stream instead of vanishing into the request/reply inbox.
    /// Only after the terminal (or the drain-deadline-scaled timeout, fail-loud) is the sink
    /// cleared so subsequent request/reply commands (ping/probe) route to the inbox again.
    pub async fn leave(
        &self,
        run_id: impl Into<String>,
        mode: LeaveMode,
    ) -> Result<(), TrainClientError> {
        let run_id = run_id.into();
        let (tx, rx) = tokio::sync::oneshot::channel();
        *self.inner.terminal.lock().expect("terminal lock") = Some((run_id.clone(), tx));
        let sent = self
            .send_oneway(Command::Leave {
                run_id: run_id.clone(),
                mode,
            })
            .await;
        if sent.is_err() {
            *self.inner.terminal.lock().expect("terminal lock") = None;
            *self.inner.pump.lock().expect("pump lock") = None;
            return sent;
        }
        // The worker's drain deadline is 10s (quiesce + snapshot + checkpoint put); triple it so
        // a slow content plane never truncates a healthy drain. A timeout is loud but not fatal:
        // the caller's teardown proceeds and the terminal (if it still lands) is idempotent.
        if tokio::time::timeout(Duration::from_secs(30), rx)
            .await
            .is_err()
        {
            tracing::warn!(
                run_id,
                "leave: no RunTerminated within the drain window; tearing down anyway"
            );
            *self.inner.terminal.lock().expect("terminal lock") = None;
        }
        *self.inner.pump.lock().expect("pump lock") = None;
        Ok(())
    }

    /// Initiate a live module upgrade for a running instance (ABI §10.3; architecture §5.4).
    ///
    /// Sends [`Command::SwitchModule`] (the run-level upgrade record has already committed to the
    /// transition chain) and awaits the terminal fact: [`Event::ModuleSwitched`] →
    /// [`SwitchOutcome::Activated`], or a fail-closed / exhausted transaction that leaves the run
    /// (the worker answers `Event::Error{class: Module, ..}`, the worker unharmed) →
    /// [`SwitchOutcome::Left`]. Any other worker fault propagates as a [`TrainClientError`].
    #[allow(clippy::too_many_arguments)]
    pub async fn switch_module(
        &self,
        run_id: impl Into<String>,
        epoch: u64,
        role: impl Into<String>,
        new_module: [u8; 32],
        grants_hash: [u8; 32],
        deadline_ms: u64,
        admitted_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple>,
    ) -> Result<SwitchOutcome, TrainClientError> {
        let cmd = Command::SwitchModule {
            run_id: run_id.into(),
            epoch,
            role: role.into(),
            new_module,
            grants_hash,
            deadline_ms,
            admitted_tuple,
        };
        let res = self
            .exchange(cmd, |ev| match ev {
                Event::ModuleSwitched {
                    epoch,
                    module,
                    retries,
                    ..
                } => Some(Ok(SwitchOutcome::Activated {
                    epoch,
                    module,
                    retries,
                })),
                Event::SwitchRefused { reason, .. } => Some(Ok(SwitchOutcome::Refused { reason })),
                _ => None,
            })
            .await;
        match res {
            Ok(outcome) => Ok(outcome),
            // A fail-closed / exhausted local transaction leaves the run: the worker answers
            // `Error{class: Module}` (unharmed). That is a normal upgrade outcome, not a fault —
            // the node churns / re-admits on it. `Module` is not a worker-replacing class, so the
            // supervised worker survives (as the spec requires: the old epoch is simply left).
            Err(TrainClientError::Worker {
                class: ErrorClass::Module,
                detail,
            }) => Ok(SwitchOutcome::Left { reason: detail }),
            Err(e) => Err(e),
        }
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
                    "daemon-vhc-host worker crash-loop: {} restarts within {:?}",
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
        Worker::spawn(&self.cfg, self.pump.clone(), self.terminal.clone()).await
    }
}

/// A live worker process: the framed writer, an event inbox fed by a reader task, the child guard.
struct Worker {
    writer: CutWriter,
    events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    child: ChildGuard,
    reader: JoinHandle<()>,
    /// The shared pump sink, cleared on teardown so the node's stream receiver closes even when
    /// the reader task is aborted (a supervisor-initiated replacement must be as observable as a
    /// child crash).
    pump: PumpSink,
}

impl Worker {
    /// Spawn the worker and block until it reports `Ready` (or fails / times out).
    async fn spawn(
        cfg: &TrainClientConfig,
        pump: PumpSink,
        terminal: TerminalSlot,
    ) -> Result<Worker, TrainClientError> {
        let session = SessionId::new("daemon-vhc-worker");
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
            .map_err(|e| {
                TrainClientError::Transient(format!("spawn daemon-vhc-host worker: {e}"))
            })?;

        let (writer, mut framed_reader) = channel.split();
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let reader_pump = pump.clone();
        let reader = tokio::spawn(async move {
            // The first event (`Ready`) always goes to the request/reply inbox so the spawn
            // handshake completes even during a streaming respawn; subsequent events route to the
            // live pump sink when one is installed (A3 event pump), else to the inbox.
            let mut first = true;
            while let Some(bytes) = framed_reader.recv().await {
                match protocol::decode::<Event>(&bytes) {
                    Ok(event) => {
                        if matches!(
                            event,
                            Event::RunTerminated { .. } | Event::CheckpointPublished { .. }
                        ) {
                            tracing::debug!(?event, "worker event received (lifecycle)");
                        }
                        // Fire the graceful-leave terminal waiter (regardless of routing): the
                        // leave caller blocks on this run's RunTerminated. Fired AFTER the event
                        // has been routed below, so every preceding drain event (above all
                        // CheckpointPublished) is already in the node's stream when the leave
                        // returns and teardown proceeds.
                        let fire_terminal = if let Event::RunTerminated { run_id, .. } = &event {
                            let mut slot = terminal.lock().expect("terminal lock");
                            if slot.as_ref().is_some_and(|(r, _)| r == run_id) {
                                slot.take().map(|(_, tx)| tx)
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let mut routed_to_pump = false;
                        if !first {
                            let sink = reader_pump.lock().expect("pump lock").clone();
                            if let Some(tx) = sink {
                                if tx.send(event.clone()).is_ok() {
                                    routed_to_pump = true;
                                } else {
                                    // The node dropped the stream (run left): clear the sink and
                                    // fall through so THIS event still reaches the inbox.
                                    *reader_pump.lock().expect("pump lock") = None;
                                }
                            }
                        }
                        if !routed_to_pump {
                            first = false;
                            if ev_tx.send(event).is_err() {
                                if let Some(tx) = fire_terminal {
                                    let _ = tx.send(());
                                }
                                break;
                            }
                        }
                        if let Some(tx) = fire_terminal {
                            let _ = tx.send(());
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "daemon-vhc-host: undecodable event frame")
                    }
                }
            }
            // The child's stream ended (process exit / stdio cut severed): dropping the sink
            // closes the node-held stream receiver, so the node OBSERVES the transport loss and
            // its run-instance reconciliation classifies the recoverable failure — never a
            // silently-dead pump behind a live-looking receiver.
            *reader_pump.lock().expect("pump lock") = None;
        });

        let mut worker = Worker {
            writer,
            events: ev_rx,
            child,
            reader,
            pump,
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
    /// The pump sink clears so any node-held stream receiver observes the teardown (the reader's
    /// own end-of-stream clear does not run when the task is aborted).
    async fn shutdown(&mut self) {
        let _ = self.send(&Command::Shutdown).await;
        self.child.shutdown().await;
        self.reader.abort();
        *self.pump.lock().expect("pump lock") = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_has_defaults() {
        let cfg = TrainClientConfig::new("/usr/bin/daemon-vhc");
        assert_eq!(cfg.max_restarts, 3);
        assert_eq!(cfg.spawn_timeout, Duration::from_secs(30));
    }

    /// A bogus worker binary makes every spawn fail; the supervisor must trip the crash-loop
    /// meltdown to `Fatal` after `max_restarts` attempts within the window (CLI-3).
    #[tokio::test]
    async fn supervisor_meltdown() {
        let mut cfg = TrainClientConfig::new("/nonexistent/daemon-vhc-worker-binary");
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
