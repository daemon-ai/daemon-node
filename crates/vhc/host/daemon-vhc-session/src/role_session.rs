// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **role session** — the single generic runtime a worker process runs for ANY role in ANY
//! run (architecture §7; the opaque-host boundary).
//!
//! A role session binds exactly these host mechanisms and nothing else:
//!
//! 1. the admitted module bytes + config + grants (the assessed admitted tuple's artifacts);
//! 2. a channel transport provider ([`daemon_vhc_net::ControlPlane`] — opaque signed frames);
//! 3. payload + artifact providers ([`daemon_vhc_net::ContentStore`] — content-addressed, no
//!    run/round/peer coordinates);
//! 4. the certified per-run signing identity (the key is the pump's §12.1 signer; inbound
//!    verification carries the mandatory certificate check);
//! 5. lifecycle: spawn, throttle (hard pause), quiesce, cancel, terminal-outcome classification;
//! 6. the journal sink.
//!
//! FORBIDDEN here (and enforced structurally by the dependency gate): decoding SDK message
//! schemas, deriving or slicing batches, counting publishes, inspecting coordinator records,
//! branching on whether the role is trainer or coordinator. Frames are signed opaque bytes in
//! both directions; the module decides everything. Both the trainer and the consensus
//! coordinator module run through this session unmodified — that is the litmus test that the
//! boundary is generic.
//!
//! Shape: the session is a background task around the major-2 event pump
//! ([`daemon_vhc_host::run::start_run`]). Outbound, the guest's already-signed published frames
//! relay verbatim onto the control plane; inbound, every frame passes the §12.1 attach
//! (signature, scope, mandatory certificate chain, dedup/gap) before opaque delivery below the
//! pump. Capability calls (payload put/get, artifact fetch) are serviced against the
//! content-addressed providers; the pump — not the provider — re-verifies every fetched object.
//! The spawning caller keeps only a [`RoleHandle`]; the command loop above never blocks on run
//! execution.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use daemon_vhc_host::run::{
    start_run, JournalSink, OpOutcome, OpRequest, PumpHandle, RunConfig, RunEnd,
};
use daemon_vhc_host::trap::TrapCode;
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_net::{ContentStore, ControlPlane};
use daemon_vhc_proto::{to_canonical_vec, PeerId, RunKeyCertificate};
use tokio::sync::{mpsc, Notify};

use crate::attach::{Attach, CertCheck, InboundVerdict};
use crate::protocol::{Event, LeaveMode, TerminalOutcome};

/// How long an unresolved inbound sequence gap may stand before the session classifies it
/// unrecoverable (no backfill plane exists yet; reordered frames usually resolve in milliseconds).
const GAP_DEADLINE: Duration = Duration::from_secs(20);

/// The bounded hold for gapped/back-pressured frames awaiting re-presentation.
const HELD_FRAMES_MAX: usize = 256;

/// The session's internal tick (pending-frame retries, guest-end polling, gap aging).
const TICK: Duration = Duration::from_millis(50);

/// The transport + provider bindings a role session services capabilities against. All three are
/// OPAQUE seams: signed frames in/out, content-addressed bytes up/down — no schema, no
/// coordinates, no credentials cross this surface.
pub struct RoleProviders {
    /// The control plane (opaque signed frames, publish/subscribe).
    pub control: Arc<dyn ControlPlane>,
    /// The payload plane (module `payload_put`/`payload_get`, content-addressed).
    pub payloads: Arc<dyn ContentStore>,
    /// The artifact plane (module `data.fetch` by committed hash; the pump verifies + slices).
    pub artifacts: Arc<dyn ContentStore>,
}

/// Everything a role task binds, moved into the session at spawn. Construction is the caller's
/// (the worker binary authors the run config from the admitted tuple's artifacts; the node
/// delivers identity by keystore reference).
pub struct RoleSessionSpec {
    /// The admitted module bytes (hash-pinned upstream by the admitted tuple).
    pub module: Vec<u8>,
    /// The engine configuration (sandbox budgets, backend lane).
    pub engine: EngineConfig,
    /// The driver-facing run binding: execution identity, per-run signing seed, admitted
    /// config/grants bytes, admitted quotas/artifact grants. The signing seed is the certified
    /// per-run key's; the pump signs every published frame with it.
    pub run: RunConfig,
    /// The session's own run-key certificate (issued by the node's base identity; distributed to
    /// peers out of band / on join).
    pub own_cert: RunKeyCertificate,
    /// The base identities trusted to certify peer run keys (from the run's genesis/Authority
    /// configuration — never ambient config).
    pub trusted_bases: Vec<PeerId>,
    /// Peer certificates known at join (later arrivals ride the control plane).
    pub peer_certs: Vec<RunKeyCertificate>,
    /// The transport + provider bindings.
    pub providers: RoleProviders,
    /// The journal sink for this incarnation.
    pub journal: Box<dyn JournalSink>,
    /// The graceful-leave drain ceiling (the quiesce deadline).
    pub drain_deadline: Duration,
}

/// A throttle level (the owner's GPU-governor lever, forwarded verbatim by the worker loop).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThrottleLevel {
    /// Hard pause: event delivery freezes at the pump (the guest parks in `next_event`) until
    /// un-paused. A paused worker actually stops — this is not advisory.
    pub paused: bool,
    /// Cooperative duty-cycle percentage (`None` = unchanged), delivered as a `Budget` advisory
    /// the module paces itself against.
    pub duty_cycle_pct: Option<u8>,
    /// Cooperative VRAM cap in MiB (`None` = unchanged), delivered on the same advisory.
    pub vram_cap_mb: Option<u32>,
}

/// A command from the worker loop into the running role task.
enum RoleCommand {
    Throttle(ThrottleLevel),
    Leave(LeaveMode),
}

/// The handle the worker keeps per spawned role — the ONLY per-run state above the session
/// (crate ownership rule: the worker binary owns the command loop and the role-handle map,
/// nothing else).
pub struct RoleHandle {
    commands: mpsc::UnboundedSender<RoleCommand>,
    task: tokio::task::JoinHandle<()>,
    generation: u64,
    run_label: String,
}

impl RoleHandle {
    /// Forward a throttle level (hard pause / cooperative duty) into the session.
    pub fn throttle(&self, level: ThrottleLevel) {
        let _ = self.commands.send(RoleCommand::Throttle(level));
    }

    /// Ask the session to leave: graceful = quiesce (drain + snapshot + checkpoint) then end;
    /// immediate = stop now. The terminal event reports the outcome either way.
    pub fn leave(&self, mode: LeaveMode) {
        let _ = self.commands.send(RoleCommand::Leave(mode));
    }

    /// The role-instance generation (the never-reused incarnation id) stamped on every event
    /// this session emits.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The run label this handle serves (the worker map's key coordinates).
    #[must_use]
    pub fn run_label(&self) -> &str {
        &self.run_label
    }

    /// Whether the role task has reached its terminal state.
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.task.is_finished()
    }

    /// Await the role task's end (the terminal event has already been emitted when this
    /// resolves). Callers bound this with a drain deadline.
    pub async fn join(self) {
        let _ = self.task.await;
    }
}

/// Spawn a role session as a background task and return immediately. The caller's command loop
/// never blocks on run execution: module compilation, the join, and the whole run happen inside
/// the spawned task, which emits generation-stamped [`Event`]s (phases, metrics, warnings) and
/// exactly one terminal [`Event::RunTerminated`].
pub fn spawn_role(
    run_label: String,
    spec: RoleSessionSpec,
    events: mpsc::UnboundedSender<Event>,
) -> RoleHandle {
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
    let generation = spec.run.identity.instance;
    let label = run_label.clone();
    let task = tokio::spawn(async move {
        let epoch = spec.run.identity.epoch;
        let _ = events.send(Event::RunPhase {
            run_id: label.clone(),
            phase: "joining".into(),
            epoch,
            round: 0,
            generation,
        });
        let outcome = run_role(&label, spec, &events, cmd_rx, generation).await;
        let _ = events.send(Event::RunTerminated {
            run_id: label,
            generation,
            outcome,
        });
    });
    RoleHandle {
        commands: cmd_tx,
        task,
        generation,
        run_label,
    }
}

/// The session body: bind, run, service, classify. Returns the terminal outcome (the task
/// wrapper emits it).
async fn run_role(
    run_label: &str,
    spec: RoleSessionSpec,
    events: &mpsc::UnboundedSender<Event>,
    mut commands: mpsc::UnboundedReceiver<RoleCommand>,
    generation: u64,
) -> TerminalOutcome {
    let RoleSessionSpec {
        module,
        engine,
        run: run_cfg,
        own_cert,
        trusted_bases,
        mut peer_certs,
        providers,
        journal,
        drain_deadline,
    } = spec;

    let identity = run_cfg.identity.clone();
    let own_sender = own_cert.body.run_key;
    if !peer_certs.contains(&own_cert) {
        peer_certs.push(own_cert);
    }

    // Bind the runtime. Setup failures are terminal: the artifacts were already admitted, so a
    // module that cannot even instantiate is not a transient environment fault.
    let worker = match Worker::new(engine) {
        Ok(w) => w,
        Err(e) => {
            return TerminalOutcome::FailedTerminal {
                reason: format!("engine construction: {e}"),
            }
        }
    };
    let run = match start_run(&worker, &module, run_cfg, journal) {
        Ok(r) => r,
        Err(e) => {
            return TerminalOutcome::FailedTerminal {
                reason: format!("run start: {e}"),
            }
        }
    };
    let pump = run.pump.clone();

    // The egress wake: the pump signals whenever guest egress lands, so this loop is
    // event-driven over published frames / op requests / metrics instead of interval-polled.
    let egress = Arc::new(Notify::new());
    {
        let egress = egress.clone();
        pump.set_egress_hook(Arc::new(move || egress.notify_one()));
    }

    // Inbound: §12.1 attach — signature, scope, MANDATORY certificate chain, dedup/gap — then
    // opaque delivery below the pump.
    let cert_check = CertCheck::new(trusted_bases, peer_certs);
    let mut attach = Attach::new(identity.run_id, identity.epoch, cert_check, pump.clone());
    let mut inbound = providers.control.subscribe();

    let _ = events.send(Event::RunPhase {
        run_id: run_label.to_string(),
        phase: "running".into(),
        epoch: identity.epoch,
        round: 0,
        generation,
    });

    // Loop state: relay cursors, held frames (gap/back-pressure), throttle level.
    let mut published_cursor = 0usize;
    let mut metrics_cursor = 0usize;
    let mut held: VecDeque<(std::time::Instant, Vec<u8>)> = VecDeque::new();
    let mut paused = false;
    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut leave_requested: Option<LeaveMode> = None;
    let mut transport_fault: Option<String> = None;

    'session: loop {
        tokio::select! {
            frame = inbound.recv() => {
                match frame {
                    Some(frame) => {
                        if !accept_inbound(
                            &mut attach, &frame, own_sender, &mut held, events, generation,
                        ) {
                            transport_fault = Some("inbound frame delivery failed".into());
                            break 'session;
                        }
                    }
                    None => {
                        // The control plane closed under us: a transport loss, recoverable by a
                        // rejoin under the node's retry budget.
                        transport_fault = Some("control plane subscription closed".into());
                        break 'session;
                    }
                }
            }
            _ = egress.notified() => {
                if let Err(reason) = relay_egress(
                    &pump, &providers, &mut published_cursor, &mut metrics_cursor,
                    events, generation,
                ).await {
                    transport_fault = Some(reason);
                    break 'session;
                }
            }
            cmd = commands.recv() => {
                match cmd {
                    Some(RoleCommand::Throttle(level)) => {
                        apply_throttle(&pump, level, &mut paused);
                    }
                    Some(RoleCommand::Leave(mode)) => {
                        leave_requested = Some(mode);
                        break 'session;
                    }
                    None => {
                        // The handle dropped without a leave: treat as an immediate leave (the
                        // worker is tearing down).
                        leave_requested = Some(LeaveMode::Immediate);
                        break 'session;
                    }
                }
            }
            _ = ticker.tick() => {
                // Re-present held frames (a gap filled by a late frame, or back-pressure that
                // drained); age out an unrecoverable gap.
                if let Some(stale) = retry_held(&mut attach, &mut held, own_sender) {
                    transport_fault = Some(stale);
                    break 'session;
                }
                if run.is_finished() {
                    break 'session;
                }
            }
        }
        if run.is_finished() {
            break 'session;
        }
    }

    // Drain any egress the guest produced before the loop ended (final publishes/metrics race
    // the terminal decision).
    let _ = relay_egress(
        &pump,
        &providers,
        &mut published_cursor,
        &mut metrics_cursor,
        events,
        generation,
    )
    .await;

    finish(
        run,
        pump,
        &providers,
        leave_requested,
        transport_fault,
        drain_deadline,
        events,
        generation,
        run_label,
    )
    .await
}

/// Verify + deliver one inbound wire frame. Own-voice echoes (a plane that self-delivers) are
/// dropped before verification; gapped and back-pressured frames are HELD for re-presentation —
/// the reliable class never silently skips. Returns `false` on a pump/journal delivery failure.
fn accept_inbound(
    attach: &mut Attach,
    frame: &[u8],
    own_sender: PeerId,
    held: &mut VecDeque<(std::time::Instant, Vec<u8>)>,
    events: &mpsc::UnboundedSender<Event>,
    generation: u64,
) -> bool {
    // Echo filter: the §12.1 frame envelope's `sender` field (mechanism, not message schema).
    // The module's own voice is already its state; a self-delivering plane must not feed it back.
    if envelope_sender(frame) == Some(own_sender.0) {
        return true;
    }
    match attach.deliver(frame) {
        Ok(InboundVerdict::Deliver { .. } | InboundVerdict::Duplicate { .. }) => true,
        Ok(InboundVerdict::Gap { .. } | InboundVerdict::Backpressure { .. }) => {
            hold_frame(held, frame);
            true
        }
        Ok(
            verdict @ (InboundVerdict::BadSignature(_)
            | InboundVerdict::TamperedPayload
            | InboundVerdict::ScopeMismatch(_)
            | InboundVerdict::Malformed(_)
            | InboundVerdict::UncertifiedSender { .. }
            | InboundVerdict::CertRevoked { .. }),
        ) => {
            // A typed refusal of one frame, never a session fault: surface and continue.
            let _ = events.send(Event::Warning {
                class: "frame_refused".into(),
                detail: format!("{verdict:?} (generation {generation})"),
            });
            true
        }
        Err(_) => false,
    }
}

/// Hold a frame for re-presentation, bounded (an unbounded hold is a memory-DoS vector).
fn hold_frame(held: &mut VecDeque<(std::time::Instant, Vec<u8>)>, frame: &[u8]) {
    if held.len() >= HELD_FRAMES_MAX {
        held.pop_front();
    }
    held.push_back((std::time::Instant::now(), frame.to_vec()));
}

/// Re-present held frames in arrival order. Returns the typed fault when a held frame has aged
/// past the gap deadline (no backfill plane exists — the session classifies retryable and the
/// node rejoins as a fresh incarnation).
fn retry_held(
    attach: &mut Attach,
    held: &mut VecDeque<(std::time::Instant, Vec<u8>)>,
    own_sender: PeerId,
) -> Option<String> {
    let mut still_held = VecDeque::new();
    while let Some((since, frame)) = held.pop_front() {
        if envelope_sender(&frame) == Some(own_sender.0) {
            continue;
        }
        match attach.deliver(&frame) {
            Ok(InboundVerdict::Gap { .. } | InboundVerdict::Backpressure { .. }) => {
                if since.elapsed() > GAP_DEADLINE {
                    return Some(
                        "inbound sequence gap unrecoverable (no backfill within the deadline)"
                            .into(),
                    );
                }
                still_held.push_back((since, frame));
            }
            // Delivered, duplicate, or refused typed: either way the hold is over.
            Ok(_) => {}
            Err(_) => return Some("held frame delivery failed".into()),
        }
    }
    *held = still_held;
    None
}

/// The §12.1 frame envelope's `sender` field, without verification (the echo-filter peek; a
/// malformed frame returns `None` and flows to the attach, which refuses it typed).
fn envelope_sender(frame: &[u8]) -> Option<[u8; 32]> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).ok()?;
    let ciborium::value::Value::Array(parts) = v else {
        return None;
    };
    let ciborium::value::Value::Map(env) = parts.first()? else {
        return None;
    };
    env.iter().find_map(|(k, val)| match (k, val) {
        (ciborium::value::Value::Text(t), ciborium::value::Value::Bytes(b)) if t == "sender" => {
            b.as_slice().try_into().ok()
        }
        _ => None,
    })
}

/// Relay pending guest egress: published frames onto the control plane verbatim (they are
/// already §12.1-signed by the certified per-run key), op requests against the content-addressed
/// providers, metrics as advisory events. Returns a fault reason on transport failure.
async fn relay_egress(
    pump: &PumpHandle,
    providers: &RoleProviders,
    published_cursor: &mut usize,
    metrics_cursor: &mut usize,
    events: &mpsc::UnboundedSender<Event>,
    generation: u64,
) -> Result<(), String> {
    let published = pump.published();
    for (_channel, _seq, frame) in published.iter().skip(*published_cursor) {
        providers
            .control
            .publish(frame)
            .await
            .map_err(|e| format!("control plane publish: {e}"))?;
    }
    *published_cursor = published.len();

    for (op, request) in pump.take_op_requests() {
        service_op(pump, providers, op, request).await?;
    }

    let metrics = pump.metrics();
    for (name, value) in metrics.iter().skip(*metrics_cursor) {
        let _ = events.send(Event::Metric {
            name: name.clone(),
            value: *value,
        });
    }
    *metrics_cursor = metrics.len();
    let _ = generation; // metrics are ambient (un-stamped); the terminal event carries the class
    Ok(())
}

/// Service one capability op against the providers. The PUMP owns every trust step (hash
/// verification, range slicing, handle minting) — the provider only moves bytes.
async fn service_op(
    pump: &PumpHandle,
    providers: &RoleProviders,
    op: u64,
    request: OpRequest,
) -> Result<(), String> {
    let outcome = match request {
        OpRequest::PayloadPut { bytes } => match providers.payloads.put_content(&bytes).await {
            Ok(_hash) => OpOutcome::PutDone,
            Err(e) => OpOutcome::Failed {
                code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                detail: format!("payload put: {e}"),
            },
        },
        OpRequest::PayloadGet { hash } => {
            match providers
                .payloads
                .get_content(&daemon_vhc_proto::Hash(hash))
                .await
            {
                Ok(bytes) => OpOutcome::GetDone { bytes },
                Err(e) => OpOutcome::Failed {
                    code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                    detail: format!("payload get: {e}"),
                },
            }
        }
        OpRequest::ArtifactFetch { hash, .. } => {
            // The provider serves the WHOLE artifact; the pump verifies against the committed
            // hash and slices the requested range before delivery.
            match providers
                .artifacts
                .get_content(&daemon_vhc_proto::Hash(hash))
                .await
            {
                Ok(artifact) => OpOutcome::FetchDone { artifact },
                Err(e) => OpOutcome::Failed {
                    code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                    detail: format!("artifact fetch: {e}"),
                },
            }
        }
        OpRequest::ArtifactRange {
            hash,
            span_off,
            span_len,
            ..
        } => {
            // A chunk-addressed shard (fold identity): the content store holds the whole
            // object under the fold key; this in-process seat slices the requested covering
            // span itself — the pump verifies every covering chunk against the registered
            // chunk map either way (the provider is untrusted by construction). A store that
            // cannot serve the span is a typed refusal, never silent bytes.
            match providers
                .artifacts
                .get_content(&daemon_vhc_proto::Hash(hash))
                .await
            {
                Ok(artifact) => {
                    let lo = usize::try_from(span_off).unwrap_or(usize::MAX);
                    let hi = lo.saturating_add(usize::try_from(span_len).unwrap_or(usize::MAX));
                    if hi <= artifact.len() {
                        OpOutcome::RangeDone {
                            bytes: artifact[lo..hi].to_vec(),
                        }
                    } else {
                        OpOutcome::Failed {
                            code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                            detail: format!(
                                "artifact range: stored object is {} bytes, span \
                                 [{span_off}, +{span_len}) does not fit",
                                artifact.len()
                            ),
                        }
                    }
                }
                Err(e) => OpOutcome::Failed {
                    code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                    detail: format!("artifact range fetch: {e}"),
                },
            }
        }
        // Direct peer streams await their live transport binding: refuse typed, never hang the
        // guest's op.
        OpRequest::StreamOpen { .. }
        | OpRequest::StreamAccept
        | OpRequest::StreamWrite { .. }
        | OpRequest::StreamRead { .. } => OpOutcome::Failed {
            code: daemon_vhc_abi::COMP_ERR_NET_UNREACHABLE,
            detail: "no direct-stream transport is bound to this session".into(),
        },
        // Compute ops are pump-internal (serviced at the call site) — they never reach the
        // embedder's request queue.
        OpRequest::TensorExport | OpRequest::TensorImport { .. } => return Ok(()),
    };
    pump.complete_op(op, outcome)
        .map_err(|e| format!("op completion: {e}"))?;
    Ok(())
}

/// Apply a throttle level. `paused` is HARD: event delivery freezes at the pump and the guest
/// parks in `next_event` — a paused worker actually stops. Duty percentage and the VRAM cap are
/// cooperative `Budget` advisories the module paces itself against.
fn apply_throttle(pump: &PumpHandle, level: ThrottleLevel, paused: &mut bool) {
    let duty = u64::from(level.duty_cycle_pct.unwrap_or(100));
    let vram_cap_bytes = u64::from(level.vram_cap_mb.unwrap_or(0)) << 20;
    // Advisory first (while delivery still flows), then the hard gate.
    let _ = pump.budget(0, 0, level.paused, duty, vram_cap_bytes);
    if level.paused && !*paused {
        pump.hold();
        *paused = true;
    } else if !level.paused && *paused {
        pump.release();
        *paused = false;
    }
}

/// End the run per cause and classify the terminal outcome.
#[allow(clippy::too_many_arguments)]
async fn finish(
    run: daemon_vhc_host::run::Run,
    pump: PumpHandle,
    providers: &RoleProviders,
    leave_requested: Option<LeaveMode>,
    transport_fault: Option<String>,
    drain_deadline: Duration,
    events: &mpsc::UnboundedSender<Event>,
    generation: u64,
    run_label: &str,
) -> TerminalOutcome {
    // A leave that arrives while the guest is paused must still drain: release the gate.
    pump.release();

    if run.is_finished() {
        // The guest ended on its own: classify its end (a leave racing a natural end still
        // reports the natural end — the module's exit is the truth).
        let end = wait_run(run).await;
        return classify_natural_end(end, transport_fault);
    }

    match (leave_requested, transport_fault) {
        (Some(LeaveMode::Graceful), _) => {
            // Quiesce: drain + module snapshot at the fence, snapshot persisted to the payload
            // plane as the leave checkpoint.
            let deadline_ms = u64::try_from(drain_deadline.as_millis()).unwrap_or(u64::MAX);
            if pump
                .quiesce(daemon_vhc_abi::QUIESCE_REASON_LEAVE, deadline_ms)
                .is_err()
            {
                let _ = pump.stop(daemon_vhc_abi::STOP_REASON_LEAVE_REQUESTED);
                let _ = wait_run(run).await;
                return TerminalOutcome::Left { checkpoint: None };
            }
            let end = wait_run(run).await;
            let quiesced = matches!(
                end,
                Ok(RunEnd::Outcome(code)) if code == daemon_vhc_abi::OUTCOME_QUIESCE_READY
            );
            if !quiesced {
                let _ = events.send(Event::Warning {
                    class: "leave_drain".into(),
                    detail: format!("drain did not quiesce cleanly: {end:?}"),
                });
                return TerminalOutcome::Left { checkpoint: None };
            }
            let checkpoint = match pump.snapshot_capture() {
                Some(capture) => persist_drain_snapshot(providers, &capture).await,
                None => None,
            };
            if let Some(hash) = &checkpoint {
                let _ = events.send(Event::CheckpointPublished {
                    round: 0,
                    hash: hash.clone(),
                    location: format!("payload/{hash}"),
                    generation,
                });
            }
            let _ = run_label;
            TerminalOutcome::Left { checkpoint }
        }
        (Some(LeaveMode::Immediate), _) => {
            let _ = pump.stop(daemon_vhc_abi::STOP_REASON_LEAVE_REQUESTED);
            let _ = wait_run(run).await;
            TerminalOutcome::Left { checkpoint: None }
        }
        (None, Some(reason)) => {
            let _ = pump.stop(daemon_vhc_abi::STOP_REASON_FAULT);
            let _ = wait_run(run).await;
            TerminalOutcome::FailedRetryable { reason }
        }
        (None, None) => {
            // The loop ended with no cause and a live guest — a session bug surfaced loudly.
            let _ = pump.stop(daemon_vhc_abi::STOP_REASON_FAULT);
            let _ = wait_run(run).await;
            TerminalOutcome::FailedTerminal {
                reason: "session loop ended without a cause".into(),
            }
        }
    }
}

/// Join the guest thread off the async runtime.
async fn wait_run(run: daemon_vhc_host::run::Run) -> Result<RunEnd, String> {
    tokio::task::spawn_blocking(move || run.wait().map_err(|e| e.to_string()))
        .await
        .map_err(|e| format!("guest join: {e}"))?
}

/// Classify a guest that ended on its own (no owner intent).
fn classify_natural_end(
    end: Result<RunEnd, String>,
    transport_fault: Option<String>,
) -> TerminalOutcome {
    if let Some(reason) = transport_fault {
        // The transport died and the guest ended with it: recoverable environment fault.
        return TerminalOutcome::FailedRetryable { reason };
    }
    match end {
        Ok(RunEnd::Outcome(0)) => TerminalOutcome::Completed { outcome: 0 },
        Ok(RunEnd::Outcome(code)) => TerminalOutcome::FailedTerminal {
            reason: format!("module ended with outcome {code}"),
        },
        Ok(RunEnd::InitRefused(code)) => TerminalOutcome::FailedTerminal {
            reason: format!("module refused init ({code})"),
        },
        Ok(RunEnd::MigrateRefused(code)) => TerminalOutcome::FailedTerminal {
            reason: format!("module refused migration ({code})"),
        },
        Ok(RunEnd::Trapped(trap)) => classify_trap(&trap),
        Err(e) => TerminalOutcome::FailedTerminal {
            reason: format!("guest thread: {e}"),
        },
    }
}

/// Trap classification: resource-budget breaches are the node's churn/preemption loop
/// (retryable); everything else is a module/admission fault (terminal).
fn classify_trap(trap: &daemon_vhc_host::trap::Trap) -> TerminalOutcome {
    match trap.code {
        TrapCode::BudgetMemory
        | TrapCode::BudgetFuel
        | TrapCode::BudgetEpoch
        | TrapCode::BudgetOps
        | TrapCode::BudgetHandles => TerminalOutcome::FailedRetryable {
            reason: format!("resource budget breached: {trap}"),
        },
        _ => TerminalOutcome::FailedTerminal {
            reason: format!("module trapped: {trap}"),
        },
    }
}

/// Persist the drain snapshot (manifest + sections, canonical CBOR) to the payload plane and
/// return its content hash (hex) — the leave checkpoint reference.
async fn persist_drain_snapshot(
    providers: &RoleProviders,
    capture: &daemon_vhc_host::run::SnapshotCapture,
) -> Option<String> {
    let doc = ciborium::value::Value::Array(vec![
        ciborium::value::Value::Bytes(capture.manifest.clone()),
        ciborium::value::Value::Array(
            capture
                .sections
                .iter()
                .map(|(name, bytes)| {
                    ciborium::value::Value::Array(vec![
                        ciborium::value::Value::Text(name.clone()),
                        ciborium::value::Value::Bytes(bytes.clone()),
                    ])
                })
                .collect(),
        ),
    ]);
    let bytes = to_canonical_vec(&doc).ok()?;
    providers
        .payloads
        .put_content(&bytes)
        .await
        .ok()
        .map(|hash| hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_vhc_host::trap::Trap;

    fn trap(code: TrapCode) -> Trap {
        Trap::new(code, "publish", None, "test")
    }

    #[test]
    fn resource_budget_traps_classify_retryable_module_traps_terminal() {
        // Budget breaches are the node's churn/preemption loop; module faults are terminal.
        for code in [
            TrapCode::BudgetMemory,
            TrapCode::BudgetFuel,
            TrapCode::BudgetEpoch,
            TrapCode::BudgetOps,
            TrapCode::BudgetHandles,
        ] {
            assert!(matches!(
                classify_trap(&trap(code)),
                TerminalOutcome::FailedRetryable { .. }
            ));
        }
        for code in [
            TrapCode::GuestPanic,
            TrapCode::PhaseViolation,
            TrapCode::GrantViolation,
            TrapCode::PayloadOverflow,
        ] {
            assert!(matches!(
                classify_trap(&trap(code)),
                TerminalOutcome::FailedTerminal { .. }
            ));
        }
    }

    #[test]
    fn natural_ends_classify_by_the_module_exit() {
        assert_eq!(
            classify_natural_end(Ok(RunEnd::Outcome(0)), None),
            TerminalOutcome::Completed { outcome: 0 }
        );
        assert!(matches!(
            classify_natural_end(Ok(RunEnd::Outcome(7)), None),
            TerminalOutcome::FailedTerminal { .. }
        ));
        assert!(matches!(
            classify_natural_end(Ok(RunEnd::InitRefused(1)), None),
            TerminalOutcome::FailedTerminal { .. }
        ));
        assert!(matches!(
            classify_natural_end(Ok(RunEnd::MigrateRefused(16)), None),
            TerminalOutcome::FailedTerminal { .. }
        ));
        // A transport fault that took the guest down with it is the recoverable class, whatever
        // the guest's own exit looked like.
        assert!(matches!(
            classify_natural_end(Ok(RunEnd::Outcome(0)), Some("plane lost".into())),
            TerminalOutcome::FailedRetryable { .. }
        ));
    }

    #[test]
    fn the_envelope_sender_peek_reads_only_the_frame_envelope() {
        // A structurally-valid frame yields its sender; junk yields None (and flows to the
        // attach, which refuses it typed).
        let envelope = ciborium::value::Value::Map(vec![(
            ciborium::value::Value::Text("sender".into()),
            ciborium::value::Value::Bytes(vec![0xAB; 32]),
        )]);
        let wire = ciborium::value::Value::Array(vec![
            envelope,
            ciborium::value::Value::Bytes(b"payload".to_vec()),
            ciborium::value::Value::Bytes(vec![0; 64]),
        ]);
        let mut bytes = Vec::new();
        ciborium::into_writer(&wire, &mut bytes).unwrap();
        assert_eq!(envelope_sender(&bytes), Some([0xAB; 32]));
        assert_eq!(envelope_sender(b"junk"), None);
    }
}
