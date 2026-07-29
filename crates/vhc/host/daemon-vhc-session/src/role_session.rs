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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use daemon_vhc_host::run::{
    DeviceProfile, EnvelopeRoleGrants, JournalSink, MigrationInput, OpOutcome, OpRequest,
    OwnerPolicy, ParticipationLane, PumpHandle, RunConfig, RunEnd, RunIdentity,
};
use daemon_vhc_host::trap::TrapCode;
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_net::{ContentStore, ControlPlane};
use daemon_vhc_proto::{
    blake3_hash, peer_id, AdmittedQuotas, CertScope, Hash, PeerId, RunKeyCertificate, SigningKey,
};
use tokio::sync::{mpsc, Notify};

use crate::attach::{Attach, CertCheck, InboundVerdict};
use crate::protocol::{AdmittedTuple, Event, LeaveMode, TerminalOutcome};

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
    /// An optional LATE-JOIN restore (§10.2/§10.3): a checkpoint snapshot to migrate the fresh
    /// module instance from before it runs (built from a registry checkpoint pointer the node
    /// resolved, its bytes fetched + hash-verified from the payload plane). `None` = a fresh
    /// start from genesis.
    pub restore: Option<daemon_vhc_host::run::MigrationInput>,
    /// The quotas this instance was admitted under — the grant-expansion baseline a live module
    /// switch compares the re-admitted quotas against (fail closed, ABI §10.3 step 3). `None`
    /// only on harness/test seats that bypass the admission funnel; the production join always
    /// records them.
    pub admitted_quotas: Option<AdmittedQuotas>,
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

/// How many rollback-and-retry cycles a switch's migrate/validate/activate steps get before the
/// session leaves the run (ABI §10.3 step 7). Each retry re-migrates from the same durable drain
/// snapshot with a fresh instance.
const SWITCH_MIGRATE_RETRIES: u32 = 1;

/// A per-seam journal opener: invoked with the incoming incarnation's execution identity, AFTER
/// the retired instance's sink has been dropped (one writer per file series — the §8.1 seam
/// continues the same log). Called once per migrate attempt (a failed attempt's sink is dropped
/// with its instance before the retry re-opens).
pub type SeamJournal = Box<dyn FnMut(&RunIdentity) -> Result<Box<dyn JournalSink>, String> + Send>;

/// Everything a live module switch binds — the worker's pre-flight output, handed into the
/// running session (ABI §10.3). The target artifacts are hash-pinned; the identity material is
/// the node-provisioned NEW incarnation's (a live upgrade advances the epoch and mints a new
/// never-reused incarnation, §8.1); the admission inputs let the session re-run owner law over
/// the new module before the fence.
pub struct SwitchBinding {
    /// The committed transition-chain epoch this switch activates (§8.1).
    pub epoch: u64,
    /// The hash-pinned target module.
    pub new_module: [u8; 32],
    /// The committed upgrade record's grants anchor — the re-derived grants document must hash
    /// to it.
    pub grants_hash: [u8; 32],
    /// The node-assessed admitted tuple for the post-switch identity (carries the node-minted
    /// new incarnation).
    pub tuple: AdmittedTuple,
    /// The target module bytes when the worker resolved them (hash-verified here either way);
    /// `None` ⇒ the session fetches by content address from its own providers.
    pub module_bytes: Option<Vec<u8>>,
    /// The new instance's admitted config bytes (the upgrade record pins module + grants; config
    /// carriage is empty until upgrade records carry one).
    pub config: Vec<u8>,
    /// The NEW incarnation's per-run signing seed (node-minted, resolved by keystore reference).
    pub signing_seed: [u8; 32],
    /// The NEW incarnation's certificate — bound to
    /// `(run, epoch, role, new incarnation, new module)` (the re-issuance handshake, §12.3
    /// [CERT-3]: an incarnation change rotates the key).
    pub own_cert: RunKeyCertificate,
    /// The genesis role grant list (channel table etc.) the new grants document derives from.
    pub role_grants: daemon_vhc_proto::genesis::RoleGrants,
    /// The envelope-derived role grants for the admission funnel's quota derivation.
    pub envelope_grants: Option<EnvelopeRoleGrants>,
    /// The participation lane owner law evaluates against (§9.4).
    pub lane: ParticipationLane,
    /// The device profile for the re-admission funnel.
    pub device: DeviceProfile,
    /// The owner policy the re-admission enforces (fail-closed).
    pub owner: OwnerPolicy,
    /// The provisioned resource authority the fence's re-admission composes with, assembled by
    /// the worker's pre-flight (the worker holds the provisioning env and the probe machinery;
    /// the fence holds the moment of truth). `None` on an un-provisioned box, where a
    /// certification-minor target refuses `EstimateNotComposable` at the fence — typed, with the
    /// running instance untouched.
    pub resources: Option<daemon_vhc_host::run::ResourceAuthorityParts>,
    /// The seam journal opener (per migrate attempt).
    pub journal: SeamJournal,
    /// The quiesce drain deadline in ms (§4.4), also the per-attempt validate window.
    pub deadline_ms: u64,
    /// The explicit migrate fuel budget (`None` = the engine default).
    pub migrate_fuel: Option<u64>,
}

/// A command from the worker loop into the running role task.
enum RoleCommand {
    Throttle(ThrottleLevel),
    Leave(LeaveMode),
    Switch(Box<SwitchBinding>),
}

/// The handle the worker keeps per spawned role — the ONLY per-run state above the session
/// (crate ownership rule: the worker binary owns the command loop and the role-handle map,
/// nothing else).
pub struct RoleHandle {
    commands: mpsc::UnboundedSender<RoleCommand>,
    task: tokio::task::JoinHandle<()>,
    /// The LIVE generation: a successful module switch advances it to the new incarnation
    /// (shared with the session task, which stamps events with the current value).
    generation: Arc<AtomicU64>,
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

    /// Run the live module switch (the ABI §10.3 upgrade transaction) against this session's
    /// held instance. Answers on the event stream: `ModuleSwitched` on activation,
    /// `SwitchRefused` when the pre-fence checks refuse (the old module keeps running), or the
    /// terminal `RunTerminated` when a post-fence failure leaves the run.
    pub fn switch(&self, binding: SwitchBinding) {
        let _ = self.commands.send(RoleCommand::Switch(Box::new(binding)));
    }

    /// The role-instance generation (the never-reused incarnation id) stamped on every event
    /// this session emits. A successful module switch advances it.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
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
    let generation = Arc::new(AtomicU64::new(spec.run.identity.instance));
    let label = run_label.clone();
    let task = {
        let generation = generation.clone();
        tokio::spawn(async move {
            let epoch = spec.run.identity.epoch;
            let _ = events.send(Event::RunPhase {
                run_id: label.clone(),
                phase: "joining".into(),
                epoch,
                round: 0,
                generation: generation.load(Ordering::SeqCst),
            });
            let outcome = run_role(&label, spec, &events, cmd_rx, &generation).await;
            let _ = events.send(Event::RunTerminated {
                run_id: label,
                generation: generation.load(Ordering::SeqCst),
                outcome,
            });
        })
    };
    RoleHandle {
        commands: cmd_tx,
        task,
        generation,
        run_label,
    }
}

/// One live module instance under the session: the running guest, its pump, the epoch-scoped
/// inbound attach, and the relay cursors. A live module switch retires one and binds the next;
/// everything else in the session (providers, command stream, event stream) carries across.
struct LiveInstance {
    run: daemon_vhc_host::run::Run,
    pump: PumpHandle,
    attach: Attach,
    identity: RunIdentity,
    own_sender: PeerId,
    egress: EgressCursors,
}

/// How far the session has relayed each of the pump's monotonically-growing egress buffers, and
/// the accumulator that assembles what it reads out of them. Per-incarnation: a module switch
/// mints a fresh instance, so the pump's buffers start empty and these start at zero together —
/// which is the invariant that keeps a cursor from pointing into a previous instance's egress.
struct EgressCursors {
    published: usize,
    metrics: usize,
    logs: usize,
    /// Assembles the guest's reserved round-outcome metrics (the opacity-safe per-round digest +
    /// barrier bookkeeping, ABI `round_metrics`) into [`Event::RoundOutcome`]s. Partial groups
    /// never straddle a module-switch fence, for the same reason the cursors do not.
    digest_accum: daemon_vhc_abi::round_metrics::RoundOutcomeAccumulator,
}

impl Default for EgressCursors {
    fn default() -> Self {
        Self {
            published: 0,
            metrics: 0,
            logs: 0,
            digest_accum: daemon_vhc_abi::round_metrics::RoundOutcomeAccumulator::new(),
        }
    }
}

/// The session body: bind, run, service, switch, classify. Returns the terminal outcome (the
/// task wrapper emits it).
#[allow(clippy::too_many_lines)]
async fn run_role(
    run_label: &str,
    spec: RoleSessionSpec,
    events: &mpsc::UnboundedSender<Event>,
    mut commands: mpsc::UnboundedReceiver<RoleCommand>,
    generation: &Arc<AtomicU64>,
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
        restore,
        mut admitted_quotas,
    } = spec;

    let mut identity = run_cfg.identity.clone();
    // The run-pinned genesis state contract's `state_chunk_size` ([SF-5]): captured before
    // `run_cfg` is consumed so a later live module switch can provision the successor's state
    // plane with it. `RunConfig::new` defaults it to 0 (state plane disabled), and the switch
    // re-admits the NEW module without re-reading the genesis contract — so without carrying this
    // the successor comes up unprovisioned: its self-sealed carried folds re-derive under
    // `chunk_size = 0` (wrong identity) and its own `state_open` traps. It is genesis-pinned, so
    // identical across the switch.
    let run_state_chunk_size = run_cfg.state_chunk_size;
    let mut own_sender = own_cert.body.run_key;
    let own_cert_record = crate::distribution::DistributionRecord::Cert(own_cert.clone());
    if !peer_certs.contains(&own_cert) {
        peer_certs.push(own_cert);
    }

    // Bind the runtime. Setup failures are terminal: the artifacts were already admitted, so a
    // module that cannot even instantiate is not a transient environment fault.
    let worker = match Worker::new(engine.clone()) {
        Ok(w) => w,
        Err(e) => {
            return TerminalOutcome::FailedTerminal {
                reason: format!("engine construction: {e}"),
            }
        }
    };
    // A late-join restore migrates the fresh instance from the checkpoint snapshot before it
    // runs (§10.3 step 4: `da_migrate` under budget between `da_init` and `da_run`); a
    // migrate-refusal is a terminal admission fault, never a silent fresh start.
    let restoring = restore.is_some();
    if let Some(mig) = &restore {
        // The by-reference families (master/ef/adamw) carry only their FamilyRef through the
        // migrate seam; the new instance registers each fold (register_state_chunks, [SF-R2])
        // and streams its windows via chunk-keyed `data@2::fetch` in `da_run`. Naming the count
        // here is the restore path's observability seam (the streaming-rehydration marker).
        use daemon_vhc_proto::det_state::CkptDocSection;
        let by_ref = mig
            .capture
            .sections
            .iter()
            .filter(|s| matches!(s, CkptDocSection::ByRef(..)))
            .count();
        tracing::info!(
            run = run_label,
            by_ref_families = by_ref,
            "streaming restore from checkpoint document: registering by-ref det-state roots \
             (register_state_chunks + chunk-keyed rehydration)"
        );
    }
    let run = match daemon_vhc_host::run::start_run_migrating(
        &worker, &module, run_cfg, journal, restore,
    ) {
        Ok(r) => r,
        // The admitted execution backend cannot serve right now (the device disappeared or
        // shrank since assess, its runtime is unstaged, or the process device-compute slot is
        // occupied): a RECOVERABLE environment fault — the node reassesses against the live
        // device inventory and reconverges — never a quiet CPU run, never terminal.
        Err(daemon_vhc_host::run::RunError::BackendUnavailable(reason)) => {
            return TerminalOutcome::FailedRetryable {
                reason: format!("execution backend unavailable at run start: {reason}"),
            }
        }
        Err(e) => {
            return TerminalOutcome::FailedTerminal {
                reason: format!(
                    "run start{}: {e}",
                    if restoring { " (restore)" } else { "" }
                ),
            }
        }
    };
    let pump = run.pump.clone();

    // The egress wake: the pump signals whenever guest egress lands, so this loop is
    // event-driven over published frames / op requests / metrics instead of interval-polled.
    // One notify serves every instance the session binds (the hook re-registers per pump).
    let egress = Arc::new(Notify::new());
    {
        let egress = egress.clone();
        pump.set_egress_hook(Arc::new(move || egress.notify_one()));
    }

    // Inbound: §12.1 attach — signature, scope, MANDATORY certificate chain, dedup/gap — then
    // opaque delivery below the pump.
    let cert_check = CertCheck::new(trusted_bases.clone(), peer_certs.clone());
    let attach = Attach::new(identity.run_id, identity.epoch, cert_check, pump.clone());
    let mut inbound = providers.control.subscribe();

    // §12.3 distribution: announce this incarnation's certificate on the control plane so peers
    // can verify our frames without an out-of-band exchange. Best-effort by design (supersession
    // is the safety floor); a WS plane additionally re-announces on every reconnect AND on a slow
    // anti-entropy cadence via the resubscribe registration made at provider construction (a peer
    // whose planes come up after this one-shot announce converges on the next cadence tick —
    // re-publishing here would be suppressed by the plane's own content-hash dedupe).
    if let Ok(bytes) = own_cert_record.to_bytes() {
        let _ = providers.control.publish(&bytes).await;
    }
    tracing::debug!(
        run = run_label,
        role = %identity.role,
        incarnation = identity.instance,
        restoring,
        sender = %hex16(&own_sender.0),
        "role session running: control plane attached, certificate announced"
    );

    let _ = events.send(Event::RunPhase {
        run_id: run_label.to_string(),
        phase: "running".into(),
        epoch: identity.epoch,
        round: 0,
        generation: generation.load(Ordering::SeqCst),
    });

    let mut current = LiveInstance {
        run,
        pump,
        attach,
        identity: identity.clone(),
        own_sender,
        egress: EgressCursors::default(),
    };

    // Loop state: held frames (gap/back-pressure), throttle level, exit causes.
    let mut held: VecDeque<(std::time::Instant, Vec<u8>)> = VecDeque::new();
    let mut paused = false;
    let mut ticker = tokio::time::interval(TICK);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut leave_requested: Option<LeaveMode> = None;
    let mut transport_fault: Option<String> = None;

    'session: loop {
        let gen_now = generation.load(Ordering::SeqCst);
        tokio::select! {
            frame = inbound.recv() => {
                match frame {
                    Some(frame) => {
                        if !accept_inbound(
                            &mut current.attach, &frame, own_sender, &mut held, events, gen_now,
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
                    &current.pump, &providers, &mut current.egress, events, gen_now,
                ).await {
                    transport_fault = Some(reason);
                    break 'session;
                }
            }
            cmd = commands.recv() => {
                match cmd {
                    Some(RoleCommand::Throttle(level)) => {
                        apply_throttle(&current.pump, level, &mut paused);
                    }
                    Some(RoleCommand::Leave(mode)) => {
                        leave_requested = Some(mode);
                        break 'session;
                    }
                    Some(RoleCommand::Switch(binding)) => {
                        // The live module switch (ABI §10.3). A paused instance must drain:
                        // release the hard gate first.
                        current.pump.release();
                        paused = false;
                        match perform_switch(
                            &worker,
                            &engine,
                            &providers,
                            &trusted_bases,
                            &peer_certs,
                            &egress,
                            current,
                            *binding,
                            admitted_quotas.as_ref(),
                            run_label,
                            events,
                            run_state_chunk_size,
                        )
                        .await
                        {
                            SwitchStep::Refused { instance, reason } => {
                                // Pre-fence refusal: the old module keeps running untouched.
                                current = instance;
                                let _ = events.send(Event::SwitchRefused {
                                    run_id: run_label.to_string(),
                                    generation: gen_now,
                                    reason,
                                });
                            }
                            SwitchStep::Activated {
                                instance,
                                retries,
                                quotas,
                            } => {
                                current = instance;
                                identity = current.identity.clone();
                                own_sender = current.own_sender;
                                admitted_quotas = quotas.or(admitted_quotas);
                                generation.store(identity.instance, Ordering::SeqCst);
                                held.clear(); // held frames belong to the retired epoch's scope
                                let _ = events.send(Event::ModuleSwitched {
                                    run_id: run_label.to_string(),
                                    epoch: identity.epoch,
                                    module: identity.module,
                                    retries,
                                    generation: identity.instance,
                                });
                            }
                            SwitchStep::Left { outcome } => {
                                // A post-fence failure left the run (§10.3 step 7): the chain is
                                // not rolled back and the old epoch is never resumed.
                                return outcome;
                            }
                        }
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
                if let Some(stale) = retry_held(&mut current.attach, &mut held, own_sender) {
                    transport_fault = Some(stale);
                    break 'session;
                }
                if current.run.is_finished() {
                    break 'session;
                }
            }
        }
        if current.run.is_finished() {
            break 'session;
        }
    }

    // Drain any egress the guest produced before the loop ended (final publishes/metrics race
    // the terminal decision).
    let gen_now = generation.load(Ordering::SeqCst);
    let _ = relay_egress(
        &current.pump,
        &providers,
        &mut current.egress,
        events,
        gen_now,
    )
    .await;

    finish(
        current.run,
        current.pump,
        &providers,
        leave_requested,
        transport_fault,
        drain_deadline,
        events,
        gen_now,
        run_label,
    )
    .await
}

/// How one switch attempt left the session.
enum SwitchStep {
    /// Refused BEFORE the fence: the old module keeps running untouched.
    Refused {
        instance: LiveInstance,
        reason: String,
    },
    /// The new instance activated (§10.3 step 6).
    Activated {
        instance: LiveInstance,
        retries: u32,
        quotas: Option<AdmittedQuotas>,
    },
    /// A post-fence failure left the run (§10.3 step 7): terminal, old epoch never resumed.
    Left { outcome: TerminalOutcome },
}

/// The ABI §10.3 upgrade transaction under the session.
///
/// Pre-fence, everything effect-free runs first — target-artifact resolution + hash pin, the
/// re-issued certificate's scope, the admitted-tuple cross-check, owner-law re-admission, the
/// fail-closed grant-containment comparison — and ANY refusal leaves the old module running
/// untouched. Only then the fence: quiesce → durable snapshot → spool capture → retire the old
/// instance → migrate the new one from the snapshot (bounded rollback-and-retry) → validate
/// (`da_migrate` returned `Ready`, the pump's embedder-visible marker) → activate (spooled
/// frames drain into the new instance; the attach re-scopes to the new epoch; the new
/// certificate announces on the plane). Past the fence there is no way back: a migrate/validate
/// failure that exhausts its retries LEAVES the run (the chain is never rolled back and the old
/// epoch is never resumed).
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn perform_switch(
    worker: &Worker,
    engine: &EngineConfig,
    providers: &RoleProviders,
    trusted_bases: &[PeerId],
    bootstrap_certs: &[RunKeyCertificate],
    egress: &Arc<Notify>,
    current: LiveInstance,
    mut binding: SwitchBinding,
    admitted_quotas: Option<&AdmittedQuotas>,
    run_label: &str,
    events: &mpsc::UnboundedSender<Event>,
    run_state_chunk_size: u64,
) -> SwitchStep {
    let old_identity = current.identity.clone();

    // ---- pre-fence checks: any refusal returns the UNTOUCHED running instance ----------------
    let refused = |instance: LiveInstance, reason: String| SwitchStep::Refused { instance, reason };

    // Target artifact: worker-resolved bytes or a content-addressed fetch from this session's
    // own providers; the hash pin is checked HERE either way (the store is untrusted).
    let module_bytes = match binding.module_bytes.take() {
        Some(bytes) => bytes,
        None => {
            let hash = Hash(binding.new_module);
            match providers.artifacts.get_content(&hash).await {
                Ok(bytes) => bytes,
                Err(_) => match providers.payloads.get_content(&hash).await {
                    Ok(bytes) => bytes,
                    Err(e) => {
                        return refused(
                            current,
                            format!(
                                "target module {} is not resolvable from the bound content \
                                 stores: {e}",
                                hash.to_hex()
                            ),
                        )
                    }
                },
            }
        }
    };
    if *blake3::hash(&module_bytes).as_bytes() != binding.new_module {
        return refused(
            current,
            "resolved target artifact does not hash to the committed module".into(),
        );
    }

    // The post-switch identity: the node-minted NEW incarnation (a live upgrade advances the
    // epoch and mints a new never-reused incarnation, §8.1) — strictly above the retiring one.
    let tuple = &binding.tuple;
    if tuple.genesis_hash != old_identity.run_id
        || tuple.role != old_identity.role
        || tuple.module_hash != binding.new_module
        || tuple.grants_hash != binding.grants_hash
    {
        return refused(
            current,
            "the post-switch admitted tuple does not bind this run/role/target".into(),
        );
    }
    if tuple.incarnation <= old_identity.instance {
        return refused(
            current,
            format!(
                "the post-switch incarnation {} does not supersede the running incarnation {} \
                 (incarnations are never reused)",
                tuple.incarnation, old_identity.instance
            ),
        );
    }
    // The re-issued certificate (the node-authored identity handshake): it must bind EXACTLY
    // the post-switch execution identity and certify the delivered per-run key.
    let expected_scope = CertScope {
        run_id: Hash(old_identity.run_id),
        epoch: binding.epoch,
        role: old_identity.role.clone(),
        instance: tuple.incarnation,
        module_hash: Hash(binding.new_module),
    };
    if binding.own_cert.body.scope != expected_scope {
        return refused(
            current,
            "the re-issued certificate binds a different execution identity than this switch"
                .into(),
        );
    }
    if binding.own_cert.body.run_key != peer_id(&SigningKey::from_bytes(&binding.signing_seed)) {
        return refused(
            current,
            "the re-issued certificate does not certify the provisioned per-run key".into(),
        );
    }
    if let Err(e) = binding.own_cert.verify_chain() {
        return refused(current, format!("re-issued certificate chain: {e}"));
    }

    // Owner-law re-admission over the new module (§10.3 step 3, evaluated ahead of the fence —
    // it is effect-free, and refusing BEFORE quiescing leaves the old module unharmed).
    let admission_engine = EngineConfig {
        backend: daemon_vhc_host::BackendKind::Cpu,
        ..engine.clone()
    };
    let admission_worker = match Worker::new(admission_engine) {
        Ok(w) => w,
        Err(e) => return refused(current, format!("re-admission engine: {e}")),
    };
    let linked = match daemon_vhc_host::linked_worlds(&admission_worker, &module_bytes) {
        Ok(l) => l,
        Err(e) => return refused(current, format!("target module linked worlds: {e}")),
    };
    let grants =
        daemon_vhc_proto::GrantsDoc::author(&linked, &binding.role_grants).to_canonical_bytes();
    if blake3_hash(&grants) != Hash(binding.grants_hash) {
        return refused(
            current,
            "re-derived grants do not match the committed record's grants anchor".into(),
        );
    }
    // The worker-assembled resource authority, when this box is provisioned: the fence re-runs
    // owner law over the same store, report and frozen binding the pre-switch assessment composed
    // with. An authentication refusal here degrades to `None` — the truthful no-authority floor —
    // and a certification-minor target then refuses `EstimateNotComposable`, typed, with the
    // running instance untouched.
    let profile = match binding.resources.as_ref() {
        Some(parts) => match parts.select() {
            Ok(profile) => Some(profile),
            Err(refusal) => {
                tracing::warn!(%refusal, "provisioned profile did not authenticate at the switch fence");
                None
            }
        },
        None => None,
    };
    let authority = match (binding.resources.as_ref(), profile.as_ref()) {
        (Some(parts), Some(profile)) => Some(parts.authority(profile)),
        _ => None,
    };
    let admission = match daemon_vhc_host::run::admit(
        &admission_worker,
        &module_bytes,
        Some(&binding.new_module),
        &binding.config,
        &grants,
        &binding.lane,
        &binding.device,
        &binding.owner,
        None,
        binding.envelope_grants.as_ref(),
        authority.as_ref(),
    ) {
        Ok(a) => a,
        Err(refusal) => return refused(current, format!("switch re-admission: {refusal}")),
    };
    // The remaining artifact-addressed tuple fields, rederived from what would actually run.
    if tuple.config_hash != *blake3::hash(&binding.config).as_bytes() {
        return refused(
            current,
            "the post-switch admitted tuple's config hash does not match the delivered config"
                .into(),
        );
    }
    // The whole minor-selected resource statement, not just a claim digest (`[DI-9]`). Below the
    // certification minor this is the same declared-claim comparison as before; at the certification
    // minor it re-derives the composition and compares every member, so a profile, capability report,
    // planner or reservation that moved between the pre-switch assessment and the fence is caught here
    // — none of which changes any artifact hash, and all of which changes what may run.
    match crate::protocol::AdmittedResource::from_admission(&admission) {
        Ok(rederived) => {
            if let Some(member) = tuple.resource.first_mismatch(&rederived) {
                return refused(
                    current,
                    format!(
                        "the post-switch admitted tuple's `{member}` does not match the \
                         re-evaluated admission"
                    ),
                );
            }
        }
        Err(e) => {
            return refused(
                current,
                format!("the re-evaluated admission could not state its resource identity: {e}"),
            )
        }
    }
    // Grant-expanding upgrades FAIL CLOSED (§10.3 step 3): the re-admitted quotas must be
    // tighten-or-equal against the quotas this instance runs under.
    match (admitted_quotas, admission.quotas.as_ref()) {
        (Some(old), Some(new)) => {
            if let Some(why) = grant_expansion(old, new) {
                return refused(
                    current,
                    format!("grant-expanding upgrade refused (fail closed): {why}"),
                );
            }
        }
        (Some(_), None) => {
            return refused(
                current,
                "the re-admission produced no quotas to verify grant containment against \
                 (fail closed)"
                    .into(),
            );
        }
        // No recorded baseline (harness/test seats that bypass the funnel): nothing to compare.
        (None, _) => {}
    }

    // ---- the fence (§10.3 steps 1–2): from here the old instance is consumed ------------------
    let LiveInstance {
        run: old_run,
        pump: old_pump,
        attach: old_attach,
        ..
    } = current;
    if old_pump
        .quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, binding.deadline_ms)
        .is_err()
    {
        return SwitchStep::Left {
            outcome: TerminalOutcome::FailedTerminal {
                reason: "switch quiesce could not be delivered".into(),
            },
        };
    }
    let end = wait_run(old_run).await;
    let quiesced = matches!(
        end,
        Ok(RunEnd::Outcome(code)) if code == daemon_vhc_abi::OUTCOME_QUIESCE_READY
    );
    if !quiesced {
        // A failed quiesce past the committed epoch leaves the run (§10.3 step 1/7): the old
        // instance is already gone and the chain is never rolled back.
        return SwitchStep::Left {
            outcome: TerminalOutcome::FailedTerminal {
                reason: format!("switch drain did not quiesce: {end:?}"),
            },
        };
    }
    let Some(capture) = old_pump.snapshot_capture() else {
        return SwitchStep::Left {
            outcome: TerminalOutcome::FailedTerminal {
                reason: "switch drain accepted no snapshot (the module has no durable state to \
                         carry across the fence)"
                    .into(),
            },
        };
    };
    let spooled = old_pump.take_spooled_frames();
    // Carry the drain snapshot's sealed det-state families into the successor's store within the
    // switch transaction ([SF-6]): lift each by-reference family's chunks out of the DRAINING
    // instance's store NOW, while `old_pump` is still alive (it is dropped just below). An
    // in-process switch is the one migrate where the same node keeps custody of canonical state
    // and publishes nothing to the content plane, so the successor serves these folds self-sealed
    // ([SF-R1]); without this carry its streamed restore fetches them as externally-sourced and
    // misses every chunk (payload miss → trap). Held in this Vec until the successor's store is
    // seeded (below), so a crash mid-switch never leaves the chunks existing nowhere.
    let carried_state: Vec<_> = capture
        .sections
        .iter()
        .filter_map(|s| match s {
            daemon_vhc_proto::det_state::CkptDocSection::ByRef(_, family) => {
                old_pump.export_sealed_family(&family.fold.0)
            }
            daemon_vhc_proto::det_state::CkptDocSection::Inline(..) => None,
        })
        .collect();
    // Retire the old instance's handles BEFORE the seam journal opens: the journal seam
    // continues one file series, and the retired sink (inside the old pump state) must drop
    // first — one writer per series.
    drop(old_attach);
    drop(old_pump);

    // ---- migrate + validate with bounded rollback-and-retry (§10.3 steps 4–5, 7) --------------
    let new_identity = RunIdentity {
        run_id: old_identity.run_id,
        epoch: binding.epoch,
        role: old_identity.role.clone(),
        instance: tuple.incarnation,
        module: binding.new_module,
    };
    let mut attempt: u32 = 0;
    let (new_run, new_pump) = loop {
        let journal = match (binding.journal)(&new_identity) {
            Ok(sink) => sink,
            Err(e) => {
                return SwitchStep::Left {
                    outcome: TerminalOutcome::FailedTerminal {
                        reason: format!("seam journal open: {e}"),
                    },
                }
            }
        };
        let mut cfg = RunConfig::new(
            new_identity.clone(),
            binding.signing_seed,
            binding.config.clone(),
            grants.clone(),
        );
        cfg.claim_bytes = admission.claim_bytes.clone();
        cfg.manifest_bytes = admission.manifest_bytes.clone();
        admission.apply_quotas(&mut cfg);
        // Provision the successor's state plane from the run-pinned genesis state contract
        // ([SF-5]): `RunConfig::new` defaults `state_chunk_size` to 0 and `apply_quotas` never
        // sets it (it is the contract, not a grant), so carry it from the run so the successor
        // serves the carried folds self-sealed and its own `state_open` is provisioned.
        cfg.state_chunk_size = run_state_chunk_size;
        let started = daemon_vhc_host::run::start_run_migrating(
            worker,
            &module_bytes,
            cfg,
            journal,
            Some(MigrationInput {
                capture: capture.clone(),
                restore: true,
                migrate_fuel: binding.migrate_fuel,
                // The sealed families carried from the draining instance (above), so the
                // successor's streamed restore resolves them self-sealed ([SF-R1]); clone because
                // a rollback-and-retry attempt re-seeds a fresh successor store.
                carried_state: carried_state.clone(),
            }),
        );
        let failure = match started {
            Ok(run) => {
                let pump = run.pump.clone();
                // Validate (§10.3 step 5): the pump's embedder-visible marker — set once
                // `da_migrate` returned `Ready` — gates activation; a refusing/trapping
                // migrate ends the guest thread instead.
                let deadline =
                    std::time::Instant::now() + Duration::from_millis(binding.deadline_ms.max(1));
                loop {
                    if pump.migrate_validated() {
                        break;
                    }
                    if run.is_finished() || std::time::Instant::now() >= deadline {
                        break;
                    }
                    tokio::time::sleep(TICK).await;
                }
                if pump.migrate_validated() {
                    break (run, pump);
                }
                if run.is_finished() {
                    format!("migrate step failed: {:?}", wait_run(run).await)
                } else {
                    // Neither validated nor torn down within the window: the engine's own
                    // migrate budget will reap the guest; retrying against a possibly-live
                    // writer is unsafe, so the transaction leaves now.
                    return SwitchStep::Left {
                        outcome: TerminalOutcome::FailedTerminal {
                            reason: "migrate neither validated nor tore down within the switch \
                                     deadline"
                                .into(),
                        },
                    };
                }
            }
            Err(e) => format!("migrating instance start: {e}"),
        };
        // Roll back (§10.3 step 7): the failed instance already tore down guest-side; the
        // durable snapshot is the recovery point. Retry under the budget or leave.
        if attempt >= SWITCH_MIGRATE_RETRIES {
            return SwitchStep::Left {
                outcome: TerminalOutcome::FailedTerminal {
                    reason: format!(
                        "switch migration exhausted its retry budget (attempt {}): {failure}",
                        attempt + 1
                    ),
                },
            };
        }
        attempt += 1;
        let _ = events.send(Event::Warning {
            class: "switch_retry".into(),
            detail: format!("rolling back and retrying the migrate step: {failure}"),
        });
    };

    // ---- activate (§10.3 step 6) ---------------------------------------------------------------
    // Spooled frames drain into the new instance verbatim (they were verified above the old
    // pump before the fence).
    for f in spooled {
        if new_pump
            .deliver_frame(
                f.channel,
                f.seq,
                f.sender,
                f.payload,
                f.original_signed_frame,
            )
            .is_err()
        {
            return SwitchStep::Left {
                outcome: TerminalOutcome::FailedTerminal {
                    reason: "spool drain into the activated instance failed".into(),
                },
            };
        }
    }
    {
        let egress = egress.clone();
        new_pump.set_egress_hook(Arc::new(move || egress.notify_one()));
    }
    // The inbound attach re-scopes to the new epoch, seeded with the bootstrap trust plus the
    // NEW certificate; peers' post-switch certificates re-arrive as distribution records.
    let mut certs = bootstrap_certs.to_vec();
    if !certs.contains(&binding.own_cert) {
        certs.push(binding.own_cert.clone());
    }
    let attach = Attach::new(
        new_identity.run_id,
        new_identity.epoch,
        CertCheck::new(trusted_bases.to_vec(), certs),
        new_pump.clone(),
    );
    // Announce the re-issued certificate (§12.3 distribution) — best-effort, like the join's.
    if let Ok(bytes) =
        crate::distribution::DistributionRecord::Cert(binding.own_cert.clone()).to_bytes()
    {
        let _ = providers.control.publish(&bytes).await;
    }
    let _ = events.send(Event::RunPhase {
        run_id: run_label.to_string(),
        phase: "running".into(),
        epoch: new_identity.epoch,
        round: 0,
        generation: new_identity.instance,
    });
    let own_sender = binding.own_cert.body.run_key;
    SwitchStep::Activated {
        instance: LiveInstance {
            run: new_run,
            pump: new_pump,
            attach,
            identity: new_identity,
            own_sender,
            egress: EgressCursors::default(),
        },
        retries: attempt,
        quotas: admission.quotas.clone(),
    }
}

/// Whether `new` expands any grant beyond `old` (the fail-closed rule, architecture §5.4).
/// Returns `Some(reason)` naming the first expanded bound, or `None` if `new` is
/// tighten-or-equal everywhere.
///
/// Numeric bounds follow the ABI §2.3 convention where **`0` means "unbounded by this grant"** —
/// the loosest value. So tightening from unbounded (`0`) to any finite bound is fine; loosening
/// a finite bound to a larger one, or to unbounded (`0`), is expansion. The granted-artifact set
/// must be a subset of the old set.
#[must_use]
pub fn grant_expansion(old: &AdmittedQuotas, new: &AdmittedQuotas) -> Option<String> {
    // `0` = unbounded (loosest). `expands(old, new)`: new is looser than old.
    fn expands(old: u64, new: u64) -> bool {
        match (old, new) {
            (o, n) if o == n => false,
            (0, _) => false, // old already unbounded — nothing is looser
            (_, 0) => true,  // new unbounded, old finite — expansion
            (o, n) => n > o, // both finite — larger is looser
        }
    }
    let checks: [(&str, u64, u64); 11] = [
        ("max_frame_bytes", old.max_frame_bytes, new.max_frame_bytes),
        ("spool_frames", old.spool_frames, new.spool_frames),
        (
            "per_sender_quota",
            old.per_sender_quota,
            new.per_sender_quota,
        ),
        ("advisory_depth", old.advisory_depth, new.advisory_depth),
        ("payload_depth", old.payload_depth, new.payload_depth),
        ("gossip_depth", old.gossip_depth, new.gossip_depth),
        (
            "max_live_handles",
            old.max_live_handles,
            new.max_live_handles,
        ),
        ("max_live_bytes", old.max_live_bytes, new.max_live_bytes),
        (
            "max_readback_bytes",
            old.max_readback_bytes,
            new.max_readback_bytes,
        ),
        (
            "max_outstanding_ops",
            old.max_outstanding_ops,
            new.max_outstanding_ops,
        ),
        (
            "compute_queue_depth",
            old.compute_queue_depth,
            new.compute_queue_depth,
        ),
    ];
    for (name, o, n) in checks {
        if expands(o, n) {
            return Some(format!(
                "grant `{name}` expands from {o} to {n} (0 = unbounded)"
            ));
        }
    }
    // Artifacts: the new allow-list must be a subset of the old (no newly-reachable artifact).
    for h in &new.granted_artifacts {
        if !old.granted_artifacts.contains(h) {
            return Some(format!(
                "granted artifact {} is not in the previously-admitted set",
                h.to_hex()
            ));
        }
    }
    None
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
    // §12.3 distribution records travel on the same plane beside frames; the classification is
    // structural (a record is a top-level map, a §12.1 frame a top-level array), so this decode
    // attempt is cheap and never speculative on frame bytes. A refused record is a typed
    // per-record advisory, never a session fault; our own echoed announcement ingests as an
    // idempotent no-op.
    if let Ok(record) = crate::distribution::DistributionRecord::from_bytes(frame) {
        match attach.ingest_distribution(record) {
            Ok(()) => {
                tracing::debug!("inbound: distribution record ingested (peer cert/revocation)")
            }
            Err(refusal) => {
                tracing::debug!(%refusal, "inbound: distribution record refused");
                let _ = events.send(Event::Warning {
                    class: "distribution_refused".into(),
                    detail: format!("{refusal} (generation {generation})"),
                });
            }
        }
        return true;
    }
    // Echo filter: the §12.1 frame envelope's `sender` field (mechanism, not message schema).
    // The module's own voice is already its state; a self-delivering plane must not feed it back.
    if envelope_sender(frame) == Some(own_sender.0) {
        return true;
    }
    match attach.deliver(frame) {
        Ok(InboundVerdict::Deliver { .. } | InboundVerdict::Duplicate { .. }) => {
            tracing::trace!(
                sender = envelope_sender(frame).map(|s| hex16(&s)),
                "inbound: frame delivered to the module"
            );
            true
        }
        Ok(InboundVerdict::Gap { .. } | InboundVerdict::Backpressure { .. }) => {
            tracing::debug!("inbound: frame held (gap/backpressure)");
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
            tracing::debug!(?verdict, "inbound: frame refused (typed)");
            let _ = events.send(Event::Warning {
                class: "frame_refused".into(),
                detail: format!("{verdict:?} (generation {generation})"),
            });
            true
        }
        Err(_) => false,
    }
}

/// A short hex prefix of an identity, for diagnostics (never a security surface).
fn hex16(bytes: &[u8; 32]) -> String {
    bytes[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// A short label for a published §12.1 frame's inner module payload — diagnostics only (the
/// session never branches on message schema; this is a best-effort peek at the opaque bytes).
fn frame_kind(frame: &[u8]) -> String {
    let Ok(ciborium::value::Value::Array(parts)) = ciborium::de::from_reader(frame) else {
        return "frame?".into();
    };
    let Some(ciborium::value::Value::Bytes(payload)) = parts.get(1) else {
        return "frame?".into();
    };
    let Ok(ciborium::value::Value::Array(items)) = ciborium::de::from_reader(payload.as_slice())
    else {
        return "opaque".into();
    };
    // A module tag frame is `[uint tag, uint round, bytes]`; a wire VhcMessage is a tagged enum
    // (a 1-element map or a variant array) — distinguish by the first element's shape.
    match items.first() {
        Some(ciborium::value::Value::Integer(n)) => format!("tag{}", i128::from(*n)),
        Some(ciborium::value::Value::Text(t)) => format!("wire:{t}"),
        _ => format!("wire/arr[{}]", items.len()),
    }
}

/// A short label for an op request kind — diagnostics only.
fn op_kind(req: &OpRequest) -> &'static str {
    match req {
        OpRequest::PayloadPut { .. } => "PayloadPut",
        OpRequest::PayloadGet { .. } => "PayloadGet",
        OpRequest::ArtifactFetch { .. } => "ArtifactFetch",
        OpRequest::ArtifactRange { .. } => "ArtifactRange",
        OpRequest::StreamOpen { .. } => "StreamOpen",
        OpRequest::StreamAccept => "StreamAccept",
        OpRequest::StreamWrite { .. } => "StreamWrite",
        OpRequest::StreamRead { .. } => "StreamRead",
        OpRequest::TensorExport => "TensorExport",
        OpRequest::TensorImport { .. } => "TensorImport",
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
///
/// Metrics carry a reserved round-outcome group (ABI `round_metrics`): the trainer guest reports
/// its per-round det digest + barrier bookkeeping as `vhc.round.<round>.<field>` `(name, f64)`
/// metrics. The session recognizes those reserved NAMES — it decodes no module frame — folds them
/// into [`Event::RoundOutcome`] as each round's group completes, and consumes them (only ordinary
/// telemetry rides on as an [`Event::Metric`]). This is the sole opacity-safe live producer of the
/// per-round digest on the product path.
async fn relay_egress(
    pump: &PumpHandle,
    providers: &RoleProviders,
    cursors: &mut EgressCursors,
    events: &mpsc::UnboundedSender<Event>,
    generation: u64,
) -> Result<(), String> {
    // Guest log lines onto the node's own stream. This is how a guest panic becomes readable:
    // the SDK's hook forwards the message here a beat before the `unreachable` trap, so the line
    // is on disk even when the run then dies (ABI §3.6).
    let logs = pump.logs();
    for (level, line) in logs.iter().skip(cursors.logs) {
        match level {
            0 => tracing::trace!(guest = %line, "guest log"),
            1 => tracing::debug!(guest = %line, "guest log"),
            2 => tracing::info!(guest = %line, "guest log"),
            3 => tracing::warn!(guest = %line, "guest log"),
            _ => tracing::error!(guest = %line, "guest log"),
        }
    }
    cursors.logs = logs.len();

    let published = pump.published();
    for (channel, seq, frame) in published.iter().skip(cursors.published) {
        tracing::trace!(channel, seq, kind = %frame_kind(frame), "egress: publishing module frame");
        providers
            .control
            .publish(frame)
            .await
            .map_err(|e| format!("control plane publish: {e}"))?;
    }
    cursors.published = published.len();

    for (op, request) in pump.take_op_requests() {
        tracing::trace!(op, request = %op_kind(&request), "egress: servicing capability op");
        service_op(pump, providers, op, request, events, generation).await?;
    }

    let metrics = pump.metrics();
    for (name, value) in metrics.iter().skip(cursors.metrics) {
        // Fold the reserved round-outcome group into a RoundOutcome the moment its digest
        // completes (all four LE words). The values are treated as opaque numbers under a reserved
        // NAMING contract — no frame is decoded here. Robust to partial groups / reordering.
        if let Some(outcome) = cursors.digest_accum.observe(name, *value) {
            let _ = events.send(Event::RoundOutcome {
                round: outcome.round,
                committed: outcome.committed,
                ingested: outcome.ingested,
                stalled: outcome.stalled,
                digest: outcome.digest,
                generation,
            });
        }
        // The reserved carrier metrics are consumed above; only ordinary telemetry is forwarded as
        // an advisory metric event (a reserved name is never surfaced as bare telemetry).
        if !daemon_vhc_abi::round_metrics::RoundOutcomeAccumulator::is_reserved(name) {
            let _ = events.send(Event::Metric {
                name: name.clone(),
                value: *value,
            });
        }
    }
    cursors.metrics = metrics.len();
    Ok(())
}

/// Service one capability op against the providers. The PUMP owns every trust step (hash
/// verification, range slicing, handle minting) — the provider only moves bytes.
async fn service_op(
    pump: &PumpHandle,
    providers: &RoleProviders,
    op: u64,
    request: OpRequest,
    events: &mpsc::UnboundedSender<Event>,
    generation: u64,
) -> Result<(), String> {
    let outcome = match request {
        OpRequest::PayloadPut { bytes } => match providers.payloads.put_content(&bytes).await {
            Ok(hash) => {
                // The periodic LIVE checkpoint cadence (spec §9): a put whose bytes carry the
                // host's own §10.2 checkpoint-document shape — a hash-consistent state manifest
                // plus its sections and a round watermark — is a mid-run full-state checkpoint,
                // surfaced so the node records the (role, live) pointer. Structural recognition
                // over the HOST's contract shape only (the module's round vocabulary stays
                // opaque); a non-checkpoint put never matches.
                if let Some(round) = live_checkpoint_watermark(&bytes) {
                    // Publish the referenced family CHUNKS to the content plane ([SF-6]/C5): a
                    // restoring peer fetches them chunk-keyed ([SF-R2]). The chunks come from THIS
                    // (publishing) instance's self-sealed store; content-addressed `put_content` is
                    // idempotent, so family chunks unchanged since a prior slot upload NOTHING
                    // (skip-on-present) — the amortized remote cost is the changed chunks + the
                    // small document, not the whole family every slot.
                    if let Ok(capture) = decode_snapshot_doc(&bytes) {
                        let mut referenced: Vec<[u8; 32]> = Vec::new();
                        for section in &capture.sections {
                            if let daemon_vhc_proto::det_state::CkptDocSection::ByRef(_, fref) =
                                section
                            {
                                referenced.push(fref.fold.0);
                                if let Some(chunks) = pump.sealed_fold_chunks(&fref.fold.0) {
                                    for (_h, chunk) in &chunks {
                                        let _ = providers.artifacts.put_content(chunk).await;
                                    }
                                }
                            }
                        }
                        // Pin the freshest checkpoint's families out of retention eviction (§8.2,
                        // C6); a superseded checkpoint's folds re-enter ordinary retention.
                        pump.repin_checkpoint(&referenced);
                    }
                    let hash = hash.to_hex();
                    let _ = events.send(Event::CheckpointPublished {
                        round,
                        hash: hash.clone(),
                        location: format!("payload/{hash}"),
                        generation,
                        kind: "live".into(),
                    });
                }
                OpOutcome::PutDone
            }
            Err(e) => {
                // Voiced HOST-SIDE too: the failure detail otherwise rides only the completion
                // into the guest, and a fail-loud guest surfaces just its panic site — the
                // transport cause would be unrecoverable from the node log.
                tracing::warn!(error = %e, "payload put failed; failing the module op");
                OpOutcome::Failed {
                    code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                    detail: format!("payload put: {e}"),
                }
            }
        },
        OpRequest::PayloadGet { hash } => {
            match providers
                .payloads
                .get_content(&daemon_vhc_proto::Hash(hash))
                .await
            {
                Ok(bytes) => OpOutcome::GetDone { bytes },
                Err(e) => {
                    tracing::warn!(error = %e, "payload get failed; failing the module op");
                    OpOutcome::Failed {
                        code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                        detail: format!("payload get: {e}"),
                    }
                }
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
            // An externally-sourced DET-STATE fold ([SF-R2]/[SF-6]) is served CHUNK-KEYED — the
            // symmetric twin of the replay-side materialization (`run/replay.rs`): the family's
            // chunks live in the content-addressed plane each under its OWN blake3, so the
            // covering span is reassembled by fetching those chunks and concatenating them, never
            // a whole-family object under the fold key. The registered `DetStateChunkMap` (held by
            // the pump — the single source of truth) decomposes the chunk-aligned span into its
            // `(chunk hash, len)` list. The pump still verifies the reassembled covering span
            // against the fold-committed hashes at completion (unchanged) — the resolver is
            // untrusted, and content-addressed `get_content` self-verifies each chunk besides.
            if let Some(map) = pump.state_chunk_map(&hash) {
                match map.covering_chunks(span_off, span_len) {
                    Ok(chunks) => {
                        let mut bytes = Vec::with_capacity(usize::try_from(span_len).unwrap_or(0));
                        let mut failure: Option<OpOutcome> = None;
                        for (chunk_hash, _len) in &chunks {
                            match providers.artifacts.get_content(chunk_hash).await {
                                Ok(chunk) => bytes.extend_from_slice(&chunk),
                                Err(e) => {
                                    failure = Some(OpOutcome::Failed {
                                        code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                                        detail: format!(
                                            "det-state chunk {} fetch: {e}",
                                            chunk_hash.to_hex()
                                        ),
                                    });
                                    break;
                                }
                            }
                        }
                        failure.unwrap_or(OpOutcome::RangeDone { bytes })
                    }
                    Err(e) => OpOutcome::Failed {
                        code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                        detail: format!("det-state covering span: {e}"),
                    },
                }
            } else {
                // A chunk-addressed corpus shard (fold identity): the content store holds the
                // whole object under the fold key; this in-process seat slices the requested
                // covering span itself — the pump verifies every covering chunk against the
                // registered chunk map either way (the provider is untrusted by construction). A
                // store that cannot serve the span is a typed refusal, never silent bytes.
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
    match &outcome {
        OpOutcome::Failed { code, detail } => {
            tracing::debug!(op, code, detail, "capability op FAILED")
        }
        other => {
            tracing::trace!(op, outcome = ?std::mem::discriminant(other), "capability op serviced")
        }
    }
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
    tracing::debug!(
        run = run_label,
        ?leave_requested,
        transport_fault = transport_fault.as_deref(),
        finished = run.is_finished(),
        "role session finishing"
    );

    if run.is_finished() {
        // The guest ended on its own: classify its end (a leave racing a natural end still
        // reports the natural end — the module's exit is the truth).
        let end = wait_run(run).await;
        tracing::debug!(
            run = run_label,
            ?end,
            "role session: guest ended on its own"
        );
        return classify_natural_end(end, transport_fault);
    }

    match (leave_requested, transport_fault) {
        (Some(LeaveMode::Graceful), _) => {
            // Quiesce: drain + module snapshot at the fence, snapshot persisted to the payload
            // plane as the leave checkpoint.
            let deadline_ms = u64::try_from(drain_deadline.as_millis()).unwrap_or(u64::MAX);
            if let Err(e) = pump.quiesce(daemon_vhc_abi::QUIESCE_REASON_LEAVE, deadline_ms) {
                tracing::debug!(run = run_label, error = %e, "graceful leave: quiesce refused; stopping without a checkpoint");
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
                tracing::debug!(
                    run = run_label,
                    ?end,
                    "graceful leave: drain did not quiesce cleanly"
                );
                let _ = events.send(Event::Warning {
                    class: "leave_drain".into(),
                    detail: format!("drain did not quiesce cleanly: {end:?}"),
                });
                return TerminalOutcome::Left { checkpoint: None };
            }
            let capture = pump.snapshot_capture();
            // The snapshot's resync watermark, when the module declared one (the live trainer's
            // `round` section: the last round this state folds — a restore never re-ingests at or
            // below it). The pointer carries it so a rejoiner knows which state it is adopting.
            let round = capture
                .as_ref()
                .and_then(|c| {
                    c.sections.iter().find_map(|s| match s {
                        daemon_vhc_proto::det_state::CkptDocSection::Inline(name, bytes)
                            if name == "round" =>
                        {
                            <[u8; 8]>::try_from(bytes.as_slice()).ok()
                        }
                        _ => None,
                    })
                })
                .map(u64::from_le_bytes)
                .filter(|w| *w != u64::MAX)
                .unwrap_or(0);
            let checkpoint = match capture {
                Some(capture) => persist_drain_snapshot(providers, &capture).await,
                None => {
                    tracing::debug!(
                        run = run_label,
                        "graceful leave: guest quiesced but staged no snapshot capture"
                    );
                    None
                }
            };
            if let Some(hash) = &checkpoint {
                let _ = events.send(Event::CheckpointPublished {
                    round,
                    hash: hash.clone(),
                    location: format!("payload/{hash}"),
                    generation,
                    kind: "drain".into(),
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
    // The allocator readout, taken here because this is the one seam every role run passes through
    // on its way to termination — so the samples are read before the pump goes away with the run.
    //
    // Without a reader the sampler was recording into a buffer discarded at teardown: the boundaries
    // were wired, the readings were real, and no run left any of it behind. Every workspace, pooling,
    // compilation and staging term a profile prices is calibrated against these figures, so an
    // unread sampler leaves those terms with nothing to rest on but conservative bounds.
    //
    // Off by default and env-gated, following the probe readout: this is calibration evidence for an
    // operator running the deployed binary on a real box, not something every production run should
    // print. It goes to stderr so it cannot be mistaken for run output on the wire.
    if std::env::var_os("DAEMON_TRAIN_ALLOCATOR_READOUT").is_some() {
        report_allocator_samples(&run.pump);
    }
    tokio::task::spawn_blocking(move || run.wait().map_err(|e| e.to_string()))
        .await
        .map_err(|e| format!("guest join: {e}"))?
}

/// Print the allocator sample series, one line per reading, in the order they were taken.
///
/// Order is preserved because the SHAPE across boundaries is the evidence: a pool that never returns
/// memory to the driver looks identical to one that does at any single point. `bytes_reserved` above
/// `bytes_in_use` is the retention a pooling term models, and `bytes_padding` separates what the
/// module asked for from what alignment made it cost, so an alignment term and a workspace term can
/// be told apart rather than folded together.
///
/// An empty series is reported as such, explicitly. It is **not** the same as a run that allocated
/// nothing — a backend that cannot report occupancy records nothing — and a reader who took absence
/// for zero would be calibrating a profile against a figure nobody measured.
fn report_allocator_samples(pump: &daemon_vhc_host::run::PumpHandle) {
    let samples = pump.allocator_samples();
    if samples.is_empty() {
        eprintln!(
            "allocator_readout: no samples — this backend cannot report allocator occupancy. This \
             is an ABSENCE, not a zero: nothing here may be read as `bytes_in_use = 0`."
        );
        return;
    }
    eprintln!(
        "allocator_readout: {} sample(s), in the order taken",
        samples.len()
    );
    for (index, (point, sample)) in samples.iter().enumerate() {
        eprintln!(
            "allocator_readout[{index}]: point={} allocs={} in_use={} padding={} reserved={}",
            point.slug(),
            sample.number_allocs,
            sample.bytes_in_use,
            sample.bytes_padding,
            sample.bytes_reserved
        );
    }
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
        // `StaleRestore` (ABI §4.5 code 3): the module refused to fold a record history gapped
        // above its restored watermark. Unlike every other nonzero outcome this is NOT
        // deterministic for the (module, plan, grant) tuple — the environment moved (rounds were
        // committed before this incarnation attached), and a retry restores a FRESHER checkpoint
        // pointer (live pointers advance every ingested round), so reconvergence is the designed
        // recovery, exactly like a churn respawn.
        Ok(RunEnd::Outcome(code)) if code == daemon_vhc_abi::OUTCOME_STALE_RESTORE => {
            TerminalOutcome::FailedRetryable {
                reason: "stale restore: the record history is gapped above the restored \
                         watermark; rejoining to restore a fresher checkpoint"
                    .into(),
            }
        }
        Ok(RunEnd::Outcome(code)) => TerminalOutcome::FailedTerminal {
            reason: format!("module ended with outcome {code}"),
        },
        Ok(RunEnd::InitRefused(code)) => TerminalOutcome::FailedTerminal {
            reason: format!("module refused init ({code})"),
        },
        Ok(RunEnd::MigrateRefused(code)) => TerminalOutcome::FailedTerminal {
            reason: format!("module refused migration ({code})"),
        },
        // Terminal, not retryable: the refusal is deterministic for this (module, plan, grant)
        // tuple, so a fresh instance would reach the same status. Retrying needs changed admitted
        // input — a different selected configuration — which is a new admission, not a restart.
        Ok(RunEnd::ExecutionGrantRejected(status)) => TerminalOutcome::FailedTerminal {
            reason: format!("module refused the execution grant (status {status})"),
        },
        Ok(RunEnd::Trapped(trap)) => classify_trap(&trap),
        Err(e) => TerminalOutcome::FailedTerminal {
            reason: format!("guest thread: {e}"),
        },
    }
}

/// Trap classification: resource-budget breaches are the node's churn/preemption loop
/// (retryable); a deferred device fault (`ComputeFault` — driver OOM, a host-side allocation
/// rejection, a device that died at bring-up) is a CAPACITY fault, likewise retryable — the
/// node reassesses against the live device inventory (the admitted claim's headroom is the
/// primary defense; the hardware findings record that a faithful driver-OOM diagnostic is not
/// distinguishable from a host-side pool rejection, so the class is judged conservatively
/// recoverable). Everything else is a module/admission fault (terminal).
fn classify_trap(trap: &daemon_vhc_host::trap::Trap) -> TerminalOutcome {
    match trap.code {
        TrapCode::BudgetMemory
        | TrapCode::BudgetFuel
        | TrapCode::BudgetEpoch
        | TrapCode::BudgetOps
        | TrapCode::BudgetHandles => TerminalOutcome::FailedRetryable {
            reason: format!("resource budget breached: {trap}"),
        },
        TrapCode::ComputeFault => TerminalOutcome::FailedRetryable {
            reason: format!("device compute fault (capacity class): {trap}"),
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
    let bytes = match encode_snapshot_doc(capture) {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(error = %e, "drain snapshot: checkpoint document encode failed");
            return None;
        }
    };
    match providers.payloads.put_content(&bytes).await {
        Ok(hash) => Some(hash.to_hex()),
        Err(e) => {
            tracing::warn!(error = %e, "drain snapshot: checkpoint document put failed");
            None
        }
    }
}

/// The content-addressed checkpoint document v2 ([SF-6]): `[manifest, [ckpt-doc-section…]]` in the
/// manifest's declared order (the object a checkpoint pointer addresses). Encoder + decoder are
/// the shared proto codec, so a late joiner rebuilds the exact [`SnapshotCapture`] the leaving
/// peer wrote — inline sections verbatim, by-ref sections as their `FamilyRef` (the referenced
/// family chunks ride the payload plane via the publisher/leave chunk upload).
pub fn encode_snapshot_doc(
    capture: &daemon_vhc_host::run::SnapshotCapture,
) -> Result<Vec<u8>, String> {
    daemon_vhc_proto::det_state::encode_checkpoint_doc(&capture.manifest, &capture.sections)
        .map_err(|e| format!("encode snapshot doc: {e}"))
}

/// Recognize a payload-plane put as a §10.2 checkpoint DOCUMENT and return its round watermark
/// (the periodic LIVE checkpoint cadence's announcement seam, spec §9). Recognition is strictly
/// structural over the HOST's own contract shapes — the `[manifest, sections]` document paired
/// with `encode_snapshot_doc`, whose manifest is the §10.2 state-manifest map — and every
/// declared section must hash-match its bytes (a coincidental or corrupt object never registers
/// a pointer). The module's round vocabulary is never decoded. `None` = not a checkpoint.
fn live_checkpoint_watermark(bytes: &[u8]) -> Option<u64> {
    use daemon_vhc_proto::det_state::CkptDocSection;
    let capture = decode_snapshot_doc(bytes).ok()?;
    // The §10.2 state-manifest map: `sections: [{name, hash, size, …}]`.
    let v: ciborium::value::Value = ciborium::de::from_reader(capture.manifest.as_slice()).ok()?;
    let ciborium::value::Value::Map(entries) = v else {
        return None;
    };
    let sections_v = entries.iter().find_map(|(k, val)| match k {
        ciborium::value::Value::Text(t) if t == "sections" => Some(val),
        _ => None,
    })?;
    let ciborium::value::Value::Array(decls) = sections_v else {
        return None;
    };
    if decls.len() != capture.sections.len() || decls.is_empty() {
        return None;
    }
    let mut watermark = None;
    for (decl, section) in decls.iter().zip(&capture.sections) {
        let ciborium::value::Value::Map(fields) = decl else {
            return None;
        };
        let field = |want: &str| {
            fields.iter().find_map(|(k, val)| match k {
                ciborium::value::Value::Text(t) if t == want => Some(val),
                _ => None,
            })
        };
        match field("name") {
            Some(ciborium::value::Value::Text(t)) if t == section.name() => {}
            _ => return None,
        }
        // The decl's `hash` cross-checks the section: for an INLINE section it is the blake3 of
        // the bytes; for a BY-REFERENCE section it is the family FOLD, and the ref must be
        // self-consistent (its fold IS the fold of its own chunk list — the host store holds
        // those chunks by construction, §7.2). A coincidental or corrupt object never matches.
        let hash_ok = match (field("hash"), section) {
            (Some(ciborium::value::Value::Bytes(h)), CkptDocSection::Inline(_, section_bytes)) => {
                h.as_slice() == blake3::hash(section_bytes).as_bytes()
            }
            (Some(ciborium::value::Value::Bytes(h)), CkptDocSection::ByRef(_, family)) => {
                family.validate().is_ok() && h.as_slice() == family.fold.0.as_slice()
            }
            _ => false,
        };
        if !hash_ok {
            return None;
        }
        if let CkptDocSection::Inline(name, section_bytes) = section {
            if name == "round" {
                watermark = <[u8; 8]>::try_from(section_bytes.as_slice())
                    .ok()
                    .map(u64::from_le_bytes)
                    .filter(|w| *w != u64::MAX);
            }
        }
    }
    // Only a doc carrying the live watermark section registers a LIVE pointer (the drain path
    // has its own event; a watermark-less doc has no restore ordering to offer).
    watermark
}

/// Decode a checkpoint document back into a [`SnapshotCapture`] (the late-join restore input),
/// via the shared checkpoint-document v2 codec ([SF-6]): inline sections carry bytes, by-ref
/// sections carry a `FamilyRef` the restoring instance registers + streams. Fails typed on any
/// structural surprise — a malformed checkpoint never yields a partial state.
pub fn decode_snapshot_doc(bytes: &[u8]) -> Result<daemon_vhc_host::run::SnapshotCapture, String> {
    let (manifest, sections) =
        daemon_vhc_proto::det_state::decode_checkpoint_doc(bytes).map_err(|e| e.to_string())?;
    Ok(daemon_vhc_host::run::SnapshotCapture { manifest, sections })
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
        // Budget breaches are the node's churn/preemption loop; device compute faults are the
        // CAPACITY class (driver OOM / allocation rejection / bring-up loss — recoverable by a
        // reassess against the live device inventory); module faults are terminal.
        for code in [
            TrapCode::BudgetMemory,
            TrapCode::BudgetFuel,
            TrapCode::BudgetEpoch,
            TrapCode::BudgetOps,
            TrapCode::BudgetHandles,
            TrapCode::ComputeFault,
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
    fn snapshot_doc_round_trips_and_refuses_malformed() {
        // The late-join restore codec: a capture encodes + decodes to the identical sections in
        // declared order; junk refuses typed (a malformed checkpoint never yields partial state).
        use daemon_vhc_proto::det_state::{CkptDocSection, FamilyRef};
        // A mix of inline + by-reference sections (the v2 forms) round-trips verbatim.
        let fref = FamilyRef {
            fold: daemon_vhc_proto::Hash([9u8; 32]),
            byte_len: 8,
            chunk_size: 8,
            chunk_hashes: vec![daemon_vhc_proto::Hash(
                *blake3::hash(b"abcdefgh").as_bytes(),
            )],
        };
        let capture = daemon_vhc_host::run::SnapshotCapture {
            manifest: b"manifest-bytes".to_vec(),
            sections: vec![
                CkptDocSection::ByRef("master".into(), fref),
                CkptDocSection::Inline("round".into(), vec![4, 5]),
            ],
        };
        let bytes = encode_snapshot_doc(&capture).expect("encode");
        let back = decode_snapshot_doc(&bytes).expect("decode");
        assert_eq!(back.manifest, capture.manifest);
        assert_eq!(back.sections, capture.sections);
        assert!(decode_snapshot_doc(b"not-a-snapshot-doc").is_err());
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
