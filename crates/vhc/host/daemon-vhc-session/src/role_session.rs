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
    blake3_hash, peer_id, to_canonical_vec, AdmittedQuotas, CertScope, Hash, PeerId,
    RunKeyCertificate, SigningKey,
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
    published_cursor: usize,
    metrics_cursor: usize,
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
    // is the safety floor); a WS plane additionally re-announces on every reconnect via the
    // resubscribe registration made at provider construction.
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
        published_cursor: 0,
        metrics_cursor: 0,
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
                    &current.pump, &providers, &mut current.published_cursor,
                    &mut current.metrics_cursor, events, gen_now,
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
        &mut current.published_cursor,
        &mut current.metrics_cursor,
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
    if tuple.claim_hash != *blake3::hash(&admission.claim_bytes).as_bytes() {
        return refused(
            current,
            "the post-switch admitted tuple's claim hash does not match the re-evaluated claim"
                .into(),
        );
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
        let started = daemon_vhc_host::run::start_run_migrating(
            worker,
            &module_bytes,
            cfg,
            journal,
            Some(MigrationInput {
                capture: capture.clone(),
                restore: true,
                migrate_fuel: binding.migrate_fuel,
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
            published_cursor: 0,
            metrics_cursor: 0,
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
async fn relay_egress(
    pump: &PumpHandle,
    providers: &RoleProviders,
    published_cursor: &mut usize,
    metrics_cursor: &mut usize,
    events: &mpsc::UnboundedSender<Event>,
    generation: u64,
) -> Result<(), String> {
    let published = pump.published();
    for (channel, seq, frame) in published.iter().skip(*published_cursor) {
        tracing::trace!(channel, seq, kind = %frame_kind(frame), "egress: publishing module frame");
        providers
            .control
            .publish(frame)
            .await
            .map_err(|e| format!("control plane publish: {e}"))?;
    }
    *published_cursor = published.len();

    for (op, request) in pump.take_op_requests() {
        tracing::trace!(op, request = %op_kind(&request), "egress: servicing capability op");
        service_op(pump, providers, op, request, events, generation).await?;
    }

    let metrics = pump.metrics();
    for (name, value) in metrics.iter().skip(*metrics_cursor) {
        let _ = events.send(Event::Metric {
            name: name.clone(),
            value: *value,
        });
    }
    *metrics_cursor = metrics.len();
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
                    c.sections.iter().find_map(|(name, bytes)| {
                        (name == "round")
                            .then(|| <[u8; 8]>::try_from(bytes.as_slice()).ok())
                            .flatten()
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

/// The content-addressed checkpoint document: `[manifest, [[name, section-bytes], …]]` in the
/// manifest's declared order (the object a checkpoint pointer addresses). Encoder + decoder are
/// paired so a late joiner rebuilds the exact [`SnapshotCapture`] the leaving peer wrote.
pub fn encode_snapshot_doc(
    capture: &daemon_vhc_host::run::SnapshotCapture,
) -> Result<Vec<u8>, String> {
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
    to_canonical_vec(&doc).map_err(|e| format!("encode snapshot doc: {e}"))
}

/// Recognize a payload-plane put as a §10.2 checkpoint DOCUMENT and return its round watermark
/// (the periodic LIVE checkpoint cadence's announcement seam, spec §9). Recognition is strictly
/// structural over the HOST's own contract shapes — the `[manifest, sections]` document paired
/// with `encode_snapshot_doc`, whose manifest is the §10.2 state-manifest map — and every
/// declared section must hash-match its bytes (a coincidental or corrupt object never registers
/// a pointer). The module's round vocabulary is never decoded. `None` = not a checkpoint.
fn live_checkpoint_watermark(bytes: &[u8]) -> Option<u64> {
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
    for (decl, (name, section_bytes)) in decls.iter().zip(&capture.sections) {
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
            Some(ciborium::value::Value::Text(t)) if t == name => {}
            _ => return None,
        }
        match field("hash") {
            Some(ciborium::value::Value::Bytes(h))
                if h.as_slice() == blake3::hash(section_bytes).as_bytes() => {}
            _ => return None,
        }
        if name == "round" {
            watermark = <[u8; 8]>::try_from(section_bytes.as_slice())
                .ok()
                .map(u64::from_le_bytes)
                .filter(|w| *w != u64::MAX);
        }
    }
    // Only a doc carrying the live watermark section registers a LIVE pointer (the drain path
    // has its own event; a watermark-less doc has no restore ordering to offer).
    watermark
}

/// Decode a checkpoint document back into a [`SnapshotCapture`] (the late-join restore input).
/// Fails typed on any structural surprise — a malformed checkpoint never yields a partial state.
pub fn decode_snapshot_doc(bytes: &[u8]) -> Result<daemon_vhc_host::run::SnapshotCapture, String> {
    let v: ciborium::value::Value =
        ciborium::de::from_reader(bytes).map_err(|e| format!("snapshot doc cbor: {e}"))?;
    let ciborium::value::Value::Array(parts) = v else {
        return Err("snapshot doc is not an array".into());
    };
    let [manifest_v, sections_v] = <[ciborium::value::Value; 2]>::try_from(parts)
        .map_err(|_| "snapshot doc arity != 2".to_string())?;
    let ciborium::value::Value::Bytes(manifest) = manifest_v else {
        return Err("snapshot manifest is not bytes".into());
    };
    let ciborium::value::Value::Array(section_vs) = sections_v else {
        return Err("snapshot sections is not an array".into());
    };
    let mut sections = Vec::with_capacity(section_vs.len());
    for entry in section_vs {
        let ciborium::value::Value::Array(kv) = entry else {
            return Err("snapshot section is not a [name, bytes] pair".into());
        };
        let [name_v, bytes_v] = <[ciborium::value::Value; 2]>::try_from(kv)
            .map_err(|_| "snapshot section arity != 2".to_string())?;
        let ciborium::value::Value::Text(name) = name_v else {
            return Err("snapshot section name is not text".into());
        };
        let ciborium::value::Value::Bytes(section) = bytes_v else {
            return Err("snapshot section value is not bytes".into());
        };
        sections.push((name, section));
    }
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
        let capture = daemon_vhc_host::run::SnapshotCapture {
            manifest: b"manifest-bytes".to_vec(),
            sections: vec![
                ("params".into(), vec![1, 2, 3]),
                ("residual".into(), vec![4, 5]),
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
