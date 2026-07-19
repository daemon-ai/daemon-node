// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

// The worker reads its module path from an env var and the module bytes from disk (developer /
// node-controlled inputs, mirroring `fake-train-worker`); the fs/env hardening bans target the
// shipped node process, not this isolated worker binary. Allowed file-wide (crate-level, so the
// `backend`/`session` submodules inherit it too).
#![allow(clippy::disallowed_methods)]
#![forbid(unsafe_code)]

//! The `daemon-vhc-worker` binary — the child side of the frozen worker protocol (§10.2).
//!
//! Speaks [`daemon_vhc_session::protocol`] `Command`/`Event` frames over the length-framed
//! [`daemon_provision::CutChannel`] stdio cut (exactly like `fake-train-worker`, and consumed by
//! `daemon-vhc-supervisor::TrainSupervisor`), driving the real `daemon-vhc-host` runtime:
//!
//! - `Probe` → a real host capability report (the frozen 66-op vocabulary; GPU absent = CPU-only).
//! - `AssessRun{envelope}` → the peer-side re-validation (spec §6.5): verify the canonical
//!   [`daemon_vhc_proto::SignedEnvelope`] carrying a **genesis envelope v2**, resolve the worker
//!   role's module by its pinned artifact hash (blake3-verified; `DAEMON_TRAIN_MODULE` overrides
//!   the source inside the signed path), then the ABI §1.3 driver selection + the A2
//!   claim()-admission funnel → `Assessed`. **A raw config-CBOR envelope is refused typed
//!   (`UnsignedEnvelopeRetired`, D0), a schema-major-1 envelope is refused typed
//!   (`EnvelopeSchemaRetired` — the v1 form cannot configure a coordinator), and a major-1
//!   module is refused typed (`AbiUnsupportedMajor`)** — the flipped A0 fixture pins the last.
//! - `JoinRun` → spawns a **role session** (`daemon_vhc_session::role_session`) as a background
//!   task and returns to the loop immediately: the command loop NEVER blocks on run execution.
//!   The worker keeps only the `{role instance → role handle}` task map; the session owns the
//!   run (opaque frame routing, capability servicing, lifecycle, terminal classification).
//! - `Throttle` → forwarded into every live session (`paused` is a hard pump-level gate; duty
//!   percentage is a cooperative budget advisory).
//! - `Leave` → graceful (quiesce + drain-snapshot checkpoint) or immediate; either way the
//!   session emits the classified terminal `RunTerminated`.
//! - `Shutdown` → cancels every live session under a bounded drain deadline, then exits.
//!
//! A trapping module surfaces as a classified terminal outcome — the worker is never harmed.

mod backend;
// The in-process self-driven join: it decodes the SDK round schemas and drives the run's
// coordinator module in-process, so it is HARNESS machinery — the production worker routes
// opaque signed frames only and carries no round vocabulary (dep-check-enforced; the
// decentralized role-session join lands at the node command surface).
#[cfg(feature = "harness")]
mod session;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use daemon_provision::{CutChannel, CutWriter};
use daemon_vhc_session::protocol::{self, Command, ErrorClass, Event, LeaveMode};
use daemon_vhc_session::role_session::{
    spawn_role, RoleHandle, RoleProviders, RoleSessionSpec, ThrottleLevel,
};

/// The graceful-leave / shutdown drain ceiling. Becomes lane configuration when the node-side
/// quiesce ceilings land; a constant keeps the drain bounded meanwhile.
const DRAIN_DEADLINE: Duration = Duration::from_secs(10);

/// Whether the node/test selected the **in-process plane** for this worker's role sessions: an
/// in-process control plane + in-memory content stores — the single-host smoke seat. The live
/// transport attach (credential-selected planes and stores) supersedes this; without either, a
/// join refuses typed (fail closed, never a silent local run).
fn in_process_plane_selected() -> bool {
    std::env::var_os("DAEMON_VHC_INPROC_PLANE").is_some_and(|v| v == "1")
}

/// Providers for the in-process plane seat.
fn in_process_providers() -> RoleProviders {
    RoleProviders {
        control: Arc::new(daemon_vhc_net::LoopbackGossip::new()),
        payloads: Arc::new(daemon_vhc_net::MemoryContentStore::new()),
        artifacts: Arc::new(daemon_vhc_net::MemoryContentStore::new()),
    }
}

/// Author the live role-session spec for a join whose credentials selected the production
/// planes: re-run the admission binding, open the durable journal home, and construct the
/// transport providers from the plane selection (WS connect fails FAST on an unreachable
/// endpoint — one dial, typed error; reconnect policy applies only to an established plane, so
/// the command loop never hangs here).
#[cfg(feature = "vhc-net")]
async fn join_live(
    resolved: &backend::ResolvedRun,
    genesis: &backend::GenesisRun,
    run_id: &str,
    coordinator: &str,
    creds: &daemon_vhc_session::protocol::SessionCredentials,
    incarnation: u64,
) -> Result<RoleSessionSpec, String> {
    use daemon_vhc_session::providers::{build_role_providers, LiveAttachInputs};

    // Credentials bind to ONE run identity: a body authored for a different genesis refuses.
    let genesis_hash = genesis.frozen.run_id().0;
    if creds.genesis_hash != genesis_hash {
        return Err(format!(
            "JoinRun for run `{run_id}`: the credentials' genesis hash does not match the \
             resolved run (credentials authored for a different run)"
        ));
    }
    let binding = backend::role_binding(resolved, genesis, run_id, incarnation)?;
    let journal = journal_sink(run_id, &binding.run.identity)?;
    let keystore = daemon_vhc_session::keystore::VhcKeystore::from_env()
        .map_err(|e| format!("identity store: {e}"))?;
    let announcement =
        daemon_vhc_session::distribution::DistributionRecord::Cert(binding.own_cert.clone())
            .to_bytes()
            .map_err(|e| format!("certificate announcement: {e}"))?;
    let providers = build_role_providers(LiveAttachInputs {
        credentials: creds,
        coordinator,
        run_label: run_id,
        own_cert_announcement: announcement,
        keystore: &keystore,
    })
    .await?;
    // Bootstrap trust: the node-authored peer certificates (e.g. the verified seat holder's)
    // plus our own; later arrivals ride the control plane as §12.3 distribution records.
    let mut peer_certs = creds.peer_certs.clone();
    if !peer_certs.contains(&binding.own_cert) {
        peer_certs.push(binding.own_cert.clone());
    }
    // Late-join restore (§9/§10.2): fetch the node-resolved checkpoint document by content
    // address, hash-verify it (the ContentStore verifies; belt-and-suspenders here), decode the
    // snapshot, and hand it to the session as a migration input — the fresh instance migrates
    // from it before running. A restore that cannot be resolved is a typed refusal (a run asked
    // to restore must not silently start fresh).
    let restore = match &creds.restore {
        None => None,
        Some(r) => {
            let hash = daemon_vhc_proto::Hash(r.hash);
            let bytes = providers.payloads.get_content(&hash).await.map_err(|e| {
                format!(
                    "fetch checkpoint {} (round {}): {e}",
                    hash.to_hex(),
                    r.round
                )
            })?;
            let capture = daemon_vhc_session::role_session::decode_snapshot_doc(&bytes)?;
            Some(daemon_vhc_host::run::MigrationInput {
                capture,
                restore: true,
                migrate_fuel: None,
            })
        }
    };
    Ok(RoleSessionSpec {
        module: resolved.module.clone(),
        // The measured backend selection materialized (no fallback: an unavailable admitted
        // backend refuses the join typed here).
        engine: backend::engine_for_join(resolved.device_min.as_ref())?,
        run: binding.run,
        own_cert: binding.own_cert,
        trusted_bases: binding.trusted_bases,
        peer_certs,
        providers,
        journal,
        drain_deadline: DRAIN_DEADLINE,
        restore,
    })
}

/// The role session's journal sink: the node-delivered DURABLE per-incarnation home
/// (`DAEMON_VHC_RUN_DIR`, ABI §8) when the run-state root reference is present — a journal that
/// cannot open refuses the join typed (a run that cannot journal must not run, §8.4) — else the
/// in-memory sink (the referenceless in-process smoke seat / unit tests).
fn journal_sink(
    run_label: &str,
    identity: &daemon_vhc_host::run::RunIdentity,
) -> Result<Box<dyn daemon_vhc_host::run::JournalSink>, String> {
    use daemon_vhc_session::journal_home::{self, DurableSink};
    use daemon_vhc_session::keystore::VhcKeystore;

    let Some(root) = journal_home::run_dir_from_env() else {
        return Ok(Box::new(daemon_vhc_host::run::MemorySink::new()));
    };
    // The sidecar encryption key comes from the node-provided identity store (a path reference —
    // key material never rides the command wire).
    let keystore = VhcKeystore::from_env()
        .map_err(|e| format!("journal home: identity store for the sidecar key: {e}"))?;
    let key = keystore
        .journal_sidecar_key()
        .map_err(|e| format!("journal home: sidecar key: {e}"))?;
    let dir = journal_home::journal_dir(&root, run_label, &identity.role, identity.instance);
    let sink = DurableSink::open(&dir, identity, *key.bytes())
        .map_err(|e| format!("journal home: open {}: {e}", dir.display()))?;
    Ok(Box::new(sink))
}

#[tokio::main]
async fn main() {
    // Consent-gated crash reporting (component = train-worker). Armed as the first action: the
    // minidump monitor re-exec's this binary with a `--crash-reporter-server` arg, and this init
    // runs the monitor server (then exits) in that copy before it touches the stdio cut. A no-op
    // unless the spawning node injected a DSN + `DAEMON_CRASH_CONSENT=1`.
    let _crash = daemon_telemetry::init_crash_reporting("train-worker");

    // Fleet-validation readout (C2): print the same `hardware()` + `device_limits()` the live
    // `Probe`/assess path computes, then exit — so a cross-built worker on a bare fleet box (Windows
    // cmd.exe, macOS, RunPod) can report its DeviceLimits without hand-framing a CBOR `Probe`.
    if std::env::var_os("DAEMON_TRAIN_PROBE").is_some() {
        println!("hardware = {:#?}", backend::hardware());
        println!("device_limits = {:#?}", backend::device_limits());
        return;
    }

    // Fleet cache-warming mode (P3 lane S): fetch the run's module/corpus by content hash from the
    // payload store into the on-disk content cache, print per-object evidence, then exit — the
    // fleet-staging entry point (replaces P2's scp pre-staging). Like DAEMON_TRAIN_PROBE, it runs on
    // a bare box with no CBOR framing.
    if std::env::var_os("DAEMON_TRAIN_PREFETCH").is_some() {
        #[cfg(feature = "vhc-net")]
        {
            if let Err(e) = backend::prefetch_main().await {
                eprintln!("daemon-vhc-worker: prefetch FAILED: {e}");
                std::process::exit(1);
            }
            return;
        }
        #[cfg(not(feature = "vhc-net"))]
        {
            eprintln!(
                "daemon-vhc-worker: DAEMON_TRAIN_PREFETCH needs a worker built with \
                 `--features vhc-net` (the store fetch path)"
            );
            std::process::exit(1);
        }
    }

    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    send(
        &writer,
        &Event::Ready {
            capabilities: backend::host_capabilities(),
        },
    )
    .await;

    // Cached across commands: the assessed run (config + module bytes). Post-sunset every
    // eligible assessment selected the major-2 driver; `run_is_v2` records it so a JoinRun after
    // an ineligible/refused assess fails loud instead of guessing.
    let mut run: Option<backend::ResolvedRun> = None;
    let mut run_is_v2 = false;
    // The role-instance task map — the ONLY per-run state this binary owns (the session library
    // owns everything else). Keyed by the wire's run label; each handle carries the generation
    // that disambiguates incarnations.
    let mut roles: HashMap<String, RoleHandle> = HashMap::new();
    // One forwarder moves every session event (phases, metrics, warnings, the terminal
    // RunTerminated) onto the stdio cut, concurrently with the command loop. Shutdown drops the
    // sender and awaits the forwarder so no terminal event is lost to process exit.
    let (role_events, mut role_events_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let forwarder = {
        let writer = writer.clone();
        tokio::spawn(async move {
            while let Some(ev) = role_events_rx.recv().await {
                send(&writer, &ev).await;
            }
        })
    };
    // The immutable admitted tuple the last assessment produced (architecture §6.3). Join
    // rederives from the artifacts it is about to run and compares against this; a mismatch
    // aborts with a typed event so the node reassesses.
    let mut assessed_tuple: Option<daemon_vhc_session::protocol::AdmittedTuple> = None;

    while let Some(bytes) = reader.recv().await {
        let cmd: Command = match protocol::decode(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("daemon-vhc-worker: undecodable command: {e}");
                continue;
            }
        };
        match cmd {
            Command::Probe => send(&writer, &Event::Probed(backend::hardware())).await,
            Command::AssessRun { envelope, role } => {
                match backend::resolve_run(&envelope, role.as_deref()).await {
                    Ok(resolved) => {
                        // The admitted tuple's non-artifact identity: the run's genesis hash + the
                        // resolved (possibly node-directed) role. The incarnation is stamped 0 =
                        // UNASSIGNED at assess: the node mints the durable incarnation and stamps
                        // it into the tuple it delivers back with JoinRun — a join whose tuple
                        // still carries 0 refuses typed (no incarnation was ever assigned).
                        let tuple_identity =
                            resolved.genesis.as_ref().map(|g| backend::TupleIdentity {
                                genesis_hash: g.frozen.run_id().0,
                                role: &g.worker_role,
                                incarnation: 0,
                            });
                        match backend::assess(
                            &resolved.module,
                            &resolved.config,
                            resolved.module_blake3.as_ref(),
                            resolved.device_min.as_ref(),
                            resolved.envelope_grants().as_ref(),
                            tuple_identity,
                        ) {
                            Ok((elig, is_v2)) => {
                                run_is_v2 = is_v2;
                                assessed_tuple = elig.admitted_tuple.clone();
                                run = Some(resolved);
                                send(&writer, &Event::Assessed(elig)).await;
                            }
                            Err(detail) => send(&writer, &worker_error(&detail)).await,
                        }
                    }
                    Err(detail) => send(&writer, &worker_error(&detail)).await,
                }
            }
            Command::JoinRun {
                run_id,
                coordinator,
                credentials,
                admitted_tuple,
                ..
            } => {
                let Some(resolved) = run.as_ref() else {
                    send(
                        &writer,
                        &worker_error("JoinRun before AssessRun: no resolved run"),
                    )
                    .await;
                    continue;
                };
                if !run_is_v2 {
                    // Unreachable for an eligible assess post-sunset (every admitted module is
                    // major 2); a caller that joins past a refused assess fails loud + typed.
                    send(
                        &writer,
                        &worker_error(
                            "JoinRun for a run whose assess did not select the major-2 driver: \
                             the v1 five-phase driver retired at the Phase-E sunset \
                             (AbiUnsupportedMajor; decisions D5) — author a major-2 module",
                        ),
                    )
                    .await;
                    continue;
                }
                // The v2 session run: a genesis (envelope-v2) run coordinated by its REAL wasm
                // coordinator module, run in-process. The v1-envelope (device-min admission pre-screen) form refused
                // typed at assess (`EnvelopeSchemaRetired`), so an assessed run always carries
                // a genesis — the guard fails loud if a caller somehow joins past a refusal.
                let Some(genesis) = resolved.genesis.as_ref() else {
                    send(
                        &writer,
                        &worker_error(
                            "JoinRun without a resolved genesis run: the in-process \
                             self-driven join serves the envelope-v2 (genesis) form only \
                             (a schema-1 envelope refuses typed at assess — \
                             EnvelopeSchemaRetired); author a genesis envelope v2",
                        ),
                    )
                    .await;
                    continue;
                };
                // Admitted-tuple integrity (architecture §6.3): the node-delivered tuple is
                // MANDATORY on the join path — it carries the node-minted, never-reused
                // incarnation this instance runs as (a tuple still stamped 0 means no
                // incarnation was ever assigned: typed refusal, never a guessed identity).
                // Rederive from the artifacts this join is about to run and compare
                // field-by-field; any artifact mismatch aborts the join with a typed event —
                // the node reassesses; a stale/swapped artifact is never run.
                let Some(expected) = admitted_tuple.or_else(|| assessed_tuple.clone()) else {
                    send(
                        &writer,
                        &worker_error(&format!(
                            "JoinRun for run `{run_id}`: no admitted tuple (assess first; the \
                             node delivers the assessed tuple with the join)"
                        )),
                    )
                    .await;
                    continue;
                };
                if expected.role != genesis.worker_role {
                    send(
                        &writer,
                        &worker_error(&format!(
                            "JoinRun for run `{run_id}`: the admitted tuple's role \
                             `{}` is not the resolved role `{}` (re-assess with the intended \
                             role directive)",
                            expected.role, genesis.worker_role
                        )),
                    )
                    .await;
                    continue;
                }
                {
                    let tuple_identity = backend::TupleIdentity {
                        genesis_hash: genesis.frozen.run_id().0,
                        role: &expected.role,
                        incarnation: expected.incarnation,
                    };
                    match backend::assess(
                        &resolved.module,
                        &resolved.config,
                        resolved.module_blake3.as_ref(),
                        resolved.device_min.as_ref(),
                        resolved.envelope_grants().as_ref(),
                        Some(tuple_identity),
                    ) {
                        Ok((rederived_elig, _)) => {
                            if let Some(rederived) = &rederived_elig.admitted_tuple {
                                if let Some(field) = expected.first_artifact_mismatch(rederived) {
                                    send(
                                        &writer,
                                        &Event::AdmittedTupleMismatch {
                                            run_id: run_id.clone(),
                                            field: field.to_string(),
                                            generation: expected.incarnation,
                                        },
                                    )
                                    .await;
                                    continue;
                                }
                            }
                        }
                        Err(detail) => {
                            send(&writer, &worker_error(&detail)).await;
                            continue;
                        }
                    }
                }
                // The production join: spawn a role session as a background task and return to
                // the loop at once. A finished predecessor for the same run is reaped; a LIVE
                // one refuses typed (the node leaves first — a silent replacement could strand
                // a running instance).
                if let Some(existing) = roles.get(&run_id) {
                    if existing.is_finished() {
                        roles.remove(&run_id);
                    } else {
                        send(
                            &writer,
                            &worker_error(&format!(
                                "JoinRun for run `{run_id}`: a role instance is already live \
                                 for this run (generation {}); leave it before re-joining",
                                existing.generation()
                            )),
                        )
                        .await;
                        continue;
                    }
                }
                // The LIVE transport attach: node-authored plane-selection credentials
                // (`SessionCredentials`) select the production planes — WS control (optionally
                // dual-plane with iroh), presigned-R2 or filesystem content stores. Any
                // construction failure refuses the join typed (fail closed, never a silent
                // local run). Bytes that are not a `SessionCredentials` mean "no live attach"
                // and fall through to the in-process seat / typed refusal below.
                #[cfg(feature = "vhc-net")]
                if let Ok(creds) =
                    daemon_vhc_session::protocol::SessionCredentials::from_bytes(&credentials)
                {
                    match join_live(
                        resolved,
                        genesis,
                        &run_id,
                        &coordinator,
                        &creds,
                        expected.incarnation,
                    )
                    .await
                    {
                        Ok(spec) => {
                            let handle = spawn_role(run_id.clone(), spec, role_events.clone());
                            roles.insert(run_id, handle);
                        }
                        Err(detail) => send(&writer, &worker_error(&detail)).await,
                    }
                    continue;
                }
                #[cfg(not(feature = "vhc-net"))]
                let _ = (&coordinator, &credentials);
                if in_process_plane_selected() {
                    match backend::role_binding(resolved, genesis, &run_id, expected.incarnation) {
                        Ok(binding) => {
                            let journal = match journal_sink(&run_id, &binding.run.identity) {
                                Ok(sink) => sink,
                                Err(detail) => {
                                    send(&writer, &worker_error(&detail)).await;
                                    continue;
                                }
                            };
                            // The measured backend selection materialized (no fallback: an
                            // unavailable admitted backend refuses the join typed).
                            let engine =
                                match backend::engine_for_join(resolved.device_min.as_ref()) {
                                    Ok(engine) => engine,
                                    Err(detail) => {
                                        send(&writer, &worker_error(&detail)).await;
                                        continue;
                                    }
                                };
                            let spec = RoleSessionSpec {
                                module: resolved.module.clone(),
                                engine,
                                run: binding.run,
                                own_cert: binding.own_cert.clone(),
                                trusted_bases: binding.trusted_bases,
                                peer_certs: vec![binding.own_cert],
                                providers: in_process_providers(),
                                journal,
                                drain_deadline: DRAIN_DEADLINE,
                                restore: None,
                            };
                            let handle = spawn_role(run_id.clone(), spec, role_events.clone());
                            roles.insert(run_id, handle);
                        }
                        Err(detail) => send(&writer, &worker_error(&detail)).await,
                    }
                    continue;
                }
                #[cfg(feature = "harness")]
                {
                    match session::join_and_run(
                        &resolved.module,
                        &resolved.config,
                        genesis,
                        &run_id,
                        &writer,
                    )
                    .await
                    {
                        Ok(()) => {}
                        Err(detail) => send(&writer, &worker_error(&detail)).await,
                    }
                }
                #[cfg(not(feature = "harness"))]
                {
                    // No transport is bound: refuse typed (fail closed, never a silent local
                    // run). The live attach requires node-authored `SessionCredentials` (and a
                    // networked worker build); the in-process plane above serves single-host
                    // smoke.
                    send(
                        &writer,
                        &worker_error(&format!(
                            "JoinRun for run `{run_id}`: no control-plane binding for a role \
                             session (no live plane-selection credentials were delivered, and \
                             the in-process smoke seat is not selected)"
                        )),
                    )
                    .await;
                }
            }
            Command::Throttle {
                vram_cap_mb,
                duty_cycle_pct,
                paused,
            } => {
                // The owner's governor lever, forwarded into every live session: `paused` is a
                // HARD pump-level gate (a paused worker actually stops); the duty percentage
                // and VRAM cap ride the cooperative budget advisory.
                let level = ThrottleLevel {
                    paused,
                    duty_cycle_pct,
                    vram_cap_mb,
                };
                for handle in roles.values() {
                    handle.throttle(level);
                }
            }
            Command::Leave { run_id, mode } => {
                // Real leave semantics: graceful = quiesce + drain-snapshot checkpoint;
                // immediate = stop now. The session emits the terminal RunTerminated either
                // way; the handle leaves the map at once (a re-join mints a new generation).
                if let Some(handle) = roles.remove(&run_id) {
                    handle.leave(mode);
                }
            }
            Command::SwitchModule { run_id, .. } => {
                // The LOCAL upgrade transaction (ABI §10.3) runs against a LIVE, held role-instance
                // — quiesce → snapshot → owner-law re-admission → migrate → validate → activate,
                // via `daemon_vhc_session::upgrade::{run_local_upgrade, LiveUpgradeSteps}`. The
                // in-process self-driven t2 join above owns its sandbox to completion (it holds no
                // instance across commands), so — exactly like `Throttle`/`Leave` — there is no
                // live instance to switch here. A running instance is driven through the node
                // command surface (`WorkerControl::switch_module`), where the transaction has the
                // live pump/admission/migrate path in hand; the long-lived in-process join that
                // would hold an instance across `SwitchModule` arrives with the coordinator
                // re-seat of the worker session.
                send(
                    &writer,
                    &worker_error(&format!(
                        "SwitchModule for run `{run_id}`: no live role-instance is held in this \
                         worker (the self-driven join runs to completion); the live upgrade \
                         transaction is driven at the node command surface \
                         (WorkerControl::switch_module) over the held instance"
                    )),
                )
                .await;
            }
            Command::Ping => send(&writer, &Event::Pong).await,
            Command::Shutdown => {
                // Cancel every live session under a bounded drain, then exit. Each session
                // still emits its terminal RunTerminated through the forwarder; dropping the
                // sender and awaiting the forwarder flushes them before the process ends.
                for (_, handle) in roles.drain() {
                    handle.leave(LeaveMode::Immediate);
                    let _ = tokio::time::timeout(DRAIN_DEADLINE, handle.join()).await;
                }
                drop(role_events);
                let _ = forwarder.await;
                break;
            }
        }
    }
}

/// A module trap / lifecycle failure surfaces as the `Module` error class (worker unharmed, §13).
pub(crate) fn worker_error(detail: &str) -> Event {
    Event::Error {
        class: ErrorClass::Module,
        detail: detail.to_string(),
    }
}

/// Encode and send an [`Event`] over the stdio cut (shared by `main` and `session`).
pub(crate) async fn send(writer: &CutWriter, event: &Event) {
    match protocol::encode(event) {
        Ok(bytes) => {
            let _ = writer.send(&bytes).await;
        }
        Err(e) => eprintln!("daemon-vhc-worker: encode event: {e}"),
    }
}
