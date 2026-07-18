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
//! - `JoinRun` → the v2 session run (`session::join_and_run`): the event pump over the
//!   run's REAL coordinator, configured from the genesis and run in-process under the same
//!   major-2 driver (consensus never runs outside the sandboxed, content-addressed module).
//! - `Leave`/`Shutdown`/`Ping` → as the protocol requires.
//!
//! A trapping module surfaces as `Event::Error{class: Module, …}` — the worker is never harmed.

mod backend;
// The in-process self-driven join: it decodes the SDK round schemas and drives the run's
// coordinator module in-process, so it is HARNESS machinery — the production worker routes
// opaque signed frames only and carries no round vocabulary (dep-check-enforced; the
// decentralized role-session join lands at the node command surface).
#[cfg(feature = "harness")]
mod session;

use daemon_provision::{CutChannel, CutWriter};
use daemon_vhc_session::protocol::{self, Command, ErrorClass, Event};

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
            Command::AssessRun { envelope } => match backend::resolve_run(&envelope).await {
                Ok(resolved) => {
                    // The admitted tuple's non-artifact identity: the run's genesis hash + the
                    // worker role this node joins as. The self-driven seat runs one incarnation
                    // per join (the node-durable incarnation counter takes over with the
                    // node-side identity delivery).
                    let tuple_identity =
                        resolved.genesis.as_ref().map(|g| backend::TupleIdentity {
                            genesis_hash: g.frozen.run_id().0,
                            role: &g.worker_role,
                            incarnation: backend::RUN_INSTANCE,
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
            },
            Command::JoinRun {
                run_id,
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
                // Admitted-tuple integrity (architecture §6.3): rederive the tuple from the
                // artifacts this join is about to run and compare field-by-field against the
                // tuple assessment produced (the node-delivered one when present, else the
                // worker's own cached assessment). Any artifact mismatch aborts the join with a
                // typed event — the node reassesses; a stale/swapped artifact is never run.
                let expected = admitted_tuple.or_else(|| assessed_tuple.clone());
                if let Some(expected) = &expected {
                    let tuple_identity = backend::TupleIdentity {
                        genesis_hash: genesis.frozen.run_id().0,
                        role: &genesis.worker_role,
                        incarnation: backend::RUN_INSTANCE,
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
                    // The production worker carries no in-process join: the self-driven session
                    // decodes SDK round schemas and hosts the run's coordinator in-process, which
                    // the opaque-host boundary forbids (the host routes signed opaque frames and
                    // services capabilities; it never understands rounds). The decentralized
                    // role-session join over real transports is the successor; until it lands,
                    // an unadorned worker refuses typed instead of running harness machinery.
                    send(
                        &writer,
                        &worker_error(&format!(
                            "JoinRun for run `{run_id}`: this worker build carries no \
                             in-process self-driven join (harness builds only); the \
                             decentralized role-session join supersedes it"
                        )),
                    )
                    .await;
                }
            }
            Command::Throttle { .. } => {
                // The v2 session owns its sandbox exclusively; a live pause is
                // preemption-as-churn at the node (stop + re-issue JoinRun on the durable
                // intent, §10.3/§10.5). Nothing to do in-process here.
            }
            Command::Leave { .. } => {
                // The v2 session run drives itself to completion within JoinRun; the durable
                // intent lives at the node. Nothing held here to tear down.
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
