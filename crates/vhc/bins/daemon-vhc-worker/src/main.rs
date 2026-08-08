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
// The fit-probe mode (`[RC-15]`): the device answers "does it fit?" and the answer is a
// content-addressed FitVerdict. Lives in this binary because the verdict names this binary's
// sealed revision identity; consumes only pre-authored opaque inputs (no round vocabulary).
mod fit_probe;
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
        plane_stats: None,
        archive_heads: None,
    }
}

/// A live-attach refusal: the operator-facing detail plus the class the worker reports it under.
///
/// Classification is load-bearing: a plane that could not be dialed/bound (or that blew its
/// bring-up deadline) is [`ErrorClass::Transient`] — a recoverable environment fault the node
/// re-converges under its retry budget — while everything else keeps the module class. Without it
/// every attach failure looked the same to the node.
#[cfg(feature = "vhc-net")]
#[derive(Debug)]
struct AttachRefusal {
    class: ErrorClass,
    detail: String,
}

#[cfg(feature = "vhc-net")]
impl From<String> for AttachRefusal {
    fn from(detail: String) -> Self {
        Self {
            class: ErrorClass::Module,
            detail,
        }
    }
}

#[cfg(feature = "vhc-net")]
impl From<daemon_vhc_session::providers::PlaneError> for AttachRefusal {
    fn from(e: daemon_vhc_session::providers::PlaneError) -> Self {
        Self {
            class: e.class(),
            detail: e.to_string(),
        }
    }
}

/// Author the live role-session spec for a join whose credentials selected the production
/// planes: re-run the admission binding, open the durable journal home, and construct the
/// transport providers from the plane selection (WS connect fails FAST on an unreachable
/// endpoint — one dial, typed error; every plane bring-up is bounded by the session crate's
/// bring-up deadline, so the command loop never hangs here).
#[cfg(feature = "vhc-net")]
async fn join_live(
    resolved: &backend::ResolvedRun,
    genesis: &backend::GenesisRun,
    run_id: &str,
    coordinator: &str,
    creds: &daemon_vhc_session::protocol::SessionCredentials,
    incarnation: u64,
) -> Result<RoleSessionSpec, AttachRefusal> {
    use daemon_vhc_session::providers::{build_role_providers, LiveAttachInputs};

    // Credentials bind to ONE run identity: a body authored for a different genesis refuses.
    let genesis_hash = genesis.frozen.run_id().0;
    if creds.genesis_hash != genesis_hash {
        return Err(format!(
            "JoinRun for run `{run_id}`: the credentials' genesis hash does not match the \
             resolved run (credentials authored for a different run)"
        )
        .into());
    }
    let binding = backend::role_binding(resolved, genesis, run_id, incarnation)?;
    let (journal, archive) = journal_sink(run_id, &binding.run.identity, &binding.trusted_bases)?;
    let keystore = daemon_vhc_session::keystore::VhcKeystore::from_env()
        .map_err(|e| format!("identity store: {e}"))?;
    let announcement =
        daemon_vhc_session::distribution::DistributionRecord::Cert(binding.own_cert.clone())
            .to_bytes()
            .map_err(|e| format!("certificate announcement: {e}"))?;
    // The node-verified seat grant rides the same anti-entropy surface as the certificate
    // ([SEAT-1] v2 grant distribution): registered as a resubscribe frame here, published on
    // attach by the session (which re-verifies before its own floor advances).
    let seat_grant_announcement = creds
        .seat_grant
        .as_ref()
        .map(|grant| {
            daemon_vhc_session::distribution::DistributionRecord::SeatGrant(Box::new(grant.clone()))
                .to_bytes()
                .map_err(|e| format!("seat grant announcement: {e}"))
        })
        .transpose()?;
    let mut providers = build_role_providers(LiveAttachInputs {
        credentials: creds,
        coordinator,
        run_label: run_id,
        own_cert_announcement: announcement,
        seat_grant_announcement,
        keystore: &keystore,
    })
    .await?;
    // The module-driven `data.fetch` seat resolves the run's GENESIS-PINNED artifacts at the urls
    // the envelope commits (the corpus manifest, tokenizer and chunk-addressed shards live at the
    // published `corpus/…` keys, not at the committed-payload plane's `payload/<hex>`); everything
    // else keeps falling through to the payload plane this returned.
    providers.artifacts = backend::pinned_artifact_plane(genesis, run_id, providers.artifacts)?;
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
                let detail = format!(
                    "fetch checkpoint {} (round {}): {e}",
                    hash.to_hex(),
                    r.round
                );
                // A transient transport fault reaching the content plane is the budget-free
                // deferral lane (Gate C) — never a semantic refusal of the restore itself.
                if e.is_transient_transport() {
                    AttachRefusal {
                        class: ErrorClass::Transient,
                        detail,
                    }
                } else {
                    AttachRefusal::from(detail)
                }
            })?;
            let capture = daemon_vhc_session::role_session::decode_snapshot_doc(&bytes)?;
            verify_restore_manifest_schema(&capture.manifest, &hash, r.round)?;
            Some(daemon_vhc_host::run::MigrationInput {
                capture,
                restore: true,
                migrate_fuel: None,
                // A content-plane late-join restore (checkpoint document from the payload plane).
                carried_state: Vec::new(),
                // The restore founds a fresh journal chain — journal its anchoring tag-10.
                anchor: true,
            })
        }
    };
    // Coordinator crash reconstruction (§8.8): a recovery directive means the seat has PUBLISHED
    // journal history — rebuild its consensus state through the sandbox BEFORE the session spawns
    // (RoleReady is reported only after the migrate step accepts the reconstructed capture). The
    // worker re-verifies the carried heads against the genesis trust itself (carriage, not
    // trust), and a reconstruction that fails refuses the join typed — a seat with a durable
    // record that silently started fresh would fork the run behind its own history.
    let restore = match &creds.reconstruct {
        None => restore,
        Some(recovery) => {
            let capture = daemon_vhc_session::reconstruct::reconstruct_coordinator(
                daemon_vhc_session::reconstruct::ReconstructSpec {
                    heads: recovery.heads.clone(),
                    run_id: daemon_vhc_proto::Hash(genesis_hash),
                    trusted: binding.trusted_bases.clone(),
                    role: binding.run.identity.role.clone(),
                    run_label: run_id.to_string(),
                    journal_root: daemon_vhc_session::journal_home::run_dir_from_env(),
                    module: resolved.module.clone(),
                    config: binding.run.config.clone(),
                    grants: binding.run.grants.clone(),
                    incarnation,
                    restore: restore.map(|m| m.capture),
                    // The node-durable journal key (§8.5): lets a same-box reconstruction
                    // decrypt the sidecar-referenced restore read-backs its own crashed
                    // incarnations recorded. Best-effort — a cold standby has no keystore
                    // and resolves those values content-addressed instead.
                    sidecar_key: daemon_vhc_session::keystore::VhcKeystore::from_env()
                        .ok()
                        .and_then(|ks| ks.journal_sidecar_key().ok())
                        .map(|k| *k.bytes()),
                    // The export's quiesce ceiling: a reconstruction drains a full replay's
                    // state, so it gets a generous fixed budget (independent of the session's
                    // leave-drain deadline).
                    deadline_ms: 60_000,
                },
                providers.payloads.clone(),
            )
            .await
            .map_err(|e| match e {
                // The typed transient lane (Gate C, defect 10): the content plane was
                // momentarily unreachable — the node defers budget-free and retries paced,
                // instead of burning the semantic retry budget on a network outage.
                daemon_vhc_session::reconstruct::ReconstructError::Transport { .. } => {
                    AttachRefusal {
                        class: ErrorClass::Transient,
                        detail: format!("coordinator reconstruction: {e}"),
                    }
                }
                other => AttachRefusal::from(format!("coordinator reconstruction: {other}")),
            })?;
            Some(daemon_vhc_host::run::MigrationInput {
                capture,
                restore: true,
                migrate_fuel: None,
                carried_state: Vec::new(),
                // The reconstruction founds a fresh journal chain — journal its anchoring tag-10.
                anchor: true,
            })
        }
    };
    // Gate B' trainer archive catch-up: the node judged the restore fence reachable only through
    // the ARCHIVED record stream. Extract the seat's historical committed records here, before
    // the session spawns — re-verifying the carried heads against genesis trust (carriage, not
    // trust) and re-hashing every segment against its attested head. The staged fold happens
    // inside the session, before live attachment. A genuine archive gap refuses the join typed;
    // a transient content-plane fault defers budget-free (Gate C).
    let catch_up = match &creds.catch_up {
        None => Vec::new(),
        Some(cu) => daemon_vhc_session::reconstruct::extract_catch_up_frames(
            &daemon_vhc_session::reconstruct::CatchUpSpec {
                heads: cu.heads.clone(),
                run_id: daemon_vhc_proto::Hash(genesis_hash),
                trusted: binding.trusted_bases.clone(),
                run_label: run_id.to_string(),
                journal_root: daemon_vhc_session::journal_home::run_dir_from_env(),
                after_round: cu.from_round,
            },
            providers.payloads.as_ref(),
        )
        .await
        .map_err(|e| match e {
            daemon_vhc_session::reconstruct::ReconstructError::Transport { .. } => AttachRefusal {
                class: ErrorClass::Transient,
                detail: format!("trainer archive catch-up: {e}"),
            },
            other => AttachRefusal::from(format!("trainer archive catch-up: {other}")),
        })?,
    };
    Ok(RoleSessionSpec {
        module: resolved.module.clone(),
        // The measured backend selection materialized (no fallback: an unavailable admitted
        // backend refuses the join typed here).
        engine: backend::engine_for_join(
            resolved.device_min.as_ref(),
            binding.admitted_host_bytes,
        )?,
        run: binding.run,
        own_cert: binding.own_cert,
        trusted_bases: binding.trusted_bases,
        peer_certs,
        seat_grant: creds.seat_grant.clone(),
        providers,
        journal,
        drain_deadline: DRAIN_DEADLINE,
        restore,
        admitted_quotas: binding.quotas,
        archive,
        catch_up,
    })
}

/// Fail closed at restore (Gate D', ABI §10.2): a checkpoint doc whose state-manifest this
/// build cannot read — or that declares a schema major with no defined restore semantics here
/// ([`daemon_vhc_proto::det_state::STATE_MANIFEST_SCHEMA_MAJOR`]) — is refused typed BEFORE
/// `da_migrate` sees it. Value-level header decode only; the manifest bytes stay verbatim for
/// the guest. Module-hash binding via the epoch transition chain is deferred (post-C2; see the
/// constant's docs — tiny-llama's manifest module commitment is currently zeroed and needs the
/// admitted hash in guest init config first).
#[cfg(feature = "vhc-net")]
fn verify_restore_manifest_schema(
    manifest: &[u8],
    doc_hash: &daemon_vhc_proto::Hash,
    round: u64,
) -> Result<(), AttachRefusal> {
    let (schema, _module) =
        daemon_vhc_proto::det_state::decode_manifest_header(manifest).map_err(|e| {
            AttachRefusal::from(format!(
                "restore checkpoint {} (round {round}): state-manifest: {e}",
                doc_hash.to_hex()
            ))
        })?;
    if schema != daemon_vhc_proto::det_state::STATE_MANIFEST_SCHEMA_MAJOR {
        return Err(AttachRefusal::from(format!(
            "restore checkpoint {} (round {round}): state-manifest schema {schema} has no \
             defined restore semantics in this build (understood: {})",
            doc_hash.to_hex(),
            daemon_vhc_proto::det_state::STATE_MANIFEST_SCHEMA_MAJOR
        )));
    }
    Ok(())
}

/// The role session's journal sink: the node-delivered DURABLE per-incarnation home
/// (`DAEMON_VHC_RUN_DIR`, ABI §8) when the run-state root reference is present — a journal that
/// cannot open refuses the join typed (a run that cannot journal must not run, §8.4) — else the
/// in-memory sink (the referenceless in-process smoke seat / unit tests).
///
/// A durable home also arms the incremental archive-publication seam (architecture §4.4) and
/// returns its [`ArchiveSpec`] half — the seal-hook stream plus the chain coordinates the
/// session's archive publisher binds. The in-memory sink has no durable chain: no spec.
fn journal_sink(
    run_label: &str,
    identity: &daemon_vhc_host::run::RunIdentity,
    trusted_bases: &[daemon_vhc_proto::PeerId],
) -> Result<
    (
        Box<dyn daemon_vhc_host::run::JournalSink>,
        Option<daemon_vhc_session::archive::ArchiveSpec>,
    ),
    String,
> {
    use daemon_vhc_session::journal_home::{self, DurableSink};
    use daemon_vhc_session::keystore::VhcKeystore;

    let Some(root) = journal_home::run_dir_from_env() else {
        return Ok((Box::new(daemon_vhc_host::run::MemorySink::new()), None));
    };
    // The sidecar encryption key comes from the node-provided identity store (a path reference —
    // key material never rides the command wire).
    let keystore = VhcKeystore::from_env()
        .map_err(|e| format!("journal home: identity store for the sidecar key: {e}"))?;
    let key = keystore
        .journal_sidecar_key()
        .map_err(|e| format!("journal home: sidecar key: {e}"))?;
    let dir = journal_home::journal_dir(&root, run_label, &identity.role, identity.instance);
    let mut sink = DurableSink::open(&dir, identity, *key.bytes())
        .map_err(|e| format!("journal home: open {}: {e}", dir.display()))?;
    let (seal_tx, seals) = tokio::sync::mpsc::unbounded_channel();
    sink.arm_seal_hook(seal_tx);
    let archive = daemon_vhc_session::archive::ArchiveSpec {
        seals,
        journal_dir: dir,
        chain_instance: sink.founding_instance(),
        round_claim: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        archived_round: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        // The genesis-trusted attestor set: the publisher's succession-link resolution may
        // link a predecessor chain published under a DIFFERENT trusted base (a seat that
        // moved boxes keeps ONE recovery lineage).
        trusted: trusted_bases.to_vec(),
    };
    Ok((Box::new(sink), Some(archive)))
}

#[tokio::main]
async fn main() {
    // Consent-gated crash reporting (component = train-worker). Armed as the first action: the
    // minidump monitor re-exec's this binary with a `--crash-reporter-server` arg, and this init
    // runs the monitor server (then exits) in that copy before it touches the stdio cut. A no-op
    // unless the spawning node injected a DSN + `DAEMON_CRASH_CONSENT=1`.
    let _crash = daemon_telemetry::init_crash_reporting("train-worker");
    // Diagnostics to stderr (stdin/stdout are the framed cut): a worker that emits nothing is
    // undebuggable in a live/multi-process run. Honors `RUST_LOG` (default off/warn).
    daemon_telemetry::init_subscriber();

    // Fleet-validation readout: print the same `hardware()` + `device_limits()` the live
    // `Probe`/assess path computes, then exit — so a cross-built worker on a bare fleet box (Windows
    // cmd.exe, macOS, RunPod) can report its DeviceLimits without hand-framing a CBOR `Probe`.
    if std::env::var_os("DAEMON_TRAIN_PROBE").is_some() {
        println!("hardware = {:#?}", backend::hardware());
        println!("device_limits = {:#?}", backend::device_limits());
        // Why there is no device, typed — instead of leaving a reader to infer it from a zero.
        // A GPU-capable box whose graphics loader is absent from this process's environment reports
        // no adapter, and read as "no accelerator" it is silently reclassified as CPU-only, which a
        // run would then admit as a CPU participant. The environment was the fault; nothing said so.
        match daemon_vhc_host::probe::wgpu_unavailability() {
            None => println!("device_availability = available"),
            Some(reason) => println!("device_availability = {reason}"),
        }
        // One allocator reading with no run apparatus at all — the bring-up-boundary sample.
        //
        // The run-path readout needs a seeded run, published modules and a genesis envelope, so the
        // allocator terms a profile is calibrated against were unobtainable without that whole
        // apparatus. This is the same reading at the same boundary, on a path an operator can invoke
        // on a bare box.
        //
        // Absence is printed as absence: a backend that cannot report occupancy records nothing, and
        // a reader taking that for zero would calibrate a profile against a figure nobody measured.
        match backend::probe_allocator_sample() {
            Some(sample) => println!("allocator_sample[after-bring-up] = {sample:#?}"),
            None => println!(
                "allocator_sample[after-bring-up] = unavailable (this backend cannot report \
                 allocator occupancy — an ABSENCE, not a zero)"
            ),
        }
        // The device heap the backend presents, and the driver's live budget for it.
        //
        // Two numbers on purpose, and only the first is supply: the heap size is a property of the
        // device and driver, while the budget moves with whatever else on the box holds memory. The
        // live figure is a pressure reading for the governor, printed here because a calibration pass
        // wants to see the gap — never because a claim should be admitted against it.
        match daemon_vhc_host::probe::probe_vulkan_heap_budget() {
            Some(heap) => println!(
                "device_heap = {} bytes advertised; live budget {}",
                heap.heap_size_bytes,
                match heap.heap_budget_bytes {
                    Some(budget) => format!("{budget} bytes (volatile — not the supply figure)"),
                    None => "not reported by this driver".to_string(),
                }
            ),
            None => println!(
                "device_heap = unavailable (no Vulkan heap could be queried on this build or box)"
            ),
        }
        // The Windows analogue of the same split: the live DXGI local budget is a pressure reading, so
        // it prints here and the report states the static derivation instead. Absent off Windows.
        if let Some(budget) = daemon_vhc_host::probe::probe_windows_local_budget_bytes() {
            println!(
                "device_local_budget = {budget} bytes (volatile — a pressure reading for the \
                 governor, not the supply figure)"
            );
        }
        // The Device Capability Report this node states about its device: the supply figure admission
        // compares a composed claim against, with the derivation that produced it named.
        //
        // Printed with its digest and its validation verdict because both are what make it evidence
        // rather than a readout: the digest is what an admitted tuple and a composition record cite,
        // and a report that does not validate must never be quietly used by anything.
        match backend::device_capability_report() {
            Some(report) => {
                println!("device_capability_report = {report:#?}");
                match report.report_digest() {
                    Ok(digest) => {
                        println!("device_capability_report.digest = {}", digest.to_hex());
                    }
                    Err(e) => println!("device_capability_report.digest = unavailable ({e})"),
                }
                match report.validate() {
                    Ok(()) => println!("device_capability_report.validates = yes"),
                    Err(e) => println!("device_capability_report.validates = NO ({e})"),
                }
            }
            None => println!(
                "device_capability_report = none (this build found no device lane, so this node \
                 states no device supply — a CPU-only participant, not a defect)"
            ),
        }
        // The revision record each probed backend makes about itself — the structure that replaces
        // a Debug-printed adapter line, readable by whatever compares it to a profile's range.
        for capability in backend::backend_inventory() {
            println!(
                "revision_record[{}] = {:#?}",
                capability.class,
                backend::revision_record(&capability)
            );
        }
        return;
    }

    // Revision-record export mode (the provisioning seam, `[PC-12]`): write each advertised
    // backend class's `BackendImplementationRevision` — the record admission will compare a
    // profile's trust envelope against, sealed-binary identity included — as canonical CBOR into
    // the named directory, then exit. This exists because profile AUTHORING lives outside this
    // binary (dev tooling, a release provisioner) while the revision record can only truthfully
    // come from the binary that will run: an author that re-derived it independently would be
    // vouching for a record nobody will ever present. One file per class:
    // `revision-<class>.cbor`.
    if let Some(dir) = std::env::var_os("DAEMON_TRAIN_REVISION_OUT") {
        let dir = std::path::PathBuf::from(dir);
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!(
                "daemon-vhc-worker: revision export: create {}: {e}",
                dir.display()
            );
            std::process::exit(1);
        }
        for capability in backend::backend_inventory() {
            let record = backend::revision_record(&capability);
            let bytes = match daemon_vhc_proto::to_canonical_vec(&record) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!(
                        "daemon-vhc-worker: revision export: encode `{}`: {e}",
                        capability.class
                    );
                    std::process::exit(1);
                }
            };
            let path = dir.join(format!("revision-{}.cbor", capability.class));
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!(
                    "daemon-vhc-worker: revision export: write {}: {e}",
                    path.display()
                );
                std::process::exit(1);
            }
            println!(
                "revision-record[{}] -> {}",
                capability.class,
                path.display()
            );
        }
        return;
    }

    // Fit-probe mode (`[RC-15]`): run the actual module on this box's actual measured backend at
    // the actual granted geometry under the actual enforced budget, and record what the device
    // said as a content-addressed FitVerdict. The env names a probe directory pre-authored by the
    // orchestrator (`xtask vhc-fit-probe`); like the other operator modes it runs on a bare box
    // with no CBOR framing. A RED verdict exits 0 (evidence, not an outage); a probe that could
    // not produce a verdict exits 1.
    if let Some(dir) = std::env::var_os(fit_probe::FIT_PROBE_ENV) {
        let dir = std::path::PathBuf::from(dir);
        if let Err(e) = fit_probe::run_fit_probe(&dir) {
            eprintln!("daemon-vhc-worker: fit probe FAILED: {e}");
            std::process::exit(1);
        }
        return;
    }

    // Fleet cache-warming mode (the artifact-distribution path): fetch the run's module/corpus by
    // content hash from the payload store into the on-disk content cache, print per-object
    // evidence, then exit — the fleet-staging entry point (replaces the earlier scp pre-staging).
    // Like DAEMON_TRAIN_PROBE, it runs on a bare box with no CBOR framing.
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
            Command::AssessRun {
                envelope,
                role,
                switch_target: Some(target),
            } => {
                // The pre-switch assessment (ABI §10.3): resolve the run to its genesis, then
                // assess the committed TARGET — the worker (which alone touches module bytes)
                // computes the post-switch tuple's claim hash. Read-only: the cached run and
                // the assessed join tuple are deliberately NOT replaced (the running instance's
                // admission stands until the switch itself lands).
                match backend::resolve_run(&envelope, role.as_deref()).await {
                    Ok(resolved) => match resolved.genesis.as_ref() {
                        Some(genesis) => {
                            match backend::assess_switch(&resolved, genesis, &target).await {
                                Ok(elig) => send(&writer, &Event::Assessed(elig)).await,
                                Err(detail) => send(&writer, &worker_error(&detail)).await,
                            }
                        }
                        None => {
                            send(
                                &writer,
                                &worker_error("switch assessment: the run carries no genesis"),
                            )
                            .await
                        }
                    },
                    Err(detail) => send(&writer, &worker_error(&detail)).await,
                }
            }
            Command::AssessRun {
                envelope,
                role,
                switch_target: None,
            } => {
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
                        // The role's envelope context: the requirements the run signed for this
                        // role, from which a provisioned box composes its resource authority.
                        let role_context =
                            resolved.genesis.as_ref().map(|g| backend::RoleContext {
                                role: &g.worker_role,
                                execution: g
                                    .env
                                    .roles
                                    .get(&g.worker_role)
                                    .and_then(|entry| entry.execution.as_ref()),
                            });
                        match backend::assess(
                            &resolved.module,
                            &resolved.config,
                            resolved.module_blake3.as_ref(),
                            resolved.device_min.as_ref(),
                            resolved.envelope_grants().as_ref(),
                            role_context,
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
                let Some(expected) = admitted_tuple
                    .map(|boxed| *boxed)
                    .or_else(|| assessed_tuple.clone())
                else {
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
                    // The same role context assess composed with, so the rederivation holds the
                    // same authority and the composed members compare byte-for-byte.
                    let role_context = backend::RoleContext {
                        role: &genesis.worker_role,
                        execution: genesis
                            .env
                            .roles
                            .get(&genesis.worker_role)
                            .and_then(|entry| entry.execution.as_ref()),
                    };
                    match backend::assess(
                        &resolved.module,
                        &resolved.config,
                        resolved.module_blake3.as_ref(),
                        resolved.device_min.as_ref(),
                        resolved.envelope_grants().as_ref(),
                        Some(role_context),
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
                        Err(refusal) => {
                            // The classified refusal (a transport bring-up fault is retryable).
                            send(
                                &writer,
                                &Event::Error {
                                    class: refusal.class,
                                    detail: refusal.detail.clone(),
                                },
                            )
                            .await;
                            // ...and the OBSERVED TERMINAL for the generation the node admitted.
                            // No role session was spawned, so nothing else will ever report this
                            // instance's exit: without the terminal the node's admitted-instance
                            // record lives on holding its ledger reservation (the Windows
                            // fleet-smoke stale-duty finding — persisted intents rehydrated on
                            // boot, refused their attach, and kept 100% duty until the operator
                            // wiped run state) and no retry is ever scheduled. A TRANSIENT-class
                            // refusal (plane dial timeout, content plane unreachable during
                            // reconstruction/restore) is the budget-free transport lane (Gate C):
                            // the node defers paced without consuming the retry budget. Every
                            // other class is retryable under the node's bounded budget, which
                            // escalates to terminal on its own.
                            let outcome = if matches!(refusal.class, ErrorClass::Transient) {
                                protocol::TerminalOutcome::FailedTransport {
                                    reason: refusal.detail,
                                }
                            } else {
                                protocol::TerminalOutcome::FailedRetryable {
                                    reason: refusal.detail,
                                }
                            };
                            send(
                                &writer,
                                &Event::RunTerminated {
                                    run_id: run_id.clone(),
                                    generation: expected.incarnation,
                                    outcome,
                                },
                            )
                            .await;
                        }
                    }
                    continue;
                }
                #[cfg(not(feature = "vhc-net"))]
                let _ = (&coordinator, &credentials);
                if in_process_plane_selected() {
                    match backend::role_binding(resolved, genesis, &run_id, expected.incarnation) {
                        Ok(binding) => {
                            let (journal, archive) = match journal_sink(
                                &run_id,
                                &binding.run.identity,
                                &binding.trusted_bases,
                            ) {
                                Ok(parts) => parts,
                                Err(detail) => {
                                    send(&writer, &worker_error(&detail)).await;
                                    continue;
                                }
                            };
                            // The measured backend selection materialized (no fallback: an
                            // unavailable admitted backend refuses the join typed).
                            let engine = match backend::engine_for_join(
                                resolved.device_min.as_ref(),
                                binding.admitted_host_bytes,
                            ) {
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
                                // The in-process plane has no registry: no seat grant at join
                                // (the floor starts ungoverned; on-plane grants govern it).
                                seat_grant: None,
                                providers: in_process_providers(),
                                journal,
                                drain_deadline: DRAIN_DEADLINE,
                                restore: None,
                                admitted_quotas: binding.quotas,
                                // The in-process plane has no archive-head store; the seal
                                // stream (when durable) is dropped and the on-disk chain
                                // remains the local record.
                                archive,
                                // No registry ⇒ no node-resolved catch-up directive.
                                catch_up: Vec::new(),
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
            Command::SwitchModule {
                run_id,
                epoch,
                role,
                new_module,
                grants_hash,
                deadline_ms,
                admitted_tuple,
            } => {
                // The LOCAL upgrade transaction (ABI §10.3) against the held role-instance. The
                // worker's half is pre-flight only — resolve the node-provisioned post-switch
                // identity, the target bytes (where a worker-side source exists), and the
                // admission inputs — every refusal typed with the running instance untouched;
                // the SESSION runs the fence (quiesce → snapshot → migrate → validate →
                // activate) and answers ModuleSwitched / SwitchRefused / a terminal
                // RunTerminated on the event stream.
                let Some(handle) = roles.get(&run_id) else {
                    // §10.3: a worker without a long-lived instance answers typed
                    // command-unsupported and attempts no migration.
                    send(
                        &writer,
                        &Event::SwitchRefused {
                            run_id: run_id.clone(),
                            generation: 0,
                            reason: "no live role-instance is held for this run".into(),
                        },
                    )
                    .await;
                    continue;
                };
                if handle.is_finished() {
                    send(
                        &writer,
                        &Event::SwitchRefused {
                            run_id: run_id.clone(),
                            generation: handle.generation(),
                            reason: "the role instance already reached its terminal state".into(),
                        },
                    )
                    .await;
                    continue;
                }
                let refusal = |reason: String| Event::SwitchRefused {
                    run_id: run_id.clone(),
                    generation: handle.generation(),
                    reason,
                };
                let Some(resolved) = run.as_ref() else {
                    send(
                        &writer,
                        &refusal("no resolved run is cached (assess first)".into()),
                    )
                    .await;
                    continue;
                };
                let Some(genesis) = resolved.genesis.as_ref() else {
                    send(
                        &writer,
                        &refusal("the cached run carries no genesis".into()),
                    )
                    .await;
                    continue;
                };
                if role != genesis.worker_role {
                    send(
                        &writer,
                        &refusal(format!(
                            "the switch targets role `{role}` but this instance holds `{}`",
                            genesis.worker_role
                        )),
                    )
                    .await;
                    continue;
                }
                let Some(tuple) = admitted_tuple else {
                    send(
                        &writer,
                        &refusal(
                            "no post-switch admitted tuple (the node assesses the target and \
                             delivers the tuple with the switch)"
                                .into(),
                        ),
                    )
                    .await;
                    continue;
                };
                match backend::switch_binding(
                    genesis,
                    &run_id,
                    epoch,
                    new_module,
                    grants_hash,
                    deadline_ms,
                    *tuple,
                    handle.generation(),
                    // The admitted role config carries unchanged across the switch (upgrade
                    // records pin module + grants; config carriage arrives when they carry one).
                    resolved.config.clone(),
                ) {
                    Ok(binding) => handle.switch(binding),
                    Err(reason) => send(&writer, &refusal(reason)).await,
                }
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

#[cfg(all(test, feature = "vhc-net"))]
mod restore_gate_tests {
    use super::verify_restore_manifest_schema;

    /// A minimal §10.2 state-manifest at the given schema, built at the CBOR-value level (the
    /// worker links no SDK manifest type — the same wall the production decode respects).
    fn manifest_bytes(schema: u64) -> Vec<u8> {
        use ciborium::value::Value;
        let manifest = Value::Map(vec![
            (Value::Text("schema".into()), Value::Integer(schema.into())),
            (Value::Text("module".into()), Value::Bytes(vec![0u8; 32])),
            (Value::Text("sections".into()), Value::Array(Vec::new())),
        ]);
        let mut out = Vec::new();
        ciborium::into_writer(&manifest, &mut out).expect("manifest encodes");
        out
    }

    #[test]
    fn a_known_schema_manifest_passes_the_restore_gate() {
        let bytes = manifest_bytes(daemon_vhc_proto::det_state::STATE_MANIFEST_SCHEMA_MAJOR);
        verify_restore_manifest_schema(&bytes, &daemon_vhc_proto::Hash([7; 32]), 4)
            .expect("the understood schema major restores");
    }

    #[test]
    fn an_unknown_schema_major_is_refused_typed_before_da_migrate() {
        // Fail closed (Gate D'): a future-format doc meets a typed refusal naming both the
        // declared and the understood major — never an undefined in-guest restore.
        let bytes = manifest_bytes(2);
        let refusal = verify_restore_manifest_schema(&bytes, &daemon_vhc_proto::Hash([7; 32]), 4)
            .expect_err("schema 2 has no defined restore semantics in this build");
        assert!(
            refusal.detail.contains("schema 2") && refusal.detail.contains("understood: 1"),
            "the refusal names the majors: {}",
            refusal.detail
        );
    }

    #[test]
    fn an_unreadable_manifest_is_refused_typed() {
        let refusal =
            verify_restore_manifest_schema(b"not-cbor", &daemon_vhc_proto::Hash([7; 32]), 4)
                .expect_err("garbage never reaches da_migrate");
        assert!(
            refusal.detail.contains("state-manifest"),
            "the refusal is attributed to the manifest decode: {}",
            refusal.detail
        );
    }
}
