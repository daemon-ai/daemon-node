// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Sandboxed coordinator reconstruction (§8.8 crash recovery) — the PRODUCT half of the D2
//! failover drill (`daemon-vhc-testkit/tests/failover.rs`), schema-free by construction.
//!
//! When a seat-role join carries a [`crate::protocol::CoordinatorRecovery`] directive, the seat
//! has published journal history: the crashed incarnation's consensus state (retained record
//! ring, delivery cursors, round height) must be REBUILT before the instance reports ready, or
//! the run resumes behind its own durable record and forks. Consensus never runs outside the
//! content-addressed module (architecture §4.1/§4.4), so the rebuild is a sandboxed replay:
//!
//! 1. **Re-verify** the node-carried archive heads against the genesis-trusted bases
//!    (carriage is bootstrap, not trust — the worker re-runs `verify_chains` +
//!    `coordinator_lineage` exactly as it re-verifies `peer_certs` and `seat_grant`).
//! 2. **Recover the record stream**: every attested sealed segment, preferring the local
//!    journal home's file (hash-verified against the head) and falling back to the content
//!    plane; then the newest chain's LOCAL unsealed tail, chained via `prev_blake3` off the
//!    last attested segment — a torn tail contributes its intact prefix only (§8.2).
//! 3. **Replay through the sandbox**: boot a throwaway instance of the pinned module under the
//!    production event-loop driver — from the genesis config (`da_init`), or migrated from a
//!    matching anchor capture when the stream's last tag-10 proves one — deliver the recovered
//!    tag-12 signed frames verbatim (original senders/seqs — a re-signed frame would corrupt
//!    per-peer accounting), then quiesce and export the rebuilt state via the typed §10.2
//!    snapshot path.
//!
//! The exported capture becomes the REAL instance's [`daemon_vhc_host::run::MigrationInput`]
//! (`anchor: true` — the reconstruction founds a fresh journal chain, §8.3). Frames inside an
//! attested segment need no per-frame re-verification: the segment bytes hash-match a head
//! signed by a certificate chained to the genesis-trusted base, and the recording relay already
//! verified each frame above the pump before journaling it.
//!
//! The replay is bounded by the archived history's length. A checkpoint-anchored fast path
//! engages when the caller supplies the resolved restore capture AND the recovered stream's
//! last tag-10 manifest byte-matches it (the proof the capture IS that journal position);
//! otherwise the full lineage replays from genesis — always correct by policy determinism.

use std::path::PathBuf;
use std::sync::Arc;

use daemon_vhc_host::run::{
    start_run_migrating, DeliverVerdict, MemorySink, MigrationInput, OpOutcome, PumpHandle, Run,
    RunConfig, RunEnd, RunIdentity, SnapshotCapture,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_journal::{scan_bytes, scan_file, Body, JournalPaths};
use daemon_vhc_net::{ContentHash, ContentStore};
use daemon_vhc_proto::{
    coordinator_lineage, verify_chains, ArchiveHeadRecord, ChainVerifyError, Hash, PeerId,
};

/// A typed reconstruction failure. A seat join carrying a recovery directive that cannot
/// reconstruct MUST refuse — silently starting fresh would fork the run behind its own record.
#[derive(Debug, thiserror::Error)]
pub enum ReconstructError {
    /// The carried heads failed worker-side re-verification against the genesis trust.
    #[error("archive head verification: {0}")]
    Verify(#[from] ChainVerifyError),
    /// A sealed segment could not be obtained/validated (local + content plane both failed,
    /// or the bytes do not match the attested hash / chain linkage).
    #[error("segment {segment} of chain {chain_instance}: {detail}")]
    Segment {
        /// The chain scope's founding incarnation.
        chain_instance: u64,
        /// The segment ordinal within the chain.
        segment: u64,
        /// What failed.
        detail: String,
    },
    /// The sandbox replay itself failed (engine/driver/guest-level).
    #[error("sandbox replay: {0}")]
    Sandbox(String),
}

/// Everything the reconstruction executor needs — resolved by the join path, verified here.
pub struct ReconstructSpec {
    /// The node-carried archive heads (the [`crate::protocol::CoordinatorRecovery`] payload) —
    /// re-verified against `trusted` before anything is read.
    pub heads: Vec<ArchiveHeadRecord>,
    /// The run's cryptographic id (the genesis hash).
    pub run_id: Hash,
    /// The genesis-trusted base identities (from the resolved envelope — never the credentials).
    pub trusted: Vec<PeerId>,
    /// The seat role whose lineage reconstructs.
    pub role: String,
    /// The run label (the journal home's directory key).
    pub run_label: String,
    /// The durable run-state root (`DAEMON_VHC_RUN_DIR`) for local segment reuse + the unsealed
    /// tail. `None` = no local journal (a cold standby) — every sealed segment fetches remote,
    /// and reconstruction reaches the archived point only.
    pub journal_root: Option<PathBuf>,
    /// The pinned module bytes (already verified against the envelope pin by the join path).
    pub module: Vec<u8>,
    /// The role's genesis config bytes (the `da_init` input, verbatim).
    pub config: Vec<u8>,
    /// The admitted grants bytes.
    pub grants: Vec<u8>,
    /// The new seat incarnation (the sandbox instance's identity; its journal is in-memory).
    pub incarnation: u64,
    /// The node-resolved checkpoint restore capture, if any — the checkpoint-anchored fast
    /// path's candidate (engaged only when the recovered stream's last tag-10 byte-matches its
    /// manifest; otherwise the full lineage replays from genesis).
    pub restore: Option<SnapshotCapture>,
    /// The quiesce drain ceiling for the state export (ms).
    pub deadline_ms: u64,
}

/// One recovered authoritative frame (a tag-12 record), verbatim.
struct RecoveredFrame {
    channel: u32,
    seq: u64,
    sender: [u8; 32],
    payload: Vec<u8>,
    original: Vec<u8>,
}

/// What the record recovery produced: the frames to replay and the anchor judgment.
struct RecoveredStream {
    /// Frames BEFORE the anchor cut (replayed only on the genesis path).
    prefix: Vec<RecoveredFrame>,
    /// Frames AFTER the anchor cut (always replayed).
    tail: Vec<RecoveredFrame>,
    /// Whether the caller's restore capture byte-matched the stream's last tag-10 (the proof
    /// the capture is exactly the state at the cut — the fast path's engagement condition).
    anchored: bool,
}

/// Rebuild the seat's consensus state from its published journal lineage through the sandbox and
/// return the exported [`SnapshotCapture`] — the real instance's migration input (`anchor: true`).
///
/// # Errors
/// A typed [`ReconstructError`]; the join must refuse on any of them (see type docs).
pub async fn reconstruct_coordinator(
    spec: ReconstructSpec,
    segments: Arc<dyn ContentStore>,
) -> Result<SnapshotCapture, ReconstructError> {
    // -- 1. re-verify the carried heads against genesis trust (carriage, not trust) --------------
    let chains = verify_chains(&spec.run_id, &spec.trusted, spec.heads.clone())?;
    let lineage = coordinator_lineage(&chains, &spec.role)?;

    // -- 2. recover the verified record stream (attested segments + the local unsealed tail) -----
    let stream = recover_stream(&spec, &lineage, segments.as_ref()).await?;

    tracing::info!(
        run = spec.run_label,
        role = spec.role,
        chains = lineage.len(),
        anchored = stream.anchored,
        prefix_frames = stream.prefix.len(),
        tail_frames = stream.tail.len(),
        "coordinator reconstruction: replaying the recovered lineage through the sandbox"
    );
    // -- 3. the sandboxed replay (blocking: the driver runs a guest thread) ----------------------
    let restore = spec.restore.clone();
    tokio::task::spawn_blocking(move || replay_capture(&spec, restore, stream))
        .await
        .map_err(|e| ReconstructError::Sandbox(format!("replay task: {e}")))?
}

/// Obtain one attested sealed segment's bytes: the local journal home's file when it
/// hash-matches the head, else the content plane (hash-verified either way).
async fn segment_bytes(
    spec: &ReconstructSpec,
    chain_instance: u64,
    head: &ArchiveHeadRecord,
    segments: &dyn ContentStore,
) -> Result<Vec<u8>, ReconstructError> {
    let seg_err = |detail: String| ReconstructError::Segment {
        chain_instance,
        segment: head.body.segment,
        detail,
    };
    if let Some(root) = &spec.journal_root {
        let dir =
            crate::journal_home::journal_dir(root, &spec.run_label, &spec.role, chain_instance);
        if let Ok(paths) = JournalPaths::open(&dir) {
            if let Ok(bytes) = read_local(&paths.segment(head.body.segment)) {
                if daemon_vhc_proto::blake3_hash(&bytes) == head.body.segment_hash {
                    return Ok(bytes);
                }
                // A local file that disagrees with the attested head is stale/torn — the
                // content plane is authoritative; fall through.
            }
        }
    }
    let bytes = segments
        .get_content(&ContentHash(head.body.segment_hash.0))
        .await
        .map_err(|e| seg_err(format!("content plane fetch: {e}")))?;
    if daemon_vhc_proto::blake3_hash(&bytes) != head.body.segment_hash {
        return Err(seg_err("content plane returned foreign bytes".into()));
    }
    Ok(bytes)
}

// The journal home is a host-owned, node-chosen directory (never attacker-influenced); this
// mirrors the journal substrate's own sanctioned raw-fs discipline.
#[allow(clippy::disallowed_methods)]
fn read_local(path: &std::path::Path) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}

/// Walk the verified lineage: scan every attested segment (cross-checking `prev_blake3` against
/// the head), then chain the newest chain's LOCAL unsealed tail; split the frames at the anchor
/// cut (the last tag-10 whose manifest byte-matches the caller's restore capture).
async fn recover_stream(
    spec: &ReconstructSpec,
    lineage: &[&daemon_vhc_proto::VerifiedChain],
    segments: &dyn ContentStore,
) -> Result<RecoveredStream, ReconstructError> {
    let anchor_manifest = spec.restore.as_ref().map(|c| c.manifest.as_slice());
    let mut prefix: Vec<RecoveredFrame> = Vec::new();
    let mut tail: Vec<RecoveredFrame> = Vec::new();
    let mut anchored = false;

    let fold_records = |records: &[daemon_vhc_journal::Record],
                        prefix: &mut Vec<RecoveredFrame>,
                        tail: &mut Vec<RecoveredFrame>,
                        anchored: &mut bool| {
        for record in records {
            match &record.body {
                Body::Snapshot(snap) => {
                    // The anchor judgment: a tag-10 that byte-matches the resolved restore
                    // capture's manifest proves the capture IS this journal position — every
                    // frame before it is already folded into the capture.
                    if anchor_manifest.is_some_and(|m| m == snap.manifest.as_slice()) {
                        prefix.append(tail);
                        *anchored = true;
                    }
                }
                Body::SignedFrame(sf) => {
                    let Some(original) = &sf.frame else {
                        continue; // an evidence-by-reference record (Phase D) carries no bytes
                    };
                    let Some(payload) = frame_payload(original) else {
                        continue; // structurally foreign evidence — never a replay input
                    };
                    tail.push(RecoveredFrame {
                        channel: u32::try_from(sf.channel).unwrap_or(u32::MAX),
                        seq: sf.seq,
                        sender: sf.sender.0,
                        payload,
                        original: original.clone(),
                    });
                }
                _ => {}
            }
        }
    };

    for chain in lineage {
        let mut last_complete: Option<[u8; 32]> = None;
        for head in &chain.heads {
            let bytes = segment_bytes(spec, chain.chain_instance, head, segments).await?;
            let scan = scan_bytes(&bytes).map_err(|e| ReconstructError::Segment {
                chain_instance: chain.chain_instance,
                segment: head.body.segment,
                detail: format!("scan: {e}"),
            })?;
            if Hash(scan.header.prev_blake3) != head.body.prev_hash {
                return Err(ReconstructError::Segment {
                    chain_instance: chain.chain_instance,
                    segment: head.body.segment,
                    detail: "segment header's prev link disagrees with the attested head".into(),
                });
            }
            fold_records(&scan.records, &mut prefix, &mut tail, &mut anchored);
            last_complete = Some(scan.complete_file_blake3);
        }

        // The NEWEST chain's local unsealed tail: segments past the last attested head, chained
        // via prev_blake3. Only the crashed box holds it; a cold standby reconstructs to the
        // archived point (which the coordinator's replay-forward semantics then catch up).
        let is_newest = std::ptr::eq(*chain, *lineage.last().expect("lineage non-empty"));
        if !is_newest {
            continue;
        }
        let Some(root) = &spec.journal_root else {
            continue;
        };
        let dir = crate::journal_home::journal_dir(
            root,
            &spec.run_label,
            &spec.role,
            chain.chain_instance,
        );
        let Ok(paths) = JournalPaths::open(&dir) else {
            continue;
        };
        let Ok(ordinals) = paths.existing_segments() else {
            continue;
        };
        let first_unpublished = chain
            .heads
            .last()
            .map_or(0, |h| h.body.segment.saturating_add(1));
        let mut prev = last_complete;
        for ord in ordinals {
            if ord < first_unpublished {
                continue;
            }
            let scan = match scan_file(paths.segment(ord)) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(
                        run = spec.run_label,
                        segment = ord,
                        error = %e,
                        "coordinator reconstruction: unreadable local tail segment; stopping the tail walk"
                    );
                    break;
                }
            };
            if prev.is_some_and(|p| p != scan.header.prev_blake3) {
                tracing::warn!(
                    run = spec.run_label,
                    segment = ord,
                    "coordinator reconstruction: local tail segment does not chain; stopping the tail walk"
                );
                break;
            }
            fold_records(&scan.records, &mut prefix, &mut tail, &mut anchored);
            prev = Some(scan.complete_file_blake3);
        }
    }

    Ok(RecoveredStream {
        prefix,
        tail,
        anchored,
    })
}

/// Extract the module-authored payload bytes from a §12.1 signed wire frame
/// (`[envelope, payload, sig]`) — structural CBOR, never a round schema.
fn frame_payload(frame: &[u8]) -> Option<Vec<u8>> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).ok()?;
    let ciborium::value::Value::Array(parts) = v else {
        return None;
    };
    match parts.into_iter().nth(1)? {
        ciborium::value::Value::Bytes(payload) => Some(payload),
        _ => None,
    }
}

/// Boot the throwaway sandbox instance, replay the recovered frames, quiesce, export.
///
/// Synchronous (the driver runs the guest on its own thread; delivery back-pressure sleeps) —
/// called from `spawn_blocking`.
fn replay_capture(
    spec: &ReconstructSpec,
    restore: Option<SnapshotCapture>,
    stream: RecoveredStream,
) -> Result<SnapshotCapture, ReconstructError> {
    let sandbox = |e: String| ReconstructError::Sandbox(e);

    let module_hash = *blake3::hash(&spec.module).as_bytes();
    let engine =
        Worker::new(EngineConfig::default()).map_err(|e| sandbox(format!("engine: {e}")))?;
    let sel = select_driver(&engine, &spec.module, Some(&module_hash))
        .map_err(|e| sandbox(format!("selection: {e}")))?;
    if sel.driver != daemon_vhc_abi::CandidateDriver::V2 {
        return Err(sandbox(format!(
            "the pinned module must select the major-2 driver, got {:?}",
            sel.driver
        )));
    }

    // The anchor decision (see module docs): migrate from the proven capture and replay only
    // the post-anchor tail, or genesis-boot and replay everything.
    let (migration, frames) = if stream.anchored {
        let capture = restore.expect("anchored implies a restore capture");
        (
            Some(MigrationInput {
                capture,
                restore: true,
                migrate_fuel: None,
                carried_state: Vec::new(),
                // The SANDBOX journal is in-memory and discarded; the anchor obligation
                // belongs to the REAL instance's founding migration, not this throwaway.
                anchor: false,
            }),
            stream.tail,
        )
    } else {
        let mut all = stream.prefix;
        all.extend(stream.tail);
        (None, all)
    };

    // A throwaway identity: the sandbox's publishes are discarded and its journal is in-memory —
    // nothing it signs ever leaves this process.
    let identity = RunIdentity {
        run_id: spec.run_id.0,
        epoch: 0,
        role: spec.role.clone(),
        instance: spec.incarnation,
        module: module_hash,
    };
    let signing_seed =
        *blake3::hash(&[&spec.run_id.0[..], b"reconstruct-sandbox"].concat()).as_bytes();
    let run_cfg = RunConfig::new(
        identity,
        signing_seed,
        spec.config.clone(),
        spec.grants.clone(),
    );
    let run = start_run_migrating(
        &engine,
        &spec.module,
        run_cfg,
        Box::new(MemorySink::new()),
        migration,
    )
    .map_err(|e| sandbox(format!("start: {e}")))?;
    let pump = run.pump.clone();

    // Deliver the recovered frames verbatim (original senders/seqs), back-pressuring on
    // SpoolFull/SenderQuota per §4.7 — never dropping, bounded per stall. Capability ops the
    // guest issues along the way are failed promptly (`fail_pending_ops`), and a guest thread
    // that ENDS mid-replay surfaces its own end at once — a dead guest can never drain the
    // queue, so waiting out the stall ceiling would report the symptom and mask the cause
    // (observed live, c15-20260806b: the un-serviced availability `payload_get` queue crossed
    // the `grant-bound.n` outstanding-op ceiling, the guest trapped GrantViolation, and every
    // retry burned the full ceiling before reporting a frame-consumption stall).
    const STALL_CEILING: std::time::Duration = std::time::Duration::from_secs(60);
    for frame in frames {
        fail_pending_ops(&pump);
        if run.is_finished() {
            return Err(guest_end_mid_replay(run));
        }
        let deadline = std::time::Instant::now() + STALL_CEILING;
        loop {
            match pump
                .deliver_frame(
                    frame.channel,
                    frame.seq,
                    frame.sender,
                    frame.payload.clone(),
                    frame.original.clone(),
                )
                .map_err(|e| sandbox(format!("deliver: {e}")))?
            {
                DeliverVerdict::Accepted => break,
                DeliverVerdict::SpoolFull | DeliverVerdict::SenderQuota => {
                    fail_pending_ops(&pump);
                    if run.is_finished() {
                        return Err(guest_end_mid_replay(run));
                    }
                    if std::time::Instant::now() >= deadline {
                        return Err(sandbox("replay spool never drained (back-pressure)".into()));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                other => return Err(sandbox(format!("unexpected deliver verdict: {other:?}"))),
            }
        }
    }

    // The PRE-QUIESCE BARRIER: opening the drain freezes still-queued frames (they spool for a
    // successor instead of folding, §4.4 — `next_event` skips `Frame` events while draining), so
    // the export must wait until the guest has PULLED every recovered frame. The guest folds each
    // frame before pulling the next event, so an empty frame queue means the last fold completes
    // before the Quiesce is observed — the exported state covers the whole recovered stream.
    let deadline = std::time::Instant::now() + STALL_CEILING;
    while pump.pending_frames() > 0 {
        fail_pending_ops(&pump);
        if run.is_finished() {
            return Err(guest_end_mid_replay(run));
        }
        if std::time::Instant::now() >= deadline {
            return Err(sandbox(format!(
                "the reconstruction instance never consumed the recovered frames \
                 ({} still pending at the stall ceiling)",
                pump.pending_frames()
            )));
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    fail_pending_ops(&pump);

    // Quiesce → export (§10.2).
    pump.quiesce(daemon_vhc_abi::QUIESCE_REASON_UPGRADE, spec.deadline_ms)
        .map_err(|e| sandbox(format!("quiesce: {e}")))?;
    let end = run
        .wait()
        .map_err(|e| sandbox(format!("guest thread: {e}")))?;
    match end {
        RunEnd::Outcome(code)
            if u64::from(code) == u64::from(daemon_vhc_abi::OUTCOME_QUIESCE_READY) => {}
        other => {
            return Err(sandbox(format!(
                "the reconstruction instance did not quiesce cleanly: {other:?}"
            )))
        }
    }
    pump.snapshot_capture()
        .ok_or_else(|| sandbox("the reconstruction instance staged no snapshot".into()))
}

/// Fail every capability op the sandbox guest has issued, promptly and typed.
///
/// The replay is HERMETIC by design: the sandbox instance's only inputs are the recovered
/// journal frames — consensus finality for every closed round rides the journaled records
/// (its own attested `RoundRecord` publications fold back in as frames), which is exactly the
/// equivalence the D2 replay oracle proves. A capability op at replay (the quorum coordinator's
/// availability-check `payload_get` per Commitment, §6.4 I6) is therefore NOT re-serviced
/// against the content plane: a re-fetch could not change any closed round (the record is
/// already folded), it would couple crash-recovery latency to payload sizes and re-introduce
/// remote failure modes mid-join, and by module policy a failed fetch is simply "no evidence".
///
/// The completions must still be DELIVERED: an op left un-answered occupies the
/// `grant-bound.n` outstanding-op ceiling, and a long recovered lineage crosses it — observed
/// live (c15-20260806b, round 9, two trainers): the 17th un-serviced availability check
/// trapped the guest `GrantViolation` mid-replay, on every retry, until the run went terminal.
fn fail_pending_ops(pump: &PumpHandle) {
    for (op, _request) in pump.take_op_requests() {
        // A completion refused by the pump (guest already gone) is moot — the guest-end check
        // at the call sites surfaces that path.
        let _ = pump.complete_op(
            op,
            OpOutcome::Failed {
                code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                detail: "reconstruction sandbox: capability ops are not serviced at replay \
                         (a failed availability fetch is no evidence)"
                    .into(),
            },
        );
    }
}

/// The guest thread ended while frames were still being replayed: surface ITS end — the
/// drain stall is the symptom, the guest's own trap/outcome is the cause.
fn guest_end_mid_replay(run: Run) -> ReconstructError {
    match run.wait() {
        Ok(end) => ReconstructError::Sandbox(format!(
            "the reconstruction instance ended mid-replay: {end:?}"
        )),
        Err(e) => ReconstructError::Sandbox(format!(
            "the reconstruction instance faulted mid-replay: {e}"
        )),
    }
}
