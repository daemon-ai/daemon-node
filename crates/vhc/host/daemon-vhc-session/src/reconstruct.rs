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
//! 3. **Replay the FULL RECORD STREAM through the §8.7 input-replay engine** — the same
//!    journal-answers-everything semantics the D2 oracle proves: delivered events verbatim
//!    (inbound frames, timer fires, **capability-op completions in journaled order** — a
//!    ceremony coordinator's §6.4 I6 storage receipts are its OWN `payload_get` completions,
//!    which exist nowhere in the frame stream), recorded read-backs, clock readings and timer
//!    arms. At the stream's end (the crash cut) a synthetic Quiesce drives the guest's own
//!    §10.2 snapshot path, and the engine assembles the exported capture.
//! 4. **Gate on decision equivalence**: every recorded outbound `Publish` (channel, seq,
//!    payload hash) must be reproduced by the replayed guest, in order. A replay that folds
//!    the history but does not re-derive the recorded decisions did NOT rebuild the recorded
//!    state — the join refuses typed. (The c15-20260806g corruption: a frames-only replay
//!    starved of its completion-borne receipts silently exported a round-0 state and the
//!    resumed coordinator re-ran the run from round 0 on a successor chain.)
//!
//! The exported capture becomes the REAL instance's [`daemon_vhc_host::run::MigrationInput`]
//! (`anchor: true` — the reconstruction founds a fresh journal chain, §8.3). Frames inside an
//! attested segment need no per-frame re-verification: the segment bytes hash-match a head
//! signed by a certificate chained to the genesis-trusted base, and the recording relay already
//! verified each frame above the pump before journaling it. Bulk payload bytes are NOT
//! re-fetched: the recorded completion is the guest-visible evidence, and a guest that would
//! actually READ the missing bytes ends the replay typed
//! ([`daemon_vhc_host::run::ReplayScript::missing_payload_placeholders`]).
//!
//! The replay is bounded by the archived history's length and replays from the lineage's
//! beginning — always correct by policy determinism. Incarnation seams inside the stream
//! (tag-13 records) split it into spans replayed sequentially: a reason-2 (upgrade-activation)
//! span migrates from the previous span's exported capture (or the node-resolved restore
//! capture when the lineage itself begins mid-history), everything else boots fresh.

use std::path::PathBuf;
use std::sync::Arc;

use daemon_vhc_host::run::{
    replay_migrating, ReplayEnd, ReplayMigration, ReplayScript, ReplayedRun, RunIdentity,
    SnapshotCapture,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_journal::{scan_bytes, scan_file, Body, JournalPaths, Record};
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
    /// The node's durable journal sidecar key (§8.5, `VhcKeystore::journal_sidecar_key`) —
    /// lets a SAME-BOX reconstruction decrypt the sidecar-referenced read-back values its own
    /// crashed incarnations recorded (a successor chain's §10.2 restore sections exceed
    /// `READBACK_INLINE_MAX` and ride sidecars, never the archive). `None` (a cold standby)
    /// falls back to content-addressed resolution against the span's rebuilt migration capture.
    pub sidecar_key: Option<[u8; 32]>,
    /// The quiesce drain ceiling for the state export (ms).
    pub deadline_ms: u64,
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
    let records = recover_records(&spec, &lineage, segments.as_ref()).await?;

    tracing::info!(
        run = spec.run_label,
        role = spec.role,
        chains = lineage.len(),
        records = records.len(),
        "coordinator reconstruction: replaying the recovered record stream through the sandbox"
    );
    // -- 3+4. full-record input replay + decision gate (blocking: synchronous wasm drive) --------
    let restore = spec.restore.clone();
    tokio::task::spawn_blocking(move || replay_capture(&spec, restore, records))
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
/// the head), then chain the newest chain's LOCAL unsealed tail — collecting the FULL record
/// stream in order (the §8.7 input-replay engine consumes events, read-backs, clocks and timer
/// arms; tag-12 evidence and tag-4 publishes feed the decision gate).
async fn recover_records(
    spec: &ReconstructSpec,
    lineage: &[&daemon_vhc_proto::VerifiedChain],
    segments: &dyn ContentStore,
) -> Result<Vec<Record>, ReconstructError> {
    let mut records: Vec<Record> = Vec::new();

    for chain in lineage {
        let mut chain_records: Vec<Record> = Vec::new();
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
            last_complete = Some(scan.complete_file_blake3);
            chain_records.extend(scan.records);
        }

        // The NEWEST chain's local unsealed tail: segments past the last attested head, chained
        // via prev_blake3. Only the crashed box holds it; a cold standby reconstructs to the
        // archived point (which the coordinator's replay-forward semantics then catch up).
        let is_newest = std::ptr::eq(*chain, *lineage.last().expect("lineage non-empty"));
        let dir = spec.journal_root.as_ref().map(|root| {
            crate::journal_home::journal_dir(
                root,
                &spec.run_label,
                &spec.role,
                chain.chain_instance,
            )
        });
        'tail: {
            if !is_newest {
                break 'tail;
            }
            let Some(dir) = &dir else {
                break 'tail;
            };
            let Ok(paths) = JournalPaths::open(dir) else {
                break 'tail;
            };
            let Ok(ordinals) = paths.existing_segments() else {
                break 'tail;
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
                prev = Some(scan.complete_file_blake3);
                chain_records.extend(scan.records);
            }
        }

        // Same-box sidecar hydration (§8.5): a record's sidecar-referenced read-back value
        // (a successor incarnation's §10.2 restore sections) lives in the chain's LOCAL
        // sidecar store, decryptable with the node's own journal key. Best-effort: what does
        // not hydrate here still resolves content-addressed against the span's migration
        // capture in `build_script`, and only THEN refuses typed.
        if let (Some(key), Some(dir)) = (spec.sidecar_key, &dir) {
            hydrate_sidecars(&mut chain_records, dir, key);
        }
        records.extend(chain_records);
    }

    Ok(records)
}

/// Inline every locally-decryptable sidecar-referenced read-back value in `records`
/// (§8.5: the store binds the owning span's execution identity, and the AEAD nonce is
/// `(record ord, instantiation counter)` — both recovered from the stream itself: the
/// span's tag-0 header and tag-13 instantiation record).
fn hydrate_sidecars(records: &mut [Record], jdir: &std::path::Path, key: [u8; 32]) {
    let sidecar_dir = match JournalPaths::open(jdir) {
        Ok(paths) => paths.sidecars(),
        Err(_) => return,
    };
    let mut store: Option<daemon_vhc_journal::SidecarStore<daemon_vhc_journal::StaticKey>> = None;
    let mut counter: u64 = 0;
    for record in records.iter_mut() {
        match &mut record.body {
            Body::RunHeader(h) => {
                let id = daemon_vhc_journal::ExecIdentity {
                    run_id: h.run_id,
                    epoch: h.epoch,
                    role: h.role.clone(),
                    instance: h.instance,
                    module: h.module,
                };
                store = daemon_vhc_journal::SidecarStore::open(
                    &sidecar_dir,
                    id,
                    daemon_vhc_journal::StaticKey::new(key),
                )
                .ok();
            }
            Body::Instantiation(inst) => counter = inst.counter,
            Body::ReadBack(r) if r.value.is_none() => {
                if let (Some(sref), Some(store)) = (&r.sidecar, &store) {
                    match store.get(sref, record.ord, counter) {
                        Ok(bytes) => r.value = Some(bytes),
                        Err(e) => tracing::debug!(
                            ord = record.ord,
                            hash = %sref.hash.to_hex(),
                            error = %e,
                            "reconstruction: local sidecar did not hydrate; deferring to \
                             content-addressed resolution"
                        ),
                    }
                }
            }
            _ => {}
        }
    }
}

/// One incarnation's slice of the recovered stream: its tag-0 header, its instantiation reason
/// (0 initial, 1 trap-restart, 2 upgrade-activation) and its records.
struct Span {
    header: Box<daemon_vhc_journal::record::RunHeader>,
    reason: u64,
    records: Vec<Record>,
}

/// Split the recovered stream at incarnation seams: every tag-0 run header starts a span (the
/// driver journals the header before the tag-13 instantiation, which carries the reason).
fn split_spans(records: Vec<Record>) -> Result<Vec<Span>, ReconstructError> {
    let mut spans: Vec<Span> = Vec::new();
    for record in records {
        if let Body::RunHeader(header) = &record.body {
            spans.push(Span {
                header: header.clone(),
                reason: 0,
                records: Vec::new(),
            });
            continue;
        }
        let Some(span) = spans.last_mut() else {
            return Err(ReconstructError::Sandbox(
                "the recovered stream begins without a tag-0 run header — not a \
                 production journal lineage"
                    .into(),
            ));
        };
        if let Body::Instantiation(inst) = &record.body {
            span.reason = inst.reason;
        }
        span.records.push(record);
    }
    if spans.is_empty() {
        return Err(ReconstructError::Sandbox(
            "the recovered stream carries no records — nothing to reconstruct".into(),
        ));
    }
    Ok(spans)
}

/// The content-addressed section values a span's sidecar-referenced read-backs resolve
/// against: every INLINE section of the capture the span migrates from, keyed by its
/// plaintext blake3 — exactly the bytes the live driver staged for `read_back(kind = 3)`
/// (§10.2 restore bindings), whose hash is what the sidecar ref records.
fn sections_by_hash(
    capture: Option<&SnapshotCapture>,
) -> std::collections::BTreeMap<[u8; 32], Vec<u8>> {
    let mut map = std::collections::BTreeMap::new();
    if let Some(capture) = capture {
        for section in &capture.sections {
            if let daemon_vhc_proto::det_state::CkptDocSection::Inline(_, bytes) = section {
                map.insert(*blake3::hash(bytes).as_bytes(), bytes.clone());
            }
        }
    }
    map
}

/// The §8.7 record → [`ReplayScript`] mapping (the same split the D2 archive replay uses):
/// delivered events verbatim (frames, timer fires, completions — in journaled order),
/// read-backs routed by kind, clock readings, timer arms/cancels, device profiles.
///
/// A read-back whose value rode a sidecar (plaintext > `READBACK_INLINE_MAX`, §8.5) is
/// resolved CONTENT-ADDRESSED from `sections`: sidecar files are node-local (their key
/// material never rides the archive, §8.5), but the only sidecar-sized read-backs a
/// coordinator records are its §10.2 restore sections (`read_back(kind = 3)` is legal
/// only during `da_migrate`) — and those bytes are exactly the migration capture the
/// reconstruction already rebuilt from the predecessor span. An unresolvable reference
/// refuses typed: replaying a guessed value would fork the seat behind its own record.
fn build_script(
    records: &[Record],
    sections: &std::collections::BTreeMap<[u8; 32], Vec<u8>>,
) -> Result<ReplayScript, ReconstructError> {
    let mut script = ReplayScript::default();
    for record in records {
        match &record.body {
            Body::Event(e) => script.events.push_back((e.at, e.frame.clone())),
            Body::ReadBack(r) => {
                let value = match (&r.value, &r.sidecar) {
                    (Some(v), _) => v.clone(),
                    (None, Some(sref)) => {
                        sections.get(sref.hash.as_bytes()).cloned().ok_or_else(|| {
                            ReconstructError::Sandbox(format!(
                                "record {}: the sidecar-referenced read-back value {} ({} B, \
                                 kind {}) resolves neither inline nor against the span's \
                                 migration capture — the lineage does not carry the bytes \
                                 this replay must re-feed",
                                record.ord,
                                sref.hash.to_hex(),
                                sref.size,
                                r.kind
                            ))
                        })?
                    }
                    (None, None) => Vec::new(),
                };
                if r.kind >= 128 {
                    // The retired bridge's reserved journal kinds: never re-fed.
                } else if r.kind == u64::from(daemon_vhc_abi::READBACK_KIND_STREAM_BYTES) {
                    script.stream_bytes.push_back((r.src, value));
                } else if r.kind == u64::from(daemon_vhc_abi::READBACK_KIND_TENSOR_EXPORT) {
                    script.tensor_exports.push_back((r.src, value));
                } else if r.kind == u64::from(daemon_vhc_abi::READBACK_KIND_STATE_SEAL) {
                    script.state_seals.push_back((r.src, value));
                } else {
                    script.readbacks.push_back((r.src, r.kind, value));
                }
            }
            Body::Clock(c) => script.clocks.push_back(c.now),
            Body::TimerArm(t) => script.timer_arms.push_back((t.id, t.delay)),
            Body::TimerCancel(t) => script.timer_cancels.push_back((t.id, t.status)),
            Body::DeviceProfile(d) => script.device_profiles.push_back(d.profile.clone()),
            _ => {}
        }
    }
    Ok(script)
}

/// The decision gate: every recorded outbound publish `(channel, seq, payload hash)` of the
/// span must be reproduced by the replayed guest, in order. The replay may legitimately
/// produce MORE than the record (a decision whose publish record was lost to the crash cut
/// re-derives), never fewer and never different — a mismatch means the replay did NOT rebuild
/// the recorded state, and resuming from it would fork the run behind its own durable record.
fn verify_decisions(
    span_index: usize,
    records: &[Record],
    replayed: &ReplayedRun,
) -> Result<(), ReconstructError> {
    let recorded: Vec<(u64, u64, [u8; 32], u64)> = records
        .iter()
        .filter_map(|r| match &r.body {
            Body::Publish(p) => Some((p.channel, p.seq, p.hash.0, r.ord)),
            _ => None,
        })
        .collect();
    if replayed.decisions.len() < recorded.len() {
        return Err(ReconstructError::Sandbox(format!(
            "span {span_index}: the replay reproduced {} of {} recorded publishes — the \
             rebuilt state is BEHIND the durable record (resuming would fork the run)",
            replayed.decisions.len(),
            recorded.len()
        )));
    }
    for (i, (rec, got)) in recorded.iter().zip(&replayed.decisions).enumerate() {
        if got.channel != rec.0 || got.seq != rec.1 || got.payload_hash != rec.2 {
            return Err(ReconstructError::Sandbox(format!(
                "span {span_index}: replayed decision {i} (channel {}, seq {}) diverges from \
                 the recorded publish at ordinal {} (channel {}, seq {}) — the replay did not \
                 re-derive the recorded history",
                got.channel, got.seq, rec.3, rec.0, rec.1
            )));
        }
    }
    Ok(())
}

/// Re-drive the recovered record stream through the §8.7 input-replay engine, span by span,
/// gate each span's decisions against its recorded publishes, and export the final state via
/// the guest's own §10.2 snapshot path (driven by the synthetic exhaustion Quiesce).
///
/// Synchronous (a replay can never block — every input it may wait for is in the script) —
/// called from `spawn_blocking`.
fn replay_capture(
    spec: &ReconstructSpec,
    restore: Option<SnapshotCapture>,
    records: Vec<Record>,
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

    let spans = split_spans(records)?;
    let total = spans.len();

    // The capture carried across spans: a reason-2 (upgrade-activation) span migrates from its
    // predecessor's export; the LEADING span may instead migrate from the node-resolved restore
    // capture (a lineage that begins mid-history). Anything else boots fresh from genesis.
    let mut carried: Option<SnapshotCapture> = restore;
    let mut exported: Option<SnapshotCapture> = None;

    for (i, span) in spans.into_iter().enumerate() {
        // The header is the journal's own claim of what ran — cross-check it against the
        // directive before re-driving anything from it (a foreign journal is a refusal, not
        // an input).
        if span.header.module.0 != module_hash {
            return Err(sandbox(format!(
                "span {i}: the recorded module {} is not the genesis-pinned coordinator",
                span.header.module.to_hex()
            )));
        }
        if span.header.run_id != spec.run_id {
            return Err(sandbox(format!(
                "span {i}: the recorded run id {} is not this run",
                span.header.run_id.to_hex()
            )));
        }
        if span.header.config != spec.config {
            return Err(sandbox(format!(
                "span {i}: the recorded da_init config differs from the genesis role config \
                 — a journal recorded under foreign configuration cannot rebuild this seat"
            )));
        }

        // Sidecar-referenced read-backs (§10.2 restore sections) resolve content-addressed
        // against the capture this span migrates from — available BEFORE the take below.
        let section_values = if span.reason == 2 {
            sections_by_hash(carried.as_ref())
        } else {
            sections_by_hash(None)
        };
        let mut script = build_script(&span.records, &section_values)?;
        script.identity = Some(RunIdentity {
            run_id: span.header.run_id.0,
            epoch: span.header.epoch,
            role: span.header.role.clone(),
            instance: span.header.instance,
            module: span.header.module.0,
        });
        // At the recorded stream's end (the crash cut) the guest quiesces through its own
        // §10.2 snapshot path and the engine assembles the export. Payload bytes are not
        // re-fetched: the recorded completion is the evidence (module docs).
        script.quiesce_at_exhaustion = Some(spec.deadline_ms);
        script.missing_payload_placeholders = true;

        let migration = if span.reason == 2 {
            let capture = carried.take().ok_or_else(|| {
                sandbox(format!(
                    "span {i} is an upgrade-activation but no predecessor capture is \
                     available (the lineage begins mid-history and the node resolved no \
                     restore) — the lineage does not anchor"
                ))
            })?;
            Some(ReplayMigration {
                capture,
                migrate_fuel: None,
            })
        } else {
            None
        };

        let replayed = replay_migrating(
            &engine,
            &spec.module,
            &span.header.config,
            &span.header.grants,
            script,
            migration,
        )
        .map_err(|e| sandbox(format!("span {i} replay: {e}")))?;

        verify_decisions(i, &span.records, &replayed)?;

        let is_final = i + 1 == total;
        match &replayed.end {
            ReplayEnd::Outcome(code)
                if u64::from(*code) == u64::from(daemon_vhc_abi::OUTCOME_QUIESCE_READY) => {}
            // A mid-lineage span may end exactly as the recording did (a trap that caused the
            // reason-1 restart, a recorded terminal outcome) — the successor span rebuilds
            // from scratch, as the live restart did.
            ReplayEnd::Outcome(_) | ReplayEnd::Trapped(_) if !is_final => {}
            other => {
                return Err(sandbox(format!(
                    "span {i}: the reconstruction replay did not reach the §10.2 export: \
                     {other:?}"
                )));
            }
        }

        carried = replayed.capture.clone();
        if is_final {
            exported = replayed.capture;
        }
    }

    exported.ok_or_else(|| {
        sandbox("the reconstruction instance staged no snapshot (§10.2 export missing)".into())
    })
}
