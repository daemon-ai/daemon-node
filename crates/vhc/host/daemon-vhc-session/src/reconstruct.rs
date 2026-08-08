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
    /// A TRANSIENT transport fault reaching the content plane (connect/timeout/reset/5xx) —
    /// preserved TYPED from the HTTP boundary instead of stringified into the `Segment` fold
    /// (Gate C, defect 10). The archive is not wrong and the join is not refused semantically:
    /// the environment is momentarily unavailable, so the caller defers budget-free and
    /// retries paced. (The c15-20260806g outage: transient R2 egress failures during
    /// reconstruction consumed the semantic retry budget and drove a healthy seat terminal.)
    #[error("transient transport fault during reconstruction ({kind}): {detail}")]
    Transport {
        /// The transport fault class, verbatim from the net boundary.
        kind: daemon_vhc_net::TransportFaultKind,
        /// Operator-facing detail (never branched on).
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
    /// The node's durable journal sidecar key (§8.5, `VhcKeystore::journal_sidecar_key`) — a
    /// same-box CACHE FAST-PATH only (Gate A): it lets this reconstruction decrypt the local
    /// sidecar files its own crashed incarnations wrote, skipping a capture replay dependency.
    /// The AUTHORITATIVE path never needs it: the only sidecar-sized read-backs a coordinator
    /// records are its §10.2 restore sections (kind 3, legal only during `da_migrate`), and
    /// those bytes are content-addressed sections of the migration capture the reconstruction
    /// rebuilds from the archived record stream itself. `None` (a cold standby / different box)
    /// is therefore fully supported, not degraded.
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
    local_dir: Option<&std::path::Path>,
    chain_instance: u64,
    head: &ArchiveHeadRecord,
    segments: &dyn ContentStore,
) -> Result<Vec<u8>, ReconstructError> {
    let seg_err = |detail: String| ReconstructError::Segment {
        chain_instance,
        segment: head.body.segment,
        detail,
    };
    if let Some(dir) = local_dir {
        if let Ok(paths) = JournalPaths::open(dir) {
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
        .map_err(|e| match e {
            // A transient transport fault stays TYPED (never folded into the semantic
            // `Segment` refusal): the caller's recovery is budget-free paced deferral.
            daemon_vhc_net::VhcNetError::Transient { kind, detail } => {
                ReconstructError::Transport { kind, detail }
            }
            other => seg_err(format!("content plane fetch: {other}")),
        })?;
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
        let dir = spec.journal_root.as_ref().map(|root| {
            crate::journal_home::journal_dir(
                root,
                &spec.run_label,
                &spec.role,
                chain.chain_instance,
            )
        });
        for head in &chain.heads {
            let bytes = segment_bytes(dir.as_deref(), chain.chain_instance, head, segments).await?;
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

        // Same-box sidecar hydration (§8.5) — the CACHE FAST-PATH (Gate A), never the
        // authority: a record's sidecar-referenced read-back value (a successor incarnation's
        // §10.2 restore sections) may live in the chain's LOCAL sidecar store, decryptable
        // with the node's own journal key and verified against its content address. What does
        // not hydrate here (a cold standby, a pruned store, a different box) resolves through
        // the AUTHORITATIVE path in `build_script`: content-addressed against the migration
        // capture rebuilt from the archived record stream. Only when BOTH miss does the
        // reconstruction refuse typed.
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
/// resolved CONTENT-ADDRESSED from `sections` — the AUTHORITATIVE archive path (Gate A):
/// sidecar files are node-local (their key material never rides the archive, §8.5), but
/// the only sidecar-sized read-backs a coordinator records are its §10.2 restore sections
/// (`read_back(kind = 3)` is legal only during `da_migrate`, and by-ref sections restore
/// through `data@2::fetch`, never through kind 3) — so those bytes are exactly the INLINE
/// sections of the migration capture the reconstruction already rebuilt from the archived
/// stream. The c15h audit confirms the domain: kind 3 is the ONLY sidecar-referenced kind
/// any coordinator chain records (trainer kind-5 tensor exports stay node-local and are
/// never reconstruction inputs). An unresolvable reference refuses typed: replaying a
/// guessed value would fork the seat behind its own record.
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

/// Whether an encoded event frame is the host's `Stop` delivery (`EV_TAG_STOP`): the frame is
/// the canonical CBOR array `[tag, ...]` and the tag is its first element.
fn is_stop_event(frame: &[u8]) -> bool {
    let Ok(ciborium::value::Value::Array(items)) =
        ciborium::de::from_reader::<ciborium::value::Value, _>(frame)
    else {
        return false;
    };
    matches!(
        items.first(),
        Some(ciborium::value::Value::Integer(i))
            if u64::try_from(*i) == Ok(daemon_vhc_abi::EV_TAG_STOP)
    )
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
        // A TRAILING recorded stop delivery (`EV_TAG_STOP`) is the host's end-of-feed marker —
        // a graceful stop that was journaled before the process exited — not a consensus
        // input: the guest returns from `da_run` on it without publishing, so re-feeding it
        // would end the replay before the synthetic reconstruction Quiesce can export
        // (§10.2). Drop it; the state to rebuild IS the state at the stop point. (A stop with
        // records after it — a same-incarnation node-restart resume — replays through the
        // live LOCAL recovery path, never through this executor.)
        if script
            .events
            .back()
            .is_some_and(|(_, frame)| is_stop_event(frame))
        {
            script.events.pop_back();
        }
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

// ---- Gate B': trainer archive catch-up extraction ------------------------------------------------

/// One archived coordinator publish extracted for a trainer's staged catch-up fold: the complete
/// §12.1-signed wire frame (delivered pump-direct, journaled as ordinary tag-12 evidence) plus
/// the delivery coordinates already parsed out of it.
#[derive(Clone, Debug)]
pub struct CatchUpFrame {
    /// The channel the coordinator published on.
    pub channel: u64,
    /// The coordinator's durable channel-scoped sequence number.
    pub seq: u64,
    /// The publishing per-run key (from the frame envelope).
    pub sender: [u8; 32],
    /// The module payload bytes (the frame's second element).
    pub payload: Vec<u8>,
    /// The complete signed wire frame, verbatim.
    pub frame: Vec<u8>,
    /// The `RoundRecord` round the frame carries (the fold-order / dedup key).
    pub round: u64,
}

/// Everything the catch-up extractor needs — resolved by the join path, verified here.
pub struct CatchUpSpec {
    /// The node-carried seat-lineage heads ([`crate::protocol::TrainerCatchUp`]) — re-verified
    /// against `trusted` before anything is read.
    pub heads: Vec<ArchiveHeadRecord>,
    /// The run's cryptographic id (the genesis hash).
    pub run_id: Hash,
    /// The genesis-trusted base identities (from the resolved envelope — never the credentials).
    pub trusted: Vec<PeerId>,
    /// The run label (the journal home's directory key, for the local-segment fast path).
    pub run_label: String,
    /// The durable run-state root for local segment reuse. `None` / no local copy = every
    /// sealed segment fetches from the content plane.
    pub journal_root: Option<PathBuf>,
    /// The restore fence: only rounds STRICTLY ABOVE this extract.
    pub after_round: u64,
}

/// Extract the seat's historical committed `RoundRecord` frames from its verified archive
/// lineage — the trainer staged-catch-up input (architecture §5.3: "the host fetches
/// record-archive segments by range … SDK replays records since the fence as ordinary frames").
///
/// Authenticity comes from the VERIFIED LINEAGE (heads signed to genesis trust, every segment's
/// bytes re-hashed against its attested head), deliberately NOT from per-frame certificate
/// liveness — a superseded coordinator incarnation's committed records are exactly the history a
/// rejoiner must fold, and its certificate is long revoked. The frame probe is structural (the
/// `frame_kind` discipline — no SDK schema linked): it reads the wire union's variant name and
/// the record's `round`, leaving the module payload opaque. Re-publishes of the same round
/// (replay-forward emissions inside the archived stream) deduplicate on the round key — a
/// committed round's record is canonical.
///
/// # Errors
/// A typed [`ReconstructError`]; a join carrying a catch-up directive that cannot extract MUST
/// refuse (a genuine archive gap is [`ReconstructError::Segment`], never a silent wedge) —
/// except the [`ReconstructError::Transport`] lane, which defers budget-free (Gate C).
pub async fn extract_catch_up_frames(
    spec: &CatchUpSpec,
    segments: &dyn ContentStore,
) -> Result<Vec<CatchUpFrame>, ReconstructError> {
    let chains = verify_chains(&spec.run_id, &spec.trusted, spec.heads.clone())?;
    let Some(role) = chains.first().map(|c| c.role.clone()) else {
        return Err(ReconstructError::Sandbox(
            "the catch-up directive carries no verified chains".into(),
        ));
    };
    if chains.iter().any(|c| c.role != role) {
        return Err(ReconstructError::Sandbox(
            "the catch-up directive mixes roles; one seat lineage expected".into(),
        ));
    }
    let lineage = coordinator_lineage(&chains, &role)?;

    let mut frames: Vec<CatchUpFrame> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for chain in &lineage {
        let dir = spec.journal_root.as_ref().map(|root| {
            crate::journal_home::journal_dir(root, &spec.run_label, &role, chain.chain_instance)
        });
        for head in &chain.heads {
            let bytes = segment_bytes(dir.as_deref(), chain.chain_instance, head, segments).await?;
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
            for record in &scan.records {
                let Body::Publish(p) = &record.body else {
                    continue;
                };
                let Some((round, sender, payload)) = round_record_frame(&p.frame) else {
                    continue;
                };
                if round > spec.after_round && seen.insert(round) {
                    frames.push(CatchUpFrame {
                        channel: p.channel,
                        seq: p.seq,
                        sender,
                        payload,
                        frame: p.frame.clone(),
                        round,
                    });
                }
            }
        }
    }
    // Chains walk founding-first, so rounds are near-sorted already; the fold order MUST be
    // ascending (the guest's resync guard folds monotonically past its watermark).
    frames.sort_by_key(|f| f.round);
    Ok(frames)
}

/// If a §12.1 wire frame carries a wire `RoundRecord`, its `(round, sender, payload)` — else
/// `None`. A structural shape probe in the `frame_kind` discipline (see
/// `role_session::published_round_record`): no SDK schema is linked; the walk reads exactly the
/// envelope's `sender`, the externally-tagged wire union's variant name and the record's
/// `round` field, all fixed by the consensus wire contract.
fn round_record_frame(frame: &[u8]) -> Option<(u64, [u8; 32], Vec<u8>)> {
    use ciborium::value::Value;
    let Ok(Value::Array(parts)) = ciborium::de::from_reader(frame) else {
        return None;
    };
    let Some(Value::Bytes(payload)) = parts.get(1) else {
        return None;
    };
    let Some(Value::Map(env)) = parts.first() else {
        return None;
    };
    let sender: [u8; 32] = env.iter().find_map(|(k, v)| match (k, v) {
        (Value::Text(t), Value::Bytes(b)) if t == "sender" => b.as_slice().try_into().ok(),
        _ => None,
    })?;
    let inner: Value = ciborium::de::from_reader(payload.as_slice()).ok()?;
    // The externally-tagged union: a 1-element map (serde's standard shape) or a
    // `[name, value]` variant array.
    let (tag, body) = match &inner {
        Value::Map(entries) if entries.len() == 1 => match &entries[0] {
            (Value::Text(t), v) => (t.as_str(), v),
            _ => return None,
        },
        Value::Array(items) if items.len() == 2 => match items.first() {
            Some(Value::Text(t)) => (t.as_str(), &items[1]),
            _ => return None,
        },
        _ => return None,
    };
    if tag != "RoundRecord" {
        return None;
    }
    let Value::Map(fields) = body else {
        return None;
    };
    let round = fields.iter().find_map(|(k, v)| match (k, v) {
        (Value::Text(t), Value::Integer(n)) if t == "round" => u64::try_from(i128::from(*n)).ok(),
        _ => None,
    })?;
    Some((round, sender, payload.clone()))
}

#[cfg(test)]
mod transport_lane_tests {
    //! Gate C (defect 10): the content-plane fetch inside reconstruction preserves the typed
    //! transient transport lane instead of stringifying it into the semantic `Segment` fold.

    use super::*;
    use async_trait::async_trait;
    use daemon_vhc_net::VhcNetError;
    use daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN;
    use daemon_vhc_proto::{peer_id, ArchiveHeadBody, CertScope, RunKeyCertificate, SigningKey};

    /// A content plane whose GETs fail with a chosen error (the outage / the pruned store).
    struct FailingPlane(fn() -> VhcNetError);

    #[async_trait]
    impl ContentStore for FailingPlane {
        async fn put_content(&self, _bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
            unreachable!("reconstruction never writes the content plane")
        }
        async fn get_content(&self, _hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
            Err((self.0)())
        }
    }

    fn head() -> ArchiveHeadRecord {
        let base = SigningKey::from_bytes(&[0xB0; 32]);
        let run_key = SigningKey::from_bytes(&[0x4A; 32]);
        let cert = RunKeyCertificate::issue(
            &base,
            CertScope {
                run_id: Hash([0x1D; 32]),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: Hash([0x2A; 32]),
            },
            peer_id(&run_key),
        )
        .expect("cert");
        ArchiveHeadRecord::publish(
            &run_key,
            cert,
            ArchiveHeadBody {
                domain: ARCHIVE_HEAD_DOMAIN.into(),
                run_id: Hash([0x1D; 32]),
                role: "coordinator".into(),
                chain_instance: 1,
                segment: 0,
                segment_hash: Hash([0xA0; 32]),
                prev_hash: Hash([0; 32]),
                records: 4,
                instance: 1,
                epoch: 0,
                module: Hash([0x2A; 32]),
                predecessor: None,
                round: None,
            },
        )
        .expect("publish")
    }

    /// A transient transport fault (connect refused, timeout, 5xx — the R2 outage shape) stays
    /// TYPED [`ReconstructError::Transport`]: the caller defers budget-free instead of burning
    /// the semantic retry budget on a network outage (the c15-20260806g terminal loop).
    #[tokio::test]
    async fn a_transient_content_plane_fault_stays_typed() {
        let plane = FailingPlane(|| VhcNetError::Transient {
            kind: daemon_vhc_net::TransportFaultKind::Connect,
            detail: "egress: request failed (connect): dial refused".into(),
        });
        // A cold standby (no local journal dir): every segment fetches remote.
        let err = segment_bytes(None, 1, &head(), &plane)
            .await
            .expect_err("the fetch fails");
        assert!(
            matches!(
                err,
                ReconstructError::Transport {
                    kind: daemon_vhc_net::TransportFaultKind::Connect,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    /// A semantic miss (the object genuinely absent / pruned) still folds into the typed
    /// `Segment` refusal — the budgeted lane: the archive is incomplete, not the network.
    #[tokio::test]
    async fn a_semantic_miss_still_folds_into_the_segment_refusal() {
        let plane =
            FailingPlane(|| VhcNetError::PayloadMiss("never stored / lifecycle-expired".into()));
        let err = segment_bytes(None, 1, &head(), &plane)
            .await
            .expect_err("the fetch fails");
        assert!(
            matches!(err, ReconstructError::Segment { segment: 0, .. }),
            "got {err:?}"
        );
    }
}

#[cfg(test)]
mod catch_up_tests {
    //! Gate B': the trainer archive catch-up extractor — verified lineage in, the fence-bounded,
    //! round-deduplicated, ascending `RoundRecord` frame stream out.

    use super::*;
    use async_trait::async_trait;
    use daemon_vhc_net::VhcNetError;
    use daemon_vhc_proto::domains::ARCHIVE_HEAD_DOMAIN;
    use daemon_vhc_proto::{peer_id, ArchiveHeadBody, CertScope, RunKeyCertificate, SigningKey};

    const RUN: [u8; 32] = [0x1D; 32];
    const MODULE: [u8; 32] = [0x2A; 32];
    const SENDER: [u8; 32] = [0xC0; 32];

    /// A content plane serving exactly the uploaded segments, content-addressed.
    struct MapPlane(std::collections::BTreeMap<[u8; 32], Vec<u8>>);

    #[async_trait]
    impl ContentStore for MapPlane {
        async fn put_content(&self, _bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
            unreachable!("extraction never writes the content plane")
        }
        async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
            self.0
                .get(&hash.0)
                .cloned()
                .ok_or_else(|| VhcNetError::PayloadMiss("absent".into()))
        }
    }

    /// A §12.1-shaped wire frame carrying a wire `RoundRecord` (structural shape only — the
    /// extractor never verifies the signature; authenticity is the attested segment).
    fn round_record_wire_frame(round: u64) -> Vec<u8> {
        use ciborium::value::Value;
        let mut payload = Vec::new();
        ciborium::into_writer(
            &Value::Map(vec![(
                Value::from("RoundRecord"),
                Value::Map(vec![(Value::from("round"), Value::from(round))]),
            )]),
            &mut payload,
        )
        .unwrap();
        let envelope = Value::Map(vec![
            (Value::from("sender"), Value::Bytes(SENDER.to_vec())),
            (Value::from("channel"), Value::from(0u64)),
            (Value::from("seq"), Value::from(round)),
        ]);
        let wire = Value::Array(vec![
            envelope,
            Value::Bytes(payload),
            Value::Bytes(vec![0u8; 64]),
        ]);
        let mut out = Vec::new();
        ciborium::into_writer(&wire, &mut out).unwrap();
        out
    }

    /// Author a real sealed coordinator segment (through the journal substrate) whose publishes
    /// carry rounds 1..=3 plus a replay-forward RE-publish of round 2, upload it to a map plane,
    /// and attest it with a head signed to a trusted base. The extractor must return exactly the
    /// post-fence rounds, deduplicated and ascending.
    #[tokio::test]
    async fn extraction_is_fence_bounded_deduplicated_and_ascending() {
        let dir = tempfile::tempdir().unwrap();
        let jroot = dir.path().join("journal");
        let id = daemon_vhc_journal::ExecIdentity {
            run_id: Hash(RUN),
            epoch: 0,
            role: "coordinator".into(),
            instance: 1,
            module: Hash(MODULE),
        };
        {
            let mut j = daemon_vhc_journal::Journal::create(
                &jroot,
                id,
                daemon_vhc_journal::StaticKey::new([0; 32]),
                daemon_vhc_journal::RotatePolicy::default(),
            )
            .unwrap();
            for round in [1u64, 2, 3, 2] {
                // rounds in publication order, with a replay-forward duplicate
                j.publish(0, b"round-payload", round_record_wire_frame(round))
                    .unwrap();
            }
            j.roll().unwrap(); // seal segment 0
        }
        // Test-only fixture read inside a tempdir — not a production fs path.
        #[allow(clippy::disallowed_methods)]
        let seg_bytes = std::fs::read(JournalPaths::open(&jroot).unwrap().segment(0)).unwrap();
        let seg_hash = daemon_vhc_proto::blake3_hash(&seg_bytes);

        let base = SigningKey::from_bytes(&[0xB0; 32]);
        let run_key = SigningKey::from_bytes(&[0x4A; 32]);
        let cert = RunKeyCertificate::issue(
            &base,
            CertScope {
                run_id: Hash(RUN),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: Hash(MODULE),
            },
            peer_id(&run_key),
        )
        .unwrap();
        let head = ArchiveHeadRecord::publish(
            &run_key,
            cert,
            ArchiveHeadBody {
                domain: ARCHIVE_HEAD_DOMAIN.into(),
                run_id: Hash(RUN),
                role: "coordinator".into(),
                chain_instance: 1,
                segment: 0,
                segment_hash: seg_hash,
                prev_hash: Hash([0; 32]),
                records: 4,
                instance: 1,
                epoch: 0,
                module: Hash(MODULE),
                predecessor: None,
                round: Some(3),
            },
        )
        .unwrap();

        let plane = MapPlane([(seg_hash.0, seg_bytes)].into_iter().collect());
        let spec = CatchUpSpec {
            heads: vec![head],
            run_id: Hash(RUN),
            trusted: vec![peer_id(&base)],
            run_label: "run-x".into(),
            journal_root: None, // a cold rejoiner: the segment fetches from the content plane
            after_round: 1,     // the restore fence
        };
        let frames = extract_catch_up_frames(&spec, &plane)
            .await
            .expect("extraction succeeds from the verified lineage");
        let rounds: Vec<u64> = frames.iter().map(|f| f.round).collect();
        assert_eq!(
            rounds,
            vec![2, 3],
            "post-fence only, the replay-forward duplicate deduplicated, ascending"
        );
        for f in &frames {
            assert_eq!(f.sender, SENDER, "the sender parses out of the envelope");
            assert_eq!(f.channel, 0);
            assert!(!f.payload.is_empty(), "the module payload rides verbatim");
            assert!(
                !f.frame.is_empty(),
                "the complete wire frame rides verbatim"
            );
        }
    }

    /// A directive whose segment is gone from BOTH planes is the genuine archive gap: a typed
    /// `Segment` refusal (the re-scoped `CheckpointStale` posture — never a silent wedge).
    #[tokio::test]
    async fn a_genuine_archive_gap_refuses_typed() {
        let base = SigningKey::from_bytes(&[0xB0; 32]);
        let run_key = SigningKey::from_bytes(&[0x4A; 32]);
        let cert = RunKeyCertificate::issue(
            &base,
            CertScope {
                run_id: Hash(RUN),
                epoch: 0,
                role: "coordinator".into(),
                instance: 1,
                module_hash: Hash(MODULE),
            },
            peer_id(&run_key),
        )
        .unwrap();
        let head = ArchiveHeadRecord::publish(
            &run_key,
            cert,
            ArchiveHeadBody {
                domain: ARCHIVE_HEAD_DOMAIN.into(),
                run_id: Hash(RUN),
                role: "coordinator".into(),
                chain_instance: 1,
                segment: 0,
                segment_hash: Hash([0xEE; 32]), // attested, but stored nowhere
                prev_hash: Hash([0; 32]),
                records: 1,
                instance: 1,
                epoch: 0,
                module: Hash(MODULE),
                predecessor: None,
                round: Some(9),
            },
        )
        .unwrap();
        let plane = MapPlane(std::collections::BTreeMap::new());
        let spec = CatchUpSpec {
            heads: vec![head],
            run_id: Hash(RUN),
            trusted: vec![peer_id(&base)],
            run_label: "run-x".into(),
            journal_root: None,
            after_round: 0,
        };
        let err = extract_catch_up_frames(&spec, &plane)
            .await
            .expect_err("a missing attested segment refuses");
        assert!(
            matches!(err, ReconstructError::Segment { segment: 0, .. }),
            "typed segment refusal, got {err:?}"
        );
    }
}
