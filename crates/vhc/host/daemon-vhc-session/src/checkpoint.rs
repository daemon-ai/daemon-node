// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Round-boundary checkpointing + desync-recovery replay (spec §9; TDD RUN-6/7 subset).
//!
//! A checkpoint is `TrainerBackend::checkpoint_save` bytes plus a [`CheckpointManifest`] — the
//! round, the blake3 of the bytes (content address, §9), and the post-round state digest (§5.6).
//! Checkpoints are stored on the payload plane under a reserved key ([`CHECKPOINT_PEER`]), so
//! [`save_checkpoint`] / [`load_checkpoint`] round-trip through the same [`PayloadStore`] the round
//! payloads use, blake3-verified on load.
//!
//! Desync recovery is **record replay** (§6.4 I1, §9): a peer whose post-round digest disagrees
//! with the consensus reloads the latest checkpoint and replays the retained `RoundRecord`s (their
//! root-verified committed sets) + payloads forward to the current round. [`resync_by_replay`] is
//! that pure fold — `checkpoint_load` then `ingest` each retained round in order — the offline
//! resync oracle. The *trigger* (this peer's digest vs the quorum/consensus digest) is
//! `daemon_vhc_observe::digest_tally` / `DesyncVerdict` (folded over the run's `Digest` messages,
//! §9) — consumed by the harness + drills, which drive this replay on `DesyncVerdict::is_desync()`;
//! the replay fold itself is here.

use std::sync::Arc;

use daemon_vhc_proto::{blake3_hash, Hash, PeerId};
use daemon_vhc_sdk_consensus::attestation::{
    AttestationLedger, AttestationPolicy, JoinEligibility,
};
use daemon_vhc_sdk_consensus::checkpoint::{
    CheckpointManifest as TypedCheckpointManifest, SectionKind,
};

use crate::backend::{StagedPayload, StateDigest, TrainerBackend};
use crate::seam::{PayloadKey, RoundId, RunId};
use daemon_vhc_net::PayloadStore;

use crate::VhcRunError;

/// The reserved payload-plane peer id under which a run's checkpoints are stored (never a real node
/// identity — node pubkeys are ed25519 points, this sentinel is not).
pub const CHECKPOINT_PEER: PeerId = PeerId([0xCC; PeerId::LEN]);

/// Section-schema tag: the `module` section bytes are the backend's opaque `checkpoint_save` bytes
/// (the authoritative bit-exact restore serialization).
pub const MODULE_SCHEMA_OPAQUE: u64 = 1;
/// Section-schema tag: the `module` section bytes are a safetensors serialization of the module's
/// typed state dict (the portable, inspectable typed export — the E1 safetensors bridge).
pub const MODULE_SCHEMA_SAFETENSORS: u64 = 2;
/// Section-schema tag for the host-owned sections (consensus digest, data cursor, journal position).
pub const HOST_SECTION_SCHEMA: u64 = 1;

/// The payload-plane sentinel peer under which the typed checkpoint manifest itself is stored (a
/// distinct reserved byte pattern, never a real identity — cf. [`CHECKPOINT_PEER`]).
#[must_use]
fn manifest_peer() -> PeerId {
    let mut b = [0xCC; PeerId::LEN];
    b[1] = 0xFF;
    PeerId(b)
}

/// The payload-plane sentinel peer under which a typed checkpoint's section of `kind` is stored.
/// Distinct per kind (`b[1] = kind.tag()`), and a manifest carries at most one section per kind, so
/// the key is collision-free and deterministically reconstructible by a late joiner from the
/// manifest alone.
#[must_use]
fn section_peer(kind: SectionKind) -> PeerId {
    let mut b = [0xCC; PeerId::LEN];
    b[1] = kind.tag() as u8;
    PeerId(b)
}

/// The payload-plane key of a typed checkpoint's **manifest** at `(run, round)` — how a late joiner
/// (E3) fetches the manifest it was pointed at, verifying against the pointer's content hash.
#[must_use]
pub fn typed_manifest_key(run: &RunId, round: RoundId) -> PayloadKey {
    PayloadKey::new(run.clone(), round, manifest_peer())
}

/// The payload-plane key of a typed checkpoint's **section** of `kind` at `(run, round)` —
/// deterministically reconstructible from the manifest alone (one section per kind), verified
/// against the section's declared blake3 on fetch.
#[must_use]
pub fn typed_section_key(run: &RunId, round: RoundId, kind: SectionKind) -> PayloadKey {
    PayloadKey::new(run.clone(), round, section_peer(kind))
}

/// The manifest of one checkpoint (§9): the round it captures, the blake3 of its bytes, and the
/// post-round state digest (§5.6) it should reproduce on reload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointManifest {
    /// The round this checkpoint captures (post-ingest state).
    pub round: RoundId,
    /// blake3 of the `checkpoint_save` bytes (content address).
    pub blake3: Hash,
    /// The `checkpoint_save` byte length — the third field of the coordinator checkpoint pointer
    /// (lane R, spec §9). Part of the both-match cross-check (identical checkpoints are the same
    /// size).
    pub size: u64,
    /// The post-round state digest this checkpoint reproduces.
    pub digest: StateDigest,
}

/// One replayed round: its committed set staged in record order (§6.4 I3), the input `ingest`
/// consumes during resync.
#[derive(Clone, Debug)]
pub struct ReplayStep {
    /// The round being replayed.
    pub round: RoundId,
    /// Its committed set, staged in record order.
    pub staged: Vec<StagedPayload>,
}

/// Save a round-boundary checkpoint: serialize the backend, content-address it, PUT it to the
/// payload plane, and return the [`CheckpointManifest`].
pub async fn save_checkpoint<P, B>(
    store: &Arc<P>,
    run: &RunId,
    backend: &B,
    round: RoundId,
    digest: StateDigest,
) -> Result<CheckpointManifest, VhcRunError>
where
    P: PayloadStore,
    B: TrainerBackend,
{
    let bytes = backend
        .checkpoint_save()
        .map_err(|e| VhcRunError::Lifecycle(format!("checkpoint_save: {e}")))?;
    let blake3 = blake3_hash(&bytes);
    let size = bytes.len() as u64;
    let key = PayloadKey::new(run.clone(), round, CHECKPOINT_PEER);
    store.put(&key, &bytes).await?;
    Ok(CheckpointManifest {
        round,
        blake3,
        size,
        digest,
    })
}

/// Load a checkpoint named by `manifest` from the payload plane (blake3-verified against the
/// manifest) into `backend`.
pub async fn load_checkpoint<P, B>(
    store: &Arc<P>,
    run: &RunId,
    backend: &mut B,
    manifest: &CheckpointManifest,
) -> Result<(), VhcRunError>
where
    P: PayloadStore,
    B: TrainerBackend,
{
    let key = PayloadKey::new(run.clone(), manifest.round, CHECKPOINT_PEER);
    let bytes = store.get(&key, &manifest.blake3).await?;
    backend
        .checkpoint_load(&bytes)
        .map_err(|e| VhcRunError::Lifecycle(format!("checkpoint_load: {e}")))?;
    Ok(())
}

/// Desync recovery (§9, I1): reload `checkpoint_bytes` into `backend`, then replay `steps` forward
/// (each round's committed set → `ingest`), returning the final post-replay digest.
///
/// A pure fold of `(checkpoint, records, payloads)` — the resync oracle. Since ingest is
/// deterministic and record-ordered, the replayed digest equals the digest an in-sync peer reached
/// (the property this recovers to).
pub fn resync_by_replay<B: TrainerBackend>(
    backend: &mut B,
    checkpoint_bytes: &[u8],
    steps: &[ReplayStep],
) -> Result<StateDigest, VhcRunError> {
    backend
        .checkpoint_load(checkpoint_bytes)
        .map_err(|e| VhcRunError::Lifecycle(format!("resync checkpoint_load: {e}")))?;
    let mut last = None;
    for step in steps {
        let digest = backend
            .ingest(step.round, &step.staged)
            .map_err(|e| VhcRunError::Lifecycle(format!("resync ingest r{}: {e}", step.round)))?;
        last = Some(digest);
    }
    last.ok_or_else(|| VhcRunError::Lifecycle("resync replay had no steps".into()))
}

// -- RUN-6: two-checkpointer both-match registration + degraded mode ----------------------------

/// The outcome of registering a round checkpoint from the elected checkpointers' uploads (§9,
/// TDD RUN-6). The spec elects **two** checkpointers that upload independently; a checkpoint
/// registers only when both agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointRegistration {
    /// Both checkpointers uploaded byte-identical manifests → registered with cross-check.
    Registered(CheckpointManifest),
    /// Only one checkpointer uploaded (the other churned) → registered, but flagged degraded (no
    /// cross-check this round; the run continues and the observer is warned).
    Degraded(CheckpointManifest),
    /// The checkpointers uploaded divergent manifests → rejected (a checkpointer is faulty).
    Mismatch,
    /// No checkpointer uploaded.
    Missing,
}

/// Register a round checkpoint from the elected checkpointers' uploaded manifests (RUN-6, §9).
///
/// Registers only on a **both-match** (all uploads byte-identical: same round, blake3, and digest).
/// A single upload registers in **degraded** mode; divergent uploads are rejected as a fault.
#[must_use]
pub fn register_checkpoint(uploads: &[CheckpointManifest]) -> CheckpointRegistration {
    match uploads {
        [] => CheckpointRegistration::Missing,
        [only] => CheckpointRegistration::Degraded(*only),
        [first, rest @ ..] => {
            if rest.iter().all(|m| m == first) {
                CheckpointRegistration::Registered(*first)
            } else {
                CheckpointRegistration::Mismatch
            }
        }
    }
}

// -- RUN-7: resync-vs-retention decision --------------------------------------------------------

/// How a desynced peer recovers, given the payload-retention floor (§6.4/§9, TDD RUN-7).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResyncPlan {
    /// Replay from the latest checkpoint forward over the retained records/payloads (I1).
    ReplayFromCheckpoint {
        /// The checkpoint round to reload from.
        from_round: RoundId,
        /// The number of rounds to replay forward to reach `current_round`.
        steps: u64,
    },
    /// The desync predates the retention floor: the records/payloads needed to replay are gone, so
    /// the peer waits for the next epoch checkpoint to rejoin (the stall ladder's terminal arm).
    WaitForEpoch,
}

/// Decide how a desynced peer at `current_round` recovers, given the latest checkpoint at
/// `checkpoint_round` and the payload `retention_rounds` floor (RUN-7). If every round since the
/// checkpoint is still retained, replay; otherwise the payloads are gone → wait for the epoch.
#[must_use]
pub fn plan_resync(
    checkpoint_round: RoundId,
    current_round: RoundId,
    retention_rounds: u64,
) -> ResyncPlan {
    let steps = current_round.saturating_sub(checkpoint_round);
    if steps <= retention_rounds {
        ResyncPlan::ReplayFromCheckpoint {
            from_round: checkpoint_round,
            steps,
        }
    } else {
        ResyncPlan::WaitForEpoch
    }
}

// -- E1: the typed checkpoint bridge (refactor §9; architecture §5.3) ---------------------------

/// The cryptographic execution identity a typed checkpoint manifest carries (architecture §5.1),
/// kept distinct from the payload-plane [`RunId`] *label* (D1 `RunLabel` vs `RunId`): the genesis
/// hash, the epoch, and the producing module hash. E2/D0 thread the real values; a pre-D0 caller
/// may pass `blake3(label)` / `epoch = 0` / the module blob hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointIdent {
    /// The run identity (genesis hash, architecture §5.1).
    pub run_id: Hash,
    /// The epoch this checkpoint captures (transition-chain head).
    pub epoch: u64,
    /// The producing module hash.
    pub module: Hash,
}

/// What one checkpoint captures at a round boundary: the round, the post-ingest state digest, and
/// the two host-owned cursors (data cursor + journal position, architecture §5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointCapture {
    /// The round this checkpoint captures (post-ingest state).
    pub round: RoundId,
    /// The post-round consensus-state digest (the det-lane agreement probe, §5.6).
    pub digest: StateDigest,
    /// How far this replica has consumed its corpus window.
    pub data_cursor: u64,
    /// The execution-identity journal ordinal the checkpoint captures (catch-up resumes here).
    pub journal_position: u64,
}

/// A saved typed checkpoint: the [`TypedCheckpointManifest`] (architecture §5.3), its content
/// address, and the coordinator-facing [`CheckpointManifest`] pointer (round + manifest hash + size
/// + digest) — the pointer a coordinator record names, now referencing the *typed manifest*.
#[derive(Clone, Debug)]
pub struct TypedCheckpoint {
    /// The sectioned, content-addressed manifest.
    pub manifest: TypedCheckpointManifest,
    /// The manifest's content address (blake3 of its canonical CBOR) — the checkpoint identity an
    /// attestation signs over ([`daemon_vhc_sdk_consensus::attestation`]).
    pub content_hash: Hash,
    /// The coordinator-facing pointer, `blake3` = [`TypedCheckpoint::content_hash`].
    pub pointer: CheckpointManifest,
}

fn to_proto_digest(d: StateDigest) -> daemon_vhc_proto::StateDigest {
    daemon_vhc_proto::StateDigest(d.0)
}

/// Save a **typed** checkpoint (architecture §5.3, Phase E): assemble the sectioned manifest, wire
/// the safetensors typed serialization into the `module` section when the backend exports a state
/// dict, content-address every section + the manifest, and PUT them to the payload plane.
///
/// Sections (architecture §5.3): `consensus` (the det-lane digest — the consensus-canonical section
/// a digest attestation signs), `module` (the module/role state — safetensors when
/// [`TrainerBackend::export_state_dict`] is `Some`, else the opaque `checkpoint_save` bytes),
/// `worker-local` (the opaque `checkpoint_save` bytes carried as the authoritative bit-exact
/// restore serialization *whenever* the `module` section is the typed safetensors export),
/// `data-cursor`, and `journal-position`. The authoritative restore path is therefore **never**
/// weaker than today's opaque round-trip.
///
/// # Errors
/// A backend serialization error, a store error, or a manifest-assembly error.
pub async fn save_typed_checkpoint<P, B>(
    store: &Arc<P>,
    run: &RunId,
    ident: &CheckpointIdent,
    backend: &B,
    capture: CheckpointCapture,
) -> Result<TypedCheckpoint, VhcRunError>
where
    P: PayloadStore,
    B: TrainerBackend,
{
    let CheckpointCapture {
        round,
        digest,
        data_cursor,
        journal_position,
    } = capture;
    let opaque = backend
        .checkpoint_save()
        .map_err(|e| VhcRunError::Lifecycle(format!("checkpoint_save: {e}")))?;
    let typed = backend
        .export_state_dict()
        .map_err(|e| VhcRunError::Lifecycle(format!("export_state_dict: {e}")))?;

    let mut builder = TypedCheckpointManifest::builder(
        ident.run_id,
        ident.epoch,
        round,
        ident.module,
        to_proto_digest(digest),
    )
    .section(
        "consensus",
        SectionKind::Consensus,
        HOST_SECTION_SCHEMA,
        digest.as_bytes(),
    );

    // The module section: the safetensors typed serialization when available (the E1 bridge), with
    // the opaque bytes retained as the authoritative worker-local restore section; otherwise the
    // opaque bytes are the module section directly.
    let mut section_bytes: Vec<(SectionKind, Vec<u8>)> = Vec::new();
    section_bytes.push((SectionKind::Consensus, digest.as_bytes().to_vec()));
    if let Some(sd) = typed {
        let st = sd
            .to_safetensors()
            .map_err(|e| VhcRunError::Lifecycle(format!("safetensors serialize: {e}")))?;
        builder = builder
            .section(
                "module",
                SectionKind::Module,
                MODULE_SCHEMA_SAFETENSORS,
                &st,
            )
            .section(
                "worker-local",
                SectionKind::WorkerLocal,
                MODULE_SCHEMA_OPAQUE,
                &opaque,
            );
        section_bytes.push((SectionKind::Module, st));
        section_bytes.push((SectionKind::WorkerLocal, opaque));
    } else {
        builder = builder.section("module", SectionKind::Module, MODULE_SCHEMA_OPAQUE, &opaque);
        section_bytes.push((SectionKind::Module, opaque));
    }

    let cursor_bytes = data_cursor.to_le_bytes().to_vec();
    let journal_bytes = journal_position.to_le_bytes().to_vec();
    builder = builder
        .section(
            "data-cursor",
            SectionKind::DataCursor,
            HOST_SECTION_SCHEMA,
            &cursor_bytes,
        )
        .section(
            "journal-position",
            SectionKind::JournalPosition,
            HOST_SECTION_SCHEMA,
            &journal_bytes,
        );
    section_bytes.push((SectionKind::DataCursor, cursor_bytes));
    section_bytes.push((SectionKind::JournalPosition, journal_bytes));

    let manifest = builder
        .build()
        .map_err(|e| VhcRunError::Lifecycle(format!("checkpoint manifest: {e}")))?;

    // Store each section on the payload plane under its per-kind sentinel key; the put returns the
    // content hash, which MUST equal the hash the manifest declared.
    for (kind, bytes) in &section_bytes {
        let key = PayloadKey::new(run.clone(), round, section_peer(*kind));
        let put = store.put(&key, bytes).await?;
        let declared = manifest.section(*kind).expect("declared section").hash;
        debug_assert_eq!(put, declared, "section {kind:?} hash");
    }

    // Store the manifest itself, content-addressed.
    let wire = manifest
        .to_wire()
        .map_err(|e| VhcRunError::Lifecycle(format!("manifest wire: {e}")))?;
    let content_hash = blake3_hash(&wire);
    let mkey = PayloadKey::new(run.clone(), round, manifest_peer());
    store.put(&mkey, &wire).await?;

    Ok(TypedCheckpoint {
        pointer: CheckpointManifest {
            round,
            blake3: content_hash,
            size: wire.len() as u64,
            digest,
        },
        content_hash,
        manifest,
    })
}

/// Restore a **typed** checkpoint (architecture §5.3, Phase E): fetch + verify the manifest against
/// the pointer's content hash, then restore the backend bit-exactly from the authoritative section
/// (the `worker-local` opaque bytes when the module section is the typed safetensors export, else
/// the opaque `module` section). The typed `module` safetensors section, when present, is fetched
/// and parsed as an integrity check (it must decode as a valid state dict). Returns the decoded
/// manifest.
///
/// # Errors
/// A store / hash-verification error, a manifest decode/validate error, a malformed safetensors
/// section, or a backend restore error.
pub async fn load_typed_checkpoint<P, B>(
    store: &Arc<P>,
    run: &RunId,
    backend: &mut B,
    pointer: &CheckpointManifest,
) -> Result<TypedCheckpointManifest, VhcRunError>
where
    P: PayloadStore,
    B: TrainerBackend,
{
    let mkey = PayloadKey::new(run.clone(), pointer.round, manifest_peer());
    let wire = store.get(&mkey, &pointer.blake3).await?;
    let manifest = TypedCheckpointManifest::from_wire(&wire)
        .map_err(|e| VhcRunError::Lifecycle(format!("manifest decode: {e}")))?;
    manifest
        .validate()
        .map_err(|e| VhcRunError::Lifecycle(format!("manifest validate: {e}")))?;
    let content_hash = manifest
        .content_hash()
        .map_err(|e| VhcRunError::Lifecycle(format!("manifest hash: {e}")))?;
    if content_hash != pointer.blake3 {
        return Err(VhcRunError::Lifecycle(
            "typed checkpoint manifest content hash != pointer".into(),
        ));
    }

    // The authoritative bit-exact restore section: worker-local opaque bytes when present, else the
    // module section (which is then guaranteed opaque).
    let restore = manifest
        .section(SectionKind::WorkerLocal)
        .or_else(|| manifest.section(SectionKind::Module))
        .ok_or_else(|| VhcRunError::Lifecycle("checkpoint has no restore section".into()))?;
    let restore_bytes = store
        .get(
            &PayloadKey::new(run.clone(), pointer.round, section_peer(restore.kind)),
            &restore.hash,
        )
        .await?;
    backend
        .checkpoint_load(&restore_bytes)
        .map_err(|e| VhcRunError::Lifecycle(format!("checkpoint_load: {e}")))?;

    // Integrity-check the typed safetensors module section, if that is what the module section is.
    if let Some(module) = manifest.section(SectionKind::Module) {
        if module.schema == MODULE_SCHEMA_SAFETENSORS {
            let st = store
                .get(
                    &PayloadKey::new(
                        run.clone(),
                        pointer.round,
                        section_peer(SectionKind::Module),
                    ),
                    &module.hash,
                )
                .await?;
            daemon_vhc_safetensors::StateDict::from_safetensors(&st).map_err(|e| {
                VhcRunError::Lifecycle(format!("typed module section not valid safetensors: {e}"))
            })?;
        }
    }

    Ok(manifest)
}

// -- E1: late-join foundation (refactor §9) -----------------------------------------------------

/// One candidate checkpoint a late joiner may restore from: its content address + the round it
/// captured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointCandidate {
    /// The typed manifest content hash (the attestation subject + payload-plane identity).
    pub content_hash: Hash,
    /// The round the checkpoint captured (post-ingest).
    pub round: RoundId,
}

/// The generalized late-join path shape (architecture §5.3; refactor §9): **admission → attested
/// checkpoint → restore/`migrate` → record-replay catch-up**. This is the manifest/attestation
/// layer E3 drives to full cold-join acceptance; it composes the E1 primitives — the attestation
/// tiers ([`AttestationPolicy`]) for the "attested checkpoint" step and [`plan_resync`] (which rides
/// the record archive / consensus replay from Phase B/D2) for the "record-replay catch-up" step.
///
/// D2's standby-coordinator drill is the same flow, so this reuses the same seams rather than
/// inventing parallel ones (refactor §9).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LateJoinPlan {
    /// No join-eligible checkpoint yet: not enough digest attestations on any candidate. The joiner
    /// waits and re-evaluates as attestations accrue (the K-digest gate, architecture §5.3).
    AwaitAttested {
        /// The eligibility verdict for the best candidate considered.
        eligibility: JoinEligibility,
    },
    /// Restore from the attestation-preferred checkpoint, then catch up by record replay.
    Restore {
        /// The chosen checkpoint (restore-attested-preferred among join-eligible candidates).
        checkpoint: Hash,
        /// The round it captured.
        from_round: RoundId,
        /// How to reach `current_round` from there (replay vs wait-for-epoch), via [`plan_resync`].
        catch_up: ResyncPlan,
    },
}

/// Plan a late join (refactor §9): choose the attestation-preferred, K-digest-eligible checkpoint
/// among `candidates`, then plan the record-replay catch-up to `current_round` under the payload
/// `retention_rounds` floor. Returns [`LateJoinPlan::AwaitAttested`] if no candidate has gathered K
/// digest attestations yet (initial join-eligibility is gated on the K-digest tier, architecture
/// §5.3).
#[must_use]
pub fn plan_late_join(
    policy: &AttestationPolicy,
    ledger: &AttestationLedger,
    candidates: &[CheckpointCandidate],
    current_round: RoundId,
    retention_rounds: u64,
) -> LateJoinPlan {
    let hashes: Vec<Hash> = candidates.iter().map(|c| c.content_hash).collect();
    match policy.preferred_checkpoint(ledger, &hashes) {
        Some(chosen) => {
            let from_round = candidates
                .iter()
                .find(|c| c.content_hash == chosen)
                .map_or(current_round, |c| c.round);
            LateJoinPlan::Restore {
                checkpoint: chosen,
                from_round,
                catch_up: plan_resync(from_round, current_round, retention_rounds),
            }
        }
        None => {
            // Report the best eligibility verdict observed for diagnostics.
            let eligibility = hashes
                .iter()
                .map(|h| policy.join_eligibility(ledger, h))
                .max_by_key(|e| match e {
                    JoinEligibility::Eligible {
                        digest_attestations,
                    } => *digest_attestations,
                    JoinEligibility::Ineligible { have, .. } => *have,
                })
                .unwrap_or(JoinEligibility::Ineligible {
                    have: 0,
                    need: policy.k_digest,
                });
            LateJoinPlan::AwaitAttested { eligibility }
        }
    }
}

/// Build the record-replay catch-up steps from **archive-recovered** [`RoundRecord`]s (refactor §9:
/// "record-replay catch-up — the archive fetch from Phase B/D2"). This is the offline twin of the
/// engine's live `resolve_record_set` path, feeding [`resync_by_replay`]: for each record, the
/// committed inline set is ordered by node-pubkey bytes (§6.4 I3 record order), every payload is
/// fetched by content hash from `fetch` and blake3+size-verified, and the record's set commitment
/// is **recomputed and checked** (the D2 consensus-replay discipline — an unverifiable record is a
/// typed error, never a silent pass).
///
/// `records` are the recovered records *after* the checkpoint round, in round order (a late joiner
/// obtains them via `daemon-vhc-observe::recover_chain_from_archive` +
/// `extract_consensus_capture`); `fetch` resolves a content hash to payload bytes (the
/// content-addressed payload plane).
///
/// # Errors
/// A typed [`VhcRunError::Lifecycle`] on a record without an inline set (the full set-object
/// resolution is the live engine's path), a missing/mismatched payload, or a set commitment that
/// does not recompute.
pub fn steps_from_round_records<F>(
    records: &[daemon_vhc_sdk_consensus::messages::RoundRecord],
    mut fetch: F,
) -> Result<Vec<ReplayStep>, VhcRunError>
where
    F: FnMut(&Hash) -> Option<Vec<u8>>,
{
    use daemon_vhc_proto::commit_set;

    let mut steps = Vec::with_capacity(records.len());
    for record in records {
        let entries = record.inline.as_ref().ok_or_else(|| {
            VhcRunError::Lifecycle(format!(
                "round {} record has no inline set (resolve the set object via the live path)",
                record.round
            ))
        })?;
        // Record order is a consensus input (§6.4 I3): sorted by node public-key bytes.
        let mut ordered = entries.clone();
        ordered.sort_by_key(|e| e.peer.0);

        let mut staged = Vec::with_capacity(ordered.len());
        let mut pairs = Vec::with_capacity(ordered.len());
        for entry in &ordered {
            let bytes = fetch(&entry.hash).ok_or_else(|| {
                VhcRunError::Lifecycle(format!(
                    "round {}: committed payload {} missing from the payload plane",
                    record.round,
                    entry.hash.to_hex()
                ))
            })?;
            if blake3_hash(&bytes) != entry.hash || bytes.len() as u64 != entry.size {
                return Err(VhcRunError::Lifecycle(format!(
                    "round {}: payload does not re-verify against its committed entry",
                    record.round
                )));
            }
            pairs.push((entry.peer, entry.hash));
            staged.push(StagedPayload {
                peer: entry.peer,
                hash: entry.hash,
                bytes,
            });
        }
        // The record's committed set commitment must recompute from the verified pairs (the D2
        // consensus-replay discipline, applied at the catch-up boundary).
        if commit_set(&pairs).commitment() != record.set {
            return Err(VhcRunError::Lifecycle(format!(
                "round {}: set commitment does not recompute from the payload set",
                record.round
            )));
        }
        steps.push(ReplayStep {
            round: record.round,
            staged,
        });
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::StubBackend;
    use daemon_vhc_net::FsPayloadStore;

    fn temp_store() -> Arc<FsPayloadStore> {
        let root = std::env::temp_dir().join(format!(
            "daemon-vhc-ckpt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        Arc::new(FsPayloadStore::open(&root, 64).unwrap())
    }

    fn staged(peer: u8, tag: &[u8]) -> StagedPayload {
        StagedPayload {
            peer: PeerId([peer; 32]),
            hash: blake3_hash(tag),
            bytes: tag.to_vec(),
        }
    }

    fn built(config: &[u8]) -> StubBackend {
        let mut b = StubBackend::new();
        b.build(config).unwrap();
        b
    }

    #[tokio::test]
    async fn checkpoint_save_load_roundtrips_through_store() {
        // RUN-6 subset: save a checkpoint, reload it into a fresh backend, and reach the same digest
        // on the next ingest.
        let store = temp_store();
        let run = RunId::new("ckpt-run");
        let mut a = built(b"cfg");
        let d0 = a.ingest(0, &[staged(1, b"x"), staged(2, b"y")]).unwrap();

        let manifest = save_checkpoint(&store, &run, &a, 0, d0).await.unwrap();
        assert_eq!(manifest.round, 0);
        assert_eq!(manifest.digest, d0);

        // Reload into a fresh (differently-built) backend, then a further identical ingest must
        // match what the original produces from the same point.
        let mut b = built(b"totally-different-config");
        load_checkpoint(&store, &run, &mut b, &manifest)
            .await
            .unwrap();
        let next = [staged(1, b"p"), staged(2, b"q")];
        assert_eq!(b.ingest(1, &next).unwrap(), a.ingest(1, &next).unwrap());
    }

    #[tokio::test]
    async fn load_rejects_tampered_checkpoint() {
        // A manifest whose blake3 does not match the stored bytes is rejected on load (§9 content
        // addressing).
        let store = temp_store();
        let run = RunId::new("ckpt-tamper");
        let a = built(b"cfg");
        let mut manifest = save_checkpoint(&store, &run, &a, 0, StateDigest([0; 16]))
            .await
            .unwrap();
        manifest.blake3 = blake3_hash(b"not the checkpoint");
        let mut fresh = StubBackend::new();
        let err = load_checkpoint(&store, &run, &mut fresh, &manifest)
            .await
            .unwrap_err();
        assert!(matches!(err, VhcRunError::Net(_)), "got {err:?}");
    }

    use daemon_vhc_proto::{peer_id, SigningKey};
    use daemon_vhc_sdk_consensus::attestation::{
        AttestationBody, AttestationLedger, AttestationPolicy, AttestationTier,
    };
    use daemon_vhc_sdk_consensus::checkpoint::SectionClass;

    fn ident() -> CheckpointIdent {
        CheckpointIdent {
            run_id: blake3_hash(b"e1-run"),
            epoch: 0,
            module: blake3_hash(b"e1-module"),
        }
    }

    #[tokio::test]
    async fn typed_checkpoint_round_trips_and_restores_bit_exactly() {
        // Save a typed sectioned checkpoint (safetensors module + opaque worker-local + consensus /
        // data-cursor / journal-position sections), reload it into a fresh backend, and reach the
        // same digest on the next ingest — the authoritative restore is never weaker than opaque.
        let store = temp_store();
        let run = RunId::new("typed-ckpt");
        let mut a = built(b"cfg");
        let d0 = a.ingest(0, &[staged(1, b"x"), staged(2, b"y")]).unwrap();

        let saved = save_typed_checkpoint(
            &store,
            &run,
            &ident(),
            &a,
            CheckpointCapture {
                round: 0,
                digest: d0,
                data_cursor: 0,
                journal_position: 42,
            },
        )
        .await
        .unwrap();
        // The manifest is sectioned per architecture §5.3, content-addressed + schema-versioned.
        assert_eq!(saved.pointer.blake3, saved.content_hash);
        assert_eq!(saved.manifest.round, 0);
        let m = &saved.manifest;
        assert_eq!(
            m.section(SectionKind::Consensus).unwrap().class,
            SectionClass::ConsensusCanonical
        );
        // StubBackend exports a state dict → the module section is the safetensors typed export, and
        // the opaque bytes are retained as the authoritative worker-local restore section.
        assert_eq!(
            m.section(SectionKind::Module).unwrap().schema,
            MODULE_SCHEMA_SAFETENSORS
        );
        assert_eq!(
            m.section(SectionKind::WorkerLocal).unwrap().schema,
            MODULE_SCHEMA_OPAQUE
        );
        assert!(m.section(SectionKind::DataCursor).is_some());
        assert!(m.section(SectionKind::JournalPosition).is_some());

        // Restore into a differently-built backend; the next identical ingest must match.
        let mut b = built(b"totally-different-config");
        let restored = load_typed_checkpoint(&store, &run, &mut b, &saved.pointer)
            .await
            .unwrap();
        assert_eq!(restored, saved.manifest);
        let next = [staged(1, b"p"), staged(2, b"q")];
        assert_eq!(b.ingest(1, &next).unwrap(), a.ingest(1, &next).unwrap());
    }

    #[tokio::test]
    async fn restore_attestation_round_trip() {
        // The NAMED attestation round-trip (refactor §9): K digest attestations gate initial
        // join-eligibility; a peer that loads the full manifest and it verifies emits a restore
        // attestation; the late-join planner then prefers the restore-attested checkpoint and plans
        // the record-replay catch-up. NOT folded into "cold join works".
        let store = temp_store();
        let run = RunId::new("attest-run");
        let id = ident();
        let mut a = built(b"cfg");
        let d0 = a.ingest(0, &[staged(1, b"x"), staged(2, b"y")]).unwrap();
        let saved = save_typed_checkpoint(
            &store,
            &run,
            &id,
            &a,
            CheckpointCapture {
                round: 0,
                digest: d0,
                data_cursor: 0,
                journal_position: 7,
            },
        )
        .await
        .unwrap();
        let ckpt = saved.content_hash;

        let policy = AttestationPolicy::new(2);
        let mut ledger = AttestationLedger::new();
        let candidates = [CheckpointCandidate {
            content_hash: ckpt,
            round: 0,
        }];

        // Before K digest attestations: not join-eligible → the planner awaits.
        assert!(matches!(
            plan_late_join(&policy, &ledger, &candidates, 3, 8),
            LateJoinPlan::AwaitAttested { .. }
        ));

        // Two live peers sign digest attestations ("declared digest == my consensus state").
        let digest_att = |sk: &SigningKey| {
            AttestationBody {
                tier: AttestationTier::Digest,
                run_id: id.run_id,
                epoch: id.epoch,
                round: 0,
                checkpoint: ckpt,
                digest: to_proto_digest(d0),
                signer: peer_id(sk),
            }
            .sign(sk)
            .unwrap()
        };
        for seed in 1..=2u8 {
            ledger
                .record(digest_att(&SigningKey::from_bytes(&[seed; 32])))
                .unwrap();
        }

        // Now join-eligible: the plan restores from this checkpoint and plans catch-up.
        let plan = plan_late_join(&policy, &ledger, &candidates, 3, 8);
        let LateJoinPlan::Restore {
            checkpoint,
            from_round,
            catch_up,
        } = plan
        else {
            panic!("expected Restore, got {plan:?}");
        };
        assert_eq!(checkpoint, ckpt);
        assert_eq!(from_round, 0);
        assert_eq!(
            catch_up,
            ResyncPlan::ReplayFromCheckpoint {
                from_round: 0,
                steps: 3,
            }
        );

        // A late joiner performs the restore step: load the full manifest (it verifies) …
        let restorer_sk = SigningKey::from_bytes(&[9u8; 32]);
        let mut joiner = built(b"fresh-joiner-config");
        load_typed_checkpoint(&store, &run, &mut joiner, &saved.pointer)
            .await
            .unwrap();
        // … then signs a RESTORE attestation ("I loaded the full manifest and it verified").
        let restore_att = AttestationBody {
            tier: AttestationTier::Restore,
            run_id: id.run_id,
            epoch: id.epoch,
            round: 0,
            checkpoint: ckpt,
            digest: to_proto_digest(d0),
            signer: peer_id(&restorer_sk),
        }
        .sign(&restorer_sk)
        .unwrap();
        ledger.record(restore_att).unwrap();
        assert!(ledger.is_restore_attested(&ckpt));

        // The restored joiner reaches the in-sync digest on the next ingest (recoverability proven).
        let next = [staged(1, b"p"), staged(2, b"q")];
        assert_eq!(
            joiner.ingest(1, &next).unwrap(),
            a.ingest(1, &next).unwrap()
        );
    }

    #[tokio::test]
    async fn late_join_catches_up_from_archived_round_records() {
        // The late-join composition end to end (refactor §9: admission → attested checkpoint →
        // restore → record-replay catch-up), with the catch-up sourced from RoundRecords as the
        // record archive recovers them (D2's seam): typed checkpoint at round 0 → records for
        // rounds 1..=2 with committed inline sets → steps_from_round_records verifies payloads +
        // recomputes set commitments → resync_by_replay reaches the in-sync digest.
        use daemon_vhc_proto::{commit_set, Seed};
        use daemon_vhc_sdk_consensus::messages::{Locator, RecordEntry, RoundRecord};
        use std::collections::BTreeMap;

        let store = temp_store();
        let run = RunId::new("late-join");
        let mut reference = built(b"cfg");
        let d0 = reference
            .ingest(0, &[staged(1, b"a0"), staged(2, b"b0")])
            .unwrap();
        let saved = save_typed_checkpoint(
            &store,
            &run,
            &ident(),
            &reference,
            CheckpointCapture {
                round: 0,
                digest: d0,
                data_cursor: 0,
                journal_position: 3,
            },
        )
        .await
        .unwrap();

        // Rounds 1..=2 proceed; the archive's records commit their sets.
        let mut payload_plane: BTreeMap<Hash, Vec<u8>> = BTreeMap::new();
        let mut records = Vec::new();
        let mut target = d0;
        for round in 1..=2u64 {
            let staged_set = [
                staged(1, format!("a{round}").as_bytes()),
                staged(2, format!("b{round}").as_bytes()),
            ];
            target = reference.ingest(round, &staged_set).unwrap();
            let mut entries = Vec::new();
            let mut pairs = Vec::new();
            for s in &staged_set {
                payload_plane.insert(s.hash, s.bytes.clone());
                entries.push(RecordEntry {
                    peer: s.peer,
                    hash: s.hash,
                    size: s.bytes.len() as u64,
                });
                pairs.push((s.peer, s.hash));
            }
            records.push(RoundRecord {
                round,
                set: commit_set(&pairs).commitment(),
                drops: Vec::new(),
                next_seed: Seed([round as u8; 32]),
                set_locator: Locator::StoreKey(format!("r{round}")),
                inline: Some(entries),
            });
        }

        // The late joiner: restore the typed checkpoint, then catch up over the records.
        let mut joiner = built(b"unrelated-config");
        load_typed_checkpoint(&store, &run, &mut joiner, &saved.pointer)
            .await
            .unwrap();
        let steps = steps_from_round_records(&records, |h| payload_plane.get(h).cloned()).unwrap();
        let ckpt_bytes = joiner.checkpoint_save().unwrap();
        let recovered = resync_by_replay(&mut joiner, &ckpt_bytes, &steps).unwrap();
        assert_eq!(recovered, target, "catch-up reaches the in-sync digest");

        // Negatives: a withheld payload and a tampered set commitment are typed errors.
        let mut missing = payload_plane.clone();
        let victim = *missing.keys().next().unwrap();
        missing.remove(&victim);
        assert!(steps_from_round_records(&records, |h| missing.get(h).cloned()).is_err());

        let mut forged = records.clone();
        forged[0].set = commit_set(&[(PeerId([9; 32]), blake3_hash(b"nope"))]).commitment();
        assert!(
            steps_from_round_records(&forged, |h| payload_plane.get(h).cloned()).is_err(),
            "a set commitment that does not recompute is refused"
        );
    }

    #[test]
    fn desync_replay_recovers_the_in_sync_digest() {
        // RUN-7 subset: a peer diverges (wrong ingest), then resyncs from a checkpoint + replays the
        // retained records/payloads → recovers the exact digest the in-sync peer reached (I1).
        let s0 = [staged(1, b"a0"), staged(2, b"b0")];
        let s1 = [staged(1, b"a1"), staged(2, b"b1")];
        let s2 = [staged(1, b"a2"), staged(2, b"b2")];

        // The in-sync reference peer.
        let mut good = built(b"cfg");
        good.ingest(0, &s0).unwrap();
        let checkpoint = good.checkpoint_save().unwrap(); // checkpoint after round 0
        good.ingest(1, &s1).unwrap();
        let target = good.ingest(2, &s2).unwrap();

        // The diverged peer: same round 0, then a *reordered* round-1 set → wrong digest.
        let mut bad = built(b"cfg");
        bad.ingest(0, &s0).unwrap();
        let diverged = bad.ingest(1, &[s1[1].clone(), s1[0].clone()]).unwrap();
        assert_ne!(diverged, target, "the peer has desynced");

        // Resync: reload the round-0 checkpoint, replay rounds 1 and 2 in record order.
        let recovered = resync_by_replay(
            &mut bad,
            &checkpoint,
            &[
                ReplayStep {
                    round: 1,
                    staged: s1.to_vec(),
                },
                ReplayStep {
                    round: 2,
                    staged: s2.to_vec(),
                },
            ],
        )
        .unwrap();
        assert_eq!(recovered, target, "replay recovers the in-sync digest (I1)");
    }
}
