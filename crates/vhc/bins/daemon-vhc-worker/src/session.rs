// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The worker's in-process join — the session run path for a major-2 module under a **genesis
//! envelope v2**: node → worker `JoinRun` → re-admission (§9.4 step 10) → `start_run` (the event
//! pump) → the run's REAL coordinator (`configure_coordinator` derives the spec from
//! the frozen genesis; the pinned, content-addressed `coordinator_quorum` blob runs in-process
//! under the same major-2 event-loop driver) → frames verified + authority-judged ABOVE the pump
//! before delivery. Consensus never runs outside the sandboxed module — the native-tick drive
//! shell this file used to carry is gone with the v1-envelope (device-min admission pre-screen) form, which now refuses
//! typed at assess (`EnvelopeSchemaRetired`).
//!
//! Seams, stated where they live:
//! - **Coordinator configuration**: the genesis coordinator role's pinned module hash + its
//!   verbatim opaque config (`{state: CoordinatorState}` — the host never interprets it) +
//!   the run's declared `AuthorityConfig`, all derived by the production
//!   `daemon_vhc_host::coordinator` seat. Every coordinator decision is
//!   signature-verified AND authority-judged (`authorize_coordinator_frame`) above the pump;
//!   the original signed frame rides as tag-12 evidence.
//! - **In-process identity contract**: every signing key resolves against the node-provided
//!   identity keystore (`daemon_vhc_session::keystore`, referenced by `DAEMON_VHC_IDENTITY_DIR`)
//!   — CSPRNG per-run keys, never a function of any public run value. The genesis author must
//!   name the keystore's coordinator run key as the envelope `SingleKey` coordinator identity
//!   (an authoring fixture reads the peer id out of the same store it hands the worker), or
//!   every coordinator frame refuses at the authority judgment (fail closed, never a fallback).
//!   The worker's per-run key is additionally CERTIFIED by the store's base identity to the
//!   full execution identity, and the certificate is persisted beside the key.
//! - **Guest-authored payloads (B1)**: the guest seals its own committed container and
//!   `payload_put`s it; this session is the async-runtime seat servicing the put, and the
//!   commitment evidence is the guest's own tag-3 voice — verified here to hash exactly the
//!   serviced bytes before it is relayed onto the coordinator plane.
//! - **Batch staging (module-wire kind-0 bytes)**: the plumbing stages REAL token windows cut
//!   from a corpus as the trainer's `[0, round, step, sequences, seq_len, tokens_le]` wrapper —
//!   window arithmetic mirrors the module's own SDK math (proto assignment + the same slicing;
//!   windowing is MODULE policy, the plumbing stages content in the module's training order).
//! - **Round digests**: from the guest's published tag-4 digest frame (`[4, round, digest16]`) —
//!   the guest's own det-lane voice, never a host-side re-derivation.
//! - **Inline replay soak** (refactor §12.6): after the run, the recorded journal is re-driven
//!   through the §8.7 replay engine and every decision must reproduce bit-for-bit — a
//!   diverging run is a join FAILURE, not a warning.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::coordinator::{
    authorize_coordinator_frame, configure_coordinator, Coordinator,
};
use daemon_vhc_host::run::{
    replay, start_run, MemorySink, ReplayEnd, ReplayScript, RunConfig, RunEnd, RunIdentity,
    SinkEntry,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, CertScope, Hash, IrohId, SigningKey,
    StateDigest,
};
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Digest, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass, VhcMessage,
};

use crate::send;
use daemon_provision::CutWriter;
use daemon_vhc_session::identity::certify_existing_key;
use daemon_vhc_session::keystore::VhcKeystore;
use daemon_vhc_session::protocol::Event;

pub(crate) use crate::backend::RUN_INSTANCE;

pub(crate) use crate::backend::derive_grants;

/// How many rounds the self-driven t2 run drives before a clean stop.
const DRIVE_ROUNDS: u64 = 2;

/// Inner steps per round — part of the fixed drive geometry: the authored genesis (the
/// coordinator state's schedule AND the trainer role's config) must match it.
const STEPS_PER_ROUND: u32 = 2;

/// Sequences per staged step batch (one micro-window per inner step; the genesis schedule's
/// `global_batch`/`steps_per_round` fix the counts).
const BATCH_SEQS: u32 = 1;

/// Tokens per sequence — the corpus's `seq_len` and the guest model's.
const SEQ_LEN: u32 = 9;

/// The harness corpus's chunk size in bytes (a multiple of the u16 token width; small enough
/// that every 36-byte shard spans several chunks — the chunk math is exercised, not vacuous).
const CHUNK_SIZE: u64 = 16;

/// The run's **artifact store** — the one unified fetch path, speaking BOTH
/// artifact classes of the corpus contract:
///
/// - **plain** objects (the corpus manifest), keyed + verified by the blake3 of their bytes;
/// - **chunk-addressed** shards, keyed by their chunk FOLD (`daemon_vhc_proto::shard_fold`) —
///   there is no whole-object hash to verify, so reads are verified per covering chunk against
///   the manifest's chunk hashes (exactly what the run pump enforces on `data@2` ranges).
struct HarnessArtifacts {
    plain: std::collections::HashMap<[u8; 32], Vec<u8>>,
    shards: std::collections::HashMap<[u8; 32], Vec<u8>>,
}

impl HarnessArtifacts {
    /// Fetch one committed artifact's bytes: plain objects whole-verified; fold-keyed shards
    /// verbatim (chunk verification happens at the read seam — the staging assembly below and
    /// the pump's own covering-chunk check both re-verify).
    fn fetch(&self, hash: &[u8; 32]) -> Result<Vec<u8>, String> {
        if let Some(bytes) = self.plain.get(hash) {
            if blake3::hash(bytes).as_bytes() != hash {
                return Err("store content does not hash to the committed value".into());
            }
            return Ok(bytes.clone());
        }
        self.shards
            .get(hash)
            .cloned()
            .ok_or_else(|| "artifact not in the harness store".to_string())
    }

    /// The committed hash set — the run's artifact grants (`RunConfig::granted_artifacts`).
    fn granted(&self) -> std::collections::BTreeSet<[u8; 32]> {
        self.plain
            .keys()
            .chain(self.shards.keys())
            .copied()
            .collect()
    }
}

/// Build the run's corpus artifacts under the CHUNK-ADDRESSED corpus contract: deterministic
/// small-vocab shards, each identified by its chunk fold, plus the canonical-CBOR
/// [`daemon_vhc_proto::CorpusManifest`] — the in-process stand-in for a published corpus whose
/// manifest hash a genesis pins. Returns the store and the manifest's committed hash.
///
/// Tokens are embedding indices, so the vocabulary is strictly below the guest model's.
fn harness_corpus_artifacts() -> Result<(HarnessArtifacts, [u8; 32]), String> {
    use daemon_vhc_proto::corpus::{
        CorpusManifest, Endianness, SequenceBoundary, ShardEntry, TokenWidth, TokenizerId,
        CORPUS_MANIFEST_FORMAT,
    };
    const SHARDS: u64 = 2;
    const TOKENS_PER_SHARD: u64 = 2 * SEQ_LEN as u64; // two sequences per shard
    const VOCAB: u64 = 64;
    let mut entries: Vec<ShardEntry> = Vec::new();
    let mut shards = std::collections::HashMap::new();
    for i in 0..SHARDS {
        let mut bytes = Vec::with_capacity(TOKENS_PER_SHARD as usize * 2);
        let mut s = 0xDAE0_7E57u64 ^ i;
        for _ in 0..TOKENS_PER_SHARD {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bytes.extend_from_slice(&(((s >> 33) % VOCAB) as u16).to_le_bytes());
        }
        let entry = CorpusManifest::author_shard(&bytes, TOKENS_PER_SHARD, CHUNK_SIZE)
            .map_err(|e| format!("author shard {i}: {e}"))?;
        shards.insert(entry.shard_hash.0, bytes);
        entries.push(entry);
    }
    // The tokenizer identity is content-addressed like any artifact; the harness corpus is
    // synthetic-token fixture data, so its "tokenizer" is a named fixture object.
    let tokenizer_bytes = b"harness-identity-tokenizer".to_vec();
    let manifest = CorpusManifest {
        format_version: CORPUS_MANIFEST_FORMAT,
        token_width: TokenWidth::U16,
        endianness: Endianness::Little,
        seq_len: SEQ_LEN,
        sequence_boundary: SequenceBoundary::WholeSequencesPerShard,
        eos_id: None,
        pad_id: None,
        chunk_size: CHUNK_SIZE,
        tokenizer: TokenizerId {
            hash: blake3_hash(&tokenizer_bytes),
            name: "harness-identity".into(),
            revision: "fixture".into(),
        },
        total_tokens: SHARDS * TOKENS_PER_SHARD,
        shards: entries,
    };
    let manifest_bytes = manifest
        .to_canonical_bytes()
        .map_err(|e| format!("manifest cbor: {e}"))?;
    let manifest_hash = *blake3::hash(&manifest_bytes).as_bytes();
    let mut plain = std::collections::HashMap::new();
    plain.insert(manifest_hash, manifest_bytes);
    plain.insert(blake3_hash(&tokenizer_bytes).0, tokenizer_bytes);
    Ok((HarnessArtifacts { plain, shards }, manifest_hash))
}

/// The staging corpus, assembled **through the artifact store** (the one fetch path): fetch the
/// canonical-CBOR manifest by its committed hash, fetch every shard by its FOLD identity, and
/// re-verify every chunk against the manifest's chunk hashes (the same covering-chunk integrity
/// rule the pump enforces on `data@2` ranges — never trust a staging layer you didn't have to).
struct StagingCorpus {
    manifest: daemon_vhc_proto::CorpusManifest,
    blobs: Vec<Vec<u8>>,
}

impl StagingCorpus {
    fn via_store(store: &HarnessArtifacts, manifest_hash: &[u8; 32]) -> Result<Self, String> {
        let manifest_bytes = store.fetch(manifest_hash)?;
        let manifest = daemon_vhc_proto::CorpusManifest::from_canonical_bytes(&manifest_bytes)
            .map_err(|e| format!("parse manifest: {e}"))?;
        let mut blobs = Vec::with_capacity(manifest.shards.len());
        for (i, entry) in manifest.shards.iter().enumerate() {
            let bytes = store.fetch(&entry.shard_hash.0)?;
            if bytes.len() as u64 != entry.byte_len {
                return Err(format!("shard {i} byte length lies"));
            }
            for (c, chunk) in bytes
                .chunks(usize::try_from(manifest.chunk_size).unwrap_or(usize::MAX))
                .enumerate()
            {
                if entry.chunk_hashes.get(c).map(|h| h.0) != Some(*blake3::hash(chunk).as_bytes()) {
                    return Err(format!("shard {i} chunk {c} does not verify"));
                }
            }
            blobs.push(bytes);
        }
        Ok(Self { manifest, blobs })
    }

    /// The token ids of sequence `id` (LE u16 extraction over the pinned geometry — the same
    /// windowing arithmetic the module-side SDK owns; the plumbing stages content only).
    fn sequence(&self, id: u64) -> Result<Vec<u32>, String> {
        let seq_len = u64::from(self.manifest.seq_len);
        let total = self.manifest.total_sequences();
        if total == 0 {
            return Err("empty corpus".into());
        }
        let mut seq = id % total;
        for (entry, blob) in self.manifest.shards.iter().zip(&self.blobs) {
            let seqs = entry.token_count / seq_len;
            if seq < seqs {
                let start = (seq * seq_len * 2) as usize;
                return Ok(blob[start..start + (seq_len as usize) * 2]
                    .chunks_exact(2)
                    .map(|c| u32::from(u16::from_le_bytes([c[0], c[1]])))
                    .collect());
            }
            seq -= seqs;
        }
        Err(format!("sequence {id} out of range"))
    }
}

/// The trainer's staged-batch wrapper: `[0, round, step, sequences, seq_len, tokens_le]`
/// (module-wire kind-0 bytes — the promoted trainer's contract).
fn batch_wrapper(round: u64, step: u32, sequences: u32, tokens: &[u32]) -> Result<Vec<u8>, String> {
    let mut le = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        le.extend_from_slice(&t.to_le_bytes());
    }
    let v = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(0u8),
        ciborium::value::Value::from(round),
        ciborium::value::Value::from(step),
        ciborium::value::Value::from(sequences),
        ciborium::value::Value::from(SEQ_LEN),
        ciborium::value::Value::Bytes(le),
    ]);
    to_canonical_vec(&v).map_err(|e| format!("batch wrapper: {e}"))
}

/// The trainer's staged committed-payload wrapper: `[1, round, peer32, payload]`.
fn update_wrapper(round: u64, peer: &[u8; 32], payload: &[u8]) -> Result<Vec<u8>, String> {
    let v = ciborium::value::Value::Array(vec![
        ciborium::value::Value::from(1u8),
        ciborium::value::Value::from(round),
        ciborium::value::Value::Bytes(peer.to_vec()),
        ciborium::value::Value::Bytes(payload.to_vec()),
    ]);
    to_canonical_vec(&v).map_err(|e| format!("update wrapper: {e}"))
}

/// Decode one published frame's module-authored `[tag, round, bytes]` payload.
fn decode_tagged(frame: &[u8]) -> Option<(u64, u64, Vec<u8>)> {
    let v: ciborium::value::Value = ciborium::de::from_reader(frame).ok()?;
    let ciborium::value::Value::Array(parts) = v else {
        return None;
    };
    let ciborium::value::Value::Bytes(payload) = parts.get(1)? else {
        return None;
    };
    let inner: ciborium::value::Value = ciborium::de::from_reader(payload.as_slice()).ok()?;
    let ciborium::value::Value::Array(items) = inner else {
        return None;
    };
    let uint = |i: usize| -> Option<u64> {
        items
            .get(i)
            .and_then(ciborium::value::Value::as_integer)
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
    };
    let bytes = match items.get(2) {
        Some(ciborium::value::Value::Bytes(b)) => b.clone(),
        _ => Vec::new(),
    };
    Some((uint(0)?, uint(1)?, bytes))
}

/// Drive one self-driven v2 run over genesis: the whole-run t2 shape under the run's real wasm
/// coordinator. Reports the guest's final tag-4 det digest.
#[allow(clippy::too_many_lines)]
pub(crate) async fn join_and_run(
    module: &[u8],
    config: &[u8],
    genesis: &crate::backend::GenesisRun,
    run_id: &str,
    writer: &CutWriter,
) -> Result<(), String> {
    let worker = Worker::new(EngineConfig::default()).map_err(|e| format!("engine: {e}"))?;
    let module_hash = *blake3::hash(module).as_bytes();

    // §9.4 step 10: re-run selection (the admission happened at assess; the byte-identity
    // re-check is the hash pin — same module bytes, same config, same derived grants).
    let sel = daemon_vhc_host::select_driver(&worker, module, Some(&module_hash))
        .map_err(|e| format!("join re-selection: {e}"))?;
    if sel.driver != daemon_vhc_abi::CandidateDriver::V2 {
        return Err("join_and_run on a non-major-2 module".into());
    }
    let worker_role_grants = &genesis.env.roles[&genesis.worker_role].grants;
    let grants = derive_grants(&worker, module, worker_role_grants)?;

    // The coordinator configuration, derived from the frozen genesis (the production seat:
    // pinned module hash, verbatim opaque config, declared AuthorityConfig, cryptographic RunId).
    let coord_spec =
        configure_coordinator(&genesis.frozen).map_err(|e| format!("coordinator config: {e}"))?;
    let coordinator_wasm = crate::backend::resolve_coordinator_module(&genesis.env).await?;

    // Identities (module docs "in-process identity contract"): every key resolves against the
    // node-provided keystore — CSPRNG per-run material, never derived from the public run id.
    // The genesis must name the keystore's coordinator run key or authority judgment refuses
    // every frame.
    let keystore = VhcKeystore::from_env().map_err(|e| format!("identity store: {e}"))?;
    let worker_key = keystore
        .run_signing_key(run_id, &genesis.worker_role, RUN_INSTANCE)
        .map_err(|e| format!("worker run key: {e}"))?;
    let worker_key_seed = worker_key.to_bytes();
    let worker_peer = peer_id(&worker_key);
    let coordinator_role = genesis
        .env
        .roles
        .iter()
        .find(|(_, r)| r.lane == "coordinator")
        .map(|(name, _)| name.clone())
        .ok_or("genesis envelope has no role with lane `coordinator`")?;
    let coord_key_seed = keystore
        .run_signing_key(run_id, &coordinator_role, RUN_INSTANCE)
        .map_err(|e| format!("coordinator run key: {e}"))?
        .to_bytes();

    // Certify the worker's per-run key to the store's base identity over the full execution
    // identity, and persist the certificate beside the key (crash recovery re-reads both).
    let base = keystore
        .base_identity()
        .map_err(|e| format!("base identity: {e}"))?;
    let certified = certify_existing_key(
        &base,
        CertScope {
            run_id: coord_spec.run_id,
            epoch: 0,
            role: genesis.worker_role.clone(),
            instance: RUN_INSTANCE,
            module_hash: Hash(module_hash),
        },
        SigningKey::from_bytes(&worker_key_seed),
    )
    .map_err(|e| format!("certify run key: {e}"))?;
    keystore
        .store_run_certificate(run_id, &genesis.worker_role, RUN_INSTANCE, &certified.cert)
        .map_err(|e| format!("persist run certificate: {e}"))?;

    // The execution identity (ABI §8.1): run_id = the genesis hash (the cryptographic RunId),
    // role = the joining worker role from the genesis role set.
    let identity = RunIdentity {
        run_id: coord_spec.run_id.0,
        epoch: 0,
        role: genesis.worker_role.clone(),
        instance: RUN_INSTANCE,
        module: module_hash,
    };
    // The t2 artifact store: ONE fetch path (B2) — the session's staging reads and any guest
    // data@2::fetch are both answered through HarnessArtifacts::fetch; the run's artifact grants are
    // exactly the store's committed hashes (which artifacts a module may touch is a grant).
    let (artifacts, manifest_hash) = harness_corpus_artifacts()?;
    let mut run_cfg = RunConfig::new(
        identity.clone(),
        worker_key_seed,
        config.to_vec(),
        grants.clone(),
    );
    run_cfg.granted_artifacts = artifacts.granted();
    // A real transformer's per-round op stream exceeds the tiny default queue depth (the guest
    // also fences per inner step to reclaim depth).
    run_cfg.compute_queue_depth = 1 << 20;
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, module, run_cfg, Box::new(sink.clone()))
        .map_err(|e| format!("v2 start_run: {e}"))?;
    let pump = run.pump.clone();

    // -- the run's REAL coordinator, in-process under the major-2 driver --------------------
    let coord_role_grants = &genesis.env.roles[&coordinator_role].grants;
    let coord_grants = derive_grants(&worker, &coordinator_wasm, coord_role_grants)?;
    let mut coord = Coordinator::start(
        &coordinator_wasm,
        &coord_spec,
        coord_grants,
        0,
        coord_key_seed,
    )?;

    // Join the roster + the ready heartbeat (the event-driven fast path: the coordinator's
    // synthetic clock advances per frame; no wall-clock manipulation exists on this drive).
    // The transport identity: the keystore's iroh secret — its own key, never the base identity
    // and never a run key (the fixed placeholder id this frame used to carry is gone).
    let iroh_id = IrohId(
        peer_id(
            &keystore
                .iroh_secret()
                .map_err(|e| format!("iroh identity: {e}"))?
                .signing_key(),
        )
        .0,
    );
    coord.deliver(
        &worker_key,
        &VhcMessage::Join(Join {
            run_id: run_id.to_string(),
            iroh_id,
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: None,
        }),
    )?;
    coord.deliver(
        &worker_key,
        &VhcMessage::Heartbeat(Heartbeat {
            round: 0,
            ready: Some(true),
        }),
    )?;

    send(
        writer,
        &Event::RunPhase {
            run_id: run_id.to_string(),
            phase: "train".to_string(),
            epoch: 0,
            round: 0,
            generation: RUN_INSTANCE,
        },
    )
    .await;

    // The corpus the staging path reads — assembled THROUGH the artifact store (one fetch path).
    let corpus = StagingCorpus::via_store(&artifacts, &manifest_hash)?;
    let mut rounds_done = 0u64;
    let mut last_round = 0u64;
    let mut last_digest = [0u8; 16];
    // Per-sender dense delivery seq into the worker pump (§12.2, channel 0).
    let mut coord_seq = 0u64;
    // Sealed bytes captured while servicing the guest's payload_put ops (the async-runtime seat).
    let mut puts: Vec<Vec<u8>> = Vec::new();
    let step_timeout = Duration::from_secs(120);

    while rounds_done < DRIVE_ROUNDS {
        // The coordinator's next authoritative decision, authority-judged above the pump.
        let (sender, evidence, msg) = coord.next_decision(step_timeout)?;
        let (auth_sender, _token) =
            authorize_coordinator_frame(&coord_spec.authority, &evidence)
                .map_err(|e| format!("coordinator frame not authoritative: {e}"))?;
        if auth_sender != sender {
            return Err("coordinator frame sender != authorized signer".into());
        }
        match &msg {
            VhcMessage::RoundOpen(ro) => {
                last_round = ro.round;
                // Corpus-backed staging (module-wire kind-0 bytes): slice the round's batch
                // window exactly as the module's SDK math does (single-peer roster ⇒ assignment
                // yields the whole window), then stage each inner step's REAL tokens from the
                // corpus, in training order.
                let interval =
                    daemon_vhc_session::data::BatchInterval::new(ro.batch.start, ro.batch.end);
                let steps =
                    daemon_vhc_session::data::slice_interval(interval, STEPS_PER_ROUND, BATCH_SEQS)
                        .map_err(|e| format!("t2 window slicing: {e}"))?;
                for (h, step) in steps.iter().enumerate() {
                    let mut tokens = Vec::new();
                    let mut sequences = 0u32;
                    for mb in &step.micro_batches {
                        for id in mb.start..mb.end {
                            tokens.extend(
                                corpus
                                    .sequence(id)
                                    .map_err(|e| format!("corpus sequence {id}: {e}"))?,
                            );
                        }
                        sequences += u32::try_from(mb.end - mb.start).unwrap_or(0);
                    }
                    pump.stage_payload(
                        batch_wrapper(ro.round, h as u32, sequences, &tokens)?,
                        None,
                    )
                    .map_err(|e| format!("stage batch: {e}"))?;
                }
                deliver_to_worker(&pump, sender, &msg, evidence.clone(), &mut coord_seq)?;

                // The guest trains, seals + puts its own container, and voices its tag-3
                // commitment hash: per round, theta (tag 2) + commitment (tag 3) then, after
                // the record, the digest (tag 4) — so publishes reach 3r + 2 here.
                let published = wait_publishes_servicing(
                    &pump,
                    (rounds_done as usize) * 3 + 2,
                    &mut puts,
                    &artifacts,
                )?;
                let sealed = puts
                    .last()
                    .cloned()
                    .ok_or("no payload_put serviced after the round's commit")?;
                // The guest-authored commitment evidence: its tag-3 voice must hash exactly the
                // serviced bytes.
                let guest_hash: [u8; 32] = published
                    .iter()
                    .rev()
                    .find_map(|(_, _, frame)| match decode_tagged(frame) {
                        Some((3, r, bytes)) if r == ro.round => Some(bytes),
                        _ => None,
                    })
                    .ok_or("no tag-3 commitment voice for the round")?
                    .as_slice()
                    .try_into()
                    .map_err(|_| "tag-3 commitment voice is not 32 bytes".to_string())?;
                if guest_hash != blake3_hash(&sealed).0 {
                    return Err(format!(
                        "guest commitment hash {} != staged bytes hash {} (evidence must be \
                         authored over the guest's own sealed bytes)",
                        Hash(guest_hash).to_hex(),
                        blake3_hash(&sealed).to_hex()
                    ));
                }
                let hash = blake3_hash(&sealed);
                let size = sealed.len() as u64;
                // Relay the guest's evidence onto the coordinator plane (worker-signed).
                coord.deliver(
                    &worker_key,
                    &VhcMessage::Commitment(Commitment {
                        round: ro.round,
                        payload: hash,
                        size,
                        locators: Vec::new(),
                    }),
                )?;
                // Availability evidence (§6.4): the storage seat's receipt (any authenticated
                // sender carries it — `on_receipt` consumes content, not signer).
                coord.deliver(
                    &worker_key,
                    &VhcMessage::StorageReceipt(StorageReceipt {
                        round: ro.round,
                        verified: vec![RecordEntry {
                            peer: worker_peer,
                            hash,
                            size,
                        }],
                    }),
                )?;
                // Stage the committed set for the barrier BEFORE the record arrives (§5.11:
                // record-listed order; single peer = the one guest-put payload).
                pump.stage_payload(update_wrapper(ro.round, &worker_peer.0, &sealed)?, None)
                    .map_err(|e| format!("stage update: {e}"))?;
            }
            VhcMessage::RoundRecord(rr) => {
                deliver_to_worker(&pump, sender, &msg, evidence.clone(), &mut coord_seq)?;
                // The guest ingests and voices its round digest (tag 4): publishes reach 3r + 3.
                let published = wait_publishes_servicing(
                    &pump,
                    (rounds_done as usize) * 3 + 3,
                    &mut puts,
                    &artifacts,
                )?;
                let digest_bytes = published
                    .iter()
                    .rev()
                    .find_map(|(_, _, frame)| match decode_tagged(frame) {
                        Some((4, r, bytes)) if r == rr.round => Some(bytes),
                        _ => None,
                    })
                    .ok_or("no tag-4 digest voice for the round")?;
                last_digest = digest_bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| "tag-4 digest is not 16 bytes".to_string())?;
                // Relay the digest to the coordinator (roster liveness/desync accounting).
                coord.deliver(
                    &worker_key,
                    &VhcMessage::Digest(Digest {
                        round: rr.round,
                        digest: StateDigest(last_digest),
                    }),
                )?;
                rounds_done += 1;
            }
            _ => { /* notes/chatter — not part of the closed t2 drive */ }
        }
    }

    // Clean stop for both sandboxes; the guest returns Outcome Ok.
    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .map_err(|e| format!("stop: {e}"))?;
    match run.wait().map_err(|e| format!("guest thread: {e}"))? {
        RunEnd::Outcome(0) => {}
        other => return Err(format!("v2 run ended {other:?}, expected Outcome(0)")),
    }
    match coord.stop()? {
        RunEnd::Outcome(0) => {}
        other => return Err(format!("coordinator ended {other:?}, expected Outcome(0)")),
    }

    // -- inline replay soak (refactor §12.6): the recorded run must re-drive bit-for-bit ---------
    let entries: Vec<SinkEntry> = sink.lock().expect("sink").entries.clone();
    let mut script = ReplayScript::from_entries(&entries);
    // The identity behind the recorded run: what `sys@2::rng_seed` re-derives from at replay
    // (in a real journal this rides the tag-0 run header).
    script.identity = Some(identity);
    let replayed = replay(&worker, module, config, &grants, script)
        .map_err(|e| format!("replay harness: {e}"))?;
    if replayed.end != ReplayEnd::Outcome(0) {
        return Err(format!("input replay diverged: {:?}", replayed.end));
    }
    let recorded: Vec<(u64, u64, [u8; 32])> = entries
        .iter()
        .filter_map(|e| match e {
            SinkEntry::Publish {
                channel,
                seq,
                payload_hash,
                ..
            } => Some((*channel, *seq, *payload_hash)),
            _ => None,
        })
        .collect();
    let redriven: Vec<(u64, u64, [u8; 32])> = replayed
        .decisions
        .iter()
        .map(|d| (d.channel, d.seq, d.payload_hash))
        .collect();
    if recorded != redriven {
        return Err(format!(
            "input replay diverged: {} recorded vs {} replayed decisions",
            recorded.len(),
            redriven.len()
        ));
    }
    send(
        writer,
        &Event::Metric {
            name: "replay_decisions".to_string(),
            value: recorded.len() as f64,
        },
    )
    .await;

    send(
        writer,
        &Event::RoundOutcome {
            round: last_round,
            committed: 1,
            ingested: 1,
            stalled: false,
            digest: last_digest,
            generation: RUN_INSTANCE,
        },
    )
    .await;
    Ok(())
}

/// Deliver one authority-judged coordinator decision into the worker pump: the decoded control
/// message's canonical bytes as the payload, the coordinator's ORIGINAL §12.1 signed frame as
/// tag-12 evidence, per-sender dense seq (§12.2 discipline, channel 0).
fn deliver_to_worker(
    pump: &daemon_vhc_host::run::PumpHandle,
    sender: [u8; 32],
    msg: &VhcMessage,
    evidence: Vec<u8>,
    seq: &mut u64,
) -> Result<(), String> {
    let payload = to_canonical_vec(msg).map_err(|e| format!("payload encode: {e}"))?;
    let verdict = pump
        .deliver_frame(0, *seq, sender, payload, evidence)
        .map_err(|e| format!("deliver: {e}"))?;
    match verdict {
        daemon_vhc_host::run::DeliverVerdict::Accepted => {
            *seq += 1;
            Ok(())
        }
        other => Err(format!(
            "coordinator frame back-pressured/refused ({other:?}) — the t2 drive never fills \
             the spool"
        )),
    }
}

/// Wait (bounded) until the pump has published at least `target` frames in total, servicing the
/// guest's ops meanwhile (the async-runtime seat): `payload_put`s' sealed bytes are captured
/// into `puts` (the barrier's staging + commitment-evidence input), and `data.fetch`s are
/// answered from the run's artifact store — the SAME `HarnessArtifacts::fetch` the session's own
/// corpus staging reads through (one fetch path; the pump re-verifies + range-slices).
fn wait_publishes_servicing(
    pump: &daemon_vhc_host::run::PumpHandle,
    target: usize,
    puts: &mut Vec<Vec<u8>>,
    artifacts: &HarnessArtifacts,
) -> Result<Vec<(u64, u64, Vec<u8>)>, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        for (op, request) in pump.take_op_requests() {
            match request {
                daemon_vhc_host::run::OpRequest::PayloadPut { bytes } => {
                    puts.push(bytes.to_vec());
                    pump.complete_op(op, daemon_vhc_host::run::OpOutcome::PutDone)
                        .map_err(|e| format!("put completion: {e}"))?;
                }
                daemon_vhc_host::run::OpRequest::ArtifactFetch { hash, .. }
                | daemon_vhc_host::run::OpRequest::ArtifactRange { hash, .. } => {
                    // The unified fetch path: the same store + verification discipline the
                    // staging corpus was assembled through. Whole-object answers serve both
                    // classes — the pump verifies (whole hash, or the registered covering
                    // chunks) and slices; the store is untrusted either way.
                    let _ = match artifacts.fetch(&hash) {
                        Ok(artifact) => pump
                            .complete_op(
                                op,
                                daemon_vhc_host::run::OpOutcome::FetchDone { artifact },
                            )
                            .map_err(|e| format!("fetch completion: {e}"))?,
                        Err(detail) => pump
                            .complete_op(
                                op,
                                daemon_vhc_host::run::OpOutcome::Failed {
                                    code: daemon_vhc_abi::COMP_ERR_STORE_REFUSED,
                                    detail,
                                },
                            )
                            .map_err(|e| format!("fetch failure completion: {e}"))?,
                    };
                }
                other => return Err(format!("unexpected op request from the guest: {other:?}")),
            }
        }
        let published = pump.published();
        if published.len() >= target {
            return Ok(published);
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out waiting for publishes ({} < {target})",
                published.len()
            ));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
