// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The worker's v2 join — the session run path for a major-2 module (refactor §5 A2, the last
//! A2 remainder): node → worker `JoinRun` → re-admission (§9.4 step 10) → `start_run` (the
//! event pump) → the NATIVE coordinator (`daemon-vhc-coordinator`'s pure `tick`, driven
//! in-process exactly as the session's `LocalCoordinator` shells it) opening rounds over the
//! run's frozen envelope → frames verified ABOVE the pump before delivery → the D3 cell-5/6
//! whole-run shape: a v2 worker module under a v1 envelope + the native coordinator.
//!
//! Phase-A seams, stated where they live:
//! - **Guest-authored payloads (the B1 sealing-gap retirement)**: the guest seals its own
//!   container (`read_back` → `create_from`) and `payload_put`s it; this session is the
//!   async-runtime seat servicing the put (capturing the bytes), and the commitment evidence fed
//!   to the coordinator is decoded FROM THE GUEST'S OWN Commitment frame — verified here to hash
//!   exactly the serviced bytes (commitment hash ≡ staged bytes) before it is relayed.
//! - **Coordinator-plane verification**: the coordinator speaks v1 `SignedMessage`s; each is
//!   signature-verified here, above the pump, and the pre-verified payload is delivered with
//!   the original signed bytes as tag-12 evidence. (Worker-plane §12.1 v2 frames verify through
//!   `daemon_vhc_session::v2_attach` — the same seam, different wire.)
//! - **Batch staging (corpus-backed since B2)**: the plumbing stages REAL token windows cut
//!   from a corpus — the same `Corpus` shape the live fetch path (`live.rs::build_corpus`,
//!   fetch-by-hash + blake3 verify + content cache) yields; the in-process t2 run assembles it
//!   deterministically instead of fetching. Window arithmetic mirrors the module's own SDK math
//!   (proto assignment + the same slicing — windowing is MODULE policy, the plumbing stages
//!   content in the module's training order); the kind-1 staging encoding is unchanged.
//! - **Inline replay soak** (refactor §12.6): after the run, the recorded journal is re-driven
//!   through the §8.7 replay engine and every decision must reproduce bit-for-bit — a
//!   diverging run is a join FAILURE, not a warning.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_vhc_host::v2::{
    replay_v2, start_run, MemorySink, ReplayEnd, ReplayScript, RunEnd, RunIdentity, SinkEntry,
    V2RunConfig,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::messages::{Join, StorageReceipt, ThroughputClass};
use daemon_vhc_proto::{
    blake3_hash, digest_state, peer_id, to_canonical_vec, CapabilitySet, Envelope, IrohId, Seed,
    SignedMessage, SigningKey, SwarmMessage,
};
use daemon_vhc_sdk_consensus::coordinator::{tick, CoordinatorState, Input, Output};

use crate::backend::GenesisRun;
use crate::send;
use daemon_provision::CutWriter;
use daemon_vhc_session::protocol::Event;

/// Which envelope form configures the v2 run (D1 deliverable 4): the v1 frozen envelope
/// (mixed-fleet cell 5 — the A2 shape) or the envelope-v2 genesis (cell 6, through the
/// transitional native-coordinator adapter that retires at D2).
pub(crate) enum V2RunSource<'a> {
    /// The v1 envelope: coordinator config from `[data]`/`[phases]` (`RunConfig::from_envelope`).
    V1(&'a Envelope),
    /// The genesis envelope: coordinator config from the coordinator role's opaque config via the
    /// cell-6 adapter (`RunConfig::from_genesis`); role grants tighten the run quotas.
    Genesis(&'a GenesisRun),
}

/// The Phase-A derived grants document (§2.6 stand-in until node-side lane config lands): the
/// admitted channel table + the worlds the Phase-A driver links. Deterministic, so assess and
/// join derive byte-identical copies (§9.4 steps 8/11 hash pinning).
pub(crate) fn derive_grants() -> Vec<u8> {
    let channels: Vec<ciborium::value::Value> = daemon_vhc_abi::PHASE_A_DEFAULT_CHANNEL_TABLE
        .iter()
        .map(|c| ciborium::value::Value::from(u64::from(c.id)))
        .collect();
    let worlds = ["vhc@2", "net@2", "sys@2", "tabi@1"]
        .iter()
        .map(|w| ciborium::value::Value::from(*w))
        .collect();
    let doc = ciborium::value::Value::Map(vec![
        (
            ciborium::value::Value::from("channels"),
            ciborium::value::Value::Array(channels),
        ),
        (
            ciborium::value::Value::from("worlds"),
            ciborium::value::Value::Array(worlds),
        ),
    ]);
    let mut b = Vec::new();
    ciborium::into_writer(&doc, &mut b).expect("grants cbor");
    b
}

/// How many rounds the self-driven t2 run drives before a clean stop.
const T2_ROUNDS: u64 = 2;

/// Sequences per staged batch (one micro-window per inner step; the envelope's
/// `global_batch`/`steps_per_round` schedule fixes the counts).
const T2_BATCH_SEQS: u32 = 1;

/// Tokens per sequence — the corpus's `seq_len` and the guest model's (`TinyLlamaCfg.seq_len`).
const T2_SEQ_LEN: u32 = 9;

/// Decode a 64-char lowercase-hex blake3 digest.
fn hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

/// The t2 run's **artifact store** — the one fetch path (B2 tier-d unification). Every corpus
/// byte the session stages AND every guest `data@2::fetch` is answered from here, through
/// [`T2Artifacts::fetch`]'s single fetch-and-verify discipline: content addressed by the
/// committed blake3, verified on every read (exactly what `live.rs::fetch_cached` + the
/// resolver do against a real store — this is the in-process seat of the same mechanism, not a
/// second path).
struct T2Artifacts {
    by_hash: std::collections::HashMap<[u8; 32], Vec<u8>>,
}

impl T2Artifacts {
    /// Fetch-and-verify one committed artifact (the whole-artifact rule the `data@2` pump also
    /// enforces — a store lie is a typed error, never silent bytes).
    fn fetch(&self, hash: &[u8; 32]) -> Result<Vec<u8>, String> {
        let bytes = self
            .by_hash
            .get(hash)
            .ok_or_else(|| "artifact not in the t2 store".to_string())?;
        if blake3::hash(bytes).as_bytes() != hash {
            return Err("t2 store content does not hash to the committed value".into());
        }
        Ok(bytes.clone())
    }

    /// The committed hash set — the run's artifact grants (`V2RunConfig::granted_artifacts`).
    fn granted(&self) -> std::collections::BTreeSet<[u8; 32]> {
        self.by_hash.keys().copied().collect()
    }
}

/// Build the t2 run's corpus artifacts: deterministic small-vocab shards + their manifest, as
/// content-addressed blobs — the in-process stand-in for the envelope's artifact map (the live
/// path fetches the same shapes by hash through `live.rs::build_corpus`). Returns the store and
/// the manifest's committed hash.
///
/// Tokens are embedding indices, so the vocabulary is strictly below the guest model's
/// (`TinyLlamaCfg::default().vocab` = 64).
fn t2_corpus_artifacts() -> Result<(T2Artifacts, [u8; 32]), String> {
    use daemon_vhc_session::data::{Manifest, ShardDesc, TokenWidth};
    const SHARDS: u64 = 2;
    const TOKENS_PER_SHARD: u64 = 2 * T2_SEQ_LEN as u64; // two sequences per shard
    const VOCAB: u64 = 64;
    let mut shards = Vec::new();
    let mut by_hash = std::collections::HashMap::new();
    for i in 0..SHARDS {
        let mut bytes = Vec::with_capacity(TOKENS_PER_SHARD as usize * 2);
        let mut s = 0xDAE0_7E57u64 ^ i;
        for _ in 0..TOKENS_PER_SHARD {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bytes.extend_from_slice(&(((s >> 33) % VOCAB) as u16).to_le_bytes());
        }
        shards.push(ShardDesc {
            name: format!("shard-{i:04}.bin"),
            bytes: bytes.len() as u64,
            tokens: TOKENS_PER_SHARD,
            blake3: blake3_hash(&bytes).to_hex(),
        });
        by_hash.insert(*blake3::hash(&bytes).as_bytes(), bytes);
    }
    let manifest = Manifest {
        token_width: TokenWidth::U16,
        seq_len: T2_SEQ_LEN,
        shards,
        tokenizer: None,
        tokenizer_revision: None,
        dataset: None,
        dataset_revision: None,
    };
    let manifest_bytes = manifest
        .to_json()
        .map_err(|e| format!("manifest json: {e}"))?
        .into_bytes();
    let manifest_hash = *blake3::hash(&manifest_bytes).as_bytes();
    by_hash.insert(manifest_hash, manifest_bytes);
    Ok((T2Artifacts { by_hash }, manifest_hash))
}

/// Assemble the staging corpus **through the artifact store** (the one fetch path): fetch the
/// manifest by its committed hash, parse it, fetch every shard by its committed hash, and let
/// `Corpus::from_parts` re-run the per-shard integrity verification — byte-for-byte the
/// `build_corpus` shape over the unified store.
fn t2_corpus_via_store(
    store: &T2Artifacts,
    manifest_hash: &[u8; 32],
) -> Result<daemon_vhc_session::data::Corpus, String> {
    let manifest_bytes = store.fetch(manifest_hash)?;
    let manifest = daemon_vhc_session::data::Manifest::from_json(
        std::str::from_utf8(&manifest_bytes).map_err(|e| format!("manifest utf8: {e}"))?,
    )
    .map_err(|e| format!("parse manifest: {e}"))?;
    let mut blobs = Vec::with_capacity(manifest.shards.len());
    for desc in &manifest.shards {
        let hash = hex32(&desc.blake3)
            .ok_or_else(|| format!("shard hash `{}` is not 64 hex chars", desc.blake3))?;
        blobs.push(store.fetch(&hash)?);
    }
    daemon_vhc_session::data::Corpus::from_parts(manifest, blobs)
        .map_err(|e| format!("t2 corpus: {e}"))
}

/// Drive one self-driven v2 run: the whole-run t2 shape. Returns the final det-lane digest.
#[allow(clippy::too_many_lines)]
pub(crate) async fn join_and_run_v2(
    module: &[u8],
    config: &[u8],
    source: &V2RunSource<'_>,
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
        return Err("join_and_run_v2 on a non-major-2 module".into());
    }
    let grants = derive_grants();

    // Identities: the worker peer signs its frames; the coordinator signs its records.
    let worker_key_seed = *blake3::hash(format!("vhc-worker/{run_id}").as_bytes()).as_bytes();
    let coord_key = SigningKey::from_bytes(
        blake3::hash(format!("vhc-coordinator/{run_id}").as_bytes()).as_bytes(),
    );
    let worker_key = SigningKey::from_bytes(&worker_key_seed);
    let worker_peer = peer_id(&worker_key);

    // The execution identity (ABI §8.1): on the genesis path the run_id IS the genesis hash (the
    // cryptographic RunId) and the role is the envelope's worker-role label; the v1 path keeps the
    // A2 label-derived stand-in.
    let identity = match source {
        V2RunSource::V1(_) => RunIdentity {
            run_id: *blake3::hash(run_id.as_bytes()).as_bytes(),
            epoch: 0,
            role: "trainer".to_string(),
            instance: 1,
            module: module_hash,
        },
        V2RunSource::Genesis(g) => RunIdentity {
            run_id: g.run_id,
            epoch: 0,
            role: g.worker_role.clone(),
            instance: 1,
            module: module_hash,
        },
    };
    // The t2 artifact store: ONE fetch path (B2) — the session's staging reads and any guest
    // data@2::fetch are both answered through T2Artifacts::fetch; the run's artifact grants are
    // exactly the store's committed hashes (which artifacts a module may touch is a grant).
    let (artifacts, manifest_hash) = t2_corpus_artifacts()?;
    let mut run_cfg = V2RunConfig::new(identity.clone(), worker_key_seed, config.to_vec(), grants);
    // Genesis path (D1 deliverable 4): tighten the run quotas from the worker role's envelope
    // grants ∩ the selected lane — the SAME derivation assess ran (byte-identical inputs), so
    // assess and join agree by construction (§9.4 step 10).
    if let V2RunSource::Genesis(g) = source {
        let role = g
            .env
            .roles
            .get(&g.worker_role)
            .ok_or("worker role absent from the genesis role set at join")?;
        let run_artifacts = g.env.artifacts.values().map(|a| a.blake3).collect();
        let quotas = daemon_vhc_proto::derive_admitted_quotas(
            &role.grants,
            &crate::backend::selected_lane().ceilings,
            &run_artifacts,
        )
        .map_err(|e| format!("join grants derivation: {e}"))?;
        daemon_vhc_host::v2::apply_admitted_quotas(&quotas, &mut run_cfg);
    }
    // The t2 corpus stands in for the run's artifact map on this in-process drive: the staged
    // shards/manifest are generated here, so the fetch allow-list is the store's committed hashes
    // (the envelope-derived set names the real artifact map the live path fetches instead).
    run_cfg.granted_artifacts = artifacts.granted();
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, module, run_cfg, Box::new(sink.clone()))
        .map_err(|e| format!("v2 start_run: {e}"))?;
    let pump = run.pump.clone();

    // -- the native coordinator, in-process: the pure tick over the run's envelope ----------------
    let params = daemon_vhc_sdk_consensus::coordinator::CoordinatorParams {
        seq_len: u64::from(T2_SEQ_LEN),
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    // The coordinator's projected config: `from_envelope` on the v1 path (cell 5), the D1
    // transitional cell-6 adapter `from_genesis` on the envelope-v2 path (retired at D2, when the
    // wasm coordinator supersedes the native tick).
    let config_c = match source {
        V2RunSource::V1(envelope) => {
            daemon_vhc_sdk_consensus::coordinator::RunConfig::from_envelope(envelope, params)
        }
        V2RunSource::Genesis(g) => daemon_vhc_sdk_consensus::coordinator::RunConfig::from_genesis(
            &g.env,
            &g.coordinator_role,
            params,
        ),
    }
    .map_err(|e| format!("coordinator config: {e}"))?;
    let envelope_hash = config_c.envelope_hash;
    // The drive's pacing knobs, read from the PROJECTED config so both envelope forms pace
    // identically (these came verbatim from `[data]`/`[phases]` on v1 and from the coordinator
    // role's opaque config on genesis).
    let warmup_s = config_c.warmup_s;
    let round_train_max_s = config_c.round_train_max_s;
    let steps_per_round = config_c.steps_per_round;
    let mut state = CoordinatorState::new(config_c, Seed([0x33; 32]), 0);
    let mut now_s = 0u64;
    let mut outbox: Vec<SwarmMessage> = Vec::new();
    let feed = |state: &mut CoordinatorState,
                input: Input,
                outbox: &mut Vec<SwarmMessage>|
     -> Result<(), String> {
        let (next, outputs) = tick(state.clone(), input);
        *state = next;
        for o in outputs {
            match o {
                Output::Publish(msg) => outbox.push(*msg),
                Output::Reject(r) => return Err(format!("coordinator rejected input: {r:?}")),
                Output::Note(_) => {}
            }
        }
        Ok(())
    };

    // Join the roster (the coordinator's real admission path: envelope hash + capabilities).
    let join = SwarmMessage::Join(Join {
        run_id: run_id.to_string(),
        iroh_id: IrohId([0x44; 32]),
        class: ThroughputClass::C1,
        capabilities: CapabilitySet::new(),
        envelope_hash: Some(envelope_hash),
    });
    let signed = SignedMessage::sign(&worker_key, daemon_vhc_proto::SWARM_PROTO_VERSION, join)
        .map_err(|e| format!("join sign: {e}"))?;
    feed(&mut state, Input::Message(signed), &mut outbox)?;

    // Clock past warmup so round 0 opens.
    now_s += warmup_s + 1;
    feed(&mut state, Input::Clock(now_s), &mut outbox)?;

    send(
        writer,
        &Event::RunPhase {
            run_id: run_id.to_string(),
            phase: "train".to_string(),
            epoch: 0,
            round: 0,
        },
    )
    .await;

    // The corpus the staging path reads — assembled THROUGH the artifact store (one fetch path).
    let corpus = t2_corpus_via_store(&artifacts, &manifest_hash)?;
    let mut rounds_done = 0u64;
    let mut last_round = 0u64;
    // Sealed bytes captured while servicing the guest's payload_put ops (the async-runtime seat).
    let mut puts: Vec<Vec<u8>> = Vec::new();
    while rounds_done < T2_ROUNDS {
        // The coordinator's next authoritative message.
        let Some(msg) = outbox.first().cloned() else {
            // Nothing pending: advance the clock (timeout-driven transitions — warmup, the
            // round cadence). Bounded so a wedged coordinator is a typed join failure.
            let mut ticks = 0u32;
            while outbox.is_empty() {
                now_s += 1;
                feed(&mut state, Input::Clock(now_s), &mut outbox)?;
                ticks += 1;
                if ticks > 10_000 {
                    return Err("coordinator went quiet before the run finished".into());
                }
            }
            continue;
        };
        outbox.remove(0);
        match &msg {
            SwarmMessage::RoundOpen(ro) => {
                last_round = ro.round;
                // Corpus-backed staging (B2): slice the round's batch window exactly as the
                // module's SDK math does (single-peer roster ⇒ assignment yields the whole
                // window), then stage each micro-window's REAL tokens from the corpus, in
                // training order — content is mechanism, the window arithmetic is the module's
                // policy mirrored (the bridge-era contract; kind-1 encoding unchanged).
                let interval =
                    daemon_vhc_session::data::BatchInterval::new(ro.batch.start, ro.batch.end);
                let steps = daemon_vhc_session::data::slice_interval(
                    interval,
                    steps_per_round,
                    T2_BATCH_SEQS,
                )
                .map_err(|e| format!("t2 window slicing: {e}"))?;
                for step in &steps {
                    for mb in &step.micro_batches {
                        let mut tokens = Vec::new();
                        for id in mb.start..mb.end {
                            tokens.extend(
                                corpus
                                    .sequence(id)
                                    .map_err(|e| format!("corpus sequence {id}: {e}"))?,
                            );
                        }
                        pump.stage_batch(&tokens, (mb.end - mb.start) as u32, T2_SEQ_LEN, None)
                            .map_err(|e| format!("stage batch: {e}"))?;
                    }
                }
                deliver_coordinator_msg(&pump, &coord_key, &msg)?;
                // The guest trains, seals its own container, payload_puts it, and voices its
                // Commitment over the completion hash: per round, one Commitment then one
                // Digest — so after round r's open, publishes reach 2r + 1. This session is the
                // async-runtime seat: it services the put, capturing the sealed bytes.
                let published = wait_publishes_servicing(
                    &pump,
                    (rounds_done as usize) * 2 + 1,
                    &mut puts,
                    &artifacts,
                )?;
                let sealed = puts
                    .last()
                    .cloned()
                    .ok_or("no payload_put serviced after commit (bridge module expected)")?;
                // The GUEST-authored commitment (the B1 sealing-gap retirement): decode the
                // guest's own frame, and verify its evidence hashes exactly the serviced bytes.
                let commitment_frame = &published[(rounds_done as usize) * 2].2;
                let SwarmMessage::Commitment(commitment) = decode_v2_payload(commitment_frame)?
                else {
                    return Err("publish 2r is not the guest's Commitment".into());
                };
                if commitment.payload != blake3_hash(&sealed) {
                    return Err(format!(
                        "guest commitment hash {} != staged bytes hash {} (evidence must be \
                         authored over the guest's own sealed bytes)",
                        commitment.payload.to_hex(),
                        blake3_hash(&sealed).to_hex()
                    ));
                }
                if commitment.round != ro.round {
                    return Err("guest commitment names the wrong round".into());
                }
                let hash = commitment.payload;
                let size = commitment.size;
                // Relay the guest's evidence onto the v1 coordinator plane (the proto-plane
                // shim, exactly as deliver_coordinator_msg shims the other direction).
                let signed = SignedMessage::sign(
                    &worker_key,
                    daemon_vhc_proto::SWARM_PROTO_VERSION,
                    SwarmMessage::Commitment(commitment),
                )
                .map_err(|e| format!("commitment sign: {e}"))?;
                feed(&mut state, Input::Message(signed), &mut outbox)?;
                // Availability evidence (§6.4): the coordinator-as-storage-client receipt.
                let receipt = SwarmMessage::StorageReceipt(StorageReceipt {
                    round: ro.round,
                    verified: vec![daemon_vhc_proto::messages::RecordEntry {
                        peer: worker_peer,
                        hash,
                        size,
                    }],
                });
                let signed =
                    SignedMessage::sign(&coord_key, daemon_vhc_proto::SWARM_PROTO_VERSION, receipt)
                        .map_err(|e| format!("receipt sign: {e}"))?;
                feed(&mut state, Input::Message(signed), &mut outbox)?;
                // Clock forward so the round closes into a record.
                now_s += round_train_max_s + 1;
                feed(&mut state, Input::Clock(now_s), &mut outbox)?;
                // Stage the committed set for the barrier BEFORE the record arrives (§5.11:
                // record-listed order; single peer = the one guest-put payload).
                pump.stage_update(sealed, None)
                    .map_err(|e| format!("stage update: {e}"))?;
            }
            SwarmMessage::RoundRecord(_) => {
                deliver_coordinator_msg(&pump, &coord_key, &msg)?;
                // The guest ingests and voices its round digest: publishes reach 2(r + 1).
                wait_publishes_servicing(
                    &pump,
                    (rounds_done as usize) * 2 + 2,
                    &mut puts,
                    &artifacts,
                )?;
                rounds_done += 1;
            }
            _ => { /* Digest/witness chatter — not part of the Phase-A closed drive */ }
        }
    }

    // Clean stop; the guest returns Outcome Ok and the final canonical state exports.
    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .map_err(|e| format!("stop: {e}"))?;
    match run.wait().map_err(|e| format!("guest thread: {e}"))? {
        RunEnd::Outcome(0) => {}
        other => return Err(format!("v2 run ended {other:?}, expected Outcome(0)")),
    }
    let final_state = pump
        .bridge_final_state()
        .ok_or("no final bridge state exported")?;
    let mut seed = [0u8; 32];
    seed[..8].copy_from_slice(&last_round.to_le_bytes());
    let digest = digest_state(&Seed(seed), 64, u32::MAX, &final_state);

    // -- inline replay soak (refactor §12.6): the recorded run must re-drive bit-for-bit ---------
    let entries: Vec<SinkEntry> = sink.lock().expect("sink").entries.clone();
    let mut script = ReplayScript::from_entries(&entries);
    // The identity behind the recorded run: what `sys@2::rng_seed` re-derives from at replay
    // (in a real journal this rides the tag-0 run header).
    script.identity = Some(identity);
    let replayed = replay_v2(&worker, module, config, &derive_grants(), script)
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
            digest: *digest.as_bytes(),
        },
    )
    .await;
    Ok(())
}

/// Verify a coordinator `SignedMessage` ABOVE the pump, then deliver the pre-verified payload
/// with the original signed bytes as tag-12 evidence (the pump's delivery contract).
fn deliver_coordinator_msg(
    pump: &daemon_vhc_host::v2::PumpHandle,
    coord_key: &SigningKey,
    msg: &SwarmMessage,
) -> Result<(), String> {
    // tick emits UNSIGNED messages; the shell signs them (LocalCoordinator's contract). Sign,
    // then verify exactly as a remote worker would before delivery — the seam under test.
    let signed = SignedMessage::sign(
        coord_key,
        daemon_vhc_proto::SWARM_PROTO_VERSION,
        msg.clone(),
    )
    .map_err(|e| format!("coordinator sign: {e}"))?;
    signed
        .verify()
        .map_err(|e| format!("coordinator frame REFUSED above the pump: {e}"))?;
    let payload = to_canonical_vec(msg).map_err(|e| format!("payload encode: {e}"))?;
    let evidence = to_canonical_vec(&signed).map_err(|e| format!("evidence encode: {e}"))?;
    let seq = next_coord_seq();
    match pump
        .deliver_frame(0, seq, peer_id(coord_key).0, payload, evidence)
        .map_err(|e| format!("deliver: {e}"))?
    {
        daemon_vhc_host::v2::DeliverVerdict::Accepted => Ok(()),
        other => Err(format!(
            "coordinator frame back-pressured/refused ({other:?}) — the t2 drive never fills \
             the spool"
        )),
    }
}

/// Coordinator-plane delivery seq (per-process monotone; the §12.2 dense-seq discipline for the
/// coordinator sender on channel 0).
fn next_coord_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Wait (bounded) until the pump has published at least `target` frames in total, servicing the
/// guest's ops meanwhile (the async-runtime seat): `payload_put`s' sealed bytes are captured
/// into `puts` (the barrier's staging + commitment-evidence input), and `data.fetch`s are
/// answered from the run's artifact store — the SAME `T2Artifacts::fetch` the session's own
/// corpus staging reads through (one fetch path; the pump re-verifies + range-slices).
fn wait_publishes_servicing(
    pump: &daemon_vhc_host::v2::PumpHandle,
    target: usize,
    puts: &mut Vec<Vec<u8>>,
    artifacts: &T2Artifacts,
) -> Result<Vec<(u64, u64, Vec<u8>)>, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        for (op, request) in pump.take_op_requests() {
            match request {
                daemon_vhc_host::v2::OpRequest::PayloadPut { bytes } => {
                    puts.push(bytes.to_vec());
                    pump.complete_op(op, daemon_vhc_host::v2::OpOutcome::PutDone)
                        .map_err(|e| format!("put completion: {e}"))?;
                }
                daemon_vhc_host::v2::OpRequest::ArtifactFetch { hash, .. } => {
                    // The unified fetch path: the same store + verification discipline the
                    // staging corpus was assembled through (the range is the pump's job).
                    // (The minted handle is the transport's concern; fetch needs no routing.)
                    let _ = match artifacts.fetch(&hash) {
                        Ok(artifact) => pump
                            .complete_op(op, daemon_vhc_host::v2::OpOutcome::FetchDone { artifact })
                            .map_err(|e| format!("fetch completion: {e}"))?,
                        Err(detail) => pump
                            .complete_op(
                                op,
                                daemon_vhc_host::v2::OpOutcome::Failed {
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

/// Decode the module-authored payload of a §12.1 signed frame `[envelope, payload, sig]` as a
/// `SwarmMessage` (the guest's voice on the control channel).
fn decode_v2_payload(frame: &[u8]) -> Result<SwarmMessage, String> {
    let v: ciborium::value::Value =
        ciborium::de::from_reader(frame).map_err(|e| format!("frame cbor: {e}"))?;
    let ciborium::value::Value::Array(parts) = v else {
        return Err("frame is not [envelope, payload, sig]".into());
    };
    let Some(ciborium::value::Value::Bytes(payload)) = parts.get(1) else {
        return Err("frame payload is not bytes".into());
    };
    daemon_vhc_proto::from_canonical_slice(payload).map_err(|e| format!("payload decode: {e}"))
}
