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
//! - **Plumbing-owned payloads** (the outbound sealing gap, resolved by B1 `payload_put`): the
//!   guest's `Commitment` frame is its voice, but the payload the barrier stages is the
//!   PLUMBING-sealed container (the v1 host-sealing relocated to the v2 commit seam), so the
//!   commitment evidence fed to the coordinator is plumbing-authored over those sealed bytes.
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

use daemon_vhc_coordinator::{tick, CoordinatorState, Input, Output};
use daemon_vhc_host::v2::{
    replay_v2, start_run, MemorySink, ReplayEnd, ReplayScript, RunEnd, RunIdentity, SinkEntry,
    V2RunConfig,
};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::messages::{Commitment, Join, StorageReceipt, ThroughputClass};
use daemon_vhc_proto::{
    blake3_hash, digest_state, peer_id, to_canonical_vec, CapabilitySet, Envelope, IrohId, Seed,
    SignedMessage, SigningKey, SwarmMessage,
};

use crate::send;
use daemon_provision::CutWriter;
use daemon_vhc_session::protocol::Event;

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

/// Build the t2 run's corpus: deterministic small-vocab shards assembled into the **same
/// `Corpus` the live fetch path yields** (`live.rs::build_corpus` — manifest + shards fetched by
/// hash, blake3-verified, windowed). The in-process t2 run has no store to fetch from, so it
/// assembles the fetch path's product deterministically; `Corpus::from_parts` re-runs the same
/// per-shard integrity verification a fetch would.
///
/// Tokens are embedding indices, so the vocabulary is strictly below the guest model's
/// (`TinyLlamaCfg::default().vocab` = 64).
fn t2_corpus() -> Result<daemon_vhc_session::data::Corpus, String> {
    use daemon_vhc_session::data::{Manifest, ShardDesc, TokenWidth};
    const SHARDS: u64 = 2;
    const TOKENS_PER_SHARD: u64 = 2 * T2_SEQ_LEN as u64; // two sequences per shard
    const VOCAB: u64 = 64;
    let mut shards = Vec::new();
    let mut blobs = Vec::new();
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
        blobs.push(bytes);
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
    daemon_vhc_session::data::Corpus::from_parts(manifest, blobs)
        .map_err(|e| format!("t2 corpus: {e}"))
}

/// Drive one self-driven v2 run: the whole-run t2 shape. Returns the final det-lane digest.
#[allow(clippy::too_many_lines)]
pub(crate) async fn join_and_run_v2(
    module: &[u8],
    config: &[u8],
    envelope: &Envelope,
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

    let identity = RunIdentity {
        run_id: *blake3::hash(run_id.as_bytes()).as_bytes(),
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: module_hash,
    };
    let run_cfg = V2RunConfig::new(identity.clone(), worker_key_seed, config.to_vec(), grants);
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, module, run_cfg, Box::new(sink.clone()))
        .map_err(|e| format!("v2 start_run: {e}"))?;
    let pump = run.pump.clone();

    // -- the native coordinator, in-process: the pure tick over the run's frozen envelope --------
    let params = daemon_vhc_coordinator::CoordinatorParams {
        seq_len: u64::from(T2_SEQ_LEN),
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let config_c = daemon_vhc_coordinator::RunConfig::from_envelope(envelope, params)
        .map_err(|e| format!("coordinator config: {e}"))?;
    let envelope_hash = config_c.envelope_hash;
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
    now_s += u64::from(envelope.phases.warmup) + 1;
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

    // The corpus the staging path reads (the fetch-path product — see `t2_corpus`).
    let corpus = t2_corpus()?;
    let mut rounds_done = 0u64;
    let mut last_round = 0u64;
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
                    envelope.data.steps_per_round,
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
                // The guest trains and voices its Commitment: per round, one Commitment then
                // one Digest — so after round r's open, publishes reach 2r + 1.
                wait_publishes(&pump, (rounds_done as usize) * 2 + 1)?;
                // Plumbing-owned sealing (module docs): the committed payload is the sealed
                // container; the coordinator's evidence is authored over THOSE bytes.
                let (_, sealed) = pump
                    .take_sealed_update()
                    .ok_or("no sealed container after commit (bridge module expected)")?;
                let hash = blake3_hash(&sealed);
                let commitment = SwarmMessage::Commitment(Commitment {
                    round: ro.round,
                    payload: hash,
                    size: sealed.len() as u64,
                    locators: Vec::new(),
                });
                let signed = SignedMessage::sign(
                    &worker_key,
                    daemon_vhc_proto::SWARM_PROTO_VERSION,
                    commitment,
                )
                .map_err(|e| format!("commitment sign: {e}"))?;
                feed(&mut state, Input::Message(signed), &mut outbox)?;
                // Availability evidence (§6.4): the coordinator-as-storage-client receipt.
                let receipt = SwarmMessage::StorageReceipt(StorageReceipt {
                    round: ro.round,
                    verified: vec![daemon_vhc_proto::messages::RecordEntry {
                        peer: worker_peer,
                        hash,
                        size: sealed.len() as u64,
                    }],
                });
                let signed =
                    SignedMessage::sign(&coord_key, daemon_vhc_proto::SWARM_PROTO_VERSION, receipt)
                        .map_err(|e| format!("receipt sign: {e}"))?;
                feed(&mut state, Input::Message(signed), &mut outbox)?;
                // Clock forward so the round closes into a record.
                now_s += u64::from(envelope.phases.round_train_max) + 1;
                feed(&mut state, Input::Clock(now_s), &mut outbox)?;
                // Stage the committed set for the barrier BEFORE the record arrives (§5.11:
                // record-listed order; single peer = the one sealed payload).
                pump.stage_update(sealed, None)
                    .map_err(|e| format!("stage update: {e}"))?;
            }
            SwarmMessage::RoundRecord(_) => {
                deliver_coordinator_msg(&pump, &coord_key, &msg)?;
                // The guest ingests and voices its round digest: publishes reach 2(r + 1).
                wait_publishes(&pump, (rounds_done as usize) * 2 + 2)?;
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
    pump.deliver_frame(0, seq, peer_id(coord_key).0, payload, evidence)
        .map_err(|e| format!("deliver: {e}"))
}

/// Coordinator-plane delivery seq (per-process monotone; the §12.2 dense-seq discipline for the
/// coordinator sender on channel 0).
fn next_coord_seq() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Wait (bounded) until the pump has published at least `target` frames in total.
fn wait_publishes(
    pump: &daemon_vhc_host::v2::PumpHandle,
    target: usize,
) -> Result<Vec<(u64, u64, Vec<u8>)>, String> {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
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
