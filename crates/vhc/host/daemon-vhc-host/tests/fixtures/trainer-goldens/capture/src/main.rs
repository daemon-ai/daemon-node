// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Capture the **native trainer goldens** as a content-addressed recorded fixture — the
//! successor drift oracle that lets the recorded v1 parity oracle retire (retirement plan §3).
//!
//! The recorded lane is the compute@2 trainer guest (`tiny-llama`): a real Burn LLaMA over
//! `Autodiff<HostBackend>`, det-lane ingest in-guest (`daemon-vhc-sdk-profiles::SparseLoco`),
//! `BarrierRound` choreography, kind-0 byte staging. The capture drives it through a single-peer
//! barrier whole-run and records, per round:
//!
//! - **trained theta** (the guest's tag-2 publish — the tolerance-class comparison surface),
//! - **the committed payload bytes** (reconstructed natively — see below — and cross-checked
//!   against the guest's tag-3 commitment hash),
//! - **the post-ingest det digest** (the guest's tag-4 publish — the equality-class oracle).
//!
//! plus the matched init, the exact model/profile config literals, and the schedule.
//!
//! ## Why the payload is reconstructed, not read off the wire
//!
//! The guest voices the blake3 HASH of its committed payload on the control channel (tag-3) and
//! externalizes the sealed bytes through `payload_put` (this capture acknowledges the put and
//! drops the bytes). To record the trainer's OWN committed payload bytes we recompute them
//! natively with the identical det-lane profile math (`SparseLoco::make_update` over the guest's
//! published theta and the running round base) and assert the reconstruction's blake3 equals the
//! guest's tag-3 hash. `daemon-vhc-det` is bit-identical wasm-vs-native (the det-lane
//! invariant), so the reconstruction is byte-exact — and the double derivation (native rebuild
//! hash-checked against the guest's voice) makes any drift a hard capture failure rather than a
//! silent divergence.
//!
//! The run is a genuine single-peer barrier: the trainer commits its own update and ingests that
//! same committed set, so the recorded digests are the trainer's autonomous native trajectory
//! (no v1 inputs feed it). The matched init was originally inherited from the v1 oracle bundle;
//! that oracle retired with the v1 parity lanes (retirement plan §3), so the init now lives IN this
//! bundle (`init.f32le.bin`) and the capture is self-contained (see ../README.md for the historical
//! provenance chain).
//!
//! ## Reproducibility
//!
//! `main` runs the whole capture TWICE and asserts every recorded byte is identical before it
//! writes anything. The determinism is total: fixed init, fixed token schedule, the ndarray
//! compute lane, and bit-exact det math.
//!
//! Dev/test harness: shells `cargo build` for the guests, so fs/process use is expected here.

// This is a fixture recorder, not product code: it reads and rewrites the golden bundle in the
// checkout it lives in, so the product's contained-root filesystem discipline does not apply (the
// same allowance the wasm-backed harness suites carry).
#![allow(clippy::disallowed_methods)]

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ciborium::value::Value;
use daemon_vhc_host::run::{start_run, DeliverVerdict, MemorySink, RunConfig, RunEnd, RunIdentity};
use daemon_vhc_host::{EngineConfig, Worker};
use daemon_vhc_proto::det_state::{
    derive_state_chunk_size, DetStateManifest, FamilyEntry, LayoutBinding,
    DET_STATE_MANIFEST_FORMAT,
};
use daemon_vhc_proto::genesis::{StateContract, StateInit};
use daemon_vhc_proto::merkle::commit_set;
use daemon_vhc_proto::{blake3_hash, to_canonical_vec, Hash, PeerId, Seed};
use daemon_vhc_sdk_consensus::messages::{
    BatchWindow, Locator, RecordEntry, RoundOpen, RoundRecord, VhcMessage,
};
use daemon_vhc_sdk_profiles::payload::PayloadLayout;
use daemon_vhc_sdk_profiles::{IngestParam, ParamView, SparseLoco, SparseLocoCfg};
use serde::{Deserialize, Serialize};

// The pinned parity shape (the trainer_parity harness shape): 1 layer, seq 9, 2 rounds x 2 inner steps,
// micro-batch 2, vocab 64, single-peer roster. Small, deterministic, fast — the frozen-pin shape
// the v1 oracle and the det-equality equality proof both use.
const ROUNDS: u64 = 2;
const STEPS_PER_ROUND: u32 = 2;
const MICRO_BATCH: u32 = 2;
const SEQ_LEN: u32 = 9;
const VOCAB: u32 = 64;
const PEER: [u8; 32] = [7u8; 32];

/// A flat, serde mirror of the guest's `ModelCfg` (the field set the guest deserializes). Kept in
/// this crate so the capture pulls no Burn dependency; the guest's `from_flat` asserts the init
/// length against its own `param_numels`, so a drifted field set fails the capture loudly.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCfgLit {
    d_model: u32,
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    vocab: u32,
    seq_len: u32,
    ffn_mult: u32,
    rope_theta: f64,
    rmsnorm_eps: f64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    adam_eps: f64,
    wd: f64,
}

// -- guest build (the shared builder every wasm-backed harness goes through) -----------------------

fn build_and_read_guest(name: &str) -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm(name)
}

// -- schedule + wire shapes (the trainer module contract) ---------------------------------------------

/// Deterministic varied tokens for `(round, step)` — identical on every path (the trainer_parity
/// schedule, so the goldens sit on the same trajectory the det-equality equality proof pins).
fn tokens_for(round: u64, step: u32) -> Vec<u32> {
    let n = u64::from(MICRO_BATCH * SEQ_LEN);
    (0..n)
        .map(|i| {
            let x = i + 1_000 * u64::from(step) + 100_000 * round + 1;
            (x.wrapping_mul(2_654_435_761) % u64::from(VOCAB)) as u32
        })
        .collect()
}

/// A staged batch wrapper: `[0, round, step, sequences, seq_len, tokens_le]`.
fn batch_wrapper(round: u64, step: u32, tokens: &[u32]) -> Vec<u8> {
    let mut le = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        le.extend_from_slice(&t.to_le_bytes());
    }
    let v = Value::Array(vec![
        Value::from(0u8),
        Value::from(round),
        Value::from(step),
        Value::from(MICRO_BATCH),
        Value::from(SEQ_LEN),
        Value::Bytes(le),
    ]);
    to_canonical_vec(&v).expect("batch wrapper")
}

/// One published frame's `[tag, round, bytes]` decoded.
fn decode_publish(frame: &[u8]) -> Option<(u64, u64, Vec<u8>)> {
    let v: Value = ciborium::de::from_reader(frame).ok()?;
    let Value::Array(parts) = v else { return None };
    let Value::Bytes(payload) = parts.get(1)? else {
        return None;
    };
    let inner: Value = ciborium::de::from_reader(payload.as_slice()).ok()?;
    let Value::Array(items) = inner else {
        return None;
    };
    let uint = |i: usize| -> Option<u64> {
        items
            .get(i)
            .and_then(Value::as_integer)
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(u64::MAX))
    };
    let bytes = match items.get(2) {
        Some(Value::Bytes(b)) => b.clone(),
        _ => Vec::new(),
    };
    Some((uint(0)?, uint(1)?, bytes))
}

fn split_params(flat: &[f32], numels: &[usize]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(numels.len());
    let mut off = 0;
    for &n in numels {
        out.push(flat[off..off + n].to_vec());
        off += n;
    }
    assert_eq!(off, flat.len(), "flat buffer matches the recorded numels");
    out
}

fn flat_le_bytes(params: &[Vec<f32>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(params.iter().map(Vec::len).sum::<usize>() * 4);
    for p in params {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn theta_from_le(bytes: &[u8], numels: &[usize]) -> Vec<Vec<f32>> {
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    split_params(&flat, numels)
}

/// The guest config map (canonical CBOR — the trainer `GuestCfg`). Init is the genesis **state
/// contract** (the run-pinned `state_chunk_size` + the artifact-form init pin the guest fetches and
/// self-seals), not inline f32s — canonical state is chunk-addressed host-side.
fn guest_cfg_bytes(
    model: &ModelCfgLit,
    profile: &SparseLocoCfg,
    chunk_size: u64,
    manifest_hash: [u8; 32],
) -> Vec<u8> {
    let contract = StateContract {
        chunk_size,
        init: StateInit::Manifest {
            manifest: Hash(manifest_hash),
        },
    };
    let map = Value::Map(vec![
        (
            Value::Text("model".into()),
            Value::serialized(model).expect("model cfg"),
        ),
        (Value::Text("peer".into()), Value::Bytes(PEER.to_vec())),
        (
            Value::Text("roster".into()),
            Value::Array(vec![Value::Bytes(PEER.to_vec())]),
        ),
        (
            Value::Text("steps_per_round".into()),
            Value::from(STEPS_PER_ROUND),
        ),
        (Value::Text("micro_batch".into()), Value::from(MICRO_BATCH)),
        (Value::Text("stall_rounds_max".into()), Value::from(2u32)),
        (
            Value::Text("profile".into()),
            Value::serialized(profile).expect("profile cfg"),
        ),
        (
            Value::Text("state".into()),
            Value::serialized(&contract).expect("state contract"),
        ),
    ]);
    to_canonical_vec(&map).expect("guest cfg")
}

// -- the capture --------------------------------------------------------------------------------

struct Captured {
    /// Per-round trained theta (tag-2), split by the canonical layout.
    trained: Vec<Vec<Vec<f32>>>,
    /// Per-round committed payload bytes (native reconstruction, hash-verified vs tag-3).
    payloads: Vec<Vec<u8>>,
    /// Per-round post-ingest det digests (tag-4).
    digests: Vec<[u8; 16]>,
}

impl PartialEq for Captured {
    fn eq(&self, other: &Self) -> bool {
        self.trained == other.trained
            && self.payloads == other.payloads
            && self.digests == other.digests
    }
}

fn wait_published(pump: &daemon_vhc_host::run::PumpHandle, n: usize, serve: &ServeCtx) {
    let deadline = Instant::now() + Duration::from_secs(180);
    while pump.published().len() < n {
        service_ops(pump, serve);
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {n} publishes (have {}); logs: {:?}",
            pump.published().len(),
            pump.logs()
        );
        std::thread::sleep(Duration::from_millis(5));
    }
    service_ops(pump, serve);
}

/// What the embedder must serve the streamed trainer: the artifact-form init boot (the pinned
/// det-state manifest whole, and the master fold's ranges sliced length-aware from the flat init
/// image), and the content-addressed committed payloads the guest fetches back at each barrier.
struct ServeCtx {
    manifest_hash: [u8; 32],
    manifest_bytes: Vec<u8>,
    master_fold: [u8; 32],
    init_flat: Vec<u8>,
    /// The fetchable committed containers (`blake3 → bytes`) and the `payload_get` ops waiting on
    /// one. [SF-R3]: a committed payload reaches the guest as a host buffer it range-reads, so the
    /// capture records the production path, never a staging shortcut.
    committed: Mutex<CommittedArchive>,
}

#[derive(Default)]
struct CommittedArchive {
    by_hash: std::collections::HashMap<[u8; 32], Vec<u8>>,
    deferred: Vec<(u64, [u8; 32])>,
}

impl ServeCtx {
    /// Author the artifact-form init from the matched init at `chunk_size`.
    fn new(init: &[Vec<f32>], chunk_size: u64) -> Self {
        let param_bytes: Vec<Vec<u8>> = init
            .iter()
            .map(|p| flat_le_bytes(std::slice::from_ref(p)))
            .collect();
        let views: Vec<&[u8]> = param_bytes.iter().map(Vec::as_slice).collect();
        let master = FamilyEntry::author(&views, chunk_size).expect("author master family");
        let numels_u64: Vec<u64> = init.iter().map(|p| p.len() as u64).collect();
        let manifest = DetStateManifest {
            format: DET_STATE_MANIFEST_FORMAT,
            run_id: Hash([0x67; 32]),
            round: 0,
            layout: LayoutBinding::of_numels(&numels_u64).expect("layout binding"),
            chunk_size,
            families: [("master".to_string(), master.clone())]
                .into_iter()
                .collect(),
        };
        let manifest_bytes = manifest.to_canonical_bytes().expect("manifest cbor");
        Self {
            manifest_hash: blake3_hash(&manifest_bytes).0,
            manifest_bytes,
            master_fold: master.fold.0,
            init_flat: param_bytes.concat(),
            committed: Mutex::new(CommittedArchive::default()),
        }
    }

    /// Make a committed container fetchable (content-addressed, as the record names it).
    fn publish_committed(&self, payload: &[u8]) {
        self.committed
            .lock()
            .expect("archive")
            .by_hash
            .insert(blake3_hash(payload).0, payload.to_vec());
    }
}

/// The embedder seat's duties: acknowledge the trainer's `payload_put` (the capture reconstructs
/// the payload natively and verifies it against the tag-3 hash, so the put's bytes are dropped),
/// serve the artifact-form init boot, and serve every record-listed `payload_get` out of the
/// content-addressed archive.
fn service_ops(pump: &daemon_vhc_host::run::PumpHandle, serve: &ServeCtx) {
    use daemon_vhc_host::run::{OpOutcome, OpRequest};
    for (op, request) in pump.take_op_requests() {
        match request {
            OpRequest::PayloadPut { .. } => {
                pump.complete_op(op, OpOutcome::PutDone).expect("put done");
            }
            OpRequest::PayloadGet { hash } => {
                serve
                    .committed
                    .lock()
                    .expect("archive")
                    .deferred
                    .push((op, hash));
            }
            OpRequest::ArtifactFetch { hash, .. } if hash == serve.manifest_hash => {
                pump.complete_op(
                    op,
                    OpOutcome::FetchDone {
                        artifact: serve.manifest_bytes.clone(),
                    },
                )
                .expect("manifest fetch done");
            }
            OpRequest::ArtifactRange {
                hash,
                span_off,
                span_len,
                ..
            } if hash == serve.master_fold => {
                let (s, e) = (span_off as usize, (span_off + span_len) as usize);
                pump.complete_op(
                    op,
                    OpOutcome::RangeDone {
                        bytes: serve.init_flat[s..e].to_vec(),
                    },
                )
                .expect("init range done");
            }
            other => panic!("unexpected op request from the trainer guest: {other:?}"),
        }
    }
    let ready: Vec<(u64, Vec<u8>)> = {
        let mut archive = serve.committed.lock().expect("archive");
        let mut pending = Vec::new();
        let mut ready = Vec::new();
        for (op, hash) in std::mem::take(&mut archive.deferred) {
            match archive.by_hash.get(&hash) {
                Some(bytes) => ready.push((op, bytes.clone())),
                None => pending.push((op, hash)),
            }
        }
        archive.deferred = pending;
        ready
    };
    for (op, bytes) in ready {
        pump.complete_op(op, OpOutcome::GetDone { bytes })
            .expect("committed payload get done");
    }
}

/// Drive the trainer guest through the single-peer barrier whole-run once, recording the goldens.
fn capture_once(
    wasm: &[u8],
    model: &ModelCfgLit,
    profile_cfg: &SparseLocoCfg,
    numels: &[usize],
    init: &[Vec<f32>],
) -> Captured {
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = daemon_vhc_host::select_driver(&worker, wasm, Some(blake3::hash(wasm).as_bytes()))
        .expect("trainer guest admitted");
    assert_eq!(sel.driver, daemon_vhc_abi::CandidateDriver::V2);
    assert_eq!(
        (sel.major, sel.minor),
        (2, daemon_vhc_abi::BUFFER_STAGE_MINOR_V2),
        "the trainer guest imports the det-state write surface and the incremental buffer staging \
         its committed container is built through"
    );

    // The fold-walk window / state chunk size is derived from the profile compression chunk
    // (§3.2), exactly as the reproduction tiers derive it from the recorded profile literals.
    let chunk_size = derive_state_chunk_size(u64::from(profile_cfg.chunk));
    let serve = ServeCtx::new(init, chunk_size);

    let identity = RunIdentity {
        run_id: [0x67; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(wasm).as_bytes(),
    };
    let mut run_cfg = RunConfig::new(
        identity,
        [0x9d; 32],
        guest_cfg_bytes(model, profile_cfg, chunk_size, serve.manifest_hash),
        Vec::new(),
    );
    // A real transformer's per-round op stream exceeds the tiny default queue depth (the guest
    // also fences per inner step to reclaim depth) — the trainer_parity setting.
    run_cfg.compute_queue_depth = 1 << 20;
    run_cfg.state_chunk_size = chunk_size;
    run_cfg.granted_artifacts.insert(serve.manifest_hash);
    run_cfg.granted_artifacts.insert(serve.master_fold);
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();
    let mut seq = 0u64;
    let sender = [9u8; 32];

    let deliver = |msg: &VhcMessage, seq: &mut u64| {
        let payload = to_canonical_vec(msg).expect("msg");
        assert_eq!(
            pump.deliver_frame(0, *seq, sender, payload.clone(), payload)
                .expect("deliver"),
            DeliverVerdict::Accepted
        );
        *seq += 1;
    };

    // The native det-lane mirror: the SAME profile math the guest runs in-guest, used only to
    // reconstruct the guest's own committed payload bytes (verified against the tag-3 hash) and
    // to advance the round base in lockstep with the guest's ingest.
    let mut profile = SparseLoco::new(profile_cfg.clone(), numels);
    let mut master = init.to_vec();
    let mut round_base = init.to_vec();
    // The container layout the guest emits its committed update through: the definition bridge
    // between the resident reference sections above and the range-addressable bytes on the wire
    // (`PayloadLayout::encode_sections`). Reconstructing THROUGH it is what makes the recorded
    // payload bytes the guest's own.
    let layout = PayloadLayout::new(profile_cfg, numels, chunk_size).expect("payload layout");

    let mut trained: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut payloads: Vec<Vec<u8>> = Vec::new();

    for round in 0..ROUNDS {
        for h in 0..STEPS_PER_ROUND {
            pump.stage_payload(batch_wrapper(round, h, &tokens_for(round, h)), None)
                .expect("stage batch");
        }
        // RoundOpen → train; the guest walks fence -> export -> publishes theta (tag 2) and its
        // commitment voice (tag 3).
        deliver(
            &VhcMessage::RoundOpen(RoundOpen {
                round,
                seed: Seed([round as u8; 32]),
                roster_digest: Hash([0; 32]),
                batch: BatchWindow {
                    start: 0,
                    end: u64::from(STEPS_PER_ROUND * MICRO_BATCH),
                },
                deadline_unix_s: 0,
            }),
            &mut seq,
        );
        wait_published(&pump, (round as usize) * 3 + 2, &serve); // + theta + commitment

        // Recover this round's theta (tag 2) and the guest's commitment hash (tag 3).
        let published = pump.published();
        let mut theta: Option<Vec<Vec<f32>>> = None;
        let mut guest_hash: Option<[u8; 32]> = None;
        for (_, _, frame) in &published {
            let Some((tag, r, bytes)) = decode_publish(frame) else {
                continue;
            };
            if r != round {
                continue;
            }
            match tag {
                2 => theta = Some(theta_from_le(&bytes, numels)),
                3 => guest_hash = Some(bytes.as_slice().try_into().expect("hash32")),
                _ => {}
            }
        }
        let theta = theta.expect("theta published this round");
        let guest_hash = guest_hash.expect("commitment published this round");

        // Reconstruct the guest's committed payload natively and verify byte-identity via the hash.
        let views: Vec<ParamView<'_>> = theta
            .iter()
            .zip(round_base.iter())
            .map(|(t, b)| ParamView {
                theta: t,
                round_base: b,
            })
            .collect();
        let sections = profile.make_update(&views);
        let payload = layout
            .encode_sections(&sections)
            .expect("the reference sections tile the container layout");
        assert_eq!(
            blake3_hash(&payload).0,
            guest_hash,
            "round {round}: the natively reconstructed payload must hash-match the guest's tag-3 \
             commitment (the det lane is bit-identical wasm-vs-native)"
        );

        // Self-ingest: publish the trainer's OWN committed container to the archive it fetches
        // from, then the single-peer record.
        let entry = RecordEntry {
            peer: PeerId(PEER),
            hash: blake3_hash(&payload),
            size: payload.len() as u64,
        };
        serve.publish_committed(&payload);
        let set: Vec<(PeerId, Hash)> = vec![(PeerId(PEER), entry.hash)];
        deliver(
            &VhcMessage::RoundRecord(RoundRecord {
                round,
                set: commit_set(&set).commitment(),
                drops: Vec::new(),
                next_seed: Seed([0; 32]),
                set_locator: Locator::StoreKey(String::new()),
                inline: Some(vec![entry]),
            }),
            &mut seq,
        );
        wait_published(&pump, (round as usize) * 3 + 3, &serve); // + digest

        // Advance the native mirror exactly as the guest's ingest does (rebase over the same
        // committed set), so the next round's base matches the guest's.
        {
            let mut params: Vec<IngestParam<'_>> = master
                .iter_mut()
                .zip(round_base.iter())
                .map(|(m, b)| IngestParam {
                    master: m,
                    round_base: b,
                })
                .collect();
            profile
                .ingest(&mut params, std::slice::from_ref(&sections))
                .expect("ingest");
        }
        round_base = master.clone();

        trained.push(theta);
        payloads.push(payload);
    }

    pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
        .expect("stop");
    match run.wait().expect("guest thread clean") {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }

    // Collect the per-round digests (tag 4).
    let mut digests: Vec<[u8; 16]> = Vec::new();
    for (_, _, frame) in pump.published() {
        if let Some((4, _r, bytes)) = decode_publish(&frame) {
            digests.push(bytes.as_slice().try_into().expect("digest16"));
        }
    }
    assert_eq!(trained.len() as u64, ROUNDS, "one theta per round");
    assert_eq!(payloads.len() as u64, ROUNDS, "one payload per round");
    assert_eq!(digests.len() as u64, ROUNDS, "one digest per round");
    Captured {
        trained,
        payloads,
        digests,
    }
}

// -- bundle writing (content-addressed, the v1-oracle manifest shape) ----------------------------

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn write_file(root: &Path, rel: &str, bytes: &[u8]) -> serde_json::Value {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
    std::fs::write(&path, bytes).expect("write fixture file");
    serde_json::json!({
        "file": rel,
        "blake3": hex(blake3_hash(bytes).as_bytes()),
        "bytes": bytes.len(),
    })
}

fn fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fixture root")
        .to_path_buf()
}

/// The matched init + config literals — carried IN this bundle (`init.f32le.bin` + the
/// `model_cfg`/`profile_cfg`/`param_*` fields of `expected.json`). They were originally inherited
/// from the recorded v1 parity oracle; that oracle retired with the v1 parity lanes (retirement
/// plan §3), so the bundle is now self-contained and a re-capture regenerates the native
/// trajectory from the bundle's OWN frozen init (idempotent — the init is written back byte-
/// identically). See ../README.md for the (now historical) provenance chain.
fn read_matched_init_and_config() -> (
    ModelCfgLit,
    SparseLocoCfg,
    Vec<usize>,
    Vec<String>,
    Vec<Vec<f32>>,
) {
    let root = fixture_root();
    let expected: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("expected.json")).expect("read"))
            .expect("trainer-goldens expected.json (the bundle carries its own matched init)");
    let model: ModelCfgLit =
        serde_json::from_value(expected["model_cfg"].clone()).expect("model cfg literals");
    let profile: SparseLocoCfg =
        serde_json::from_value(expected["profile_cfg"].clone()).expect("profile cfg literals");
    let numels: Vec<usize> = expected["param_numels"]
        .as_array()
        .expect("numels")
        .iter()
        .map(|n| usize::try_from(n.as_u64().expect("numel")).expect("usize"))
        .collect();
    let names: Vec<String> = expected["param_names"]
        .as_array()
        .expect("names")
        .iter()
        .map(|n| n.as_str().expect("name").to_string())
        .collect();
    let init_rel = expected["init"]["file"].as_str().expect("init file");
    let init_bytes = std::fs::read(root.join(init_rel)).expect("read matched init");
    let init_flat: Vec<f32> = init_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let init = split_params(&init_flat, &numels);
    (model, profile, numels, names, init)
}

fn main() {
    let root = fixture_root();
    let commit = std::env::var("CAPTURE_COMMIT").unwrap_or_else(|_| "UNSET".into());

    // Matched init + config literals are carried in this bundle (originally inherited from the now-
    // retired v1 oracle — see read_matched_init_and_config); the trained trajectory, payloads, and
    // digests are the compute@2 trainer's OWN native output.
    let (model, profile_cfg, numels, names, init) = read_matched_init_and_config();
    assert_eq!(
        numels.iter().sum::<usize>(),
        init.iter().map(Vec::len).sum::<usize>(),
        "recorded numels match the init length"
    );

    let wasm = build_and_read_guest("tiny_llama");

    // Capture twice and prove byte-identity before writing anything.
    let first = capture_once(&wasm, &model, &profile_cfg, &numels, &init);
    let second = capture_once(&wasm, &model, &profile_cfg, &numels, &init);
    assert!(
        first == second,
        "double-capture is not byte-identical — the trainer lane is not reproducible; \
         STOP and investigate (do NOT record a non-reproducible golden)"
    );
    let cap = first;

    // -- write the content-addressed bundle ------------------------------------------------------
    let init_entry = write_file(&root, "init.f32le.bin", &flat_le_bytes(&init));
    let trained_entries: Vec<serde_json::Value> = cap
        .trained
        .iter()
        .enumerate()
        .map(|(r, t)| {
            write_file(
                &root,
                &format!("trained-round-{r}.f32le.bin"),
                &flat_le_bytes(t),
            )
        })
        .collect();
    let payload_entries: Vec<serde_json::Value> = cap
        .payloads
        .iter()
        .enumerate()
        .map(|(r, p)| write_file(&root, &format!("payload-round-{r}.bin"), p))
        .collect();

    let expected = serde_json::json!({
        "captured_from": {
            "commit": commit,
            "tree": "vhc/trainer-goldens (the current tree — the live compute@2 trainer lane)",
            "trainer": "compute@2 tiny-llama (real Burn LLaMA over Autodiff<HostBackend>, \
                        SparseLoco det-lane ingest in-guest, BarrierRound, kind-0 staging)",
            "compute_lane": "host ndarray ComputeRunner (the compute@2 host execution backend at \
                             this commit; EngineConfig.backend does not select the compute@2 tier)",
            "capture": "tests/fixtures/trainer-goldens/capture (see ../README.md)",
        },
        "provenance": "HISTORICAL (the v1 parity oracle retired with the v1 parity lanes, \
                       retirement plan §3): v1 parity oracle (matched init + config literals) -> \
                       trainer_parity det-equality green at the original capture commit -> these goldens. The \
                       matched init now lives in this self-contained bundle (init.f32le.bin). See \
                       ../README.md for the full historical chain and the exact comparison surface.",
        "module": {
            "name": "tiny_llama.wasm",
            "blake3": hex(blake3_hash(&wasm).as_bytes()),
            "bytes": wasm.len(),
            "source": "guests/target build at capture (byte-identical across checkout paths via \
                       the guests workspace rustc remap; pinned in guests.blake3 at the commit)",
        },
        "schedule": {
            "rounds": ROUNDS,
            "steps_per_round": STEPS_PER_ROUND,
            "micro_batch": MICRO_BATCH,
            "seq_len": SEQ_LEN,
            "vocab": VOCAB,
            "peer": "[7; 32]",
            "roster": "single-peer (self-ingest barrier)",
            "tokens": "token[i] of (round, step) = ((i + 1000*step + 100000*round + 1) \
                       * 2654435761) % 64",
        },
        "model_cfg": model,
        "profile_cfg": profile_cfg,
        "param_names": names,
        "param_numels": numels,
        "init": init_entry,
        "trained": trained_entries,
        "payloads": payload_entries,
        "digests": cap.digests.iter().map(|d| hex(d)).collect::<Vec<_>>(),
    });
    let json = serde_json::to_string_pretty(&expected).expect("expected json");
    std::fs::write(root.join("expected.json"), json + "\n").expect("write expected.json");

    println!(
        "captured native trainer goldens into {} (module {} … digests {})",
        root.display(),
        &hex(blake3_hash(&wasm).as_bytes())[..16],
        cap.digests
            .iter()
            .map(|d| hex(d))
            .collect::<Vec<_>>()
            .join(", "),
    );
    println!("double-capture reproducibility: byte-identical across two runs");
}
