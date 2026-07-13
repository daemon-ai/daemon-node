// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The flagship MVP scenario, compressed for CI: 3 in-process `WasmBackend` peers (real host training
// over the tiny-llama `.wasm`) driven by the **real** `daemon_swarm_coordinator::tick` loop, one run
// per comm profile (sparse_loco / diloco / demo). Each round every peer trains H inner steps on its
// own data slice, seals a payload, and ingests the record-ordered committed set; the coordinator
// admits the roster, opens rounds, and finalizes each from the peers' signed commitments +
// coordinator storage-receipt evidence (§6.4 I6). We assert:
//
//   * all three peers' post-ingest digests are **bit-identical every round** (the MVP consensus
//     claim — data-parallel peers reconverge because every profile rebases to the round base and
//     applies the canonical aggregate);
//   * the digest transcript **evolves** (the swarm is learning, not frozen) and the tiny-llama
//     **loss decreases** over the run (the host reverse-mode autodiff, HOST-9, actually learns);
//   * the coordinator produced one `RoundRecord` per round and `daemon_swarm_observe::replay`
//     re-derives them byte-for-byte from the recorded `tick` input trace (PROTO-20) — the replay
//     oracle, now over a **wasm-backed** run.
//
// The `daemon-train-worker` **binary** path is exercised separately (real spawn + envelope seam) in
// `daemon-train/tests/worker_protocol.rs`; here the peers run the WasmBackend in-process so the
// per-profile matrix stays CI-light (no subprocess per peer).
//
// The guest `.wasm` is located via `SWARM_TEST_GUEST_DIR` else built on demand; this is a dev/test
// harness that shells `cargo build` for the guests, so the fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use blake3::hash as blake3_hash_raw;
use ciborium::into_writer;

use daemon_swarm_coordinator::{
    tick, CoordinatorParams, CoordinatorState, Input, Output, RunConfig,
};
use daemon_swarm_observe::{genesis_seed, replay};
use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition, ENVELOPE_SCHEMA_MAJOR,
};
use daemon_swarm_proto::messages::{
    Commitment, Digest, Join, Locator, RecordEntry, RoundRecord, StorageReceipt, ThroughputClass,
};
use daemon_swarm_proto::{
    peer_id, CapabilitySet, Hash, IrohId, PeerId, SignedMessage, SigningKey, StateDigest,
    SwarmMessage, SwarmProtoVersion, SWARM_PROTO_VERSION,
};
use daemon_swarm_run::backend::{
    BatchRef, StagedPayload, StateDigest as RunDigest, StepCtx, TrainerBackend,
};
use daemon_train::{EngineConfig, WasmBackend, WasmBackendConfig};
use daemon_train_sdk::models::TinyLlamaCfg;

// -- guest module loading (mirrors daemon-train/tests) ------------------------------------------

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("SWARM_TEST_GUEST_DIR") {
        return PathBuf::from(dir);
    }
    guests_root().join("target/wasm32-unknown-unknown/release")
}

static BUILD: Once = Once::new();

fn ensure_built() {
    BUILD.call_once(|| {
        if std::env::var("SWARM_TEST_GUEST_DIR").is_ok() {
            return;
        }
        let status = Command::new("cargo")
            .current_dir(guests_root())
            // Clear the devShell's `CARGO_TARGET_DIR` (pinned to the parent checkout) so the guests
            // build into their own `guests/target/` where `guest_dir()` reads them.
            .env_remove("CARGO_TARGET_DIR")
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
}

fn tiny_llama_wasm() -> Vec<u8> {
    let path = guest_dir().join("tiny_llama.wasm");
    if !path.exists() {
        ensure_built();
    }
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn cbor(cfg: &TinyLlamaCfg) -> Vec<u8> {
    let mut b = Vec::new();
    into_writer(cfg, &mut b).expect("cbor");
    b
}

// -- run parameters -----------------------------------------------------------------------------

const PEERS: usize = 3;
const ROUNDS: u64 = 6;
const SEQ: u32 = 8;
const SEQS: u32 = 2;
const VOCAB: u32 = 64;
const RUN_ID: &str = "wasm-profile-e2e";

fn peer_key(i: usize) -> SigningKey {
    SigningKey::from_bytes(&[0x21 + i as u8; 32])
}

fn coordinator_key() -> SigningKey {
    SigningKey::from_bytes(&[0xC0; 32])
}

fn tiny_cfg(profile: &str) -> TinyLlamaCfg {
    TinyLlamaCfg {
        n_layers: 1,
        seq_len: SEQ + 1,
        vocab: VOCAB,
        profile: profile.to_string(),
        ..TinyLlamaCfg::default()
    }
}

/// Peer-`i` deterministic token ids (`< VOCAB`); each peer trains its own fixed data slice so the run
/// is genuinely data-parallel (the peers reconverge every round through the consensus fold).
fn tokens(peer: usize) -> Vec<u32> {
    let salt = (peer as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..SEQ * SEQS)
        .map(|i| {
            ((u64::from(i).wrapping_mul(2_654_435_761).wrapping_add(salt)) % u64::from(VOCAB))
                as u32
        })
        .collect()
}

fn make_backend(config: &[u8]) -> WasmBackend {
    let mut b = WasmBackend::new(WasmBackendConfig {
        wasm: tiny_llama_wasm(),
        engine: EngineConfig::default(),
    })
    .expect("construct WasmBackend");
    b.build(config).expect("da_build");
    b
}

/// A minimal tiny-llama run envelope for `RunConfig::from_envelope` + the observe replay genesis.
fn envelope() -> Envelope {
    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        "tiny-llama.wasm".to_string(),
        Artifact {
            url: "file://tiny-llama.wasm".to_string(),
            blake3: Hash([0u8; 32]),
        },
    );
    artifacts.insert(
        "manifest.json".to_string(),
        Artifact {
            url: "file://manifest.json".to_string(),
            blake3: Hash([0u8; 32]),
        },
    );
    Envelope {
        run: RunSection {
            schema: ENVELOPE_SCHEMA_MAJOR,
            run_id: RUN_ID.to_string(),
            min_peers: PEERS as u32,
            max_peers: PEERS as u32,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "tiny-llama.wasm".to_string(),
            abi: "tensor-abi@1".to_string(),
            config: ciborium::value::Value::Null,
        },
        artifacts,
        data: DataSection {
            manifest: "manifest.json".to_string(),
            steps_per_round: 3,
            global_batch: GlobalBatch {
                start: 12,
                end: 12,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(ROUNDS),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 0,
            uplink_mbps_min: 0,
            downlink_mbps_min: 0,
            disk_gb_min: 0,
            throughput_floor: "c1".to_string(),
            update_mb_max: 64,
            capabilities: Vec::new(),
            payload_store: "r2".to_string(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 1,
            round_train_max: 1_000,
            round_witness: 1_000,
            cooldown: 1,
            epoch_rounds: 0,
            checkpoint_every_epochs: 0,
            stall_rounds_max: 2,
            payload_retention_rounds: 8,
        },
    }
}

fn params() -> CoordinatorParams {
    CoordinatorParams {
        seq_len: 1,
        witness_target: 0, // every peer witnesses
        overlap_bps: 0,
        k_absences: 100, // no drops in this short run
        verification_percent: 0,
        authorized: Vec::new(),
    }
}

/// The impure-but-tiny coordinator driver: feed one input to the pure `tick`, record it in the
/// replay trace, and return the outputs. The coordinator's own signed `RoundRecord` publications are
/// appended to the trace as the replay oracle (as a real gossip driver would broadcast them).
struct Coord {
    state: Option<CoordinatorState>,
    version: SwarmProtoVersion,
    key: SigningKey,
    trace: Vec<Input>,
    records: Vec<RoundRecord>,
}

impl Coord {
    fn apply(&mut self, input: Input) {
        self.trace.push(input.clone());
        let (next, outputs) = tick(self.state.take().unwrap(), input);
        self.state = Some(next);
        for o in outputs {
            if let Output::Publish(msg) = o {
                if let SwarmMessage::RoundRecord(rr) = *msg {
                    // Broadcast (sign) the record + record it in the trace as the replay oracle.
                    let signed = SignedMessage::sign(
                        &self.key,
                        self.version,
                        SwarmMessage::RoundRecord(rr.clone()),
                    )
                    .expect("sign record");
                    self.trace.push(Input::Message(signed));
                    self.records.push(rr);
                }
            }
        }
    }

    fn state(&self) -> &CoordinatorState {
        self.state.as_ref().unwrap()
    }
}

fn sign(k: &SigningKey, version: SwarmProtoVersion, payload: SwarmMessage) -> SignedMessage {
    SignedMessage::sign(k, version, payload).expect("sign")
}

/// Run the flagship scenario for `profile`: returns `(digests_per_round, round_losses, replay_ok)`.
fn run_profile(profile: &str) -> (Vec<RunDigest>, Vec<f32>, u64) {
    let config = cbor(&tiny_cfg(profile));
    let mut peers: Vec<WasmBackend> = (0..PEERS).map(|_| make_backend(&config)).collect();
    let steps = peers[0].steps_per_round().expect("steps_per_round");

    let keys: Vec<SigningKey> = (0..PEERS).map(peer_key).collect();
    let ids: Vec<PeerId> = keys.iter().map(peer_id).collect();
    let version = SWARM_PROTO_VERSION;

    let env = envelope();
    let run_config = RunConfig::from_envelope(&env, params()).expect("run config");
    let envelope_hash = run_config.envelope_hash;
    let seed = genesis_seed(&env).expect("genesis seed");
    let mut coord = Coord {
        state: Some(CoordinatorState::new(run_config, seed, 0)),
        version,
        key: coordinator_key(),
        trace: Vec::new(),
        records: Vec::new(),
    };

    // Admission: each peer joins under the frozen-envelope hash (§6.5), then clock past warmup.
    for k in &keys {
        let join = Join {
            run_id: RUN_ID.to_string(),
            iroh_id: IrohId([0x22; 32]),
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: Some(envelope_hash),
        };
        coord.apply(Input::Message(sign(k, version, SwarmMessage::Join(join))));
    }
    let now = coord.state().now_s;
    let warmup = coord.state().config.warmup_s;
    coord.apply(Input::Clock(now + 1)); // WaitingForMembers -> Warmup
    coord.apply(Input::Clock(now + warmup + 2)); // Warmup -> RoundTrain (opens round 0)

    let mut digests = Vec::new();
    let mut round_losses = Vec::new();

    for round in 0..ROUNDS {
        // Each peer trains its own data slice, then seals its round payload.
        let mut payloads: Vec<(PeerId, Vec<u8>)> = Vec::new();
        let mut losses = Vec::new();
        for (i, b) in peers.iter_mut().enumerate() {
            let batch = BatchRef {
                tokens: tokens(i),
                seq_len: SEQ,
            };
            let mut last = f32::NAN;
            for step in 0..steps {
                let stats = b
                    .train_step(
                        &batch,
                        StepCtx {
                            inner_step: step,
                            mb_index: 0,
                            mb_count: 1,
                            step_seqs: SEQS,
                        },
                    )
                    .expect("train_step");
                last = stats.loss;
                b.inner_update(step).expect("inner_update");
            }
            losses.push(last);
            let payload = b.make_update(round).expect("make_update");
            payloads.push((ids[i], payload));
        }
        round_losses.push(losses.iter().sum::<f32>() / losses.len() as f32);

        // The committed set in record order (§6.4 I3: sorted by node pubkey bytes).
        payloads.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
        let staged: Vec<StagedPayload> = payloads
            .iter()
            .map(|(peer, bytes)| StagedPayload {
                peer: *peer,
                hash: Hash(*blake3_hash_raw(bytes).as_bytes()),
                bytes: bytes.clone(),
            })
            .collect();

        // Every peer ingests the identical record-ordered set → its post-ingest digest.
        let mut round_digests = Vec::new();
        for b in &mut peers {
            round_digests.push(b.ingest(round, &staged).expect("ingest"));
        }
        for (i, d) in round_digests.iter().enumerate() {
            assert_eq!(
                *d,
                round_digests[0],
                "{profile} r{round}: peer {i} digest must match peer 0 ({} vs {})",
                d.to_hex(),
                round_digests[0].to_hex()
            );
        }
        digests.push(round_digests[0]);

        // Drive the real coordinator: each peer commits its payload; the coordinator issues a
        // storage receipt (availability evidence) and finalizes the round event-driven.
        for (peer, bytes) in &payloads {
            let hash = Hash(*blake3_hash_raw(bytes).as_bytes());
            let commit = Commitment {
                round,
                payload: hash,
                size: bytes.len() as u64,
                locators: vec![Locator::StoreKey(format!("runs/{RUN_ID}/r{round}"))],
            };
            let signer = keys.iter().find(|k| peer_id(k) == *peer).unwrap();
            coord.apply(Input::Message(sign(
                signer,
                version,
                SwarmMessage::Commitment(commit),
            )));
        }
        // Peers publish their post-ingest digest (round still active in RoundWitness).
        for (i, k) in keys.iter().enumerate() {
            let d = Digest {
                round,
                digest: StateDigest(*round_digests[i].as_bytes()),
            };
            coord.apply(Input::Message(sign(k, version, SwarmMessage::Digest(d))));
        }
        // Coordinator-as-storage-client receipts finalize the round (§6.4 I6).
        for (peer, bytes) in &payloads {
            let hash = Hash(*blake3_hash_raw(bytes).as_bytes());
            let sr = StorageReceipt {
                round,
                verified: vec![RecordEntry {
                    peer: *peer,
                    hash,
                    size: bytes.len() as u64,
                }],
            };
            coord.apply(Input::Message(sign(
                &coordinator_key(),
                version,
                SwarmMessage::StorageReceipt(sr),
            )));
        }
    }

    assert_eq!(
        coord.records.len() as u64,
        ROUNDS,
        "{profile}: the coordinator finalized one record per round"
    );

    // PROTO-20: the observe replay oracle re-derives every RoundRecord from the recorded trace.
    let report = replay(&env, params(), coord.trace.into_iter())
        .unwrap_or_else(|e| panic!("{profile}: replay diverged: {e}"));
    assert_eq!(
        report.rounds_verified, ROUNDS,
        "{profile}: replay verified every round record (PROTO-20)"
    );

    (digests, round_losses, report.rounds_verified)
}

fn assert_run(profile: &str) {
    let (digests, losses, verified) = run_profile(profile);
    println!(
        "[{profile}] rounds={ROUNDS} peers={PEERS} verified={verified} loss {:.4}->{:.4} per_round={:?} digest[0]={} digest[last]={}",
        losses.first().unwrap(),
        losses.last().unwrap(),
        losses.iter().map(|l| (l * 1e4).round() / 1e4).collect::<Vec<_>>(),
        digests.first().unwrap().to_hex(),
        digests.last().unwrap().to_hex(),
    );
    assert_eq!(digests.len() as u64, ROUNDS);
    assert_eq!(verified, ROUNDS);
    // The transcript evolves (the swarm is learning, not a vacuous constant).
    assert!(
        digests.windows(2).any(|w| w[0] != w[1]),
        "{profile}: the digest transcript must evolve across rounds"
    );
    // Loss decreases: the host reverse-mode autodiff (HOST-9) actually learns from the data.
    let first = *losses.first().unwrap();
    let last = *losses.last().unwrap();
    assert!(
        first.is_finite() && last.is_finite(),
        "{profile}: losses are finite (per_round={losses:?})"
    );
    assert!(
        last < first,
        "{profile}: mean loss must decrease over the run (first={first}, last={last}, per_round={losses:?})"
    );
}

#[test]
fn flagship_sparse_loco_wasm_backed_run() {
    assert_run("sparse_loco");
}

#[test]
fn flagship_diloco_wasm_backed_run() {
    assert_run("diloco");
}

#[test]
fn flagship_demo_wasm_backed_run() {
    assert_run("demo");
}
