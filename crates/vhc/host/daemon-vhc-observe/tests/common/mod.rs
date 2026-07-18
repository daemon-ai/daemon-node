// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared fixture helpers for the replay-oracle tests: author a deterministic multi-round driving
//! script and run it through the **real `coordinator-quorum` module** (via the
//! [`WasmCoordinatorSandbox`]) to capture the coordinator's published decisions — the same sandbox
//! the oracle re-derives through. Consensus never runs natively here : both fixture generation
//! and verification drive the sandboxed module.

// The sandbox shells `cargo build` for the guests workspace (the established testkit pattern); the
// fixtures also touch temp dirs in the callers, so allow the fs/process bans test-wide.
#![allow(clippy::disallowed_methods)]
// Not every test consumes every helper; keep them all available without per-item dead-code churn.
#![allow(dead_code)]

use std::collections::BTreeMap;

use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::messages::{
    Commitment, Heartbeat, Join, RecordEntry, RoundRecord, SignedMessage, StorageReceipt,
    ThroughputClass, VhcMessage,
};
use daemon_vhc_proto::{
    blake3_hash, peer_id, CapabilitySet, Hash, IrohId, PeerId, Seed, SigningKey, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig};

use daemon_vhc_observe::genesis_seed;
pub use daemon_vhc_session::replay_sandbox::WasmCoordinatorSandbox;

/// A signing key from a one-byte seed (test identities).
#[must_use]
pub fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

/// The coordinator's frame-signing key (the run's `SingleKey` authority in the fixtures).
#[must_use]
pub fn coord_key() -> SigningKey {
    key(1)
}

/// Build the run's coordinator config: continuous rounds with huge phase deadlines, so only the
/// event-driven fast paths fire (the run is a pure function of the message sequence). A literal
/// [`RunConfig`] — the run's genesis-derived config the coordinator module is initialized with.
#[must_use]
pub fn run_config(run_id: &str) -> RunConfig {
    // The run-identity hash the seed + module identity derive from (the genesis hash domain).
    let envelope_hash = blake3_hash(run_id.as_bytes());
    RunConfig {
        run_id: run_id.into(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash,
        required_capabilities: CapabilitySet::new(),
        min_peers: 2,
        max_peers: 4,
        warmup_s: 1_000_000,
        round_train_max_s: 1_000_000,
        round_witness_s: 1_000_000,
        cooldown_s: 1_000_000,
        epoch_rounds: 0,
        stall_rounds_max: 2,
        global_batch: GlobalBatch {
            start: 4,
            end: 4,
            ramp_rounds: 1,
        },
        stop: StopCondition::Rounds(1_000_000),
        steps_per_round: 2,
        seq_len: 9,
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    }
}

/// The genesis-derived initial coordinator state (config + genesis-hash seed).
#[must_use]
pub fn initial_state(run_id: &str) -> CoordinatorState {
    let config = run_config(run_id);
    let seed: Seed = genesis_seed(&config.envelope_hash);
    CoordinatorState::new(config, seed, 0)
}

/// One authored update payload for `(peer_index, round)` and its content hash.
#[must_use]
pub fn payload_for(peer_index: usize, round: u64) -> (Vec<u8>, Hash) {
    let bytes = format!("update/{peer_index}/{round}").into_bytes();
    let hash = blake3_hash(&bytes);
    (bytes, hash)
}

/// A generated fixture: the initial state, the ordered driving messages (fed to the module), the
/// module's published `RoundRecord`s (the oracle), and the committed payload bytes.
pub struct Fixture {
    pub run_id: String,
    pub initial: CoordinatorState,
    pub worker_keys: Vec<SigningKey>,
    pub peers: Vec<PeerId>,
    pub driving: Vec<SignedMessage>,
    /// Every decision the module published, in order (`RoundOpen` + `RoundRecord`).
    pub published: Vec<VhcMessage>,
    pub records: Vec<RoundRecord>,
    pub payloads: BTreeMap<Hash, Vec<u8>>,
}

fn sign(k: &SigningKey, m: VhcMessage) -> SignedMessage {
    SignedMessage::sign(k, VHC_PROTO_VERSION, m).expect("sign")
}

/// Author the deterministic driving script and run it through the real coordinator module to
/// capture its published decisions — the fixture both the recorded run and its replay derive from.
///
/// Script (the all-committed → all-evidenced fast path, per round): join the two workers, ready
/// heartbeats to open round 0, then per round two commitments (with real payload bytes) + one
/// covering storage receipt that finalizes the record and opens the next round.
pub fn run_fixture(sandbox: &WasmCoordinatorSandbox, run_id: &str, rounds: u64) -> Fixture {
    let initial = initial_state(run_id);
    let envelope_hash = initial.config.envelope_hash;
    let worker_keys = vec![key(2), key(3)];
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();

    let mut driving: Vec<SignedMessage> = Vec::new();
    let mut payloads: BTreeMap<Hash, Vec<u8>> = BTreeMap::new();

    for k in &worker_keys {
        driving.push(sign(
            k,
            VhcMessage::Join(Join {
                run_id: run_id.into(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: Some(envelope_hash),
            }),
        ));
    }
    for k in &worker_keys {
        driving.push(sign(
            k,
            VhcMessage::Heartbeat(Heartbeat {
                round: 0,
                ready: Some(true),
            }),
        ));
    }
    for round in 0..rounds {
        let mut entries = Vec::new();
        for (i, k) in worker_keys.iter().enumerate() {
            let (bytes, hash) = payload_for(i, round);
            payloads.insert(hash, bytes.clone());
            driving.push(sign(
                k,
                VhcMessage::Commitment(Commitment {
                    round,
                    payload: hash,
                    size: bytes.len() as u64,
                    locators: Vec::new(),
                }),
            ));
            entries.push(RecordEntry {
                peer: peers[i],
                hash,
                size: bytes.len() as u64,
            });
        }
        driving.push(sign(
            &worker_keys[0],
            VhcMessage::StorageReceipt(StorageReceipt {
                round,
                verified: entries,
            }),
        ));
    }

    // Drive the real coordinator module over the script and capture its published decisions.
    let published =
        replay_run(sandbox, &initial, &driving, rounds as usize).expect("coordinator sandbox run");
    let records: Vec<RoundRecord> = published
        .iter()
        .filter_map(|m| match m {
            VhcMessage::RoundRecord(r) => Some(r.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        records.len() as u64,
        rounds,
        "the coordinator module must finalize {rounds} records"
    );

    Fixture {
        run_id: run_id.into(),
        initial,
        worker_keys,
        peers,
        driving,
        published,
        records,
        payloads,
    }
}

/// Run the sandbox once (fixture generation) — a thin wrapper over the [`CoordinatorSandbox`] seam.
pub fn replay_run(
    sandbox: &WasmCoordinatorSandbox,
    initial: &CoordinatorState,
    driving: &[SignedMessage],
    expected_records: usize,
) -> Result<Vec<VhcMessage>, daemon_vhc_observe::ReplayError> {
    use daemon_vhc_observe::CoordinatorSandbox as _;
    sandbox.replay_run(initial, driving, expected_records)
}

/// A fresh sandbox over the `coordinator-quorum` guest (built once per process).
#[must_use]
pub fn coordinator_sandbox() -> WasmCoordinatorSandbox {
    WasmCoordinatorSandbox::from_built_guest().expect("build coordinator-quorum guest")
}
