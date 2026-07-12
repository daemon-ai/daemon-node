// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared test scaffolding for the coordinator scenarios.

#![allow(dead_code)]

use daemon_swarm_proto::envelope::{GlobalBatch, StopCondition};
use daemon_swarm_proto::messages::{
    AttestEntry, Attestation, Commitment, Digest, Heartbeat, Join, Locator, RecordEntry,
    SignedMessage, StorageReceipt, Straggle, StraggleStatus, SwarmMessage, ThroughputClass,
};
use daemon_swarm_proto::sign::Signed;
use daemon_swarm_proto::{
    commit_set, peer_id, CapabilitySet, Hash, IrohId, PeerId, Seed, SigningKey, StateDigest,
    SwarmProtoVersion, SWARM_PROTO_VERSION,
};

use daemon_swarm_coordinator::{
    ControlAction, ControlRequest, CoordinatorState, Input, Output, Phase, RunConfig,
};

pub const RUN_ID: &str = "test-run";

pub fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

pub fn pid(seed: u8) -> PeerId {
    peer_id(&key(seed))
}

pub fn base_config() -> RunConfig {
    RunConfig {
        run_id: RUN_ID.to_string(),
        proto_version: SWARM_PROTO_VERSION,
        envelope_hash: Hash([0x11; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: 2,
        max_peers: 8,
        warmup_s: 10,
        round_train_max_s: 100,
        round_witness_s: 30,
        cooldown_s: 20,
        epoch_rounds: 3,
        stall_rounds_max: 2,
        global_batch: GlobalBatch {
            start: 100,
            end: 100,
            ramp_rounds: 0,
        },
        stop: StopCondition::Rounds(1_000),
        steps_per_round: 4,
        seq_len: 1,
        witness_target: 4,
        overlap_bps: 0,
        k_absences: 3,
        verification_percent: 0,
        authorized: Vec::new(),
    }
}

pub fn new_state(config: RunConfig) -> CoordinatorState {
    CoordinatorState::new(config, Seed([0xAB; 32]), 0)
}

// ----- message builders -----

pub fn join_msg(k: &SigningKey) -> SignedMessage {
    let j = Join {
        run_id: RUN_ID.to_string(),
        iroh_id: IrohId([0x22; 32]),
        class: ThroughputClass::C2,
        capabilities: CapabilitySet::new(),
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Join(j)).unwrap()
}

pub fn join_msg_version(k: &SigningKey, version: SwarmProtoVersion) -> SignedMessage {
    let j = Join {
        run_id: RUN_ID.to_string(),
        iroh_id: IrohId([0x22; 32]),
        class: ThroughputClass::C2,
        capabilities: CapabilitySet::new(),
    };
    SignedMessage::sign(k, version, SwarmMessage::Join(j)).unwrap()
}

pub fn payload_hash(seed: u8) -> Hash {
    Hash([seed; 32])
}

pub fn commitment_msg(k: &SigningKey, round: u64, payload_seed: u8) -> SignedMessage {
    let c = Commitment {
        round,
        payload: payload_hash(payload_seed),
        size: 1_000,
        locators: vec![Locator::StoreKey("k".to_string())],
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Commitment(c)).unwrap()
}

pub fn receipt_msg(coord: &SigningKey, round: u64, entries: &[(PeerId, u8)]) -> SignedMessage {
    let verified = entries
        .iter()
        .map(|(p, seed)| RecordEntry {
            peer: *p,
            hash: payload_hash(*seed),
            size: 1_000,
        })
        .collect();
    let sr = StorageReceipt { round, verified };
    SignedMessage::sign(coord, SWARM_PROTO_VERSION, SwarmMessage::StorageReceipt(sr)).unwrap()
}

pub fn attestation_msg(
    witness: &SigningKey,
    round: u64,
    entries: &[(PeerId, u8)],
) -> SignedMessage {
    let inline: Vec<AttestEntry> = entries
        .iter()
        .map(|(p, seed)| AttestEntry {
            peer: *p,
            hash: payload_hash(*seed),
        })
        .collect();
    let pairs: Vec<(PeerId, Hash)> = entries
        .iter()
        .map(|(p, seed)| (*p, payload_hash(*seed)))
        .collect();
    let a = Attestation {
        round,
        set: commit_set(&pairs).commitment(),
        inline: Some(inline),
    };
    SignedMessage::sign(witness, SWARM_PROTO_VERSION, SwarmMessage::Attestation(a)).unwrap()
}

pub fn digest_msg(k: &SigningKey, round: u64, digest_seed: u8) -> SignedMessage {
    let d = Digest {
        round,
        digest: StateDigest([digest_seed; 16]),
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Digest(d)).unwrap()
}

pub fn straggle_msg(k: &SigningKey, round: u64) -> SignedMessage {
    let s = Straggle {
        round,
        status: StraggleStatus::Stalled,
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Straggle(s)).unwrap()
}

pub fn heartbeat_msg(k: &SigningKey, round: u64) -> SignedMessage {
    let h = Heartbeat { round, ready: None };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Heartbeat(h)).unwrap()
}

/// A heartbeat that also signals model-readiness during `Warmup` (Wave-3 additive `ready` flag).
pub fn ready_heartbeat_msg(k: &SigningKey, round: u64) -> SignedMessage {
    let h = Heartbeat {
        round,
        ready: Some(true),
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Heartbeat(h)).unwrap()
}

pub fn control(k: &SigningKey, action: ControlAction) -> Signed<ControlRequest> {
    Signed::seal(
        k,
        ControlRequest {
            run_id: RUN_ID.to_string(),
            action,
        },
    )
    .unwrap()
}

// ----- drivers -----

/// Apply a sequence of inputs, returning the final state and the outputs of the last input.
pub fn drive(mut state: CoordinatorState, inputs: Vec<Input>) -> (CoordinatorState, Vec<Output>) {
    let mut last = Vec::new();
    for inp in inputs {
        let (s, o) = daemon_swarm_coordinator::tick(state, inp);
        state = s;
        last = o;
    }
    (state, last)
}

/// Join `keys`, then clock past warmup so the run opens round 0 (`RoundTrain`).
pub fn to_first_round(config: RunConfig, keys: &[SigningKey]) -> CoordinatorState {
    let mut state = new_state(config);
    for k in keys {
        let (s, _) = daemon_swarm_coordinator::tick(state, Input::Message(join_msg(k)));
        state = s;
    }
    // enter Warmup
    let (s, _) = daemon_swarm_coordinator::tick(state, Input::Clock(1));
    state = s;
    // exit Warmup -> RoundTrain (warmup_s = 10)
    let (s, _) = daemon_swarm_coordinator::tick(state, Input::Clock(20));
    state = s;
    assert_eq!(state.phase, Phase::RoundTrain, "expected RoundTrain");
    state
}

/// Count `Output::Publish` matching a predicate on the payload.
pub fn publishes(outputs: &[Output]) -> Vec<&SwarmMessage> {
    outputs
        .iter()
        .filter_map(|o| match o {
            Output::Publish(m) => Some(m.as_ref()),
            _ => None,
        })
        .collect()
}
