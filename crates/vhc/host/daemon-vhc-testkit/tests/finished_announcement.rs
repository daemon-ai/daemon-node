// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// **The completion announcement — the closure contract** (defect 5 of the c15 drills): a run
// whose stop condition is met must END, observably, for every party:
//
// 1. the coordinator PUBLISHES a signed `Finished` decision on the records channel (the trainers
//    do not know the stop condition — it lives in the coordinator's config — so this frame is
//    their only way to learn the run is over), and
// 2. the coordinator guest's `da_run` RETURNS outcome 0 of its own accord (no host stop), which
//    the session classifies `Completed`.
//
// Before the fix the SDK emitted `Notice::Finished` as an advisory note, the wrapper dropped
// every note ("advisory"), and `run()` had no Finished exit: c15d committed all 24 authored
// rounds and then idled its 1 Hz timer forever — trainers parked in `next_event` waiting for a
// round 24 that would never open, the node never leaving `run_state=running`.
//
// Dev/test harness: shells `cargo build` for guests.
#![allow(clippy::disallowed_methods)]

use std::time::Duration;

use ciborium::value::Value;

use daemon_vhc_host::run::RunEnd;
use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::{
    blake3_hash, peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, Seed, SigningKey,
    StateDigest, VHC_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig};
use daemon_vhc_sdk_consensus::messages::{
    Commitment, Digest, Heartbeat, Join, RecordEntry, StorageReceipt, ThroughputClass,
};
use daemon_vhc_sdk_consensus::VhcMessage;
use daemon_vhc_sdk_consensus::{AuthorityConfig, SingleKey, Topology, DEFAULT_RECORDS_CHANNEL};
use daemon_vhc_testkit::genesis_run::phase_a_grants;
use daemon_vhc_testkit::{Coordinator, CoordinatorSpec};

fn coordinator_quorum_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("coordinator_quorum")
}

/// One round, then stop. Deadlines are event-count seconds (the deterministic synthetic clock);
/// only the cooldown is expected to elapse — every other transition is evidence-driven.
const COOLDOWN_S: u64 = 3;

fn run_config() -> RunConfig {
    RunConfig {
        run_id: "finished-announcement".to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
        required_capabilities: CapabilitySet::new(),
        min_peers: 2,
        max_peers: 4,
        warmup_s: 1_000_000, // exited early by ready heartbeats
        round_train_max_s: 1_000_000,
        round_witness_s: 1_000_000,
        cooldown_s: COOLDOWN_S,
        epoch_rounds: 0,
        stall_rounds_max: 2,
        global_batch: GlobalBatch {
            start: 4,
            end: 4,
            ramp_rounds: 1,
        },
        stop: StopCondition::Rounds(1),
        steps_per_round: 2,
        seq_len: 9,
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    }
}

fn spec_for(wasm: &[u8], state: &CoordinatorState, run_id: Hash) -> CoordinatorSpec {
    let config_bytes = {
        let v = Value::Map(vec![(
            Value::Text("state".into()),
            Value::serialized(state).expect("state value"),
        )]);
        to_canonical_vec(&v).expect("config cbor")
    };
    CoordinatorSpec {
        module_hash: Hash(*blake3::hash(wasm).as_bytes()),
        config_bytes,
        authority: AuthorityConfig {
            topology: Topology::SingleKey(SingleKey::new(peer_id(&SigningKey::from_bytes(
                blake3::hash(b"finished/authority").as_bytes(),
            )))),
            records_channel: DEFAULT_RECORDS_CHANNEL,
        },
        run_id,
    }
}

/// The whole closure contract in one drive: last record → cooldown elapses → the signed
/// `Finished` publish (carrying the committed-round count) → `da_run` returns outcome 0 with no
/// host stop.
#[test]
fn a_stopped_run_publishes_finished_and_the_guest_returns_completed() {
    let wasm = coordinator_quorum_wasm();
    let initial = CoordinatorState::new(run_config(), Seed([0x33; 32]), 0);
    let run_id = blake3_hash(b"finished-announcement/run");
    let spec = spec_for(&wasm, &initial, run_id);
    let worker_keys = [
        SigningKey::from_bytes(blake3::hash(b"finished/worker/0").as_bytes()),
        SigningKey::from_bytes(blake3::hash(b"finished/worker/1").as_bytes()),
    ];
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();

    let mut coord = Coordinator::start(
        &wasm,
        &spec,
        phase_a_grants(),
        0,
        *blake3::hash(b"finished/run-key").as_bytes(),
    )
    .expect("coordinator start");

    // Join + ready → warmup exits early → round 0 opens.
    for k in &worker_keys {
        coord
            .deliver(
                k,
                &VhcMessage::Join(Join {
                    run_id: "finished-announcement".into(),
                    iroh_id: IrohId([0x44; 32]),
                    class: ThroughputClass::C1,
                    capabilities: CapabilitySet::new(),
                    envelope_hash: None,
                }),
            )
            .expect("join");
    }
    for k in &worker_keys {
        coord
            .deliver(
                k,
                &VhcMessage::Heartbeat(Heartbeat {
                    round: 0,
                    ready: Some(true),
                }),
            )
            .expect("ready heartbeat");
    }

    // Round 0: both commitments, both digests, the covering receipt → the record commits and
    // `stop_reached` (Rounds(1)) sends the run to Cooldown.
    let mut entries = Vec::new();
    for (i, k) in worker_keys.iter().enumerate() {
        let bytes = format!("update/{i}/0").into_bytes();
        let hash = blake3_hash(&bytes);
        coord
            .deliver(
                k,
                &VhcMessage::Commitment(Commitment {
                    round: 0,
                    payload: hash,
                    size: bytes.len() as u64,
                    locators: Vec::new(),
                }),
            )
            .expect("commitment");
        entries.push(RecordEntry {
            peer: peers[i],
            hash,
            size: bytes.len() as u64,
        });
    }
    for k in &worker_keys {
        coord
            .deliver(
                k,
                &VhcMessage::Digest(Digest {
                    round: 0,
                    digest: StateDigest([0x11; 16]),
                }),
            )
            .expect("digest");
    }
    coord
        .deliver(
            &worker_keys[0],
            &VhcMessage::StorageReceipt(StorageReceipt {
                round: 0,
                verified: entries,
            }),
        )
        .expect("receipt");

    // Elapse the cooldown on the synthetic event-count clock: each delivered frame advances it
    // one second. The guest may return mid-loop, so a refused late delivery is fine.
    for _ in 0..=COOLDOWN_S + 1 {
        let _ = coord.deliver(
            &worker_keys[0],
            &VhcMessage::Heartbeat(Heartbeat {
                round: 0,
                ready: Some(true),
            }),
        );
        std::thread::sleep(Duration::from_millis(20));
        let finished_published = coord
            .published()
            .iter()
            .any(|(_, _, _, m)| matches!(m, VhcMessage::Finished(_)));
        if finished_published {
            break;
        }
    }

    // 1. The completion is a PUBLISHED, signed decision carrying the committed-round count.
    let published = coord.published();
    let finished = published
        .iter()
        .find_map(|(_, _, _, m)| match m {
            VhcMessage::Finished(f) => Some(*f),
            _ => None,
        })
        .expect("the coordinator PUBLISHES Finished (not an advisory note)");
    assert_eq!(finished.rounds, 1, "the run committed rounds 0..1");

    // 2. The guest returns of its own accord — outcome 0, no host stop (`Completed`).
    let end = coord.wait_end().expect("guest thread joins");
    assert!(
        matches!(end, RunEnd::Outcome(0)),
        "a finished run RETURNS Completed instead of idling its timer forever (got {end:?})"
    );
}
