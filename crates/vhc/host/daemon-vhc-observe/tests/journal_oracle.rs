// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator-oracle migration onto the journal substrate (refactor §5 A1 acceptance): the
//! existing replay oracle passes **unchanged in semantics** over journal-backed capture. This test
//! drives a real coordinator run, replays it both the in-memory way ([`replay_capture`]) and over the
//! crash-safe segmented [`Journal`] ([`replay_over_journal`]), and asserts the two [`ReplayReport`]s
//! are byte-identical — proving the oracle's pinned behavior is preserved on the new substrate.

// Uses a throwaway temp dir for the journal fixtures; test-scoped fs allow (Phase-4 guardrail
// targets production paths).
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use ciborium::value::Value;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition,
};
use daemon_vhc_proto::messages::{
    Commitment, Heartbeat, Join, Locator, RecordEntry, SignedMessage, StorageReceipt, SwarmMessage,
    ThroughputClass,
};
use daemon_vhc_proto::{
    peer_id, CapabilitySet, Hash, IrohId, PeerId, SigningKey, SWARM_PROTO_VERSION,
};

use daemon_vhc_sdk_consensus::coordinator::{
    tick, CoordinatorParams, CoordinatorState, Input, Output, RunConfig,
};

use daemon_vhc_observe::journal::oracle::{record_capture, replay_over_journal};
use daemon_vhc_observe::journal::record::ExecIdentity;
use daemon_vhc_observe::journal::sidecar::StaticKey;
use daemon_vhc_observe::journal::store::{Journal, RotatePolicy};
use daemon_vhc_observe::{genesis_seed, replay_capture, MessageLog, RunCapture};

const RUN_ID: &str = "oracle-journal-run";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn sample_envelope(stop_rounds: u64) -> Envelope {
    let mut artifacts = BTreeMap::new();
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: "r2://runs/obs/mod.wasm".into(),
            blake3: Hash([1; 32]),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: "r2://runs/obs/manifest.json".into(),
            blake3: Hash([2; 32]),
        },
    );
    Envelope {
        run: RunSection {
            schema: 1,
            run_id: RUN_ID.into(),
            min_peers: 2,
            max_peers: 2,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".into(),
            abi: "tensor-abi@1".into(),
            config: Value::Map(vec![(
                Value::Text("profile".into()),
                Value::Text("stub".into()),
            )]),
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".into(),
            steps_per_round: 4,
            global_batch: GlobalBatch {
                start: 100,
                end: 100,
                ramp_rounds: 0,
            },
            stop: StopCondition::Rounds(stop_rounds),
        },
        requirements: Requirements {
            vram_mb_min: 8000,
            ram_gb_min: 16,
            uplink_mbps_min: 10,
            downlink_mbps_min: 50,
            disk_gb_min: 20,
            throughput_floor: "c1".into(),
            update_mb_max: 40,
            capabilities: vec!["tensor-abi@1".into()],
            payload_store: "r2".into(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 300,
            round_train_max: 900,
            round_witness: 60,
            cooldown: 120,
            epoch_rounds: 100,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 8,
        },
    }
}

fn join_msg(k: &SigningKey) -> SignedMessage {
    let j = Join {
        run_id: RUN_ID.into(),
        iroh_id: IrohId([0x22; 32]),
        class: ThroughputClass::C1,
        capabilities: CapabilitySet::from_tokens(["tensor-abi@1"]).unwrap(),
        envelope_hash: None,
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Join(j)).unwrap()
}

fn ready_heartbeat(k: &SigningKey, round: u64) -> SignedMessage {
    let h = Heartbeat {
        round,
        ready: Some(true),
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Heartbeat(h)).unwrap()
}

fn payload_hash(round: u64) -> Hash {
    Hash([(round as u8) + 1; 32])
}

fn commitment_msg(k: &SigningKey, round: u64) -> SignedMessage {
    let c = Commitment {
        round,
        payload: payload_hash(round),
        size: 1_000,
        locators: vec![Locator::StoreKey("k".into())],
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Commitment(c)).unwrap()
}

fn receipt_msg(coord: &SigningKey, round: u64, peers: &[PeerId]) -> SignedMessage {
    let verified = peers
        .iter()
        .map(|p| RecordEntry {
            peer: *p,
            hash: payload_hash(round),
            size: 1_000,
        })
        .collect();
    let sr = StorageReceipt { round, verified };
    SignedMessage::sign(coord, SWARM_PROTO_VERSION, SwarmMessage::StorageReceipt(sr)).unwrap()
}

fn drive(
    state: CoordinatorState,
    trace: &mut Vec<Input>,
    coord: &SigningKey,
    input: Input,
) -> CoordinatorState {
    trace.push(input.clone());
    let (next, outputs) = tick(state, input);
    for out in outputs {
        if let Output::Publish(msg) = out {
            if let SwarmMessage::RoundRecord(r) = *msg {
                let signed =
                    SignedMessage::sign(coord, SWARM_PROTO_VERSION, SwarmMessage::RoundRecord(r))
                        .unwrap();
                trace.push(Input::Message(signed));
            }
        }
    }
    next
}

fn live_run(
    env: &Envelope,
    params: &CoordinatorParams,
    rounds: u64,
) -> (CoordinatorState, Vec<Input>) {
    let config = RunConfig::from_envelope(env, params.clone()).unwrap();
    let seed = genesis_seed(env).unwrap();
    let coord = key(200);
    let ks = [key(1), key(2)];
    let pids: Vec<PeerId> = ks.iter().map(peer_id).collect();

    let mut state = CoordinatorState::new(config, seed, 0);
    let mut trace = Vec::new();
    for k in &ks {
        state = drive(state, &mut trace, &coord, Input::Message(join_msg(k)));
    }
    state = drive(state, &mut trace, &coord, Input::Clock(1)); // → Warmup
    for k in &ks {
        state = drive(
            state,
            &mut trace,
            &coord,
            Input::Message(ready_heartbeat(k, 0)),
        );
    }
    for r in 0..rounds {
        for k in &ks {
            state = drive(
                state,
                &mut trace,
                &coord,
                Input::Message(commitment_msg(k, r)),
            );
        }
        state = drive(
            state,
            &mut trace,
            &coord,
            Input::Message(receipt_msg(&coord, r, &pids)),
        );
    }
    (state, trace)
}

fn tempdir() -> std::path::PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let mut base = std::env::temp_dir();
    base.push(format!(
        "dvhc-journal-oracle-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn ident() -> ExecIdentity {
    ExecIdentity {
        run_id: Hash([0xC0; 32]),
        epoch: 0,
        role: "coordinator".into(),
        instance: 1,
        module: Hash([0xCD; 32]),
    }
}

/// The oracle re-derives the run identically over journal-backed capture as it does in-memory.
#[test]
fn oracle_parity_over_journal_substrate() {
    let env = sample_envelope(3);
    let params = CoordinatorParams::default();
    let (_, trace) = live_run(&env, &params, 3);

    // The wire log: every signed message (incl. the coordinator's own published RoundRecords).
    let mut log = MessageLog::new(RUN_ID);
    for input in &trace {
        if let Input::Message(sm) = input {
            log.append(sm.clone());
        }
    }

    // The driving capture: initial state + driving inputs (no RoundRecord/RoundOpen — those are the
    // oracle, re-supplied from the log).
    let initial = CoordinatorState::new(
        RunConfig::from_envelope(&env, params).unwrap(),
        genesis_seed(&env).unwrap(),
        0,
    );
    let driving: Vec<Input> = trace
        .into_iter()
        .filter(|i| {
            !matches!(
                i,
                Input::Message(sm)
                    if matches!(sm.payload, SwarmMessage::RoundRecord(_) | SwarmMessage::RoundOpen(_))
            )
        })
        .collect();
    let capture = RunCapture::new(initial, driving);

    // In-memory oracle (the pinned path).
    let report_mem = replay_capture(capture.clone(), &log).expect("in-memory replay");

    // Journal-backed oracle: record the capture onto a crash-safe segmented journal, then replay.
    let dir = tempdir();
    let mut journal = Journal::create(
        dir.join("j"),
        ident(),
        StaticKey::new([5u8; 32]),
        RotatePolicy { max_records: 4 },
    )
    .unwrap();
    record_capture(&mut journal, &ident(), &capture).expect("record capture onto journal");
    let report_journal = replay_over_journal(&journal, &log).expect("journal-backed replay");

    // Byte-identical: same records, same rounds_verified, same final state hash (pinned behavior).
    assert_eq!(
        report_journal, report_mem,
        "oracle parity over the journal substrate"
    );
    assert_eq!(report_journal.rounds_verified, 3);

    // And it survives a reopen (crash-safe substrate): reopening the journal replays identically.
    drop(journal);
    let reopened = Journal::open(
        dir.join("j"),
        ident(),
        StaticKey::new([5u8; 32]),
        RotatePolicy { max_records: 4 },
    )
    .unwrap();
    let report_reopened = replay_over_journal(&reopened, &log).expect("replay after reopen");
    assert_eq!(
        report_reopened, report_mem,
        "oracle parity persists across reopen"
    );
}
