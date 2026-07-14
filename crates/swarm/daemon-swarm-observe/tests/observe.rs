// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-observe` behavior tests (spec §6.4/§14; TDD §3.9 + PROTO-20):
//! `log_roundtrip_canonical`, `replay_matches_live_run`, `replay_detects_tampered_record`,
//! `digest_quorum_flags_outlier`, plus run-health projection.

use std::collections::BTreeMap;

use ciborium::value::Value;
use daemon_swarm_proto::envelope::{
    Access, Artifact, DataSection, Envelope, ExperimentSection, GlobalBatch, Phases, Requirements,
    RoundMode, RunSection, StopCondition,
};
use daemon_swarm_proto::messages::{
    Commitment, Digest, Heartbeat, Join, Locator, RecordEntry, SignedMessage, StorageReceipt,
    Straggle, StraggleStatus, SwarmMessage, ThroughputClass,
};
use daemon_swarm_proto::{
    peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, SigningKey, StateDigest,
    SWARM_PROTO_VERSION,
};

use daemon_swarm_coordinator::{
    tick, CoordinatorParams, CoordinatorState, Input, Output, RunConfig,
};

use daemon_swarm_observe::desync::digest_tally_from_log;
use daemon_swarm_observe::{
    digest_tally, genesis_seed, replay, replay_capture, replay_from_state, MessageKind, MessageLog,
    ReplayError, RunCapture, RunHealth,
};

const RUN_ID: &str = "obs-run";

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn pid(seed: u8) -> PeerId {
    peer_id(&key(seed))
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

// ----- message builders -----

fn join_msg(k: &SigningKey) -> SignedMessage {
    let j = Join {
        run_id: RUN_ID.into(),
        iroh_id: IrohId([0x22; 32]),
        class: ThroughputClass::C1,
        // Must advertise the envelope's required capabilities to be admitted (§6.5).
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

fn digest_msg(k: &SigningKey, round: u64, d: StateDigest) -> SignedMessage {
    let x = Digest { round, digest: d };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Digest(x)).unwrap()
}

fn straggle_msg(k: &SigningKey, round: u64) -> SignedMessage {
    let s = Straggle {
        round,
        status: StraggleStatus::Stalled,
    };
    SignedMessage::sign(k, SWARM_PROTO_VERSION, SwarmMessage::Straggle(s)).unwrap()
}

// ----- a live coordinator run over the event-driven happy path (zero timeouts) -----

/// Drive `input` through `tick`, appending it to the oracle trace, then append any published
/// `RoundRecord` as a signed oracle message (as the coordinator would broadcast it).
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

/// Run `rounds` complete rounds and return `(final_state, oracle_trace)`. Warmup exits on peer
/// readiness (Wave-3 additive), so the whole trace needs exactly one clock (to enter Warmup).
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

// ----- OBS: message-log roundtrip -----

#[test]
fn log_roundtrip_canonical() {
    let (_, trace) = live_run(&sample_envelope(3), &CoordinatorParams::default(), 3);
    let mut log = MessageLog::new(RUN_ID);
    for input in &trace {
        if let Input::Message(sm) = input {
            log.append(sm.clone());
        }
    }
    assert!(!log.is_empty());

    // Write → read is lossless and preserves arrival order + run id.
    let mut bytes = Vec::new();
    log.write_to(&mut bytes).unwrap();
    let read = MessageLog::read_from(&mut bytes.as_slice()).unwrap();
    assert_eq!(read, log);
    assert_eq!(read.run_id(), RUN_ID);

    // Framing is canonical: a second write is byte-identical.
    let mut bytes2 = Vec::new();
    read.write_to(&mut bytes2).unwrap();
    assert_eq!(bytes, bytes2);

    // (round, kind) index: every round has exactly one published RoundRecord.
    for round in log.rounds() {
        let records = log.by_round_kind(round, MessageKind::RoundRecord).count();
        assert_eq!(records, 1, "one record per round");
    }
    // There are three RoundRecords total across the run.
    assert_eq!(log.by_kind(MessageKind::RoundRecord).count(), 3);
    // Joins are roster-scoped (no round) — not returned by `by_round`.
    assert_eq!(
        log.by_round(0)
            .filter(|m| matches!(m.payload, SwarmMessage::Join(_)))
            .count(),
        0
    );
}

// ----- OBS / PROTO-20: replay reproduces the live coordinator -----

#[test]
fn replay_matches_live_run() {
    let env = sample_envelope(3);
    let params = CoordinatorParams::default();
    let (live_final, trace) = live_run(&env, &params, 3);

    let report = replay(&env, params, trace.into_iter()).expect("replay must reproduce the run");
    assert_eq!(report.rounds_verified, 3, "all recorded records re-derived");
    assert_eq!(report.records.len(), 3);

    // The re-derived final state is byte-identical to the live coordinator's (I1 replayability).
    let live_hash = daemon_swarm_proto::blake3_hash(&to_canonical_vec(&live_final).unwrap());
    assert_eq!(report.final_state_hash, live_hash);
}

// ----- OBS: RunCapture replays a recorded run (the --observe / swarm-replay path) -----

#[test]
fn run_capture_replays_recorded_run() {
    let env = sample_envelope(3);
    let params = CoordinatorParams::default();
    let (live_final, trace) = live_run(&env, &params, 3);

    // The node-visible message log: every signed message on the wire (incl. the coordinator's own
    // published RoundRecords, which live_run appends to the trace as it would broadcast them).
    let mut log = MessageLog::new(RUN_ID);
    for input in &trace {
        if let Input::Message(sm) = input {
            log.append(sm.clone());
        }
    }

    // The reproducible driver capture: the initial state + the driving inputs (messages + clocks),
    // WITHOUT the coordinator's own RoundRecord publications (those are the oracle, re-supplied from
    // the log by `replay_capture`).
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
    let capture = RunCapture::new(initial.clone(), driving.clone());

    // The capture round-trips through its on-disk framing byte-identically.
    let mut bytes = Vec::new();
    capture.write_to(&mut bytes).unwrap();
    let read = RunCapture::read_from(&mut bytes.as_slice()).unwrap();
    assert_eq!(read, capture);

    // replay_capture re-derives every logged RoundRecord byte-identically (digest equality).
    let report = replay_capture(capture, &log).expect("recorded run must re-derive");
    assert_eq!(
        report.rounds_verified, 3,
        "all 3 recorded records re-derived"
    );
    let live_hash = daemon_swarm_proto::blake3_hash(&to_canonical_vec(&live_final).unwrap());
    assert_eq!(
        report.final_state_hash, live_hash,
        "final state byte-identical"
    );

    // replay_from_state over driving-inputs-only produces the records with nothing to compare
    // (no oracle in the stream) — it still re-derives 3 records, verifying 0.
    let bare = replay_from_state(initial, driving.into_iter()).unwrap();
    assert_eq!(bare.records.len(), 3);
    assert_eq!(bare.rounds_verified, 0);
    assert_eq!(bare.final_state_hash, live_hash);
}

// ----- OBS: a tampered record is caught, first-divergence pinpointed -----

#[test]
fn replay_detects_tampered_record() {
    let env = sample_envelope(3);
    let params = CoordinatorParams::default();
    let (_, mut trace) = live_run(&env, &params, 3);
    let coord = key(200);

    // Tamper the first recorded RoundRecord (round 0): claim a spurious drop, then re-sign so the
    // frame is still valid — only the consensus content diverges.
    let mut tampered_round = None;
    for input in &mut trace {
        if let Input::Message(sm) = input {
            if let SwarmMessage::RoundRecord(r) = &sm.payload {
                let mut bad = r.clone();
                tampered_round = Some(bad.round);
                bad.drops.push(pid(9));
                *sm = SignedMessage::sign(
                    &coord,
                    SWARM_PROTO_VERSION,
                    SwarmMessage::RoundRecord(bad),
                )
                .unwrap();
                break;
            }
        }
    }
    assert_eq!(tampered_round, Some(0));

    match replay(&env, params, trace.into_iter()) {
        Err(ReplayError::Diverged(d)) => {
            assert_eq!(d.round, 0);
            assert!(d.rederived.is_some());
            assert!(d.recorded.drops.contains(&pid(9)));
            assert!(!d.rederived.unwrap().drops.contains(&pid(9)));
        }
        other => panic!("expected a divergence at round 0, got {other:?}"),
    }
}

// ----- OBS: digest tally flags the outlier -----

#[test]
fn digest_quorum_flags_outlier() {
    let good = StateDigest([0xAA; 16]);
    let bad = StateDigest([0xBB; 16]);
    let reports = vec![(pid(1), good), (pid(2), good), (pid(3), bad)];

    let verdict = digest_tally(5, reports, 2); // quorum = 2
    assert_eq!(verdict.quorum_digest, Some(good));
    assert_eq!(verdict.outliers, vec![pid(3)]);
    assert_eq!(verdict.reporters, 3);
    assert!(!verdict.agreed);
    assert!(verdict.is_desync());

    // Full agreement → no outliers, not a desync.
    let all_good = vec![(pid(1), good), (pid(2), good), (pid(3), good)];
    let ok = digest_tally(5, all_good, 2);
    assert!(ok.agreed);
    assert!(ok.outliers.is_empty());
    assert!(!ok.is_desync());

    // Same, folded straight from a message log.
    let mut log = MessageLog::new(RUN_ID);
    log.append(digest_msg(&key(1), 5, good));
    log.append(digest_msg(&key(2), 5, good));
    log.append(digest_msg(&key(3), 5, bad));
    let from_log = digest_tally_from_log(&log, 5, 2);
    assert_eq!(from_log.quorum_digest, Some(good));
    assert_eq!(from_log.outliers, vec![pid(3)]);
    assert!(from_log.is_desync());
}

// ----- OBS: run-health projection -----

#[test]
fn run_health_projects_per_round_facts() {
    let (_, trace) = live_run(&sample_envelope(2), &CoordinatorParams::default(), 2);
    let mut log = MessageLog::new(RUN_ID);
    for input in &trace {
        if let Input::Message(sm) = input {
            log.append(sm.clone());
        }
    }
    // Add per-round observability messages the coordinator run doesn't itself emit.
    let good = StateDigest([7; 16]);
    for r in 0..2u64 {
        log.append(digest_msg(&key(1), r, good));
        log.append(digest_msg(&key(2), r, good));
    }
    log.append(straggle_msg(&key(2), 1));

    let health = RunHealth::from_log(&log);
    assert_eq!(health.run_id, RUN_ID);
    assert_eq!(health.rounds.len(), 2);
    for rh in &health.rounds {
        assert_eq!(rh.committed, 2, "both peers committed + evidenced");
        assert!(rh.finalized);
        assert!(rh.drops.is_empty());
        assert_eq!(rh.digest_reporters, 2);
        assert!(rh.digest_agreed);
    }
    assert_eq!(health.rounds[1].stragglers, vec![pid(2)]);
    assert!(health.rounds[0].stragglers.is_empty());
}
