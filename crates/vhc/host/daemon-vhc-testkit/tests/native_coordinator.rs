// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The reusable native-coordinator building block (refactor §6, tier-1): from a run envelope, the
// testkit's NativeCoordinator admits a worker and — once warmup elapses — opens round 0. This is
// the "native coordinator" half the tiny-llama-v2 barrier-round whole run composes with the
// event-loop driver (the A2 t2 shape lifted into testkit infra); it links host/* only.

use ciborium::value::Value;
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, ExperimentSection, GlobalBatch, Phases, Requirements, RoundMode,
    RunSection, StopCondition,
};
use daemon_vhc_proto::{Envelope, Hash, SigningKey, SwarmMessage};
use daemon_vhc_testkit::NativeCoordinator;

fn envelope() -> Envelope {
    let mut artifacts = std::collections::BTreeMap::new();
    artifacts.insert(
        "experiment.wasm".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([1; 32]),
        },
    );
    artifacts.insert(
        "data.manifest".to_string(),
        Artifact {
            url: "file:///dev/null".into(),
            blake3: Hash([2; 32]),
        },
    );
    Envelope {
        run: RunSection {
            schema: 1,
            run_id: "testkit-coord".into(),
            min_peers: 1,
            max_peers: 4,
            access: Access::Org,
        },
        experiment: ExperimentSection {
            module: "experiment.wasm".into(),
            abi: "tensor-abi@1".into(),
            config: Value::Null,
        },
        artifacts,
        data: DataSection {
            manifest: "data.manifest".into(),
            steps_per_round: 2,
            global_batch: GlobalBatch {
                start: 2,
                end: 2,
                ramp_rounds: 1,
            },
            stop: StopCondition::Tokens(1_000_000),
        },
        requirements: Requirements {
            vram_mb_min: 0,
            ram_gb_min: 1,
            uplink_mbps_min: 1,
            downlink_mbps_min: 1,
            disk_gb_min: 1,
            throughput_floor: "c1".into(),
            update_mb_max: 8,
            capabilities: vec![],
            payload_store: "r2".into(),
        },
        phases: Phases {
            round_mode: RoundMode::Barrier,
            warmup: 1,
            round_train_max: 60,
            round_witness: 1,
            cooldown: 1,
            epoch_rounds: 10,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 4,
        },
    }
}

#[test]
fn native_coordinator_admits_a_worker_and_opens_round_zero() {
    let env = envelope();
    let coord_key = SigningKey::from_bytes(&[0x42; 32]);
    let worker_key = SigningKey::from_bytes(&[0x7A; 32]);
    let mut coord = NativeCoordinator::from_envelope(&env, coord_key, 9)
        .expect("coordinator config from envelope");

    coord
        .join(&worker_key, "testkit-coord")
        .expect("worker admitted");

    // Advance the clock (1 s per step, bounded) until the coordinator opens a round — the same
    // timeout-driven drive the A2 t2 whole run uses (warmup then the round cadence).
    let mut opened = None;
    'drive: for _ in 0..10_000 {
        coord.advance_clock(1).expect("advance clock");
        while let Some(msg) = coord.next_message() {
            if matches!(msg, SwarmMessage::RoundOpen(_)) {
                opened = Some(msg);
                break 'drive;
            }
        }
    }
    let round_open = opened.expect("the native coordinator opened a round");
    let SwarmMessage::RoundOpen(ref ro) = round_open else {
        unreachable!()
    };
    assert_eq!(ro.round, 0, "round 0 opened");

    // The coordinator signs its authoritative messages (the §12 evidentiary envelope a worker
    // verifies above its pump) — the sign path the whole-run harness relies on.
    let signed = coord.sign(round_open).expect("coordinator signs");
    signed.verify().expect("coordinator signature verifies");
}
