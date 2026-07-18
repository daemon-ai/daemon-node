// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The D2 rig back-pressure prerequisite (refactor §8/D2 acceptance): deterministic
// SpoolFull / SenderQuota cases need a PUMP-LEVEL pause/hold control, because a live guest drains
// the authoritative spool as fast as frames arrive and can never be pushed to the bound. The
// PumpHandle::hold/release control (host-side) freezes delivery so the embedder can fill the spool
// past `spool_frames` / a sender's quota and observe the typed back-pressure verdicts, then release
// and confirm the guest drains cleanly (the reliable class is bounded but NEVER drops — §4.7).
//
// Drain target: the coordinator-quorum guest with the event-driven clock (tick_period_ms = 0) arms
// no timer, so it parks in next_event immediately with no timer-delivery race — the clean fixture
// for a delivery hold.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::{Duration, Instant};

use ciborium::value::Value;

use daemon_vhc_abi::{CandidateDriver, STOP_REASON_RUN_COMPLETE};
use daemon_vhc_host::run::{start_run, DeliverVerdict, MemorySink, RunConfig, RunEnd, RunIdentity};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::envelope::{GlobalBatch, StopCondition};
use daemon_vhc_proto::{to_canonical_vec, CapabilitySet, Hash, VHC_PROTO_VERSION};
use daemon_vhc_sdk_consensus::coordinator::{CoordinatorState, RunConfig as CoordinatorRunConfig};
use std::sync::{Arc, Mutex};

fn guests_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../guests")
        .canonicalize()
        .expect("guests workspace path")
}

fn guest_remap_rustflags() -> String {
    let root = guests_root();
    let checkout = root.ancestors().nth(3).unwrap_or(&root).to_path_buf();
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cargo"));
    format!(
        "--remap-path-prefix={}=/daemon-node --remap-path-prefix={}=/cargo",
        checkout.display(),
        cargo_home.display(),
    )
}

static BUILD: Once = Once::new();

fn coordinator_quorum_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/coordinator_quorum.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn coordinator_config() -> Vec<u8> {
    // The coordinator's opaque config is an authored `RunConfig` + genesis `CoordinatorState`,
    // exactly the guest's `da_init` shape (`{state: …}`) — the same way a genesis envelope carries
    // the coordinator role's config verbatim. Authored directly here (the coordinator module's
    // opaque config under envelope v2; the v1 `[data]`/`[phases]` projection left the envelope at
    // D0). Phase deadlines are effectively infinite: the coordinator-quorum guest runs on the
    // event-driven fast path (`tick_period_ms = 0`) and parks in `next_event`, which is exactly the
    // clean delivery-hold fixture this drill needs.
    let run_config = CoordinatorRunConfig {
        run_id: "pump-hold".to_string(),
        proto_version: VHC_PROTO_VERSION,
        envelope_hash: Hash([0u8; 32]),
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
        seq_len: 1,
        witness_target: 0,
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let state = CoordinatorState::new(run_config, daemon_vhc_proto::Seed([0x33; 32]), 0);
    let state_val = Value::serialized(&state).expect("state value");
    let init = Value::Map(vec![(Value::Text("state".into()), state_val)]);
    to_canonical_vec(&init).expect("init cbor")
}

#[test]
fn hold_forces_deterministic_spoolfull_and_senderquota_then_releases_clean() {
    let wasm = coordinator_quorum_wasm();
    let module_hash = *blake3::hash(&wasm).as_bytes();
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    assert_eq!(
        select_driver(&engine, &wasm, Some(&module_hash))
            .expect("selection")
            .driver,
        CandidateDriver::V2
    );

    let identity = RunIdentity {
        run_id: *blake3::hash(b"pump-hold").as_bytes(),
        epoch: 0,
        role: "coordinator".to_string(),
        instance: 0,
        module: module_hash,
    };
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let mut run_cfg = RunConfig::new(
        identity,
        *blake3::hash(b"pump-hold/key").as_bytes(),
        coordinator_config(),
        daemon_vhc_testkit::genesis_run::phase_a_grants(),
    );
    // Tiny bounds so the back-pressure verdicts hit deterministically.
    run_cfg.spool_frames = 3;
    run_cfg.per_sender_quota = 2;

    let run = start_run(&engine, &wasm, run_cfg, Box::new(sink.clone())).expect("start_run");
    let pump = run.pump.clone();

    // Freeze delivery: the guest cannot drain, so the spool fills deterministically.
    pump.hold();

    let sender_a = [0xA1u8; 32];
    let sender_b = [0xB2u8; 32];
    let sender_c = [0xC3u8; 32];
    let frame = |n: u8| vec![n; 8];

    // Sender A fills its per-sender quota (2), then the 3rd from A is refused SenderQuota — the
    // spool (2/3) is not yet full, so it is the *sender* bound that bites.
    assert_eq!(
        pump.deliver_frame(0, 0, sender_a, frame(0), frame(0))
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 1, sender_a, frame(1), frame(1))
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 2, sender_a, frame(2), frame(2))
            .unwrap(),
        DeliverVerdict::SenderQuota,
        "3rd frame from A exceeds per_sender_quota=2 (spool not yet full)"
    );

    // Sender B takes the last spool slot (spool now 3/3), then any further frame — from any
    // sender — is SpoolFull. Neither is ever dropped (§4.7): the caller holds + retries.
    assert_eq!(
        pump.deliver_frame(0, 0, sender_b, frame(3), frame(3))
            .unwrap(),
        DeliverVerdict::Accepted
    );
    assert_eq!(
        pump.deliver_frame(0, 0, sender_c, frame(4), frame(4))
            .unwrap(),
        DeliverVerdict::SpoolFull,
        "spool at capacity (3) back-pressures every further deliver"
    );

    // Release: the guest drains the 3 spooled frames (it ignores undecodable payloads), which frees
    // spool slots so a re-tried deliver is Accepted — the reliable class recovered, nothing dropped.
    pump.release();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match pump
            .deliver_frame(0, 1, sender_c, frame(5), frame(5))
            .unwrap()
        {
            DeliverVerdict::Accepted => break,
            DeliverVerdict::SpoolFull | DeliverVerdict::SenderQuota => {
                assert!(
                    Instant::now() < deadline,
                    "guest never drained after release"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            other => panic!("unexpected verdict after release: {other:?}"),
        }
    }

    pump.stop(STOP_REASON_RUN_COMPLETE).expect("stop");
    assert!(matches!(
        run.wait().expect("guest thread"),
        RunEnd::Outcome(0)
    ));
}
