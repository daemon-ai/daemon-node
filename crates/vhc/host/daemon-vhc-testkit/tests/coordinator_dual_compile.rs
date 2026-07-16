// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The D2 dual-compilation identity gate (refactor §8/D2 acceptance): the PRODUCTION
// `coordinator_quorum.wasm` blob, run under the real major-2 event-loop driver, produces
// byte-for-byte the SAME coordinator decisions (published RoundOpen/RoundRecord frames) as the
// NATIVE `daemon_vhc_sdk_consensus::coordinator::tick` over identical inputs, across a 20-round
// run. Both sides link the SAME `tick` (relocated into sdk-consensus at D2); this asserts that its
// wasm32 compilation (NaN canonicalization, deterministic core semantics) never diverges from the
// native compilation — the guarantee that lets consensus be a module.
//
// The "existing observe tests are the seed" (refactor §8/D2): those run `tick` natively over an
// Input trace; this adds the wasm arm and compares.
//
// Determinism: `tick` takes time only as `Input::Clock`, and the coordinator guest advances a
// SYNTHETIC clock one tick per delivered event (a pure function of the event count, never
// wall-clock — the pump's logical clock is wall-clock, so a live-timer run would not be
// reproducible). The native reference mirrors that exactly (one `tick(Clock)` after each
// authenticated message). The run is driven entirely by event-driven fast paths (ready-heartbeat
// → open, all-committed → witness, all-evidenced → record), so no timeout ever fires.
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
// fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use ciborium::value::Value;

use daemon_vhc_abi::{CandidateDriver, STOP_REASON_RUN_COMPLETE};
use daemon_vhc_host::v2::{
    start_run, DeliverVerdict, MemorySink, RunEnd, RunIdentity, V2RunConfig,
};
use daemon_vhc_host::{select_driver, EngineConfig, Worker};
use daemon_vhc_proto::envelope::{
    Access, Artifact, DataSection, ExperimentSection, GlobalBatch, Phases, Requirements, RoundMode,
    RunSection, StopCondition,
};
use daemon_vhc_proto::messages::{
    Commitment, Heartbeat, Join, RecordEntry, StorageReceipt, SwarmMessage, ThroughputClass,
};
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, peer_id, to_canonical_vec, CapabilitySet, Envelope, Hash,
    IrohId, PeerId, Seed, SignedMessage, SigningKey, SWARM_PROTO_VERSION,
};
use daemon_vhc_sdk_consensus::coordinator::{
    tick, tick_authenticated, CoordinatorParams, CoordinatorState, Input, Output, RunConfig,
};
use daemon_vhc_sdk_consensus::{Authorized, DEFAULT_RECORDS_CHANNEL};

// -- guest build (the established testkit pattern) -------------------------------------------------

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
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/coordinator_quorum.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

// -- the run: envelope, state, and a deterministic 20-round input script --------------------------

const ROUNDS: u64 = 20;
const RUN_ID: &str = "d2-dual-compile";

/// A continuous-round envelope: `epoch_rounds = 0` (no epoch boundary) and huge phase deadlines so
/// only the event-driven fast paths fire — the run is a pure function of the message sequence.
fn continuous_envelope() -> Envelope {
    let mut artifacts = BTreeMap::new();
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
            run_id: RUN_ID.into(),
            min_peers: 2,
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
                start: 4,
                end: 4,
                ramp_rounds: 1,
            },
            stop: StopCondition::Rounds(1_000_000),
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
            warmup: 1_000_000,
            round_train_max: 1_000_000,
            round_witness: 1_000_000,
            cooldown: 1_000_000,
            epoch_rounds: 0,
            checkpoint_every_epochs: 1,
            stall_rounds_max: 2,
            payload_retention_rounds: 4,
        },
    }
}

fn initial_state() -> (CoordinatorState, Envelope) {
    let envelope = continuous_envelope();
    let params = CoordinatorParams {
        seq_len: 9,
        witness_target: 0, // 0 = every peer witnesses; the run evidences via StorageReceipts
        overlap_bps: 0,
        k_absences: 8,
        verification_percent: 0,
        authorized: Vec::new(),
    };
    let config = RunConfig::from_envelope(&envelope, params).expect("run config");
    let state = CoordinatorState::new(config, Seed([0x33; 32]), 0);
    (state, envelope)
}

/// One authenticated message in the input script: the (host-authenticated) signer, its signing key
/// (to author the tag-12 evidence frame the guest run journals), and the payload.
struct ScriptMsg {
    signer: PeerId,
    key: SigningKey,
    msg: SwarmMessage,
}

/// Build the deterministic 20-round script: join the two workers, exit warmup via ready
/// heartbeats, then per round two commitments + one storage receipt (the all-committed →
/// all-evidenced fast path that finalizes a record and opens the next round).
// The `for i in 0..2` loops index the roster/key vectors through the `push` closure deliberately
// (the closure captures both `peers` and `worker_keys`), so the index is the natural driver here.
#[allow(clippy::needless_range_loop)]
fn build_script(envelope_hash: Hash) -> Vec<ScriptMsg> {
    let worker_keys: Vec<SigningKey> = (0..2)
        .map(|i| {
            SigningKey::from_bytes(
                blake3::hash(format!("d2-dual/worker/{i}").as_bytes()).as_bytes(),
            )
        })
        .collect();
    let peers: Vec<PeerId> = worker_keys.iter().map(peer_id).collect();
    let mut script = Vec::new();
    let push = |script: &mut Vec<ScriptMsg>, i: usize, msg: SwarmMessage| {
        script.push(ScriptMsg {
            signer: peers[i],
            key: worker_keys[i].clone(),
            msg,
        });
    };

    // Join both workers (admitted during WaitingForMembers), then ready-heartbeats to open round 0.
    for i in 0..2 {
        push(
            &mut script,
            i,
            SwarmMessage::Join(Join {
                run_id: RUN_ID.into(),
                iroh_id: IrohId([0x44; 32]),
                class: ThroughputClass::C1,
                capabilities: CapabilitySet::new(),
                envelope_hash: Some(envelope_hash),
            }),
        );
    }
    for i in 0..2 {
        push(
            &mut script,
            i,
            SwarmMessage::Heartbeat(Heartbeat {
                round: 0,
                ready: Some(true),
            }),
        );
    }

    // Per round: a commitment from each worker (all-committed) + one storage receipt covering the
    // set (all-evidenced) — finalize the record, open the next round.
    for round in 0..ROUNDS {
        let mut entries = Vec::new();
        for i in 0..2 {
            let payload = blake3_hash(format!("update/{}/{round}", i).as_bytes());
            let size = 64;
            push(
                &mut script,
                i,
                SwarmMessage::Commitment(Commitment {
                    round,
                    payload,
                    size,
                    locators: Vec::new(),
                }),
            );
            entries.push(RecordEntry {
                peer: peers[i],
                hash: payload,
                size,
            });
        }
        // Availability evidence for the whole set (from worker 0, any authenticated sender works).
        push(
            &mut script,
            0,
            SwarmMessage::StorageReceipt(StorageReceipt {
                round,
                verified: entries,
            }),
        );
    }
    script
}

// -- native reference + wasm arm ------------------------------------------------------------------

fn collect(outputs: &[Output], out: &mut Vec<SwarmMessage>) {
    for o in outputs {
        if let Output::Publish(m) = o {
            out.push((**m).clone());
        }
    }
}

/// Fold the native `tick` over the script exactly as the guest loop does: for each message, apply
/// the authenticated dispatch, then advance the synthetic clock by one and apply `tick(Clock)`.
fn run_native(mut state: CoordinatorState, script: &[ScriptMsg]) -> Vec<SwarmMessage> {
    let version = state.config.proto_version;
    let mut published = Vec::new();
    let mut now_s = 0u64;
    for sm in script {
        // The same D1 token the guest mints for a host-verified authoritative-channel delivery.
        let token = Authorized::from_authoritative_channel(DEFAULT_RECORDS_CHANNEL);
        let (next, outputs) = tick_authenticated(state, sm.signer, version, sm.msg.clone(), token);
        state = next;
        collect(&outputs, &mut published);
        now_s += 1;
        let (next, outputs) = tick(state, Input::Clock(now_s));
        state = next;
        collect(&outputs, &mut published);
    }
    published
}

/// Encode the guest's `da_init` config: `{ "state": <CoordinatorState> }` (the guest defaults
/// `tick_period_ms`/`control_channel` to 0 — the deterministic event-driven clock, channel 0).
fn guest_config(state: &CoordinatorState) -> Vec<u8> {
    let state_val = Value::serialized(state).expect("state to cbor value");
    let init = Value::Map(vec![(Value::Text("state".into()), state_val)]);
    to_canonical_vec(&init).expect("init cbor")
}

/// Decode a worker/coordinator publish stream: each entry is `(channel, seq, signed-frame bytes)`;
/// the signed frame is `[envelope, payload, sig]` and the payload decodes to a `SwarmMessage`.
fn decode_published(published: &[(u64, u64, Vec<u8>)]) -> Vec<SwarmMessage> {
    published
        .iter()
        .filter_map(|(_, _, frame)| {
            let v: Value = ciborium::de::from_reader(frame.as_slice()).ok()?;
            let Value::Array(parts) = v else { return None };
            let Value::Bytes(payload) = parts.get(1)? else {
                return None;
            };
            from_canonical_slice::<SwarmMessage>(payload).ok()
        })
        .collect()
}

/// Run the production `coordinator_quorum.wasm` under the real driver, deliver the script as
/// host-verified frames, and return the coordinator's published decisions in order.
fn run_wasm(
    wasm: &[u8],
    state: &CoordinatorState,
    script: &[ScriptMsg],
    expected: usize,
) -> Vec<SwarmMessage> {
    let module_hash = *blake3::hash(wasm).as_bytes();
    let engine = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&engine, wasm, Some(&module_hash)).expect("selection");
    assert_eq!(
        sel.driver,
        CandidateDriver::V2,
        "coordinator-quorum must select the major-2 driver"
    );
    let identity = RunIdentity {
        run_id: *blake3::hash(RUN_ID.as_bytes()).as_bytes(),
        epoch: 0,
        role: "coordinator".to_string(),
        instance: 0,
        module: module_hash,
    };
    let key_seed = *blake3::hash(b"d2-dual/coordinator-frame-key").as_bytes();
    let config = guest_config(state);
    let grants = daemon_vhc_testkit::barrier::phase_a_grants();
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run_cfg = V2RunConfig::new(identity, key_seed, config, grants);
    let run = start_run(&engine, wasm, run_cfg, Box::new(sink.clone())).expect("start_run");
    let pump = run.pump.clone();

    // Deliver every script message as a host-verified authoritative frame (channel 0), one dense
    // seq per sender, back-pressuring on a full spool (never dropping a consensus input).
    let mut seqs: BTreeMap<[u8; 32], u64> = BTreeMap::new();
    for sm in script {
        let payload = to_canonical_vec(&sm.msg).expect("payload cbor");
        let signed =
            SignedMessage::sign(&sm.key, SWARM_PROTO_VERSION, sm.msg.clone()).expect("sign");
        let evidence = to_canonical_vec(&signed).expect("evidence cbor");
        let seq = seqs.entry(sm.signer.0).or_insert(0);
        loop {
            match pump
                .deliver_frame(0, *seq, sm.signer.0, payload.clone(), evidence.clone())
                .expect("deliver")
            {
                DeliverVerdict::Accepted => break,
                DeliverVerdict::SpoolFull | DeliverVerdict::SenderQuota => {
                    std::thread::sleep(Duration::from_millis(2));
                }
                other => panic!("unexpected deliver verdict: {other:?}"),
            }
        }
        *seq += 1;
    }

    // Wait (bounded) for the guest to drain the script into `expected` publishes.
    let deadline = Instant::now() + Duration::from_secs(60);
    while pump.published().len() < expected {
        assert!(
            Instant::now() < deadline,
            "coordinator guest produced {} of {expected} publishes before timeout",
            pump.published().len()
        );
        std::thread::sleep(Duration::from_millis(5));
    }

    pump.stop(STOP_REASON_RUN_COMPLETE).expect("stop");
    let end = run.wait().expect("guest thread");
    assert!(
        matches!(end, RunEnd::Outcome(0)),
        "coordinator guest clean outcome, got {end:?}"
    );
    decode_published(&pump.published())
}

#[test]
fn wasm_coordinator_matches_native_tick_over_20_rounds() {
    let wasm = coordinator_quorum_wasm();
    let (state, envelope) = initial_state();
    let envelope_hash = state.config.envelope_hash;
    // Sanity: the state we hand the guest is the one built from this envelope.
    assert_eq!(
        envelope_hash,
        RunConfig::from_envelope(&envelope, CoordinatorParams::default())
            .unwrap()
            .envelope_hash
    );
    let script = build_script(envelope_hash);

    let native = run_native(state.clone(), &script);
    // The run must actually exercise 20 rounds: RoundOpen(0) + 20 × (RoundRecord + next RoundOpen).
    let records = native
        .iter()
        .filter(|m| matches!(m, SwarmMessage::RoundRecord(_)))
        .count();
    assert_eq!(records as u64, ROUNDS, "native drove {ROUNDS} records");
    let opens = native
        .iter()
        .filter(|m| matches!(m, SwarmMessage::RoundOpen(_)))
        .count();
    assert_eq!(
        opens as u64,
        ROUNDS + 1,
        "one open per round plus the trailing open"
    );

    let wasm_out = run_wasm(&wasm, &state, &script, native.len());

    assert_eq!(
        wasm_out.len(),
        native.len(),
        "wasm and native produced a different number of decisions"
    );
    for (i, (w, n)) in wasm_out.iter().zip(native.iter()).enumerate() {
        assert_eq!(
            to_canonical_vec(w).unwrap(),
            to_canonical_vec(n).unwrap(),
            "decision {i} diverged: wasm {w:?} vs native {n:?}"
        );
    }
}
