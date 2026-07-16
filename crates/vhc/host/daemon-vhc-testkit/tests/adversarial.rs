// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Adversarial-suite seeds (B3 sitting 2; architecture §4.2's conformance surface, ahead of
// D1/D2): trace-driven fault injection against the PRODUCTION `tiny_llama_v2.wasm` blob through
// the testkit's deterministic `FaultPlan` — one pinned case per rig primitive proving the
// harness shape. The full `Authority` adversarial suites (partitions, equivocation, withheld
// records, conflicting histories) are D1/D2 deliverables; they grow on this rig.
//
// Pinned cases:
// 1. duplicate RoundRecord → the round driver's ingest watermark dedups: ingested exactly once,
//    identical end state to the clean run, §8.7 replay green under the fault.
// 2. delayed committed payloads → the record arrives unmintable → the guest voices
//    Straggle{fetching} → the payloads arrive at the next open → CaughtUp digest — the straggle
//    ladder exercised end-to-end, replay green.
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
// fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_vhc_proto::messages::SwarmMessage;
use daemon_vhc_testkit::{
    barrier_whole_run, BarrierSpec, FaultAction, FaultPlan, FaultRule, FrameKind,
};

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

fn tiny_llama_v2_wasm() -> Vec<u8> {
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
    let path = guests_root().join("target/wasm32-unknown-unknown/release/tiny_llama_v2.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn digests_for(msgs: &[SwarmMessage], round: u64) -> usize {
    msgs.iter()
        .filter(|m| matches!(m, SwarmMessage::Digest(d) if d.round == round))
        .count()
}

fn straggles_for(msgs: &[SwarmMessage], round: u64) -> usize {
    msgs.iter()
        .filter(|m| matches!(m, SwarmMessage::Straggle(s) if s.round == round))
        .count()
}

/// Pinned case 1 — a byte-identical duplicate `RoundRecord` (same seq, same signed frame) is
/// **ingested exactly once**: the round driver's watermark (`rr.round <= last_ingested`) is the
/// dedup under test (the Phase-A pump has no §4.7 dedup machinery yet — that is B1's bounded
/// delivery classes; the round-driver watermark is the layer that must already hold). The
/// faulted run ends in the identical det-lane state as the clean run, and its journal §8.7
/// replays bit-for-bit (replay re-feeds the delivered sequence, duplicate included).
#[test]
fn duplicate_round_record_is_ingested_once() {
    let wasm = tiny_llama_v2_wasm();
    let run_id = "testkit-adv-dup-record";

    // Clean baseline (same run identity, so every derived seed/key matches).
    let clean = barrier_whole_run(&wasm, &BarrierSpec::new(run_id, 1, 2)).expect("clean run");
    assert!(clean.is_green());

    // Faulted run: round 0's record delivered twice to worker 0.
    let mut spec = BarrierSpec::new(run_id, 1, 2);
    spec.faults = FaultPlan {
        rules: vec![FaultRule {
            worker: 0,
            round: 0,
            kind: FrameKind::Record,
            action: FaultAction::Duplicate,
        }],
        delay_payload_staging: Vec::new(),
    };
    let faulted = barrier_whole_run(&wasm, &spec).expect("faulted run");

    let w = &faulted.workers[0];
    assert!(w.replay_matched, "§8.7 replay green under the duplicate");
    assert_eq!(
        digests_for(&w.messages, 0),
        1,
        "round 0 ingested exactly once despite the duplicate record"
    );
    assert_eq!(digests_for(&w.messages, 1), 1);
    assert_eq!(
        w.digest, clean.workers[0].digest,
        "the duplicate changed nothing: identical det-lane end state"
    );
    assert!(faulted.is_green());
}

/// Pinned case 2 — a record whose committed payloads are **delayed** past it: the guest cannot
/// mint the committed set, voices `Straggle{fetching}` (the observable straggle-path decision),
/// then catches up when the payloads stage at the next open — a `CaughtUp` digest for the
/// stalled round, the identical det-lane end state as the clean run, and a §8.7-green journal
/// (the straggle detour is part of the recorded decision stream and must replay bit-for-bit).
#[test]
fn delayed_committed_payloads_straggle_then_catch_up() {
    let wasm = tiny_llama_v2_wasm();
    let run_id = "testkit-adv-delayed-payloads";

    let clean = barrier_whole_run(&wasm, &BarrierSpec::new(run_id, 1, 2)).expect("clean run");
    assert!(clean.is_green());

    // Faulted run: round 0's committed payloads reach worker 0 only at round 1's open.
    let mut spec = BarrierSpec::new(run_id, 1, 2);
    spec.faults = FaultPlan {
        rules: Vec::new(),
        delay_payload_staging: vec![(0, 0)],
    };
    let faulted = barrier_whole_run(&wasm, &spec).expect("faulted run");

    let w = &faulted.workers[0];
    assert!(
        w.replay_matched,
        "§8.7 replay green through the straggle detour"
    );
    assert_eq!(
        straggles_for(&w.messages, 0),
        1,
        "the guest voiced the straggle for the unmintable round"
    );
    assert_eq!(
        digests_for(&w.messages, 0),
        1,
        "round 0 caught up exactly once"
    );
    assert_eq!(digests_for(&w.messages, 1), 1, "round 1 unaffected");
    // The straggle is an extra recorded decision: 2×(commitment+digest) + 1 straggle.
    assert_eq!(w.replay_decisions, 5);
    assert!(faulted.is_green());

    // Catch-up parity (the FLIPPED pin — defect found by this rig in B3 sitting 2, fixed by
    // B1's §5.9 IngestPhase state machine, flip witnessed on the merge of `vhc-integration`
    // @ 372cd0a: the previous `assert_ne!` divergence pin went red with byte-identical
    // digests). v1 parity holds through the straggle detour: the ingest epilogue
    // (`snapshot_round_bases`) now fires at the first boundary after the ingest math — so when
    // `BarrierRound::on_round_open` ingests round r and trains round r+1 within one slice,
    // round r+1 commits against the POST-ingest round base, exactly as v1's separate
    // `Instance::ingest` export boundary guaranteed. The detour changes the path, never the
    // state. (B1's own `catch_up_after_straggle_reproduces_v1_digests_cpu` pins the same
    // property against the v1 engine oracle; this pin exercises it through the adversarial
    // rig's payload-delay fault.)
    assert_eq!(
        w.digest, clean.workers[0].digest,
        "the straggle detour must change the path, not the state: identical det-lane end state \
         (v1 catch-up parity, §5.9 ingest epilogue at the ingest→training boundary)"
    );
}
