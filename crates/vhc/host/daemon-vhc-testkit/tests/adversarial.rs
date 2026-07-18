// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// Adversarial-suite seeds (architecture §4.2's conformance surface): trace-driven fault injection
// against the PRODUCTION `tiny_llama_c3.wasm` compute@2 trainer **under the production
// `coordinator_quorum.wasm` coordinator** (the cell-8 whole-run drive), through the shared
// deterministic `FaultPlan` — one pinned case per rig primitive proving the harness shape. The
// full `Authority` adversarial suites (partitions, equivocation, withheld records) live in
// `adversarial_authority.rs`; they drive the same coordinator blob.
//
// Consensus runs only in the sandboxed, content-addressed module: both the coordinator and the
// workers execute under the real major-2 event-loop driver, and the fault is injected purely at
// the network seat (coordinator→worker frame delivery / committed-payload staging). No native
// coordinator tick backs these drills.
//
// Pinned cases:
// 1. duplicate RoundRecord → the round driver's ingest watermark dedups: ingested exactly once,
//    identical guest-voiced det-lane end state to the clean run, §8.7 replay green under the
//    fault.
// 2. delayed committed payloads → the record arrives unmintable → the round driver stalls (the
//    compute@2 trainer voices no digest for the stalled round) → the payloads arrive at the next
//    open → the guest ingests the stalled round and folds its digest into state — observable as
//    the ABSENT per-round tag-4 voice plus the preserved final-state agreement with the clean
//    run; replay green through the detour.
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
// fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_vhc_testkit::{
    cell8_whole_run, Cell8Spec, FaultAction, FaultPlan, FaultRule, FrameKind,
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

fn guest_wasm(name: &str) -> Vec<u8> {
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
    let path = guests_root().join(format!("target/wasm32-unknown-unknown/release/{name}.wasm"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Pinned case 1 — a byte-identical duplicate `RoundRecord` (same seq, same signed frame) is
/// **ingested exactly once**: the round driver's watermark (`rr.round <= last_ingested`) is the
/// dedup under test. The faulted run — driven through the real wasm coordinator — ends in the
/// identical guest-voiced det-lane state as the clean run, and its journal §8.7 replays
/// bit-for-bit (replay re-feeds the delivered sequence, duplicate included).
#[test]
fn duplicate_round_record_is_ingested_once() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama_c3");
    let run_label = "cell8-adv-dup-record";

    // Clean baseline (same run label, so every derived seed/key matches).
    let clean = cell8_whole_run(&coordinator, &worker, &Cell8Spec::new(run_label, 1, 2))
        .expect("clean run");
    assert!(clean.is_green());

    // Faulted run: round 0's record delivered twice to worker 0.
    let mut spec = Cell8Spec::new(run_label, 1, 2);
    spec.faults = FaultPlan {
        rules: vec![FaultRule {
            worker: 0,
            round: 0,
            kind: FrameKind::Record,
            action: FaultAction::Duplicate,
        }],
        delay_payload_staging: Vec::new(),
    };
    let faulted = cell8_whole_run(&coordinator, &worker, &spec).expect("faulted run");

    let w = &faulted.workers[0];
    assert!(w.replay_matched, "§8.7 replay green under the duplicate");
    assert_eq!(
        w.digests_for(0),
        1,
        "round 0 ingested exactly once despite the duplicate record (one tag-4 voice)"
    );
    assert_eq!(w.digests_for(1), 1);
    assert_eq!(
        w.digest, clean.workers[0].digest,
        "the duplicate changed nothing: identical guest-voiced det-lane end state"
    );
    assert!(faulted.is_green());
}

/// Pinned case 2 — a record whose committed payloads are **delayed** past it: the round driver
/// cannot mint the committed set and stalls (the compute@2 trainer voices NO digest for the
/// stalled round — the straggle detour is observable as the absent tag-4), then catches up when
/// the payloads stage at the next open: the stalled round's ingest folds into the guest state
/// (the §5.9 ingest epilogue fires at the ingest→training boundary), so the FINAL digest equals
/// the clean run's — the detour changes the path, never the state — and the journal §8.7 replays
/// bit-for-bit through it.
#[test]
fn delayed_committed_payloads_stall_then_catch_up() {
    let coordinator = guest_wasm("coordinator_quorum");
    let worker = guest_wasm("tiny_llama_c3");
    let run_label = "cell8-adv-delayed-payloads";

    let clean = cell8_whole_run(&coordinator, &worker, &Cell8Spec::new(run_label, 1, 2))
        .expect("clean run");
    assert!(clean.is_green());

    // Faulted run: round 0's committed payloads reach worker 0 only at round 1's open.
    let mut spec = Cell8Spec::new(run_label, 1, 2);
    spec.faults = FaultPlan {
        rules: Vec::new(),
        delay_payload_staging: vec![(0, 0)],
    };
    let faulted = cell8_whole_run(&coordinator, &worker, &spec).expect("faulted run");

    let w = &faulted.workers[0];
    assert!(
        w.replay_matched,
        "§8.7 replay green through the straggle detour"
    );
    assert_eq!(
        w.digests_for(0),
        0,
        "the stalled round's digest is folded at catch-up, never voiced per-round"
    );
    assert_eq!(w.digests_for(1), 1, "round 1 voices its digest normally");
    // The detour's recorded decisions: theta+commitment per round + the single (final) digest.
    assert_eq!(w.replay_decisions, 5);
    assert!(faulted.is_green());

    // Catch-up parity: the detour changes the path, never the state — the final guest-voiced
    // det digest equals the clean run's (which ingested round 0 on time).
    assert_eq!(
        w.digest, clean.workers[0].digest,
        "the straggle detour must change the path, not the state: identical det-lane end state"
    );
}
