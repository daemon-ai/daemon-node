// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The richer tier-2 whole-run gate (refactor §6, B3 sitting 2): the PRODUCTION
// `tiny_llama_v2.wasm` blob (the macro-emitted BarrierRound guest) under the testkit's
// in-process native coordinator — real rounds (train → commit → record → barrier ingest),
// journaled, §8.7 replay-verified, with the guest config authored SDK-FREE (raw canonical CBOR;
// the dependency wall: the testkit never links sdk/*). Generalizes the A2 t2 `v2_join` run out
// of `daemon-vhc-worker` into reusable testkit infrastructure — and extends it to the
// multi-worker shape: 2 wasm workers under one coordinator, with cross-worker det-lane digest
// agreement as the whole-run oracle.
//
// Dev/test harness: shells `cargo build` for the guests (the established pattern), so the
// fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;

use daemon_vhc_host::v2::RunEnd;
use daemon_vhc_testkit::{barrier_whole_run, BarrierSpec};

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

/// The single-worker barrier whole-run: two full rounds under the native coordinator, journaled,
/// §8.7 replay-verified — the A2 t2 shape as reusable testkit infra with an SDK-free config.
#[test]
fn single_worker_tiny_llama_whole_run_replays_bit_for_bit() {
    let wasm = tiny_llama_v2_wasm();
    let spec = BarrierSpec::new("testkit-barrier-1w", 1, 2);
    let report = barrier_whole_run(&wasm, &spec).expect("whole run completes");

    assert_eq!(report.rounds_done, 2);
    assert_eq!(report.workers.len(), 1);
    let w = &report.workers[0];
    assert!(
        matches!(w.end, RunEnd::Outcome(0)),
        "clean outcome, got {:?}",
        w.end
    );
    // Per round: one Commitment + one Digest = 4 decisions over 2 rounds.
    assert_eq!(w.replay_decisions, 4, "2 rounds × (commitment + digest)");
    assert!(
        w.replay_matched,
        "§8.7 input replay reproduced every decision"
    );
    assert!(report.is_green());
}

/// The multi-worker whole-run: 2 production wasm workers under one native coordinator — each
/// trains its assigned split, commits, and barrier-ingests the full committed set; the det-lane
/// digest agreement across workers is the whole-run oracle (architecture §3.6 claim 2).
#[test]
fn two_workers_agree_on_the_det_lane_digest() {
    let wasm = tiny_llama_v2_wasm();
    let spec = BarrierSpec::new("testkit-barrier-2w", 2, 2);
    let report = barrier_whole_run(&wasm, &spec).expect("whole run completes");

    assert_eq!(report.rounds_done, 2);
    assert_eq!(report.workers.len(), 2);
    for (i, w) in report.workers.iter().enumerate() {
        assert!(
            matches!(w.end, RunEnd::Outcome(0)),
            "worker {i} clean outcome, got {:?}",
            w.end
        );
        assert!(w.replay_matched, "worker {i} §8.7 replay matched");
        assert_eq!(
            w.replay_decisions, 4,
            "worker {i}: 2 × (commitment + digest)"
        );
    }
    // The det-lane agreement: both workers ingested the identical committed set in record order
    // and hold bit-identical final consensus state.
    assert_eq!(
        report.workers[0].digest, report.workers[1].digest,
        "cross-worker det-lane digest agreement"
    );
    assert!(report.is_green());
}
