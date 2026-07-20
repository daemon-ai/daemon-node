// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The Phase-C **model-agnostic** whole-run gate (refactor §7: "a non-LLaMA toy authored with zero
// host changes … wire it through the testkit as a lane"; tier-2): the `toy_mlp.wasm` PRODUCTION
// blob — a two-layer MLP trained by SGD, authored purely over `daemon-vhc-sdk-compute` +
// `daemon-vhc-sdk` — runs under the SAME testkit driver + simulated capability providers +
// `compute@2` runner the LLaMA reference uses, is journaled end-to-end, and replays bit-for-bit
// through the §8.7 engine. The lane exists to prove the compute ABI is model-agnostic at the
// whole-run level: nothing in the testkit knows the model; the op stream is dispatched by shape.
//
// Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Once;
use std::time::Duration;

use daemon_vhc_abi::DEFAULT_CHANNEL_CONTROL_ID;
use daemon_vhc_host::run::{RunEnd, RunIdentity};
use daemon_vhc_testkit::run::{whole_run, RunSpec};

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

fn toy_mlp_wasm() -> Vec<u8> {
    BUILD.call_once(|| {
        let status = Command::new("cargo")
            .current_dir(guests_root())
            .env_remove("CARGO_TARGET_DIR")
            .env_remove("RUSTC_WRAPPER")
            .env("RUSTFLAGS", guest_remap_rustflags())
            .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
            .status()
            .expect("run cargo for guests (dev shell provides the wasm target)");
        assert!(status.success(), "building guest modules failed");
    });
    let path = guests_root().join("target/wasm32-unknown-unknown/release/toy_mlp.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Whole-run gate: the production MLP blob trains over `compute@2` under the testkit driver, emits
/// one signed publish (its exported `W1`), stops clean, and replays bit-for-bit — with zero host
/// code specific to the model.
#[test]
fn toy_mlp_whole_run_journals_and_replays_bit_for_bit() {
    let wasm = toy_mlp_wasm();
    let worker = daemon_vhc_testkit::worker().expect("testkit worker");

    let identity = RunIdentity {
        run_id: [0x71; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    // Config byte 0 = 3 SGD steps; the module publishes its trained W1 exactly once.
    let spec = RunSpec {
        timeout: Duration::from_secs(60),
        ..RunSpec::self_driven(identity, [0x53; 32], vec![3u8], Vec::new(), 1)
    };

    let report = whole_run(&worker, &wasm, spec).expect("whole run completes");

    assert!(
        matches!(report.end, RunEnd::Outcome(code) if code == daemon_vhc_abi::OUTCOME_OK),
        "clean outcome, got {:?}",
        report.end
    );
    assert_eq!(
        report.recorded_publishes.len(),
        1,
        "one publish: the trained W1"
    );
    let (channel, seq, _) = report.recorded_publishes[0];
    assert_eq!(channel, u64::from(DEFAULT_CHANNEL_CONTROL_ID));
    assert_eq!(seq, 0, "durable seq dense from 0");
    assert!(
        report.replay.matched,
        "input replay diverged: replay {:?} ended {:?} vs recorded {:?}",
        report.replay.decisions, report.replay.end, report.recorded_publishes
    );
    assert!(
        report.is_green(),
        "whole run green (compute@2, model-agnostic)"
    );
}
