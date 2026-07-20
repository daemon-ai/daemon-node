// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The first testkit whole-run gate (refactor §6, §10 gate table; tier-2): the SPARTA-shaped
// `toy_averager.wasm` PRODUCTION blob (timers + publish, no rounds, no coordinator) runs under the
// real host event-loop driver with simulated capability providers, is journaled end-to-end, and is
// then re-driven through the §8.7 input-replay engine — every decision (channel + seq + payload
// hash) and the terminal outcome must reproduce bit-for-bit. This generalizes the A2 t2 join-run's
// inline replay soak (refactor §12.6) into reusable testkit infrastructure.
//
// Dev/test harness: shells `cargo build` for the guests (the same pattern as the host crate's
// event_loop test) and reads the `.wasm`, so the fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::time::Duration;

use daemon_vhc_abi::DEFAULT_CHANNEL_CONTROL_ID;
use daemon_vhc_host::run::{RunEnd, RunIdentity};
use daemon_vhc_testkit::run::{whole_run, RunSpec};

fn toy_averager_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("toy_averager")
}

/// The whole-run gate: run the production toy blob under the testkit driver → 3 signed publishes →
/// clean stop → §8.7 replay reproduces every decision bit-for-bit.
#[test]
fn toy_averager_whole_run_journals_and_replays_bit_for_bit() {
    let wasm = toy_averager_wasm();
    let worker = daemon_vhc_testkit::worker().expect("testkit worker");

    let identity = RunIdentity {
        run_id: [0xC0; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    // Config byte 0 = 3: average over three timer ticks (three publishes).
    let spec = RunSpec {
        timeout: Duration::from_secs(60),
        ..RunSpec::self_driven(identity, [0x51; 32], vec![3u8], b"grants-tbd".to_vec(), 3)
    };

    let report = whole_run(&worker, &wasm, spec).expect("whole run completes");

    // The live run ended cleanly.
    assert!(
        matches!(report.end, RunEnd::Outcome(code) if code == daemon_vhc_abi::OUTCOME_OK),
        "clean outcome, got {:?}",
        report.end
    );
    // Exactly three publishes, dense seqs from 0 on the control channel.
    assert_eq!(report.recorded_publishes.len(), 3, "three gossip publishes");
    for (i, (channel, seq, _)) in report.recorded_publishes.iter().enumerate() {
        assert_eq!(*channel, u64::from(DEFAULT_CHANNEL_CONTROL_ID));
        assert_eq!(*seq, i as u64, "durable seq dense + monotone from 0");
    }
    // The §8.7 soak reproduced every decision bit-for-bit — the gate.
    assert!(
        report.replay.matched,
        "input replay diverged: replay {:?} ended {:?} vs recorded {:?}",
        report.replay.decisions, report.replay.end, report.recorded_publishes
    );
    assert!(
        report.is_green(),
        "whole run green (clean outcome + replay match)"
    );
}

/// A production blob whose journal replays clean is the reusable regression the testkit owns: run
/// twice and confirm each run independently replays bit-for-bit (the live payloads themselves are
/// wall-clock-derived, so cross-run equality is NOT claimed — only per-run replay reproduction, the
/// §8.7 contract).
#[test]
fn every_run_independently_replays_clean() {
    let wasm = toy_averager_wasm();
    let worker = daemon_vhc_testkit::worker().expect("testkit worker");
    for instance in 1..=2u64 {
        let identity = RunIdentity {
            run_id: [0xC1; 32],
            epoch: 0,
            role: "trainer".to_string(),
            instance,
            module: *blake3::hash(&wasm).as_bytes(),
        };
        let spec = RunSpec::self_driven(identity, [0x52; 32], vec![2u8], Vec::new(), 2);
        let report = whole_run(&worker, &wasm, spec).expect("whole run");
        assert!(report.is_green(), "run {instance} green");
        assert_eq!(report.recorded_publishes.len(), 2);
    }
}
