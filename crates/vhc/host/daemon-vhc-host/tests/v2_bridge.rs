// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The tabi@1-bridge-under-major-2 conformance (ABI §2.5; A2 choreography move): the same frozen
// dispatch the v1 driver links, genericized over the v2 store —
//
//   - registration in da_init + tensor ops inside event slices work; the read-out scalar
//     (ones + ones-param = 2.0) leaves as a signed publish, and its nr-class result is journaled
//     under the §2.7 reserved kind 128;
//   - registration inside a slice traps PhaseViolation (§2.5 rule 1 — the phase table is gone,
//     the temporal rule stands);
//   - a slice-class handle retained across a Delivered boundary traps StaleHandle (§7.1 —
//     wholesale arena clearing at slice end, the v1 finish_entry rule relocated).
//
// The load-bearing companion proof is the A0 frozen fixture lane: the v1 driver's behavior is
// byte-for-byte untouched by the genericization (same code, second monomorphization).
#![allow(clippy::disallowed_methods)]

use std::path::PathBuf;
use std::process::Command;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use daemon_vhc_host::v2::{start_run, MemorySink, RunEnd, RunIdentity, SinkEntry, V2RunConfig};
use daemon_vhc_host::{select_driver, EngineConfig, TrapCode, Worker};

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

fn bridge_wasm() -> Vec<u8> {
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
    let path = guests_root().join("target/wasm32-unknown-unknown/release/test_bridge_v2.wasm");
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// `(how the run ended, the journal sink, the published signed frames)`.
type ModeRun = (RunEnd, Arc<Mutex<MemorySink>>, Vec<(u64, u64, Vec<u8>)>);

fn run_mode(mode: u8, instance: u64) -> ModeRun {
    let wasm = bridge_wasm();
    let worker = Worker::new(EngineConfig::default()).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("bridge module is a major-2 candidate (§1.2), admitted");
    assert_eq!(sel.driver, daemon_vhc_abi::CandidateDriver::V2);

    let identity = RunIdentity {
        run_id: [0xDD; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let run_cfg = V2RunConfig::new(identity, [0x71; 32], vec![mode], Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())).expect("start");
    let pump = run.pump.clone();

    // Happy path publishes once; give it time, then stop. Trap modes end on their own.
    if mode == 0 {
        let deadline = Instant::now() + Duration::from_secs(30);
        while pump.published().is_empty() {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for the publish"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE)
            .expect("stop");
    }
    let end = run.wait().expect("guest thread clean");
    let published = pump.published();
    (end, sink, published)
}

#[test]
fn bridge_ops_run_in_slices_and_the_scalar_readout_is_journaled() {
    let (end, sink, published) = run_mode(0, 1);
    match end {
        RunEnd::Outcome(0) => {}
        other => panic!("expected Outcome(0), got {other:?}"),
    }
    // ones([1]) + ones-param w = 2.0, published as 8 LE bytes under the §12.1 envelope.
    assert_eq!(published.len(), 1);
    let frame: ciborium::value::Value =
        ciborium::de::from_reader(published[0].2.as_slice()).expect("frame");
    let ciborium::value::Value::Array(parts) = frame else {
        panic!("frame shape")
    };
    let ciborium::value::Value::Bytes(payload) = &parts[1] else {
        panic!("payload")
    };
    assert_eq!(
        f64::from_le_bytes(payload.as_slice().try_into().unwrap()),
        2.0
    );

    // The scalar@1 nr-class result is journaled under the §2.7 reserved kind 128 with the exact
    // delivered bytes — the input-replay substrate for bridge readouts.
    let entries = &sink.lock().expect("sink").entries;
    assert!(
        entries.iter().any(|e| matches!(
            e,
            SinkEntry::ReadBack { kind: 128, value, .. }
                if f64::from_le_bytes(value.as_slice().try_into().unwrap()) == 2.0
        )),
        "scalar@1 journaled under kind 128"
    );
}

#[test]
fn registration_inside_a_slice_traps_phase_violation() {
    let (end, _sink, _published) = run_mode(1, 2);
    match end {
        RunEnd::Trapped(trap) => {
            assert_eq!(trap.code, TrapCode::PhaseViolation);
            assert!(
                trap.detail.contains("da_init") || trap.detail.contains("registration"),
                "{}",
                trap.detail
            );
        }
        other => panic!("expected PhaseViolation, got {other:?}"),
    }
}

#[test]
fn slice_class_handle_across_a_slice_boundary_traps_stale_handle() {
    let (end, _sink, _published) = run_mode(2, 3);
    match end {
        RunEnd::Trapped(trap) => assert_eq!(trap.code, TrapCode::StaleHandle),
        other => panic!("expected StaleHandle, got {other:?}"),
    }
}
