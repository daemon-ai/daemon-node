// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Driver-level backend selection: a run whose admitted backend cannot serve refuses TYPED at
//! `start_run` (`RunError::BackendUnavailable`) — never a silent ndarray run, never a panic.
//!
//! The wgpu case is feature-gated but HARDWARE-INDEPENDENT: on a GPU-less runner the
//! availability probe refuses (no adapter); on a GPU runner this binary holds the process
//! device-compute slot first, so the pre-spawn slot peek refuses. Either way the typed
//! `BackendUnavailable` is the observable. (Own test binary on purpose: the device-compute
//! slot is process-global, so holding it here can never interfere with the end-to-end device
//! tiers in the goldens binary.)
//!
//! Dev/test harness: shells `cargo build` for the guests, so fs/process bans are allowed
//! file-wide.
#![allow(clippy::disallowed_methods)]
#![cfg(feature = "wgpu")]

use std::sync::{Arc, Mutex};

use daemon_vhc_host::run::{start_run, MemorySink, RunConfig, RunError, RunIdentity};
use daemon_vhc_host::{DeviceComputeGuard, EngineConfig, Worker};

fn guest(name: &str) -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm(name)
}

/// An unavailable admitted wgpu backend (device-less runner OR an occupied device-compute
/// slot) refuses `start_run` with the typed `BackendUnavailable` — before any guest code and
/// before the run header is journaled.
#[test]
fn unavailable_wgpu_backend_refuses_start_run_typed() {
    // Hold the slot when it is free: on a GPU runner this forces the slot-peek refusal; on a
    // GPU-less runner the availability probe refuses first. Both are the same typed surface.
    let _slot = DeviceComputeGuard::acquire().ok();

    let wasm = guest("test_compute_v2");
    let worker = Worker::new(EngineConfig {
        backend: daemon_vhc_host::BackendKind::Wgpu,
        ..EngineConfig::default()
    })
    .expect("engine");
    let identity = RunIdentity {
        run_id: [0x71; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: 1,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let run_cfg = RunConfig::new(identity, [0x9c; 32], Vec::new(), Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let err = match start_run(&worker, &wasm, run_cfg, Box::new(sink.clone())) {
        Err(e) => e,
        Ok(_) => panic!("an unservable admitted backend must refuse the run"),
    };
    assert!(
        matches!(err, RunError::BackendUnavailable(_)),
        "typed BackendUnavailable, got {err:?}"
    );
    // The refusal precedes the run header: nothing was journaled for this instance.
    assert!(
        sink.lock().expect("sink").entries.is_empty(),
        "the refusal must precede any journal write"
    );
}
