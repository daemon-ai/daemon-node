// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
//! A panicking guest tells the host WHAT it panicked about (ABI §3.6).
//!
//! A wasm guest's panic reaches the host as a bare `unreachable`: the payload and its
//! `file:line:col` live in linear memory the trap tears down. Three fleet cycles were spent
//! bisecting an anonymous `GuestPanic` by hand — offline symbolization of a stripped, LTO'd
//! module does not align with a name-preserving rebuild, so the backtrace's function indices are
//! not evidence. The SDK's `main!` therefore arms a panic hook at the top of `da_run` that pushes
//! the message out through the advisory `sys@2::log` sink before the panic runtime aborts.
//!
//! What this gate holds:
//!
//! - the message and its `file:line:col` reach the host, for the three shapes guest code panics
//!   in (a literal `expect`, a formatted `panic!`, an `unreachable!`);
//! - they arrive on BOTH surfaces: the guest log stream (what an operator greps) and the TYPED
//!   failure's detail (what the node reports), so neither surface alone is load-bearing;
//! - the trap is still classified `GuestPanic` — forwarding must not re-class the failure into
//!   whatever the log call itself might raise;
//! - a guest that does not panic logs nothing, so arming the hook is not itself egress;
//! - an OUT-OF-MEMORY abort names its byte count. This is the class that actually cost the field
//!   cycles, and it is the one a panic hook cannot see: `handle_alloc_error` prints to a stderr
//!   the sandbox does not have and calls `abort()` without ever consulting the hook. Linear memory
//!   is capped at the module's admitted claim, so it is also the likeliest way a guest dies — the
//!   SDK's allocator wrapper reports the failed size on the way out.
//!
//! Dev/test harness: shells `cargo build` for the guests (the `event_loop.rs` pattern), so the
//! fs/process bans are allowed file-wide.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, Mutex};

use ciborium::value::Value;
use daemon_vhc_host::run::{start_run, MemorySink, RunConfig, RunEnd, RunIdentity};
use daemon_vhc_host::{select_driver, EngineConfig, TrapCode, Worker};

fn guest_wasm() -> Vec<u8> {
    daemon_vhc_guest_build::guest_wasm("test_panic_v2")
}

fn config(mode: u64) -> Vec<u8> {
    let v = Value::Map(vec![(Value::from("mode"), Value::from(mode))]);
    let mut b = Vec::new();
    ciborium::into_writer(&v, &mut b).expect("config cbor");
    b
}

/// The linear-memory cap the OOM mode runs under: enough for the module's image, far under the
/// gigabyte it then asks for.
const OOM_MODE_MEMORY_CAP: u64 = 8 << 20;

/// Run the guest at `mode` to its end, returning `(end, guest log lines)`.
fn drive(mode: u64) -> (RunEnd, Vec<(u32, String)>) {
    let wasm = guest_wasm();
    let engine = if mode == 4 {
        EngineConfig::default().with_claimed_memory(OOM_MODE_MEMORY_CAP)
    } else {
        EngineConfig::default()
    };
    let worker = Worker::new(engine).expect("engine");
    let sel = select_driver(&worker, &wasm, Some(blake3::hash(&wasm).as_bytes()))
        .expect("the panic guest is admitted");
    assert_eq!((sel.major, sel.minor), (2, 0));

    let identity = RunIdentity {
        run_id: [0x9A; 32],
        epoch: 0,
        role: "trainer".to_string(),
        instance: mode,
        module: *blake3::hash(&wasm).as_bytes(),
    };
    let run_cfg = RunConfig::new(identity, [0x77; 32], config(mode), Vec::new());
    let sink = Arc::new(Mutex::new(MemorySink::new()));
    let run = start_run(&worker, &wasm, run_cfg, Box::new(sink)).expect("start");
    let pump = run.pump.clone();

    // The control mode pulls until Stop; the panicking modes never reach a pull. Asking for a
    // stop either way keeps the two paths on one harness.
    let _ = pump.stop(daemon_vhc_abi::STOP_REASON_RUN_COMPLETE);
    let end = run.wait().expect("guest thread clean");
    let logs = pump.logs();
    (end, logs)
}

fn trap_of(end: &RunEnd) -> &daemon_vhc_host::Trap {
    match end {
        RunEnd::Trapped(t) => t,
        other => panic!("expected a trapped run, got {other:?}"),
    }
}

/// The forwarded line as the operator sees it: exactly one, at error level, prefixed.
fn forwarded_line(logs: &[(u32, String)]) -> &str {
    let panics: Vec<&(u32, String)> = logs
        .iter()
        .filter(|(_, m)| m.starts_with(daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX))
        .collect();
    assert_eq!(
        panics.len(),
        1,
        "exactly one forwarded panic line, got {logs:?}"
    );
    assert_eq!(
        panics[0].0,
        daemon_vhc_abi::LOG_LEVEL_ERROR,
        "a forwarded panic is an error-level line"
    );
    &panics[0].1
}

/// Mode 0 — `Option::expect` with a literal: the exact shape of the trainer's staging asserts.
#[test]
fn a_literal_expect_reaches_the_host_with_its_message_and_location() {
    let (end, logs) = drive(0);

    let line = forwarded_line(&logs);
    assert!(
        line.contains("all segments fetched"),
        "the panic message is forwarded verbatim: {line}"
    );
    assert!(
        line.contains("test-panic-v2/src/lib.rs:"),
        "the panic's file:line:col is forwarded: {line}"
    );

    let trap = trap_of(&end);
    assert_eq!(
        trap.code,
        TrapCode::GuestPanic,
        "forwarding must not re-class the trap: {trap}"
    );
    assert!(
        trap.detail.contains("all segments fetched"),
        "the TYPED failure carries the message, not just the log: {trap}"
    );
    assert!(
        trap.detail.contains("test-panic-v2/src/lib.rs:"),
        "the typed failure carries the location: {trap}"
    );
}

/// Mode 1 — a formatted payload (`panic!("{…}")`): the message must arrive INTERPOLATED, not as
/// its format string. This is the shape an `assert_eq!` takes, and the shape a stripped module's
/// backtrace is least able to identify.
#[test]
fn a_formatted_panic_reaches_the_host_interpolated() {
    let (end, logs) = drive(1);

    let line = forwarded_line(&logs);
    assert!(
        line.contains("staged 1 batches for a 30-step round"),
        "the formatted payload is interpolated before forwarding: {line}"
    );

    let trap = trap_of(&end);
    assert_eq!(trap.code, TrapCode::GuestPanic);
    assert!(trap.detail.contains("staged 1 batches for a 30-step round"));
}

/// Mode 2 — `unreachable!` with a literal.
#[test]
fn an_unreachable_reaches_the_host_with_its_message() {
    let (end, logs) = drive(2);

    let line = forwarded_line(&logs);
    assert!(
        line.contains("the streaming trainer defers ingest via begin_ingest"),
        "{line}"
    );

    let trap = trap_of(&end);
    assert_eq!(trap.code, TrapCode::GuestPanic);
    assert!(trap
        .detail
        .contains("the streaming trainer defers ingest via begin_ingest"));
}

/// Mode 4 — the allocation abort. No panic hook runs for this one, so the SDK's allocator wrapper
/// is the only thing standing between an operator and an anonymous `unreachable`. The byte count
/// is the whole diagnosis: it names which buffer went geometry-scaled.
#[test]
fn an_allocation_past_the_memory_cap_names_the_size_it_could_not_get() {
    let (end, logs) = drive(4);

    let line = forwarded_line(&logs);
    assert!(
        line.contains("memory allocation of 1073741824 bytes failed"),
        "the exhausted allocation's SIZE is what identifies it: {line}"
    );

    let trap = trap_of(&end);
    assert_eq!(
        trap.code,
        TrapCode::GuestPanic,
        "an allocation abort still reaches the host as a guest panic: {trap}"
    );
    assert!(
        trap.detail
            .contains("memory allocation of 1073741824 bytes"),
        "the typed failure carries the size, so a node operator never needs the guest logs: {trap}"
    );
}

/// Mode 3 — the control: arming the hook is not itself egress, and a clean run stays clean.
#[test]
fn a_guest_that_does_not_panic_forwards_nothing() {
    let (end, logs) = drive(3);

    assert!(
        matches!(end, RunEnd::Outcome(0)),
        "the control mode ends Ok, got {end:?}"
    );
    assert!(
        logs.iter()
            .all(|(_, m)| !m.starts_with(daemon_vhc_abi::GUEST_PANIC_LOG_PREFIX)),
        "no panic line on a clean run: {logs:?}"
    );
}
