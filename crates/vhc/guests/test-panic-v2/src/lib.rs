// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `test-panic-v2` — the panic-forwarding conformance guest (declares abi **2.0**).
//!
//! A wasm guest's panic reaches the host as a bare `unreachable`: the payload and its
//! `file:line:col` live in linear memory the trap tears down, so a `GuestPanic` says only that
//! *something* panicked. The SDK's `main!` arms a panic hook at the top of `da_run` that pushes
//! the message out through `sys@2::log` first (ABI §3.6); this guest panics on purpose so a gate
//! can assert the message actually arrives.
//!
//! Driven by its config `mode`, one per shape real guest code panics in:
//!
//! - **0 — a literal `expect` on `None`**: `Option::expect`, the `stage_fetched_batches` shape —
//!   a `&'static str` payload, no formatting.
//! - **1 — a formatted `panic!`**: an interpolated payload, the `assert_eq!`/`format!` shape.
//! - **2 — `unreachable!` with a literal**: the "this arm cannot be taken" shape.
//! - **3 — no panic**: the control. Pulls until `Stop` and returns `Ok`, so a gate can prove the
//!   hook is silent on the happy path (arming forwarding must not put a line in the log by
//!   itself).
//! - **4 — an allocation past the sandbox cap**: the one guest failure a panic hook cannot see.
//!   Rust routes it through `handle_alloc_error` → `abort()`, never the hook, so the SDK's
//!   allocator wrapper reports the size instead.
//!
//! The panic happens inside `da_run`, never in `da_init`: capability imports are illegal during
//! `da_init` (§6.6), which is exactly why the hook is armed for `da_run` alone.

use daemon_vhc_sdk::module::{GuestModule, ModuleDecl};

const EV_STOP: u64 = 4;

struct TestPanic {
    mode: u64,
}

impl GuestModule for TestPanic {
    fn decl() -> ModuleDecl {
        ModuleDecl {
            name: "test-panic-v2",
            version: env!("CARGO_PKG_VERSION"),
            abi_minor: 0,
            channels: vec![0],
            host_state_bytes: 0,
            host_scratch_bytes: 0,
            device_state_bytes: 0,
            device_scratch_bytes: 0,
        }
    }

    fn init(config: &[u8], _grants: &[u8]) -> Result<Self, u32> {
        let Ok(ciborium::value::Value::Map(entries)) = ciborium::from_reader(config) else {
            return Err(16);
        };
        let mode = entries
            .iter()
            .find_map(|(k, v)| match k {
                ciborium::value::Value::Text(t) if t == "mode" => v.as_integer(),
                _ => None,
            })
            .map(|n| u64::try_from(i128::from(n)).unwrap_or(0))
            .unwrap_or(0);
        Ok(Self { mode })
    }

    fn run(&mut self) -> u32 {
        match self.mode {
            0 => {
                let staged: Option<u32> = None;
                let _ = staged.expect("all segments fetched");
            }
            1 => {
                let (want, got) = (30u32, 1u32);
                panic!("staged {got} batches for a {want}-step round");
            }
            2 => unreachable!("the streaming trainer defers ingest via begin_ingest"),
            4 => {
                // Past any admitted claim a test lane grants, so linear memory cannot grow to it.
                // `black_box` keeps the allocation from being optimized away.
                let huge: Vec<u8> = vec![0u8; 1 << 30];
                std::hint::black_box(&huge);
            }
            // Mode 3 (and anything else): never panics — the control. Pull until Stop.
            _ => {}
        }
        let mut buf: Vec<u8> = Vec::with_capacity(64);
        loop {
            if daemon_vhc_sdk::next_event(&mut buf).tag == EV_STOP {
                return 0;
            }
        }
    }
}

daemon_vhc_sdk::main!(TestPanic);
