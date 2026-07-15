// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-testkit` — the host-side integration runner (architecture §6).
//!
//! The host-side twin of `sdk/daemon-vhc-sim`: where vhc-sim runs NATIVE policy code, the testkit
//! runs the **production wasm blobs** under wasmtime + simulated capability providers. It links
//! `host/*` only and never `sdk/*` — the two layers sit on opposite sides of the wasm boundary, so
//! conflating them would smuggle a forbidden `sdk/* -> host/*` dependency (enforced by
//! `xtask vhc-dep-check`).
//!
//! It generalizes the A2 t2 join-run test (`daemon-vhc-worker`'s `v2_session`) into reusable
//! infrastructure:
//!
//! - [`run`] — [`run::whole_run`]: start a major-2 module under the real host event-loop driver,
//!   journal every §8 observation through an in-memory sink, drain it to a clean stop, then
//!   re-drive the recorded journal through the §8.7 input-replay engine and assert every decision
//!   (publish channel + seq + payload hash) and the terminal outcome reproduce bit-for-bit. A
//!   diverging replay is a gate FAILURE, never a warning (refactor §12.6).
//! - [`coordinator`] — [`coordinator::NativeCoordinator`]: the reusable "native coordinator" half —
//!   the pure `daemon-vhc-coordinator` `tick` state machine wrapped with envelope config, clock
//!   advancement, and signing, so a whole run can be driven by a real coordinator in-process (the
//!   "native coordinator + wasm workers" shape ahead of the D2 wasm coordinator).
//!
//! The first whole-run gate wired into tier-2 CI is the SPARTA-shaped `toy_averager.wasm` production
//! blob (timers + publish, no coordinator) — deterministic, journaled, replay-verified.

pub mod coordinator;
pub mod run;

pub use coordinator::NativeCoordinator;
pub use run::{whole_run, ReplayReport, RunSpec, WholeRunReport};

use daemon_vhc_host::{EngineConfig, Worker};

/// Build a fresh host [`Worker`] (the wasmtime sandbox) for the testkit — the substrate every
/// whole-run and replay is driven on.
///
/// # Errors
/// If the wasmtime engine fails to construct.
pub fn worker() -> Result<Worker, String> {
    Worker::new(EngineConfig::default()).map_err(|e| format!("testkit engine: {e}"))
}
