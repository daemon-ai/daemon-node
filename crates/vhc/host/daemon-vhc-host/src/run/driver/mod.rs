// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The major-2 event-loop driver (ABI §3–§6, §11–§12) — Phase A's closed capability subset.
//!
//! The inversion itself: the host calls `da_init` once, then `da_run` exactly once, and from then
//! on **drives nothing** — the guest pulls events through the blocking `next_event` import while
//! the host routes, journals, and enforces budgets (architecture §3.1).
//!
//! ## Threading (ABI §11)
//!
//! The guest runs on **one dedicated OS thread per role-instance** ([`start_run`] spawns it); that
//! thread owns the wasmtime `Store`, is the only thread that ever calls into wasm, and is the only
//! thread that drops the `Store` (§11.1/§11.3). The embedder (the session's async runtime; the
//! tier-1 tests) talks to the run through a [`PumpHandle`]: enqueue inbound frames, stage
//! payloads, deliver budget/stop/quiesce — a bounded, condvar-signalled queue the guest thread
//! blocks on inside `next_event` (§11.2). Timers need no external waker: the parked `next_event`
//! wait times out at the earliest armed deadline and fires due timers itself, inside the pump
//! lock, in deterministic `(fire_at, timer_id)` order.
//!
//! ## Born audited (ABI §8)
//!
//! Every observation flows through the [`JournalSink`] seam before the guest can see it: the
//! delivered event frame (tag 1, written before delivery — §8.4 rule 4) with the original signed
//! wire frame beside it (tag 12, §8.6), every `read_back` value (tag 2), every clock reading
//! (tag 3), every publish (tag 4, committed before `publish` returns — §6.2), timer arms/cancels
//! (tags 5/6), advisory drops (tag 7 via the sink's drop hook), instantiation + `da_init`
//! (tags 13/11), and the terminal fact (tag 9). There is no unjournaled mode.
//!
//! ## Budgets (ABI §5.5/§5.6)
//!
//! Fuel, the op count, and the readback-byte allowance reset at each `Delivered` return of
//! `next_event` (a `NeedCapacity` return resets nothing); the epoch deadline re-arms at the same
//! point, so a guest parked inside `next_event`/`read_back` is never epoch-killed for waiting —
//! the watchdog covers in-slice spins only, unchanged from v1.
//!
//! ## Deliberate Phase-A bounds (recorded)
//!
//! - The `tabi@1` compute bridge is RETIRED: a module importing the namespace is refused typed
//!   (`BridgeRetired`) at the §1.3 front door and again here at start; compute crosses the
//!   boundary through the `compute@2` world only.
//! - `snapshot_state` returns `SectionMissing` during a drain (no state-manifest verification yet
//!   — the §10.2 protocol lands with the migrate scaffolding); outside a drain it traps
//!   `PhaseViolation` per §6.6. `stage_state` is fully functional.
//! - Inbound frames arrive through [`PumpHandle::deliver_frame`] pre-verified: signature
//!   verification/dedup/gap detection are the session pump's admission-side jobs (the
//!   choreography sitting); this driver journals the original signed frame it is handed (§8.6).

//!
//! Decomposed by concern: [`config`] (run configuration + typed outcomes), [`pump`] (the event
//! pump + §4.7 queue policies), [`host`] (the guest-side store data + §6.6 legality gate),
//! [`linker`] (the per-world import bodies), [`chunks`] (chunk-map decode/verify),
//! [`migration`] (the §10.2 wire helpers), and [`lifecycle`] (spawn/instantiate/run/teardown).
//! Everything below re-exports at its original `driver::` path.

mod chunks;
mod config;
mod host;
mod lifecycle;
mod linker;
mod migration;
mod pump;
#[cfg(test)]
mod tests;

pub(crate) use chunks::decode_chunk_descriptor;
pub use config::{
    DeliverVerdict, MigrationInput, OpOutcome, RunConfig, RunEnd, RunError, RunIdentity,
    SnapshotCapture, SpooledFrame,
};
pub use host::{derive_rng_seed, host_crypto_hash, host_crypto_verify};
pub use lifecycle::{start_run, start_run_migrating, Run};
pub(crate) use migration::{build_migration_descriptor, RestoreBinding};
pub(crate) use pump::BufferStreams;
pub use pump::PumpHandle;
