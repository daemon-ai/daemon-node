// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-sim` — the SDK-side simulation layer (architecture §6).
//!
//! Two layers of simulation live on opposite sides of the wasm boundary. This is the **SDK-side**
//! one: native implementations of the four capability worlds plus a deterministic discrete-event
//! runtime that drives NATIVE policy code — whole runs of a coordinator + N workers — for fast
//! iteration and native coordinator tests. The host-side twin (`host/daemon-vhc-testkit`) runs the
//! **production wasm blobs** under wasmtime; conflating the two would smuggle a forbidden
//! `sdk/* -> host/*` dependency, so this crate links SDK crates only and never `host/*`.
//!
//! ## The worlds (native)
//!
//! - [`backend`] — the reference CPU backend: the shared [`daemon_vhc_det`] fixed-order fp32
//!   kernels (bit-identical to the host's det lane).
//! - [`net`] — a virtual network: channel pub/sub with **trace-driven** latency, churn, and
//!   session-length models (deterministic, seeded).
//! - [`corpus`] — a virtual corpus: deterministic token windows by `(peer, cursor)`.
//! - [`clock`] — a virtual logical clock and one-shot timers (ABI §6.3/§6.5 semantics).
//!
//! ## The runtime
//!
//! [`sim`] is a single-threaded, deterministic **discrete-event simulator**. A module implements
//! [`sim::SimModule`] (`init` + `on_event`) and is driven through a [`sim::SimCtx`] that exposes the
//! Phase-A closed capability subset natively — `publish`, `set_timer`/`cancel_timer`, `now`,
//! `emit_metric`, `log` — with the same durable-seq (§12.2) and logical-clock (§6.5) semantics the
//! host enforces. A [`sim::Simulator`] runs N modules over the virtual worlds under a
//! [`net::Trace`], collecting a deterministic decision transcript; running the same setup twice
//! yields byte-identical transcripts, the SDK-side analogue of the host's §8.7 input replay.
//!
//! The callback event-handler shape (rather than the wasm loop's blocking `next_event` pull) is the
//! deterministic single-threaded native form; the same *algorithm* ships as a wasm blob (run under
//! the testkit) authored against `daemon-vhc-sdk-v2`'s raw event loop.

pub mod backend;
pub mod clock;
pub mod corpus;
pub mod net;
pub mod sim;
pub mod toys;

pub use net::{Trace, VirtualNet};
pub use sim::{PublishedFrame, RunLimits, RunTranscript, SimCtx, SimEvent, SimModule, Simulator};

/// The SDK driver + scaffolding surface native policy code is authored against, re-exported so a
/// vhc-sim consumer links one crate. Re-exports (not just deps) so the dependency-direction gate
/// sees these `sdk/* -> sdk/*` edges as used.
pub mod drivers {
    pub use daemon_vhc_sdk_rounds::*;
    #[doc(inline)]
    pub use daemon_vhc_sdk_v2 as v2;
}
