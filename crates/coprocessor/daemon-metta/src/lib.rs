// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-metta` — the supervised out-of-process **symbolic coprocessor** (library + binary).
//!
//! MeTTa ([hyperon](https://github.com/trueagi-io/hyperon-experimental)) is `Rc`-based and therefore
//! `!Send`/`!Sync`: its runner cannot live across an `.await` or move between threads. So — exactly
//! like the local-inference worker (`daemon-infer`) isolates a crash-prone engine in a supervised
//! child — MeTTa runs in this separate process, and *inside* the process the runner lives on one
//! dedicated OS actor thread. The async stdio loop bridges decoded [`protocol`] frames to that
//! thread over channels; the engine value never crosses the boundary.
//!
//! The worker exposes the full `metta-symbolic-coprocessor` SKILL op set ([`worker::Worker`]) over a
//! pure-Rust, engine-independent state layer ([`state`]) — spaces, an append-only journal with
//! snapshot CAS, and the candidate -> active -> rollback procedure lifecycle. The MeTTa evaluation
//! itself sits behind the `hyperon` cargo feature ([`engine`]); the default build compiles a
//! pure-Rust [`engine::FallbackEngine`] (structural match + an arithmetic subset), so
//! `cargo test --workspace` never drags in the engine. Consumers that need only the wire types (the
//! supervised client + the tool) depend on this crate with `default-features = false`.

#![forbid(unsafe_code)]

pub mod engine;
pub mod protocol;
pub mod state;
pub mod worker;

pub use protocol::{Command, Event, OpResponse};
pub use state::MettaState;
pub use worker::Worker;
