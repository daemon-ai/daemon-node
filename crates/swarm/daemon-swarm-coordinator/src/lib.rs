// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-swarm-coordinator` — the purified coordinator state machine (spec §6.2, §6.4, §11.2).
//!
//! Wave 2 ships this as a **pure library**: the [`tick`] state machine plus its
//! canonical-CBOR-serializable types. `tick(state, input) -> (state', outputs)` is I/O-free —
//! time enters as [`Input::Clock`], signed evidence as [`Input::Message`], operator intents as
//! [`Input::Control`] — so the same logic runs identically in a local server, a private node, and the
//! cloud Durable Object (§11.2), is property-testable, and is the foundation of the offline replay
//! oracle (I1, TDD PROTO-20). The deterministic per-round assignment it relies on lives in the
//! wasm-clean [`daemon_swarm_proto::assignment`] module (re-exported below).
//!
//! The runnable local coordinator (axum/WS wiring, the tick loop over a real clock and transport)
//! is Wave 3 — lane R owns `bins/`. This crate never performs I/O and never signs (see the ledger).

#![forbid(unsafe_code)]

pub mod admission;
pub mod commit;
pub mod config;
pub mod epoch;
pub mod io;
pub mod state;
pub mod tick;

use std::error::Error;
use std::fmt;

pub use admission::{admit, JoinCandidate};
pub use config::{CoordinatorParams, RunConfig, K_ABSENCES_DEFAULT};
pub use epoch::{ready_to_update_epoch, EpochInputs, EpochTrigger};
pub use io::{AdmissionReject, ControlAction, ControlRequest, Input, Notice, Output, Rejection};
pub use state::{
    ClientState, CoordinatorState, Member, Phase, RoundRing, RoundState, NUM_STORED_ROUNDS,
};
pub use tick::tick;

// Re-export the assignment seam so consumers get committee/batch math without a second import
// (it is the proto crate's authority; the coordinator does not fork it).
pub use daemon_swarm_proto::assignment::{
    assign_batches, deterministic_shuffle, elect_checkpointer, global_batch_at, seeded_lcg,
    select_committee, select_verifiers, witness_quorum, Committee, Lcg,
};

/// Errors surfaced by the coordinator library.
///
/// Hand-rolled (no `thiserror`) to keep the crate lean + wasm-clean, matching the proto convention.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoordinatorError {
    /// A proto-contract step failed (canonicalization, capability parse, envelope validation).
    Proto(daemon_swarm_proto::SwarmProtoError),
    /// The run configuration was inconsistent.
    Config(String),
}

impl fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Proto(e) => write!(f, "coordinator proto error: {e}"),
            Self::Config(d) => write!(f, "coordinator config error: {d}"),
        }
    }
}

impl Error for CoordinatorError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Proto(e) => Some(e),
            Self::Config(_) => None,
        }
    }
}

impl From<daemon_swarm_proto::SwarmProtoError> for CoordinatorError {
    fn from(e: daemon_swarm_proto::SwarmProtoError) -> Self {
        Self::Proto(e)
    }
}
