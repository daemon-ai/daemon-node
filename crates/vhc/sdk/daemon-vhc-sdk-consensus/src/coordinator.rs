// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The coordinator driver — the purified consensus state machine (architecture §4.1, §6/§7;
//! refactor §8/D2).
//!
//! Relocated at **D2** from the dissolved host-side `daemon-vhc-coordinator` crate into the
//! consensus SDK layer, its architectural home (architecture §7: `sdk/daemon-vhc-sdk-consensus`
//! holds "the coordinator drivers"). The move is what lets **one** coordinator implementation be:
//!
//! - compiled to wasm32 in `guests/coordinator-quorum` (the launch coordinator module) — the guest
//!   links this crate for both the assignment math (moved here at D0) and this `tick`;
//! - run natively by `sdk/daemon-vhc-sim` for fast native-policy coordination in tests;
//! - run natively as the **dual-compilation identity reference** the D2 wasm-coordinator gate
//!   compares the guest against (refactor §8/D2 acceptance: "wasm `coordinator-quorum` vs native
//!   `tick` on identical inputs").
//!
//! `tick(state, input) -> (state', outputs)` is a total, I/O-free function: time enters as
//! [`Input::Clock`], signed evidence as [`Input::Message`], operator intents as [`Input::Control`].
//! Identical `(state, input)` always yields identical `(state', outputs)` — the replay-oracle and
//! dual-compilation foundation. It never signs and never touches the network: whose records count
//! is an `Authority` question (D1, this crate) the harness/guest resolves above `tick`; today's
//! implicit trust is the envelope-named coordinator identity (`SingleKey`), verified where a signed
//! frame enters as [`Input::Message`].

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
pub use config::{CoordinatorParams, CoordinatorRoleConfig, RunConfig, K_ABSENCES_DEFAULT};
pub use epoch::{ready_to_update_epoch, EpochInputs, EpochTrigger};
pub use io::{AdmissionReject, ControlAction, ControlRequest, Input, Notice, Output, Rejection};
pub use state::{
    ClientState, CoordinatorState, Member, Phase, RoundRing, RoundState, NUM_STORED_ROUNDS,
};
pub use tick::{tick, tick_authenticated};

// Re-export the assignment seam so consumers get committee/batch math without a second import
// (it is this crate's own lower layer; the coordinator does not fork it). Preserves the ergonomics
// of the dissolved `daemon-vhc-coordinator` crate's identical re-export.
pub use crate::assignment::{
    assign_batches, deterministic_shuffle, elect_checkpointer, global_batch_at, seeded_lcg,
    select_committee, select_verifiers, witness_quorum, Committee, Lcg,
};

/// Errors surfaced by the coordinator driver.
///
/// Hand-rolled (no `thiserror`) to keep the crate lean + wasm-clean, matching the proto convention.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CoordinatorError {
    /// A proto-contract step failed (canonicalization, capability parse, envelope validation).
    Proto(daemon_vhc_proto::SwarmProtoError),
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

impl From<daemon_vhc_proto::SwarmProtoError> for CoordinatorError {
    fn from(e: daemon_vhc_proto::SwarmProtoError) -> Self {
        Self::Proto(e)
    }
}
