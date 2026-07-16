// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Epoch-advance semantics (spec §6.2; TDD PROTO-17, ported from hivemind's `ready_to_update_epoch`).
//!
//! Hivemind advances an epoch on any of three disjuncts (batch target reached / a peer already leads
//! into a later epoch / the ETA to the target is exhausted), with the DHT removed. This is the pure
//! predicate; the barrier-mode `tick` wires **`BatchTarget`** as its epoch boundary and keeps the
//! other two disjuncts as the tested porting artifact for the pipelined mode (§6.4).

/// Which disjunct (if any) makes an epoch ready to advance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochTrigger {
    /// The epoch's round/batch target was reached.
    BatchTarget,
    /// A peer already reports a later epoch (global lead).
    GlobalLead,
    /// The estimated rounds-to-target reached zero (ETA).
    Eta,
    /// Not ready.
    None,
}

/// Inputs to the epoch-advance decision (hivemind's three disjuncts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpochInputs {
    /// Rounds completed in the current epoch.
    pub rounds_this_epoch: u64,
    /// The epoch length (`[phases].epoch_rounds`); `0` disables the batch-target disjunct.
    pub epoch_rounds: u64,
    /// How many epochs ahead the furthest-reported peer is (0 = no lead).
    pub peer_epoch_lead: u64,
    /// Estimated rounds remaining to the epoch target (0 = ETA exhausted).
    pub eta_rounds_remaining: u64,
}

/// The epoch-advance decision (§6.2; TDD PROTO-17).
#[must_use]
pub fn ready_to_update_epoch(i: &EpochInputs) -> EpochTrigger {
    if i.epoch_rounds > 0 && i.rounds_this_epoch >= i.epoch_rounds {
        return EpochTrigger::BatchTarget;
    }
    if i.peer_epoch_lead >= 1 {
        return EpochTrigger::GlobalLead;
    }
    if i.eta_rounds_remaining == 0 {
        return EpochTrigger::Eta;
    }
    EpochTrigger::None
}
