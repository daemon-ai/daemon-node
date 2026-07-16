// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-sdk-consensus` — the consensus SDK layer (architecture §6/§7; refactor §8/D0).
//!
//! Created at D0 with the **assignment math** that previously lived in `daemon-vhc-proto`
//! (`proto::assignment`) — the D0 point where `daemon-vhc-proto` becomes algorithm-free wire
//! mechanism and its consumers (the replay oracle via the coordinator, the coordinator itself,
//! the session's retained v1 `RoundEngine`, the testkit's barrier harness, and the SDK round
//! drivers) relink here. The math is unchanged byte-for-byte: the golden vectors
//! (`tests/assignment_golden.rs`) moved with it and still pin the LCG/shuffle output.
//!
//! Roadmap (architecture §7): **D1 lands `Authority` (`SingleKey`, `ThresholdKeys`) + the typed
//! `AuthorityConfig`, and the `Committed<T>` mint here** ([`authority`], [`committed`]); D2 lands the
//! coordinator drivers (`DeadlineQuorumCoordinator` …) and consumes this `Authority` surface (it
//! stubs against [`authority::SingleKey`] until this track merges — this track merges first).
//!
//! wasm32-clean by construction: the only dependency is `daemon-vhc-proto` (wire types + blake3 +
//! the `verify_sig` crypto primitive), so this crate compiles for guests and hosts alike — the
//! "linked identically by worker drivers, coordinator drivers, simulator, and replay" property
//! (architecture §8 authority table).

#![forbid(unsafe_code)]

pub mod assignment;
pub mod authority;
pub mod committed;

pub use assignment::{
    advance_cursor, assign_batches, class_weight, deterministic_shuffle, elect_checkpointer,
    elect_checkpointers, global_batch_at, seeded_lcg, select_committee, select_verifiers,
    witness_quorum, Committee, Lcg, ASSIGN_SALT, CHECKPOINTER_SALT, VERIFIER_SALT, WITNESS_SALT,
    WITNESS_TARGET_DEFAULT,
};
pub use authority::{
    AuthError, Authority, AuthorityConfig, AuthorityContract, Authorized, FaultThreshold, Finality,
    Reconfiguration, RecordSig, SingleKey, ThresholdKeys, Topology, DEFAULT_RECORDS_CHANNEL,
};
pub use committed::{
    Committed, CommittedItem, HostStaged, MintError, PayloadCheck, PayloadRepr, PayloadSource,
};
