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
//! Roadmap (architecture §7): D1 lands `Authority` (`SingleKey`, `ThresholdKeys`) and the
//! `Committed<T>` mint here; **D2 lands the coordinator driver** ([`coordinator`]) — the pure
//! `tick` state machine relocated from the dissolved host-side `daemon-vhc-coordinator` crate
//! (refactor §8/D2), so one implementation serves the wasm `coordinator-quorum` guest, the native
//! `vhc-sim` coordination, and the dual-compilation identity reference alike.
//!
//! wasm32-clean by construction: the dependencies are `daemon-vhc-proto` (wire types + blake3) and
//! `serde` (derive, for the canonical-CBOR-serializable coordinator state), so this crate compiles
//! for guests and hosts alike — the "linked identically by worker drivers, coordinator drivers,
//! simulator, and replay" property (architecture §8 authority table).

#![forbid(unsafe_code)]

pub mod assignment;
pub mod coordinator;

pub use assignment::{
    advance_cursor, assign_batches, class_weight, deterministic_shuffle, elect_checkpointer,
    elect_checkpointers, global_batch_at, seeded_lcg, select_committee, select_verifiers,
    witness_quorum, Committee, Lcg, ASSIGN_SALT, CHECKPOINTER_SALT, VERIFIER_SALT, WITNESS_SALT,
    WITNESS_TARGET_DEFAULT,
};
