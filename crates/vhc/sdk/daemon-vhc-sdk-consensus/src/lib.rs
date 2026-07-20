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
//! The layer is complete as of D1+D2: **D1 landed `Authority` (`SingleKey`, `ThresholdKeys`) +
//! the typed `AuthorityConfig`, and the `Committed<T>` mint** ([`authority`], [`committed`]);
//! **D2 landed the coordinator driver** ([`coordinator`]) — the pure `tick` state machine
//! relocated from the dissolved host-side `daemon-vhc-coordinator` crate (refactor §8/D2), so one
//! implementation serves the wasm `coordinator-quorum` guest, the native `vhc-sim` coordination,
//! and the dual-compilation identity reference alike. The coordinator's authenticated-dispatch
//! seam consumes D1's `Authority` surface (the D2 sitting-3 reconciliation).
//!
//! **The round message schemas live HERE, not in the proto (architecture §7 rule 1).** The
//! round-vocabulary move relocated the seven round messages + the membership messages ([`messages`]), the
//! round state-digest schedule ([`digest`]), and the committed-set object ([`record_set`]) out of
//! `daemon-vhc-proto` into this crate: `daemon-vhc-proto` retains wire *mechanism* only (canonical
//! CBOR, signing, hashes, merkle commitments, genesis envelope, grants, certificates/revocations),
//! while the algorithm vocabulary — what a round *is* — belongs to the SDK layer the modules link.
//! Production host crates must not link this crate (dep-check-enforced): hosts route opaque signed
//! bytes; only modules (guests), the SDK, and explicitly-exempted harness/oracle tooling decode
//! these schemas. The CBOR encodings are byte-identical to the pre-move proto encodings (the
//! `daemon-vhc.cddl` conformance suite moved with the types and still pins them).
//!
//! wasm32-clean by construction: the dependencies are `daemon-vhc-proto` (wire mechanism + blake3 +
//! the `verify_sig` crypto primitive), `serde` (derive, for the canonical-CBOR-serializable
//! coordinator state), and `xxhash-rust` (the round state digest), so this crate compiles for
//! guests and hosts alike — the "linked identically by worker drivers, coordinator drivers,
//! simulator, and replay" property (architecture §8 authority table).

#![forbid(unsafe_code)]

pub mod assignment;
pub mod attestation;
pub mod authority;
pub mod checkpoint;
pub mod committed;
pub mod coordinator;
pub mod digest;
pub mod fold_walk;
pub mod messages;
pub mod record_set;

pub use assignment::{
    advance_cursor, assign_batches, class_weight, deterministic_shuffle, elect_checkpointer,
    elect_checkpointers, global_batch_at, seeded_lcg, select_committee, select_verifiers,
    witness_quorum, Committee, Lcg, ASSIGN_SALT, CHECKPOINTER_SALT, VERIFIER_SALT, WITNESS_SALT,
    WITNESS_TARGET_DEFAULT,
};
pub use attestation::{
    AttestationBody, AttestationError, AttestationLedger, AttestationPolicy, AttestationTier,
    JoinEligibility, SignedAttestation, ATTEST_DOMAIN,
};
pub use authority::{
    AuthError, Authority, AuthorityConfig, AuthorityContract, Authorized, FaultThreshold, Finality,
    Reconfiguration, RecordSig, SingleKey, ThresholdKeys, Topology, DEFAULT_RECORDS_CHANNEL,
};
pub use checkpoint::{
    CheckpointError, CheckpointManifest as TypedCheckpointManifest, CheckpointManifestBuilder,
    CheckpointSection, SectionClass, SectionKind, CHECKPOINT_MANIFEST_SCHEMA,
};
pub use committed::{
    Committed, CommittedItem, HostStaged, MintError, PayloadCheck, PayloadRepr, PayloadSource,
};
pub use digest::{
    derive_schedule, digest_state, digest_with_schedule, DigestCarry, DigestSchedule, StateLayout,
};
pub use fold_walk::{windows, FoldWalk, SliceActions, UnexpectedCompletion, Window};
pub use messages::{SignedMessage, VhcMessage};
pub use record_set::RecordSet;
