// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-vhc-proto` — the vhc-training wire **mechanism** (algorithm-free, round-vocabulary-free).
//!
//! Canonical CBOR codec, the genesis (schema-2) run envelope + freeze/verify (the retired v1
//! envelope form survives only as the outer schema-major read that types its refusal),
//! capability-set admission, merkle set commitments, grants, certificates + revocations, and the
//! [`VhcProtoVersion`]. This crate is the shared mechanism ground for the host, the participant
//! runtime, and the (wasm32) coordinator DO — see `docs/specs/vhc-architecture-spec.md` §7.
//!
//! **Algorithm-free AND round-vocabulary-free (dep-check-enforced):** the deterministic assignment
//! math moved to `sdk/daemon-vhc-sdk-consensus` first (architecture §7 rule 1 — "daemon-vhc-proto
//! stays algorithm-free: no assignment math, no round vocabulary"); the round message schemas
//! (`RoundOpen`/`Commitment`/…/`VhcMessage`/`SignedMessage`), the round state-digest schedule, and
//! the `record-set.cbor` object followed when the round vocabulary moved out — they live in
//! `daemon_vhc_sdk_consensus::{messages, digest, record_set}` now. Hosts route the resulting
//! frames as opaque signed bytes; only modules and SDK layers decode them.
//!
//! **wasm32-clean by construction:** the only dependencies are `serde`, `ciborium`, `blake3`,
//! and `ed25519-dalek` — no `tokio`, Burn, or wasmtime — so it builds for the
//! `wasm32-unknown-unknown` coordinator target (§11.2). Signing uses only deterministic
//! ed25519 operations (no RNG on the crate's non-test paths).

#![forbid(unsafe_code)]

pub mod bytes;
pub mod canonical;
pub mod capability;
pub mod cert;
pub mod crypto;
pub mod domains;
pub mod envelope;
pub mod error;
pub mod genesis;
pub mod grants;
pub mod hash;
pub mod merkle;
pub mod revocation;
pub mod sign;
pub mod transition;
pub mod version;

pub use bytes::{Hash, IrohId, PeerId, Root, Seed, Signature, StateDigest};
pub use canonical::{from_canonical_slice, to_canonical_vec};
pub use capability::{Capability, CapabilitySet};
pub use cert::{
    verify_certified_sender, CertError, CertScope, RunKeyCertBody, RunKeyCertificate,
    CERT_DOMAIN_V2,
};
pub use crypto::{
    hash as crypto_hash, verify_sig, VerifyOutcome, HASH_LEN, VERIFY_PUBLIC_KEY_LEN,
    VERIFY_SIGNATURE_LEN,
};
pub use envelope::{DeviceMinimums, SignedEnvelope};
pub use error::VhcProtoError;
pub use genesis::{
    peek_schema, BufferReq, ChannelDecl, ControlTransport, EventCap, EventCaps, FrozenGenesis,
    GenesisEnvelope, GrantBound, Identities, MigrationGrant, RoleEntry, RoleGrants, RunSection,
    SnapshotArtifact, TransportSelection, WorldGrant, GENESIS_SCHEMA_MAJOR,
};
pub use grants::{derive_admitted_quotas, AdmittedQuotas, GrantsDoc, GrantsError, LaneCeilings};
pub use hash::blake3_hash;
pub use merkle::{commit_set, MembershipProof, SetCommitment, SetCommitmentTree};
pub use revocation::{
    RevocationError, RevocationLedger, RunKeyRevocation, RunKeyRevocationBody, REVOCATION_DOMAIN_V2,
};
pub use sign::{peer_id, sign_canonical, verify_canonical, Signed, SigningKey, VerifyingKey};
pub use transition::{
    EpochDescriptor, TransitionChain, TransitionError, UpgradeAuthority, UpgradeRecord,
    UpgradeRecordBody, UpgradeSig, UPGRADE_RECORD_DOMAIN_V2,
};
pub use version::{VhcProtoVersion, VHC_PROTO_VERSION};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_renders() {
        let err = VhcProtoError::Validation("round out of range".into());
        assert!(err.to_string().contains("validation failed"));
    }
}
