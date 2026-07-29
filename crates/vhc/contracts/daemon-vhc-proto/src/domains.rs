// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The vhc **domain-separation registry**: every blake3/signature domain string in the
//! subsystem, as named constants — the single place a derivation domain is spelled.
//!
//! # Format (normative)
//!
//! Every domain string is `daemon-vhc/<domain>/<semver>`, where `<domain>` is a single
//! kebab-case segment naming the derivation, and `<semver>` is a full SemVer 2.0 version
//! (`MAJOR.MINOR.PATCH`). Fresh domains start at `1.0.0`.
//!
//! A domain string is a **domain-separation input**: every byte of it is derivation-affecting.
//! Two uses of the same hash/signature primitive over the same payload MUST produce unrelated
//! outputs whenever their domains differ, and a domain string therefore names an *identity*,
//! not an API:
//!
//! - **MAJOR** bumps when the derivation scheme behind the tag changes in any way (input
//!   layout, hash construction, meaning of the output). The old string keeps naming the old
//!   scheme forever; the new scheme gets a new string.
//! - **MINOR**/**PATCH** exist for completeness of the SemVer format; a change that would
//!   only warrant a minor/patch bump of an API is still a *different derivation* here, so in
//!   practice bumps are MAJOR. They stay `0.0` unless the spec defines a compatible revision.
//!
//! Consumers MUST reference these constants — never inline a domain string at a call site.
//! (The ABI-major-anchored wire domains — `daemon-vhc/frame/2`, `daemon-vhc/cert/2`,
//! `daemon-vhc/upgrade/2`, `daemon-vhc/rng/2` — are NOT in this registry: their trailing `2`
//! is the `da_abi` major, a wire-contract identifier specified in the module-ABI spec, and
//! they live beside their contracts.)

/// Salt for the witness-committee shuffle (§6.3).
pub const WITNESS_SALT: &[u8] = b"daemon-vhc/witness/1.0.0";
/// Salt for the batch-assignment shuffle (§6.3).
pub const ASSIGN_SALT: &[u8] = b"daemon-vhc/assign/1.0.0";
/// Salt for the verifier-committee shuffle (§12).
pub const VERIFIER_SALT: &[u8] = b"daemon-vhc/verifier/1.0.0";
/// Salt for the checkpointer (tie-breaker) election (§9).
pub const CHECKPOINTER_SALT: &[u8] = b"daemon-vhc/checkpointer/1.0.0";

/// Label hashed to form the empty merkle root (the "no leaves" sentinel).
pub const MERKLE_EMPTY_ROOT_LABEL: &[u8] = b"daemon-vhc/merkle-empty/1.0.0";

/// Domain-separation tag bound into every checkpoint-attestation preimage (so a signature is
/// scoped to checkpoint attestation and cannot be replayed as any other signed frame).
pub const CHECKPOINT_ATTESTATION_DOMAIN: &str = "daemon-vhc/checkpoint-attestation/1.0.0";

/// Domain tag for the coordinator genesis-seed derivation: `blake3(domain ++ genesis_hash)`.
pub const GENESIS_SEED_DOMAIN: &[u8] = b"daemon-vhc/genesis-seed/1.0.0";

/// Domain tag for the gossip-topic derivation: `blake3(domain ++ genesis_hash)` → topic id.
pub const GOSSIP_TOPIC_DOMAIN: &[u8] = b"daemon-vhc/gossip-topic/1.0.0";

/// Domain-separation tag bound into every coordinator seat-lease preimage (spec §6.3; the
/// Authority-signed fenced lease). Distinct from the certificate domain so a lease signature can
/// never be replayed as a certificate or a frame signature, and vice versa.
///
/// **Retired scheme** (v1, `fencing_token == incarnation`): the string keeps naming the old
/// derivation forever; nothing authors it. Kept for the explicit v1 *interpretation* of archived
/// state ([`crate::seat::SeatLeaseBodyV1`]) — never for live authority.
pub const SEAT_LEASE_DOMAIN_V1: &str = "daemon-vhc/seat-lease/1.0.0";

/// Domain-separation tag bound into every seat-lease preimage at scheme v2: the lease separates
/// the run-role-global `leadership_term` (the seat's CAS/fencing order) from the claimant's
/// node-local `execution_incarnation` — two different order relations the v1 scheme conflated
/// (`fencing_token == incarnation`), which broke as soon as leadership could move across base
/// identities whose local counters are unrelated. A new domain string because the preimage
/// layout AND the meaning changed (MAJOR bump per the registry rules above).
pub const SEAT_LEASE_DOMAIN: &str = "daemon-vhc/seat-lease/2.0.0";

/// Retired v1 seat-release scheme tag (see [`SEAT_LEASE_DOMAIN_V1`]).
pub const SEAT_RELEASE_DOMAIN_V1: &str = "daemon-vhc/seat-release/1.0.0";

/// Domain-separation tag bound into every seat-release preimage at scheme v2 (the claimant's
/// signed statement that it gives the seat up). Distinct from [`SEAT_LEASE_DOMAIN`] so a release
/// signature can never be replayed as a live lease.
pub const SEAT_RELEASE_DOMAIN: &str = "daemon-vhc/seat-release/2.0.0";

/// Domain-separation tag bound into every iroh roster-record preimage (the registry-served
/// signed reachability statement). Distinct from the seat/cert/frame domains so a roster
/// signature can never be replayed as any other signed object, and vice versa.
pub const ROSTER_RECORD_DOMAIN: &str = "daemon-vhc/roster-record/1.0.0";

/// Domain-separation tag folded into every corpus **shard identity** (the chunk-addressed data
/// contract): `shard_hash = blake3(domain ++ u64le(chunk_size) ++ u64le(token_count) ++
/// u64le(byte_len) ++ c_0 ++ … ++ c_{n-1})` where `c_i = blake3(chunk_i bytes)`. The fold — not
/// a plain content hash of the shard bytes — IS the shard's artifact identity, which is what
/// makes a byte range verifiable from the covering chunks alone (whole-shard verify-on-first-
/// touch is rejected for streaming). The domain prefix keeps the fold from colliding with any
/// other blake3 derivation in the subsystem.
pub const CORPUS_SHARD_DOMAIN: &[u8] = b"daemon-vhc/corpus-shard/1.0.0";

/// Domain-separation tag folded into every det-state **family identity** (the chunk-addressed
/// canonical-state contract — the corpus custody chain instantiated for training state):
/// `family_fold = blake3(domain ++ u64le(chunk_size) ++ u64le(byte_len) ++ c_0 ++ … ++ c_{n-1})`
/// where `c_i = blake3(chunk_i bytes)`. As with [`CORPUS_SHARD_DOMAIN`], the fold — not a plain
/// content hash of the family bytes — IS the artifact identity, so byte ranges verify from the
/// covering chunks alone. The fold deliberately omits the corpus fold's `token_count`
/// (corpus-specific); det-state geometry is `(chunk_size, byte_len)`.
pub const DET_STATE_DOMAIN: &[u8] = b"daemon-vhc/det-state/1.0.0";

/// Fixed key seed for the observe replay sandbox's coordinator frame signer (the replay oracle
/// compares published payloads, never transport signatures, so any fixed seed serves).
pub const REPLAY_SANDBOX_FRAME_KEY_SEED: &[u8] = b"daemon-vhc/replay-sandbox-frame-key/1.0.0";

/// Fixed key seed for the in-process harness's coordinator frame signer (same rationale as
/// [`REPLAY_SANDBOX_FRAME_KEY_SEED`]).
pub const HARNESS_FRAME_KEY_SEED: &[u8] = b"daemon-vhc/harness-frame-key/1.0.0";

#[cfg(test)]
mod tests {
    /// Every registry entry conforms to `daemon-vhc/<domain>/<semver>` with a full
    /// `MAJOR.MINOR.PATCH` version.
    #[test]
    fn registry_strings_conform_to_the_specified_format() {
        let entries: &[&[u8]] = &[
            super::WITNESS_SALT,
            super::ASSIGN_SALT,
            super::VERIFIER_SALT,
            super::CHECKPOINTER_SALT,
            super::MERKLE_EMPTY_ROOT_LABEL,
            super::CHECKPOINT_ATTESTATION_DOMAIN.as_bytes(),
            super::SEAT_LEASE_DOMAIN.as_bytes(),
            super::SEAT_RELEASE_DOMAIN.as_bytes(),
            super::ROSTER_RECORD_DOMAIN.as_bytes(),
            super::GENESIS_SEED_DOMAIN,
            super::GOSSIP_TOPIC_DOMAIN,
            super::CORPUS_SHARD_DOMAIN,
            super::DET_STATE_DOMAIN,
            super::REPLAY_SANDBOX_FRAME_KEY_SEED,
            super::HARNESS_FRAME_KEY_SEED,
        ];
        for raw in entries {
            let s = core::str::from_utf8(raw).expect("domain strings are utf-8");
            let parts: Vec<&str> = s.split('/').collect();
            assert_eq!(parts.len(), 3, "{s}: must be daemon-vhc/<domain>/<semver>");
            assert_eq!(parts[0], "daemon-vhc", "{s}: prefix");
            assert!(
                !parts[1].is_empty()
                    && parts[1]
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{s}: kebab-case domain segment"
            );
            let ver: Vec<&str> = parts[2].split('.').collect();
            assert_eq!(ver.len(), 3, "{s}: full SemVer MAJOR.MINOR.PATCH");
            for comp in ver {
                assert!(
                    !comp.is_empty() && comp.chars().all(|c| c.is_ascii_digit()),
                    "{s}: numeric SemVer components"
                );
            }
        }
    }

    /// Distinct registry entries are distinct strings (domain separation actually separates).
    #[test]
    fn registry_strings_are_pairwise_distinct() {
        let entries: &[&[u8]] = &[
            super::WITNESS_SALT,
            super::ASSIGN_SALT,
            super::VERIFIER_SALT,
            super::CHECKPOINTER_SALT,
            super::MERKLE_EMPTY_ROOT_LABEL,
            super::CHECKPOINT_ATTESTATION_DOMAIN.as_bytes(),
            super::SEAT_LEASE_DOMAIN.as_bytes(),
            super::SEAT_RELEASE_DOMAIN.as_bytes(),
            super::ROSTER_RECORD_DOMAIN.as_bytes(),
            super::GENESIS_SEED_DOMAIN,
            super::GOSSIP_TOPIC_DOMAIN,
            super::CORPUS_SHARD_DOMAIN,
            super::DET_STATE_DOMAIN,
            super::REPLAY_SANDBOX_FRAME_KEY_SEED,
            super::HARNESS_FRAME_KEY_SEED,
        ];
        for (i, a) in entries.iter().enumerate() {
            for b in entries.iter().skip(i + 1) {
                assert_ne!(a, b, "duplicate domain string");
            }
        }
    }
}
