// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The normative `sys@2` crypto-primitive semantics (`hash`, `verify_sig`).
//!
//! Architecture §3.2/§3.7 and §7: the `sys@` world exposes **crypto primitive accelerations**
//! (`verify_sig`, `hash`) "following the same pattern as the det lane: semantics pinned by a
//! dual-compiled contracts crate, in-guest fallback always available, host import as the fast
//! path." `daemon-vhc-proto` is that dual-compiled crate (architecture §7 repository structure:
//! proto carries "crypto primitive semantics (verify_sig, hash)"), so this module is the **single
//! definition** both sides link:
//!
//! - the **in-guest fallback** is this module compiled to `wasm32` inside the SDK — always
//!   available, so a module can verify signatures / hash bytes with zero host support (the
//!   compatibility path, architecture §3.2);
//! - the **host acceleration** (`sys@2::hash` / `sys@2::verify_sig`, wired in
//!   `daemon-vhc-host::v2::driver`) is this module compiled natively — the fast path.
//!
//! Because both paths are the *same source*, host-op ≡ in-guest-op is bit-exact **by
//! construction** (exactly as `daemon-vhc-det` makes the det lane bit-exact by sharing one
//! implementation); the tier-1 conformance gate (host ≡ in-guest bitwise, refactor §6) asserts it
//! standingly rather than trusting two implementations to agree.
//!
//! These are pure, deterministic functions of their inputs — a hash of guest bytes, a verification
//! of a guest-supplied `(key, sig, message)`. Unlike a clock reading or a payload fetch they carry
//! **no nondeterministic host observation**, so the host import results are **not journaled**: at
//! replay the verifier re-executes them over the (already replay-reproduced) guest linear memory
//! and gets the identical answer (the `dc`/`dd` replay classes, ABI §2.7).

use crate::bytes::{PeerId, Signature};
use crate::hash::blake3_hash;
use crate::sign::verify_bytes;

/// The digest length (bytes) of the `sys@2::hash` primitive — blake3-256, the swarm's universal
/// content-address width (`hash.rs`, architecture §3.4). Fixed here so host and guest agree on the
/// output span without negotiating it.
pub const HASH_LEN: usize = 32;

/// The length (bytes) of an ed25519 public key accepted by `verify_sig`.
pub const VERIFY_PUBLIC_KEY_LEN: usize = 32;

/// The length (bytes) of an ed25519 signature accepted by `verify_sig`.
pub const VERIFY_SIGNATURE_LEN: usize = 64;

/// The outcome of [`verify_sig`], as a small closed enum with **assigned numeric values** so the
/// `sys@2::verify_sig` import can return it as a `u32` status the guest fails-closed on (ABI §5.2).
///
/// The tri-state deliberately distinguishes a well-formed-but-invalid signature from a malformed
/// input: a module's `Authority` (architecture §4.2) branches on "this record does not verify"
/// differently from "this key/signature is structurally garbage", and collapsing them would hide
/// the latter behind the former.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum VerifyOutcome {
    /// The signature verifies against the key over the message (ed25519 `verify_strict`).
    Valid = 0,
    /// The inputs are well-formed but the signature does not verify.
    Invalid = 1,
    /// The public key or signature was structurally malformed (wrong length / not on-curve).
    Malformed = 2,
}

impl VerifyOutcome {
    /// The assigned numeric status the import returns (ABI §5.2 — statuses are fixed, unknown
    /// values fail closed guest-side).
    #[must_use]
    pub fn code(self) -> u32 {
        self as u32
    }
}

/// `sys@2::hash` — the blake3-256 content hash of `data`.
///
/// The one normative definition shared by the host accel and the in-guest fallback. Delegates to
/// [`blake3_hash`] (the swarm's content-address function) so there is exactly one hash on both
/// sides of the boundary.
#[must_use]
pub fn hash(data: &[u8]) -> [u8; HASH_LEN] {
    blake3_hash(data).0
}

/// `sys@2::verify_sig` — verify an ed25519 `signature` by `public_key` over `message`.
///
/// Returns the tri-state [`VerifyOutcome`]. A wrong-length key or signature is
/// [`VerifyOutcome::Malformed`] (never a panic / trap — the boundary rejects bad lengths cleanly);
/// a structurally-valid but non-verifying pair is [`VerifyOutcome::Invalid`]. Uses
/// [`verify_bytes`]'s `verify_strict` semantics (rejects small-order / malleable edge cases),
/// matching the consensus intent that a signature be unambiguous (`sign.rs`).
#[must_use]
pub fn verify_sig(public_key: &[u8], signature: &[u8], message: &[u8]) -> VerifyOutcome {
    let Ok(pk): Result<[u8; VERIFY_PUBLIC_KEY_LEN], _> = public_key.try_into() else {
        return VerifyOutcome::Malformed;
    };
    let Ok(sig): Result<[u8; VERIFY_SIGNATURE_LEN], _> = signature.try_into() else {
        return VerifyOutcome::Malformed;
    };
    // `PeerId::from_bytes` failure (not on-curve) surfaces as `Malformed`; a well-formed key whose
    // signature does not verify is `Invalid`. `verify_bytes` collapses both into one error, so we
    // pre-screen the key structurally to keep the tri-state honest.
    if crate::sign::VerifyingKey::from_bytes(&pk).is_err() {
        return VerifyOutcome::Malformed;
    }
    match verify_bytes(&PeerId(pk), &Signature(sig), message) {
        Ok(()) => VerifyOutcome::Valid,
        Err(_) => VerifyOutcome::Invalid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sign::{peer_id, sign_canonical, SigningKey};

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn hash_matches_blake3_and_is_deterministic() {
        assert_eq!(hash(b""), blake3_hash(b"").0);
        assert_eq!(hash(b"round-42"), hash(b"round-42"));
        assert_ne!(hash(b"round-42"), hash(b"round-43"));
        assert_eq!(hash(b"").len(), HASH_LEN);
    }

    #[test]
    fn verify_sig_accepts_a_valid_signature() {
        let sk = key(7);
        let msg = b"a committed record";
        let sig = sign_canonical(&sk, &msg.as_slice()).unwrap();
        // sign_canonical signs the canonical CBOR of the value; verify over the same bytes.
        let bytes = crate::canonical::to_canonical_vec(&msg.as_slice()).unwrap();
        assert_eq!(
            verify_sig(&peer_id(&sk).0, &sig.0, &bytes),
            VerifyOutcome::Valid
        );
    }

    #[test]
    fn verify_sig_rejects_a_tampered_message() {
        let sk = key(9);
        let sig = sign_canonical(&sk, &b"original".as_slice()).unwrap();
        assert_eq!(
            verify_sig(&peer_id(&sk).0, &sig.0, b"tampered"),
            VerifyOutcome::Invalid
        );
    }

    #[test]
    fn verify_sig_flags_malformed_lengths() {
        let sk = key(3);
        let sig = sign_canonical(&sk, &b"x".as_slice()).unwrap();
        assert_eq!(
            verify_sig(&[0u8; 31], &sig.0, b"x"),
            VerifyOutcome::Malformed
        );
        assert_eq!(
            verify_sig(&peer_id(&sk).0, &[0u8; 63], b"x"),
            VerifyOutcome::Malformed
        );
    }

    #[test]
    fn verify_outcome_codes_are_stable() {
        assert_eq!(VerifyOutcome::Valid.code(), 0);
        assert_eq!(VerifyOutcome::Invalid.code(), 1);
        assert_eq!(VerifyOutcome::Malformed.code(), 2);
    }
}
