// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The `sys@2` crypto-acceleration conformance gate (tier-1; architecture §3.2/§3.7, refactor §6).
//
// The det-lane pattern applied to crypto: the `hash`/`verify_sig` semantics are pinned by ONE
// dual-compiled contract (`daemon_vhc_proto::crypto`). The **in-guest fallback** is that contract
// compiled to wasm32 (always available — a module can hash/verify with zero host support); the
// **host acceleration** (`daemon_vhc_host::v2::driver::host_crypto_{hash,verify}`, the exact body
// the `sys@2::hash`/`verify_sig` imports run) is that contract compiled natively. Because both are
// the same source and blake3/ed25519 are integer arithmetic under wasm's deterministic core
// semantics, host-op ≡ in-guest-op is bit-exact **by construction** — exactly how
// `daemon-vhc-det` makes the det lane bit-identical by sharing one implementation.
//
// This gate asserts that standingly: it locks the host accel bodies to the normative contract over
// a wide, deterministic input sweep (guarding against a later host-side re-implementation drifting
// from the contract the guest fallback pins), plus known-answer vectors and the tri-state
// verify semantics. It is CPU-deterministic (no wasm host, no GPU, no network) — a `swarm-ci-det`
// citizen.

use daemon_vhc_host::v2::driver::{host_crypto_hash, host_crypto_verify};
use daemon_vhc_proto::canonical::to_canonical_vec;
use daemon_vhc_proto::crypto::{self, VerifyOutcome};
use daemon_vhc_proto::sign::{peer_id, sign_canonical, SigningKey};

/// A cheap deterministic byte stream (an LCG) so the sweep is reproducible without a fuzz dep.
fn pseudo_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

#[test]
fn host_hash_equals_the_normative_contract_and_blake3() {
    // Empty and the official blake3 empty-input vector.
    assert_eq!(host_crypto_hash(b""), crypto::hash(b""));
    assert_eq!(
        host_crypto_hash(b""),
        *blake3::hash(b"").as_bytes(),
        "host hash must be blake3-256"
    );
    // A wide deterministic sweep over lengths 0..=2048 and content — the host accel body MUST
    // equal the shared `daemon_vhc_proto::crypto` contract the in-guest fallback also compiles.
    for seed in 0..256u64 {
        let len = (seed as usize * 8) % 2049;
        let data = pseudo_bytes(seed, len);
        assert_eq!(
            host_crypto_hash(&data),
            crypto::hash(&data),
            "host hash diverged from the contract at seed {seed} (len {len})"
        );
        assert_eq!(host_crypto_hash(&data), *blake3::hash(&data).as_bytes());
    }
    assert_eq!(host_crypto_hash(b"").len(), crypto::HASH_LEN);
}

#[test]
fn host_verify_matches_the_contract_valid_invalid_malformed() {
    for seed in 0..128u64 {
        let sk = SigningKey::from_bytes(&pseudo_bytes(seed, 32).try_into().unwrap());
        let pk = peer_id(&sk).0;
        // Sign the canonical encoding of a payload; verify over those exact bytes (VALID).
        let payload = pseudo_bytes(seed ^ 0xABCD, (seed as usize) % 512);
        let msg = to_canonical_vec(&payload.as_slice()).unwrap();
        let sig = sign_canonical(&sk, &payload.as_slice()).unwrap().0;

        // The host accel body agrees with the normative contract on all three inputs, and the
        // codes are the assigned VerifyOutcome values (0 valid / 1 invalid / 2 malformed).
        assert_eq!(
            host_crypto_verify(&pk, &sig, &msg),
            VerifyOutcome::Valid.code()
        );
        assert_eq!(
            host_crypto_verify(&pk, &sig, &msg),
            crypto::verify_sig(&pk, &sig, &msg).code()
        );

        // A tampered message: well-formed inputs, does not verify → Invalid (1).
        let mut tampered = msg.clone();
        tampered.push(0xFF);
        assert_eq!(
            host_crypto_verify(&pk, &sig, &tampered),
            VerifyOutcome::Invalid.code()
        );

        // Malformed lengths (2), never a panic/trap at the contract boundary.
        assert_eq!(
            host_crypto_verify(&pk[..31], &sig, &msg),
            VerifyOutcome::Malformed.code()
        );
        assert_eq!(
            host_crypto_verify(&pk, &sig[..63], &msg),
            VerifyOutcome::Malformed.code()
        );
    }
}
