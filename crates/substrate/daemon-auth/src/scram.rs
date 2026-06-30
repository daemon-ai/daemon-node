// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! SCRAM-SHA-256 stored-credential derivation (RFC 5802 / RFC 7677).
//!
//! SCRAM lets a client prove knowledge of a password without sending it, and lets the server store
//! only *derived* material (never the password, never the `SaltedPassword`, never the `ClientKey`).
//! On a password set we compute and persist exactly the four values RFC 5802 §3 names — `salt`,
//! `iterations`, `StoredKey`, `ServerKey` — which is precisely the shape `rsasl`'s server-side
//! `ScramStoredPassword` property consumes (see `daemon-host`'s authenticator).
//!
//! ```text
//! SaltedPassword := PBKDF2-HMAC-SHA256(SASLprep(password), salt, iterations, dkLen = 32)
//! ClientKey      := HMAC-SHA256(SaltedPassword, "Client Key")
//! StoredKey      := SHA-256(ClientKey)
//! ServerKey      := HMAC-SHA256(SaltedPassword, "Server Key")
//! ```
//!
//! Argon2id (`password_credentials`) stays the source of truth for the PLAIN/login path; this is the
//! parallel wire-KDF representation for SCRAM. The two are derived from the same password at the same
//! time (see [`crate::store::AuthStore`]).

use crate::error::{Error, Result};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

/// The advertised SCRAM mechanism name (also the `scram_credentials.mechanism` row key).
pub const SCRAM_SHA_256: &str = "SCRAM-SHA-256";

/// The PBKDF2 iteration count for newly-derived SCRAM material. RFC 5802 mandates a floor of 4096;
/// it is the wire-KDF work factor (independent of the Argon2id at-rest hash), and is persisted per
/// row so raising it later does not invalidate existing credentials.
pub const SCRAM_DEFAULT_ITERATIONS: u32 = 4096;

/// Salt length in bytes for newly-derived SCRAM material.
pub const SCRAM_SALT_LEN: usize = 16;

/// SHA-256 output / SCRAM key length in bytes.
const KEY_LEN: usize = 32;

/// The persisted SCRAM-SHA-256 material for one user (the `scram_credentials` row, minus identity).
/// Only these four values are stored — never the password, `SaltedPassword`, or `ClientKey`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScramMaterial {
    /// Per-credential random salt.
    pub salt: Vec<u8>,
    /// PBKDF2 iteration count this material was derived with.
    pub iterations: u32,
    /// `SHA-256(HMAC(SaltedPassword, "Client Key"))` — verifies the client proof.
    pub stored_key: Vec<u8>,
    /// `HMAC(SaltedPassword, "Server Key")` — signs the server's final message.
    pub server_key: Vec<u8>,
}

/// SASLprep (RFC 4013) the password the way an rsasl client does before deriving its proof, so the
/// stored `StoredKey`/`ServerKey` verify against that client. A password that cannot be prepped
/// (prohibited output / bidi violation) is rejected rather than silently passed through.
fn saslprep(password: &str) -> Result<String> {
    stringprep::saslprep(password)
        .map(|cow| cow.into_owned())
        .map_err(|e| Error::PasswordHash(format!("saslprep: {e}")))
}

/// `HMAC-SHA256(key, msg)`.
fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; KEY_LEN] {
    let mut mac =
        <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC accepts keys of any length");
    mac.update(msg);
    mac.finalize().into_bytes().into()
}

/// `SaltedPassword := Hi(SASLprep(password), salt, iterations)` (PBKDF2-HMAC-SHA256, dkLen 32).
fn salted_password(prepped: &str, salt: &[u8], iterations: u32) -> [u8; KEY_LEN] {
    let mut out = [0u8; KEY_LEN];
    pbkdf2::pbkdf2::<Hmac<Sha256>>(prepped.as_bytes(), salt, iterations, &mut out)
        .expect("PBKDF2 output length is valid for SHA-256");
    out
}

/// Derive `(StoredKey, ServerKey)` from an already-computed `SaltedPassword` (RFC 5802 §3).
pub fn keys_from_salted_password(salted: &[u8]) -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
    let client_key = hmac_sha256(salted, b"Client Key");
    let stored_key: [u8; KEY_LEN] = Sha256::digest(client_key).into();
    let server_key = hmac_sha256(salted, b"Server Key");
    (stored_key, server_key)
}

/// Derive full [`ScramMaterial`] for `password` against a given `salt` + `iterations`. The salt is
/// caller-supplied so callers can re-derive deterministically (tests / RFC vectors); production uses
/// [`derive_scram_material`] which mints a fresh random salt.
pub fn derive_with_salt(password: &str, salt: &[u8], iterations: u32) -> Result<ScramMaterial> {
    let prepped = saslprep(password)?;
    let salted = salted_password(&prepped, salt, iterations);
    let (stored_key, server_key) = keys_from_salted_password(&salted);
    Ok(ScramMaterial {
        salt: salt.to_vec(),
        iterations,
        stored_key: stored_key.to_vec(),
        server_key: server_key.to_vec(),
    })
}

/// Derive [`ScramMaterial`] for `password` with a fresh random 16-byte salt and the default
/// iteration count — the production path called on every password set.
pub fn derive_scram_material(password: &str) -> Result<ScramMaterial> {
    let mut salt = [0u8; SCRAM_SALT_LEN];
    getrandom::getrandom(&mut salt).map_err(|e| Error::Entropy(e.to_string()))?;
    derive_with_salt(password, &salt, SCRAM_DEFAULT_ITERATIONS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    fn b64(s: &str) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .expect("valid base64 test vector")
    }

    /// RFC 7677 §3 SCRAM-SHA-256 worked example. Anchors the derivation to the standard: from
    /// `password = "pencil"`, the published `salt`, and `i = 4096`, the `ServerKey` must produce the
    /// published `ServerSignature` and the `StoredKey` must reconstruct from the published
    /// `ClientProof` — i.e. both derived keys match an independent RFC implementation.
    #[test]
    fn rfc7677_scram_sha256_vector() {
        // From RFC 7677 §3:
        //   C: n,,n=user,r=rOprNGfwEbeRWgbNEkqO
        //   S: r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096
        //   C: c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,p=dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=
        //   S: v=6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=
        let salt = b64("W22ZaJ0SNY7soEsUEjb6gQ==");
        let iterations = 4096;
        let client_first_bare = "n=user,r=rOprNGfwEbeRWgbNEkqO";
        let server_first = "r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0,\
                            s=W22ZaJ0SNY7soEsUEjb6gQ==,i=4096";
        let client_final_no_proof = "c=biws,r=rOprNGfwEbeRWgbNEkqO%hvYDpWUa2RaTCAfuxFIlj)hNlF$k0";
        let auth_message = format!("{client_first_bare},{server_first},{client_final_no_proof}");

        let material = derive_with_salt("pencil", &salt, iterations).unwrap();

        // (a) ServerKey check: ServerSignature = HMAC(ServerKey, AuthMessage) == published `v`.
        let server_signature = hmac_sha256(&material.server_key, auth_message.as_bytes());
        assert_eq!(
            server_signature.as_slice(),
            b64("6rriTRBi23WpRR/wtup+mMhUZUn/dB5nLTJRsjl95G4=").as_slice(),
            "ServerKey-derived ServerSignature must equal the RFC 7677 `v=`"
        );

        // (b) StoredKey check: reconstruct ClientKey from the published ClientProof and assert
        //     SHA-256(ClientKey) == our StoredKey.
        //     ClientSignature = HMAC(StoredKey, AuthMessage); ClientKey = ClientProof XOR ClientSignature.
        let client_proof = b64("dHzbZapWIk4jUhN+Ute9ytag9zjfMHgsqmmiz7AndVQ=");
        let client_signature = hmac_sha256(&material.stored_key, auth_message.as_bytes());
        let client_key: Vec<u8> = client_proof
            .iter()
            .zip(client_signature.iter())
            .map(|(p, s)| p ^ s)
            .collect();
        let reconstructed_stored: [u8; KEY_LEN] = Sha256::digest(&client_key).into();
        assert_eq!(
            reconstructed_stored.as_slice(),
            material.stored_key.as_slice(),
            "StoredKey must reconstruct from the RFC 7677 ClientProof"
        );
    }

    #[test]
    fn derivation_is_deterministic_for_a_fixed_salt() {
        let salt = [7u8; SCRAM_SALT_LEN];
        let a = derive_with_salt("correct horse", &salt, SCRAM_DEFAULT_ITERATIONS).unwrap();
        let b = derive_with_salt("correct horse", &salt, SCRAM_DEFAULT_ITERATIONS).unwrap();
        assert_eq!(a, b);
        // A different password yields different keys under the same salt.
        let c = derive_with_salt("battery staple", &salt, SCRAM_DEFAULT_ITERATIONS).unwrap();
        assert_ne!(a.stored_key, c.stored_key);
    }

    #[test]
    fn random_salt_differs_per_derivation() {
        let a = derive_scram_material("pw").unwrap();
        let b = derive_scram_material("pw").unwrap();
        assert_ne!(a.salt, b.salt, "each derivation mints a fresh salt");
        assert_eq!(a.salt.len(), SCRAM_SALT_LEN);
        assert_eq!(a.stored_key.len(), KEY_LEN);
        assert_eq!(a.server_key.len(), KEY_LEN);
    }

    #[test]
    fn non_ascii_password_is_saslprepped_consistently() {
        // SASLprep maps some characters; the derivation must be stable across calls.
        let salt = [3u8; SCRAM_SALT_LEN];
        let a = derive_with_salt("pä55wörd", &salt, SCRAM_DEFAULT_ITERATIONS).unwrap();
        let b = derive_with_salt("pä55wörd", &salt, SCRAM_DEFAULT_ITERATIONS).unwrap();
        assert_eq!(a, b);
    }
}
