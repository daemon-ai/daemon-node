// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! SPAKE2 (RFC 9382), suite **SPAKE2-P256-SHA256-HKDF-HMAC**, for LAN pairing
//! (daemon-pairing-spec.md §3).
//!
//! A balanced PAKE: both sides derive the same secret scalar `w` from the short pairing code
//! ([`PasswordScalar::derive`], pairing spec §3.2), exchange one blinded EC share each (`pA` from
//! the initiator A, `pB` from the responder B), and prove — via HMAC confirmation MACs over the
//! protocol transcript — that they hold the same code without ever enabling offline guessing.
//! The daemon binds the exchange to its TLS channel by feeding both leaf-certificate
//! fingerprints (+ the device name) into the confirmation-key derivation AAD (pairing spec §3.3),
//! so a MITM relaying across two TLS legs cannot make both MACs verify.
//!
//! Roles are fixed by the pairing spec: the app is **party A** (identity [`IDENT_APP`]), the node
//! is **party B** ([`IDENT_NODE`]). Both implementations (this crate and the C++ app side) are
//! gated by the RFC 9382 Appendix B test vectors, and the C++ side additionally replays this
//! crate's deterministic transcript fixture.
//!
//! Implementation notes: P-256 has cofactor `h = 1`, so the RFC's `h*x` / `h*y` multiplications
//! are identity operations. Shares are uncompressed SEC1 (65 bytes, `0x04` prefix) per the RFC's
//! ciphersuite table. Received shares are validated (on-curve, correct encoding) before use, and
//! a degenerate shared point aborts the exchange. `p256::Scalar` is `Copy`, so ephemeral scalars
//! cannot be reliably zeroized in place; the intermediate secret *byte* buffers (`Ka`, the
//! confirmation keys, the HKDF output) are zeroized, and an [`Exchange`] is consumed on `finish`.

use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use p256::elliptic_curve::bigint::{Encoding, NonZero, U384};
use p256::elliptic_curve::group::Group;
use p256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use p256::elliptic_curve::PrimeField;
use p256::{AffinePoint, EncodedPoint, ProjectivePoint};
// Re-exported because it appears in the public `with_scalar`/`scalar_from_bytes` signatures
// (fixture-generation callers name it without a direct p256 dependency).
pub use p256::Scalar;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

/// The ciphersuite this crate implements (RFC 9382 Table 1, row 1).
pub const SUITE: &str = "SPAKE2-P256-SHA256-HKDF-HMAC";

/// Party A's identity string in the daemon ceremony: the app initiates (pairing spec §3.3).
pub const IDENT_APP: &[u8] = b"daemon-app";
/// Party B's identity string in the daemon ceremony: the node responds (pairing spec §3.3).
pub const IDENT_NODE: &[u8] = b"daemon-node";

/// An encoded share (`pA`/`pB`): uncompressed SEC1, `0x04 || X || Y`.
pub const SHARE_LEN: usize = 65;
/// A confirmation MAC (`cA`/`cB`): HMAC-SHA256 output.
pub const CONFIRM_LEN: usize = 32;
/// The shared-secret output `Ke`: half a SHA-256 digest.
pub const KEY_LEN: usize = 16;

/// The HKDF salt for password-to-scalar derivation (pairing spec §3.2).
const W_SALT: &[u8] = b"daemon-pair-v1";
/// The HKDF info for password-to-scalar derivation (pairing spec §3.2).
const W_INFO: &[u8] = b"spake2-w";

/// RFC 9382 §6, P-256 point M (compressed SEC1; seed "1.2.840.10045.3.1.7 point generation
/// seed (M)").
const M_COMPRESSED: [u8; 33] = [
    0x02, 0x88, 0x6e, 0x2f, 0x97, 0xac, 0xe4, 0x6e, 0x55, 0xba, 0x9d, 0xd7, 0x24, 0x25, 0x79, 0xf2,
    0x99, 0x3b, 0x64, 0xe1, 0x6e, 0xf3, 0xdc, 0xab, 0x95, 0xaf, 0xd4, 0x97, 0x33, 0x3d, 0x8f, 0xa1,
    0x2f,
];
/// RFC 9382 §6, P-256 point N (compressed SEC1; seed "1.2.840.10045.3.1.7 point generation
/// seed (N)").
const N_COMPRESSED: [u8; 33] = [
    0x03, 0xd8, 0xbb, 0xd6, 0xc6, 0x39, 0xc6, 0x29, 0x37, 0xb0, 0x4d, 0x99, 0x7f, 0x38, 0xc3, 0x77,
    0x07, 0x19, 0xc6, 0x29, 0xd7, 0x01, 0x4d, 0x49, 0xa2, 0x4b, 0x4f, 0x98, 0xba, 0xa1, 0x29, 0x2b,
    0x49,
];

/// The P-256 group order `n`, zero-padded to 48 bytes for the wide (mod-bias-free) reduction.
const ORDER_U384_HEX: &str = "00000000000000000000000000000000\
                              ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551";

/// What can go wrong in an exchange. Deliberately coarse: protocol callers map every variant to
/// the same indistinguishable `pairing-failed` wire reason (pairing spec §4).
#[derive(Debug, thiserror::Error)]
pub enum PakeError {
    /// A received share was not a valid uncompressed P-256 point.
    #[error("invalid peer share: not a valid uncompressed P-256 point")]
    InvalidPoint,
    /// The unblinded shared point degenerated to the identity (a crafted or corrupt share).
    #[error("degenerate shared point")]
    DegenerateKey,
    /// The OS CSPRNG failed (ephemeral scalar generation).
    #[error("no OS randomness: {0}")]
    Entropy(String),
}

fn m_point() -> ProjectivePoint {
    decode_fixed(&M_COMPRESSED)
}

fn n_point() -> ProjectivePoint {
    decode_fixed(&N_COMPRESSED)
}

fn decode_fixed(compressed: &[u8; 33]) -> ProjectivePoint {
    let ep = EncodedPoint::from_bytes(compressed).expect("RFC constant parses");
    let affine =
        Option::<AffinePoint>::from(AffinePoint::from_encoded_point(&ep)).expect("RFC constant");
    ProjectivePoint::from(affine)
}

/// Reduce 48 big-endian bytes modulo the group order (the mod-bias-free wide reduction both
/// `derive` and ephemeral-scalar generation use; 16 extra bytes make the bias negligible).
fn scalar_from_wide(bytes48: &[u8; 48]) -> Scalar {
    let order = Option::from(NonZero::new(U384::from_be_hex(ORDER_U384_HEX)))
        .expect("group order is nonzero");
    let wide = U384::from_be_slice(bytes48);
    let reduced = wide.rem(&order);
    let be = reduced.to_be_bytes();
    let mut repr = [0u8; 32];
    repr.copy_from_slice(&be[16..48]);
    Option::<Scalar>::from(Scalar::from_repr(repr.into())).expect("reduced value is in range")
}

/// The shared secret scalar `w`, derived identically by both sides.
#[derive(Clone)]
pub struct PasswordScalar(Scalar);

impl PasswordScalar {
    /// Pairing spec §3.2: `w = int(HKDF-SHA256(salt="daemon-pair-v1", ikm=code,
    /// info="spake2-w", L=48)) mod p`, over the canonical (uppercase, separator-stripped,
    /// confusable-folded) code bytes. Canonicalization is the caller's job — this crate never
    /// sees UI formatting.
    pub fn derive(canonical_code: &[u8]) -> Self {
        let hk = Hkdf::<Sha256>::new(Some(W_SALT), canonical_code);
        let mut okm = [0u8; 48];
        hk.expand(W_INFO, &mut okm)
            .expect("48 bytes is valid HKDF length");
        let w = Self(scalar_from_wide(&okm));
        okm.zeroize();
        w
    }

    /// A `w` given directly as a 32-byte big-endian scalar `< n` — the RFC test vectors and the
    /// cross-implementation fixture provide `w` this way. `None` if out of range.
    pub fn from_bytes(bytes: &[u8; 32]) -> Option<Self> {
        Option::<Scalar>::from(Scalar::from_repr((*bytes).into())).map(Self)
    }

    /// The scalar as 32 big-endian bytes — emitted into the cross-implementation transcript
    /// fixture so the C++ side can gate its §3.2 derivation step independently of the exchange.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0.to_repr().into()
    }
}

/// Decode a caller-provided 32-byte big-endian ephemeral scalar `< n` (`None` if out of range).
/// For [`Exchange::with_scalar`] callers ONLY — the RFC vector tests and the deterministic
/// cross-implementation fixture; production exchanges draw their scalars internally.
pub fn scalar_from_bytes(bytes: &[u8; 32]) -> Option<Scalar> {
    Option::from(Scalar::from_repr((*bytes).into()))
}

/// Which side of the exchange this is (RFC 9382: A uses M and goes first; B uses N).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The initiator (the app in the daemon ceremony).
    A,
    /// The responder (the node in the daemon ceremony).
    B,
}

/// One in-flight SPAKE2 exchange for one party. Single-use: [`Exchange::finish`] consumes it.
pub struct Exchange {
    role: Role,
    w: Scalar,
    secret: Scalar,
    id_a: Vec<u8>,
    id_b: Vec<u8>,
    share: [u8; SHARE_LEN],
}

impl Exchange {
    /// Start as party A (initiator) with a fresh ephemeral scalar.
    pub fn new_a(w: &PasswordScalar, id_a: &[u8], id_b: &[u8]) -> Result<Self, PakeError> {
        Ok(Self::with_scalar(Role::A, w, random_scalar()?, id_a, id_b))
    }

    /// Start as party B (responder) with a fresh ephemeral scalar.
    pub fn new_b(w: &PasswordScalar, id_a: &[u8], id_b: &[u8]) -> Result<Self, PakeError> {
        Ok(Self::with_scalar(Role::B, w, random_scalar()?, id_a, id_b))
    }

    /// Start with a caller-provided ephemeral scalar. For the RFC vector tests and the
    /// deterministic cross-implementation fixture ONLY — production callers use
    /// [`Exchange::new_a`]/[`Exchange::new_b`] (ephemeral reuse breaks the protocol's security).
    pub fn with_scalar(
        role: Role,
        w: &PasswordScalar,
        secret: Scalar,
        id_a: &[u8],
        id_b: &[u8],
    ) -> Self {
        let public = ProjectivePoint::GENERATOR * secret;
        let blind = match role {
            Role::A => m_point(),
            Role::B => n_point(),
        };
        let share_point = public + blind * w.0;
        let encoded = share_point.to_affine().to_encoded_point(false);
        let mut share = [0u8; SHARE_LEN];
        share.copy_from_slice(encoded.as_bytes());
        Self {
            role,
            w: w.0,
            secret,
            id_a: id_a.to_vec(),
            id_b: id_b.to_vec(),
            share,
        }
    }

    /// This party's share (`pA` for role A, `pB` for role B): uncompressed SEC1.
    pub fn share(&self) -> &[u8; SHARE_LEN] {
        &self.share
    }

    /// Consume the peer's share and derive the transcript keys and confirmation MACs.
    /// `aad` is the additional authenticated data bound into the confirmation-key derivation —
    /// the daemon ceremony's channel binding (pairing spec §3.3). The peer's share is validated
    /// (encoding + on-curve) before any arithmetic.
    pub fn finish(self, peer_share: &[u8], aad: &[u8]) -> Result<Finished, PakeError> {
        let peer_point = decode_share(peer_share)?;
        // Unblind the peer's share with the OTHER party's fixed element, then apply our
        // ephemeral scalar: A computes x*(pB - w*N), B computes y*(pA - w*M). h = 1 on P-256.
        let peer_blind = match self.role {
            Role::A => n_point(),
            Role::B => m_point(),
        };
        let k_point = (peer_point - peer_blind * self.w) * self.secret;
        if bool::from(k_point.is_identity()) {
            return Err(PakeError::DegenerateKey);
        }
        let k_encoded = k_point.to_affine().to_encoded_point(false);

        // The RFC 9382 §3.3 transcript: A, B, pA, pB, K, w — each 8-byte-LE length-prefixed;
        // w big-endian, padded to the group-order length (32 bytes).
        let (pa, pb): (&[u8], &[u8]) = match self.role {
            Role::A => (&self.share, peer_share),
            Role::B => (peer_share, &self.share),
        };
        let mut tt = Vec::with_capacity(256);
        push_len_prefixed(&mut tt, &self.id_a);
        push_len_prefixed(&mut tt, &self.id_b);
        push_len_prefixed(&mut tt, pa);
        push_len_prefixed(&mut tt, pb);
        push_len_prefixed(&mut tt, k_encoded.as_bytes());
        push_len_prefixed(&mut tt, &self.w.to_bytes());

        // Ke || Ka = Hash(TT); KcA || KcB = HKDF(Ka, nil, "ConfirmationKeys" || AAD).
        let digest = Sha256::digest(&tt);
        let mut ke = [0u8; KEY_LEN];
        ke.copy_from_slice(&digest[..KEY_LEN]);
        let mut ka = [0u8; KEY_LEN];
        ka.copy_from_slice(&digest[KEY_LEN..]);
        let hk = Hkdf::<Sha256>::new(None, &ka);
        let mut info = Vec::with_capacity(16 + aad.len());
        info.extend_from_slice(b"ConfirmationKeys");
        info.extend_from_slice(aad);
        let mut okm = [0u8; 32];
        hk.expand(&info, &mut okm)
            .expect("32 bytes is valid HKDF length");
        let ca = hmac_sha256(&okm[..16], &tt);
        let cb = hmac_sha256(&okm[16..], &tt);
        ka.zeroize();
        okm.zeroize();

        let (confirm_local, confirm_peer) = match self.role {
            Role::A => (ca, cb),
            Role::B => (cb, ca),
        };
        Ok(Finished {
            ke,
            confirm_local,
            confirm_peer,
        })
    }
}

/// The completed exchange: the shared key and both confirmation MACs, from one party's view.
pub struct Finished {
    ke: [u8; KEY_LEN],
    confirm_local: [u8; CONFIRM_LEN],
    confirm_peer: [u8; CONFIRM_LEN],
}

impl Finished {
    /// The shared secret `Ke`. Unused for data in the daemon ceremony v1 (the mutually-confirmed
    /// TLS channel carries everything), but it IS the protocol output.
    pub fn key(&self) -> &[u8; KEY_LEN] {
        &self.ke
    }

    /// The confirmation MAC this party sends (`cA` when role A, `cB` when role B).
    pub fn local_confirmation(&self) -> &[u8; CONFIRM_LEN] {
        &self.confirm_local
    }

    /// Constant-time verification of the peer's confirmation MAC. `false` means the peer does
    /// not hold the same code / channel — the exchange MUST be aborted.
    #[must_use]
    pub fn verify_peer_confirmation(&self, mac: &[u8]) -> bool {
        mac.len() == CONFIRM_LEN && bool::from(self.confirm_peer.ct_eq(mac))
    }
}

fn push_len_prefixed(tt: &mut Vec<u8>, data: &[u8]) {
    tt.extend_from_slice(&(data.len() as u64).to_le_bytes());
    tt.extend_from_slice(data);
}

fn hmac_sha256(key: &[u8], message: &[u8]) -> [u8; CONFIRM_LEN] {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(message);
    mac.finalize().into_bytes().into()
}

/// A fresh uniform scalar in `[0, n)` via the same wide reduction as [`PasswordScalar::derive`].
fn random_scalar() -> Result<Scalar, PakeError> {
    let mut wide = [0u8; 48];
    getrandom::fill(&mut wide).map_err(|e| PakeError::Entropy(e.to_string()))?;
    let s = scalar_from_wide(&wide);
    wide.zeroize();
    Ok(s)
}

/// Validate + decode a peer share: exactly 65 bytes, `0x04` (uncompressed) prefix, on-curve.
fn decode_share(bytes: &[u8]) -> Result<ProjectivePoint, PakeError> {
    if bytes.len() != SHARE_LEN || bytes[0] != 0x04 {
        return Err(PakeError::InvalidPoint);
    }
    let ep = EncodedPoint::from_bytes(bytes).map_err(|_| PakeError::InvalidPoint)?;
    let affine = Option::<AffinePoint>::from(AffinePoint::from_encoded_point(&ep))
        .ok_or(PakeError::InvalidPoint)?;
    Ok(ProjectivePoint::from(affine))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn scalar(hex32: &str) -> Scalar {
        let bytes: [u8; 32] = unhex(hex32).try_into().unwrap();
        Option::<Scalar>::from(Scalar::from_repr(bytes.into())).unwrap()
    }

    /// One RFC 9382 Appendix B vector, both sides, checked end to end: share encodings, `Ke`,
    /// and both confirmation MACs.
    #[allow(clippy::too_many_arguments)]
    fn check_vector(
        id_a: &[u8],
        id_b: &[u8],
        w_hex: &str,
        x_hex: &str,
        y_hex: &str,
        pa_hex: &str,
        pb_hex: &str,
        ke_hex: &str,
        ca_hex: &str,
        cb_hex: &str,
    ) {
        let w = PasswordScalar(scalar(w_hex));
        let a = Exchange::with_scalar(Role::A, &w, scalar(x_hex), id_a, id_b);
        let b = Exchange::with_scalar(Role::B, &w, scalar(y_hex), id_a, id_b);
        assert_eq!(a.share().as_slice(), unhex(pa_hex), "pA encoding");
        assert_eq!(b.share().as_slice(), unhex(pb_hex), "pB encoding");

        let pb = *b.share();
        let pa = *a.share();
        let fin_a = a.finish(&pb, b"").expect("A finishes");
        let fin_b = b.finish(&pa, b"").expect("B finishes");

        assert_eq!(fin_a.key().as_slice(), unhex(ke_hex), "Ke (A)");
        assert_eq!(fin_b.key().as_slice(), unhex(ke_hex), "Ke (B)");
        assert_eq!(fin_a.local_confirmation().as_slice(), unhex(ca_hex), "cA");
        assert_eq!(fin_b.local_confirmation().as_slice(), unhex(cb_hex), "cB");
        assert!(
            fin_a.verify_peer_confirmation(&unhex(cb_hex)),
            "A verifies cB"
        );
        assert!(
            fin_b.verify_peer_confirmation(&unhex(ca_hex)),
            "B verifies cA"
        );
        assert!(
            !fin_a.verify_peer_confirmation(&unhex(ca_hex)),
            "wrong MAC rejected"
        );
    }

    #[test]
    fn rfc9382_vector_server_client() {
        check_vector(
            b"server",
            b"client",
            "2ee57912099d31560b3a44b1184b9b4866e904c49d12ac5042c97dca461b1a5f",
            "43dd0fd7215bdcb482879fca3220c6a968e66d70b1356cac18bb26c84a78d729",
            "dcb60106f276b02606d8ef0a328c02e4b629f84f89786af5befb0bc75b6e66be",
            "04a56fa807caaa53a4d28dbb9853b9815c61a411118a6fe516a8798434751470\
             f9010153ac33d0d5f2047ffdb1a3e42c9b4e6be662766e1eeb4116988ede5f912c",
            "0406557e482bd03097ad0cbaa5df82115460d951e3451962f1eaf4367a420676\
             d09857ccbc522686c83d1852abfa8ed6e4a1155cf8f1543ceca528afb591a1e0b7",
            "0e0672dc86f8e45565d338b0540abe69",
            "58ad4aa88e0b60d5061eb6b5dd93e80d9c4f00d127c65b3b35b1b5281fee38f0",
            "d3e2e547f1ae04f2dbdbf0fc4b79f8ecff2dff314b5d32fe9fcef2fb26dc459b",
        );
    }

    #[test]
    fn rfc9382_vector_nil_a() {
        check_vector(
            b"",
            b"client",
            "0548d8729f730589e579b0475a582c1608138ddf7054b73b5381c7e883e2efae",
            "403abbe3b1b4b9ba17e3032849759d723939a27a27b9d921c500edde18ed654b",
            "903023b6598908936ea7c929bd761af6039577a9c3f9581064187c3049d87065",
            "04a897b769e681c62ac1c2357319a3d363f610839c4477720d24cbe32f5fd8\
             5f44fb92ba966578c1b712be6962498834078262caa5b441ecfa9d4a9485720e918a",
            "04e0f816fd1c35e22065d5556215c097e799390d16661c386e0ecc84593974\
             a61b881a8c82327687d0501862970c64565560cb5671f696048050ca66ca5f8cc7fc",
            "642f05c473c2cd79909f9a841e2f30a7",
            "47d29e6666af1b7dd450d571233085d7a9866e4d49d2645e2df975489521232b",
            "3313c5cefc361d27fb16847a91c2a73b766ffa90a4839122a9b70a2f6bd1d6df",
        );
    }

    #[test]
    fn rfc9382_vector_nil_b() {
        check_vector(
            b"server",
            b"",
            "626e0cdc7b14c9db3e52a0b1b3a768c98e37852d5db30febe0497b14eae8c254",
            "07adb3db6bc623d3399726bfdbfd3d15a58ea776ab8a308b00392621291f9633",
            "b6a4fc8dbb629d4ba51d6f91ed1532cf87adec98f25dd153a75accafafedec16",
            "04f88fb71c99bfffaea370966b7eb99cd4be0ff1a7d335caac4211c4afd855e2\
             e15a873b298503ad8ba1d9cbb9a392d2ba309b48bfd7879aefd0f2cea6009763b0",
            "040c269d6be017dccb15182ac6bfcd9e2a14de019dd587eaf4bdfd353f031101\
             e7cca177f8eb362a6e83e7d5e729c0732e1b528879c086f39ba0f31a9661bd34db",
            "005184ff460da2ce59062c87733c299c",
            "bc9f9bbe99f26d0b2260e6456e05a86196a3307ec6663a18bf6ac825736533b2",
            "c2370e1bf813b086dff0d834e74425a06e6390f48f5411900276dcccc5a297ec",
        );
    }

    #[test]
    fn rfc9382_vector_nil_both() {
        check_vector(
            b"",
            b"",
            "7bf46c454b4c1b25799527d896508afd5fc62ef4ec59db1efb49113063d70cca",
            "8cef65df64bb2d0f83540c53632de911b5b24b3eab6cc74a97609fd659e95473",
            "d7a66f64074a84652d8d623a92e20c9675c61cb5b4f6a0063e4648a2fdc02d53",
            "04a65b367a3f613cf9f0654b1b28a1e3a8a40387956c8ba6063e8658563890f4\
             6ca1ef6a676598889fc28de2950ab8120b79a5ef1ea4c9f44bc98f585634b46d66",
            "04589f13218822710d98d8b2123a079041052d9941b9cf88c6617ddb2fcc0494\
             662eea8ba6b64692dc318250030c6af045cb738bc81ba35b043c3dcb46adf6f58d",
            "fc6374762ba5cf11f4b2caa08b2cd1b9",
            "dfb4db8d48ae5a675963ea5e6c19d98d4ea028d8e898dad96ea19a80ade95dca",
            "d0f0609d1613138d354f7e95f19fb556bf52d751947241e8c7118df5ef0ae175",
        );
    }

    /// The daemon ceremony end to end with random ephemerals: same code + same AAD agree;
    /// a wrong code fails confirmation; a different AAD (the anti-relay channel binding)
    /// fails confirmation even with the right code.
    #[test]
    fn daemon_ceremony_round_trip_and_binding() {
        let w = PasswordScalar::derive(b"ABCDE12345");
        let aad = b"server-fp-32-bytes...client-fp-32-bytes...Pixel 9";

        let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).unwrap();
        let b = Exchange::new_b(&w, IDENT_APP, IDENT_NODE).unwrap();
        let (pa, pb) = (*a.share(), *b.share());
        let fin_a = a.finish(&pb, aad).unwrap();
        let fin_b = b.finish(&pa, aad).unwrap();
        assert_eq!(fin_a.key(), fin_b.key());
        assert!(fin_a.verify_peer_confirmation(fin_b.local_confirmation()));
        assert!(fin_b.verify_peer_confirmation(fin_a.local_confirmation()));

        // Wrong code: shares exchange fine, confirmation MACs never verify.
        let w_bad = PasswordScalar::derive(b"ABCDE12346");
        let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).unwrap();
        let b = Exchange::new_b(&w_bad, IDENT_APP, IDENT_NODE).unwrap();
        let (pa, pb) = (*a.share(), *b.share());
        let fin_a = a.finish(&pb, aad).unwrap();
        let fin_b = b.finish(&pa, aad).unwrap();
        assert!(!fin_a.verify_peer_confirmation(fin_b.local_confirmation()));
        assert!(!fin_b.verify_peer_confirmation(fin_a.local_confirmation()));

        // Same code, different AAD on each side (two TLS legs): confirmation fails — the
        // channel binding that defeats a relaying MITM.
        let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).unwrap();
        let b = Exchange::new_b(&w, IDENT_APP, IDENT_NODE).unwrap();
        let (pa, pb) = (*a.share(), *b.share());
        let fin_a = a.finish(&pb, b"leg-one-certs").unwrap();
        let fin_b = b.finish(&pa, b"leg-two-certs").unwrap();
        assert!(!fin_a.verify_peer_confirmation(fin_b.local_confirmation()));
        assert!(!fin_b.verify_peer_confirmation(fin_a.local_confirmation()));
    }

    /// §3.2 derivation is deterministic and code-sensitive.
    #[test]
    fn derive_w_deterministic() {
        let w1 = PasswordScalar::derive(b"ABCDE12345");
        let w2 = PasswordScalar::derive(b"ABCDE12345");
        let w3 = PasswordScalar::derive(b"ABCDE12344");
        assert_eq!(w1.0, w2.0);
        assert_ne!(w1.0, w3.0);
    }

    /// Malformed peer shares are rejected before any arithmetic.
    #[test]
    fn invalid_shares_rejected() {
        let w = PasswordScalar::derive(b"ABCDE12345");
        let bad_lengths: &[&[u8]] = &[b"", &[0x04; 10], &[0x04; 64], &[0x04; 66]];
        for bad in bad_lengths {
            let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).unwrap();
            assert!(matches!(a.finish(bad, b""), Err(PakeError::InvalidPoint)));
        }
        // Right length, compressed prefix: rejected (the suite fixes uncompressed).
        let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).unwrap();
        let mut compressed_prefix = [0u8; SHARE_LEN];
        compressed_prefix[0] = 0x02;
        assert!(matches!(
            a.finish(&compressed_prefix, b""),
            Err(PakeError::InvalidPoint)
        ));
        // Right shape, off-curve coordinates: rejected.
        let a = Exchange::new_a(&w, IDENT_APP, IDENT_NODE).unwrap();
        let mut off_curve = [0u8; SHARE_LEN];
        off_curve[0] = 0x04;
        off_curve[64] = 0x01;
        assert!(matches!(
            a.finish(&off_curve, b""),
            Err(PakeError::InvalidPoint)
        ));
    }
}
