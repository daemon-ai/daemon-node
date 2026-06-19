//! The verifiable trace journal: theater-style hash-linking upgraded with dCBOR + Gordian Envelope.
//!
//! Each trace event becomes a **Gordian [`Envelope`]** (subject = a kind label; assertions =
//! `session`, `epoch`, `seq`, `trace`, `detail`, `timestamp`) serialized as **deterministic CBOR**
//! so its bytes — and therefore its digest — are reproducible. An entry's [`ContentHash`] is the
//! envelope's digest-tree root.
//!
//! A `(session, epoch)` **segment** is itself an envelope: its subject binds the prior epoch's root
//! (a rolling hash chain across incarnations), and it carries one assertion per entry whose object
//! is that entry's envelope. The segment envelope's digest is therefore a Merkle root folding every
//! entry digest plus the prior root — the segment [`MerkleRoot`]. That root is signed with an
//! ed25519 key ([`bc_components`]); [`verify_segment`] recomputes every entry hash and the root from
//! the stored bytes and checks the signature and the cross-epoch link, so any mutation — to an
//! entry, the set of entries, or the chain — is detected.
//!
//! The store persists only the opaque envelope bytes, the content hashes, and the 32-byte roots;
//! all crypto lives here (layout §3 keeps the DAG root crypto-free).

use bc_components::{PrivateKeyBase, Signature, Signer, SigningPrivateKey, SigningPublicKey, Verifier};
use bc_envelope::prelude::*;
use daemon_common::{ContentHash, Epoch, MerkleRoot, SessionId, TraceId};

/// The genesis "prior root" for the first epoch of a session (epoch 0 chains onto zero).
pub const GENESIS_ROOT: MerkleRoot = MerkleRoot::new([0u8; 32]);

/// A single trace event to be journaled. The host builds these from its `ManageEvent` stream
/// (keeping `daemon-telemetry` free of a `daemon-supervision` dependency).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRecord {
    /// The session the event belongs to.
    pub session: SessionId,
    /// The incarnation epoch (segment key).
    pub epoch: Epoch,
    /// Monotonic per-`(session, epoch)` sequence number.
    pub seq: u64,
    /// The correlation trace context active when the event occurred.
    pub trace: TraceId,
    /// A short kind label (the envelope subject), e.g. `"started"`, `"usage"`, `"finished"`.
    pub kind: String,
    /// Human/structured detail for the event.
    pub detail: String,
    /// Milliseconds since the Unix epoch when the event was recorded.
    pub timestamp_ms: u64,
}

fn digest_to_32(d: &Digest) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(d.data());
    out
}

/// Build the Gordian Envelope for one trace record (deterministic by construction).
fn entry_envelope(rec: &TraceRecord) -> Envelope {
    Envelope::new(rec.kind.clone())
        .add_assertion("session", rec.session.as_str())
        .add_assertion("epoch", rec.epoch.0)
        .add_assertion("seq", rec.seq)
        .add_assertion("trace", rec.trace.0)
        .add_assertion("detail", rec.detail.clone())
        .add_assertion("timestamp", rec.timestamp_ms)
}

/// Encode a trace record to its opaque dCBOR bytes and content hash (the envelope's digest).
///
/// The returned bytes are what `daemon-store` persists; the [`ContentHash`] is stored alongside so
/// a verifier can detect byte-level tampering before even recomputing the segment root.
pub fn encode_entry(rec: &TraceRecord) -> (Vec<u8>, ContentHash) {
    let env = entry_envelope(rec);
    let hash = ContentHash::new(digest_to_32(&env.digest()));
    (env.to_cbor_data(), hash)
}

/// Build the segment envelope: prior root in the subject, one assertion per entry envelope.
fn segment_envelope(
    session: &SessionId,
    epoch: Epoch,
    prior: MerkleRoot,
    entries: &[Envelope],
) -> Envelope {
    let mut env = Envelope::new(format!("trace-segment:{session}:{}", epoch.0))
        .add_assertion("prior_root", prior.to_hex());
    for entry in entries {
        env = env.add_assertion("entry", entry.clone());
    }
    env
}

/// The inputs needed to (re)compute and verify a `(session, epoch)` segment: the prior epoch's
/// root and the per-entry `(seq, dCBOR bytes, stored content hash)` as loaded from the store.
#[derive(Clone, Debug)]
pub struct SegmentInput<'a> {
    /// The session the segment belongs to.
    pub session: &'a SessionId,
    /// The epoch the segment covers.
    pub epoch: Epoch,
    /// The prior epoch's committed root (or [`GENESIS_ROOT`] for epoch 0).
    pub prior: MerkleRoot,
    /// The segment's entries as loaded: `(seq, envelope dCBOR bytes, stored content hash)`.
    pub entries: &'a [(u64, Vec<u8>, ContentHash)],
}

fn decode_envelope(bytes: &[u8]) -> Result<Envelope, VerifyError> {
    let cbor = CBOR::try_from_data(bytes).map_err(|_| VerifyError::Decode)?;
    Envelope::try_from(cbor).map_err(|_| VerifyError::Decode)
}

/// Recompute the segment [`MerkleRoot`] from decoded entry envelopes + the prior root. Also returns
/// the rebuilt entry envelopes so callers can avoid decoding twice.
fn recompute_root(input: &SegmentInput<'_>) -> Result<MerkleRoot, VerifyError> {
    let mut envs = Vec::with_capacity(input.entries.len());
    for (_seq, bytes, stored_hash) in input.entries {
        let env = decode_envelope(bytes)?;
        // Byte/content tamper: the stored content hash must equal the decoded envelope's digest.
        let recomputed = ContentHash::new(digest_to_32(&env.digest()));
        if recomputed != *stored_hash {
            return Err(VerifyError::ContentHashMismatch);
        }
        envs.push(env);
    }
    let seg = segment_envelope(input.session, input.epoch, input.prior, &envs);
    Ok(MerkleRoot::new(digest_to_32(&seg.digest())))
}

/// Compute the committed root for a freshly-built segment (sealing path, host side).
pub fn segment_root(input: &SegmentInput<'_>) -> Result<MerkleRoot, VerifyError> {
    recompute_root(input)
}

/// An ed25519 signing key for sealing segment roots.
pub struct TraceSigner {
    private: SigningPrivateKey,
    public: SigningPublicKey,
}

impl TraceSigner {
    /// Generate a fresh ed25519 key (derived from a random [`PrivateKeyBase`]).
    pub fn generate() -> Self {
        let private = PrivateKeyBase::new().ed25519_signing_private_key();
        let public = private
            .public_key()
            .expect("ed25519 private key yields a public key");
        Self { private, public }
    }

    /// The verifying half, for distribution to verifiers.
    pub fn verifying_key(&self) -> VerifyingKey {
        VerifyingKey(self.public.clone())
    }

    /// Sign a segment root, returning detached signature bytes (dCBOR of the [`Signature`]).
    pub fn sign_root(&self, root: &MerkleRoot) -> Vec<u8> {
        let msg: &[u8] = root.as_bytes();
        let sig: Signature = self
            .private
            .sign(&msg)
            .expect("ed25519 signing is infallible for valid keys");
        sig.to_cbor_data()
    }
}

/// The public half of a [`TraceSigner`], used to verify sealed segment roots.
#[derive(Clone)]
pub struct VerifyingKey(SigningPublicKey);

impl VerifyingKey {
    fn verify_root(&self, root: &MerkleRoot, signature: &[u8]) -> bool {
        let Ok(cbor) = CBOR::try_from_data(signature) else {
            return false;
        };
        let Ok(sig) = Signature::try_from(cbor) else {
            return false;
        };
        let msg: &[u8] = root.as_bytes();
        self.0.verify(&sig, &msg)
    }
}

/// Why a segment failed to verify.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum VerifyError {
    /// A stored entry's bytes did not decode as a Gordian Envelope.
    #[error("a trace entry did not decode as an envelope")]
    Decode,
    /// A stored content hash did not match the entry envelope's digest (entry tampered).
    #[error("trace entry content hash mismatch (entry tampered)")]
    ContentHashMismatch,
    /// The recomputed segment root did not match the committed root (entries/chain tampered).
    #[error("recomputed segment root does not match the committed root")]
    RootMismatch,
    /// The signature over the committed root did not verify.
    #[error("segment root signature verification failed")]
    BadSignature,
}

/// Verify a sealed `(session, epoch)` segment end-to-end:
/// 1. every entry's bytes decode and match their stored content hash;
/// 2. the recomputed Merkle root (folding entries + the prior root) equals `committed_root`;
/// 3. `signature` is a valid ed25519 signature over `committed_root` by `key`.
///
/// Because the prior root is folded into the recomputed root, passing the wrong prior (a broken
/// cross-epoch link) surfaces as [`VerifyError::RootMismatch`].
pub fn verify_segment(
    input: &SegmentInput<'_>,
    committed_root: &MerkleRoot,
    signature: &[u8],
    key: &VerifyingKey,
) -> Result<(), VerifyError> {
    let recomputed = recompute_root(input)?;
    if &recomputed != committed_root {
        return Err(VerifyError::RootMismatch);
    }
    if !key.verify_root(committed_root, signature) {
        return Err(VerifyError::BadSignature);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(session: &str, epoch: u64, seq: u64, kind: &str) -> TraceRecord {
        TraceRecord {
            session: SessionId::new(session),
            epoch: Epoch(epoch),
            seq,
            trace: TraceId(0xC0FFEE),
            kind: kind.into(),
            detail: format!("detail-{seq}"),
            timestamp_ms: 1_000 + seq,
        }
    }

    fn build_segment(
        records: &[TraceRecord],
    ) -> Vec<(u64, Vec<u8>, ContentHash)> {
        records
            .iter()
            .map(|r| {
                let (bytes, hash) = encode_entry(r);
                (r.seq, bytes, hash)
            })
            .collect()
    }

    #[test]
    fn seal_and_verify_round_trip() {
        let session = SessionId::new("verifiable");
        let records = [
            rec("verifiable", 0, 0, "started"),
            rec("verifiable", 0, 1, "usage"),
            rec("verifiable", 0, 2, "finished"),
        ];
        let entries = build_segment(&records);
        let input = SegmentInput {
            session: &session,
            epoch: Epoch(0),
            prior: GENESIS_ROOT,
            entries: &entries,
        };

        let root = segment_root(&input).unwrap();
        let signer = TraceSigner::generate();
        let sig = signer.sign_root(&root);

        verify_segment(&input, &root, &sig, &signer.verifying_key())
            .expect("a faithfully sealed segment verifies");
    }

    #[test]
    fn root_is_deterministic() {
        let session = SessionId::new("det");
        let records = [rec("det", 0, 0, "a"), rec("det", 0, 1, "b")];
        let e1 = build_segment(&records);
        let e2 = build_segment(&records);
        let i1 = SegmentInput { session: &session, epoch: Epoch(0), prior: GENESIS_ROOT, entries: &e1 };
        let i2 = SegmentInput { session: &session, epoch: Epoch(0), prior: GENESIS_ROOT, entries: &e2 };
        assert_eq!(segment_root(&i1).unwrap(), segment_root(&i2).unwrap());
    }

    #[test]
    fn tampering_an_entry_is_detected() {
        let session = SessionId::new("tamper");
        let records = [rec("tamper", 0, 0, "started"), rec("tamper", 0, 1, "finished")];
        let mut entries = build_segment(&records);
        let input = SegmentInput { session: &session, epoch: Epoch(0), prior: GENESIS_ROOT, entries: &entries };
        let root = segment_root(&input).unwrap();
        let signer = TraceSigner::generate();
        let sig = signer.sign_root(&root);

        // Mutate the bytes of entry 1 (without updating its stored content hash).
        entries[1].1[0] ^= 0xFF;
        let tampered = SegmentInput { session: &session, epoch: Epoch(0), prior: GENESIS_ROOT, entries: &entries };
        let err = verify_segment(&tampered, &root, &sig, &signer.verifying_key()).unwrap_err();
        assert!(matches!(err, VerifyError::Decode | VerifyError::ContentHashMismatch | VerifyError::RootMismatch));
    }

    #[test]
    fn cross_epoch_chain_links() {
        let session = SessionId::new("chain");
        // Epoch 0.
        let r0 = [rec("chain", 0, 0, "started"), rec("chain", 0, 1, "finished")];
        let e0 = build_segment(&r0);
        let i0 = SegmentInput { session: &session, epoch: Epoch(0), prior: GENESIS_ROOT, entries: &e0 };
        let root0 = segment_root(&i0).unwrap();

        // Epoch 1 chains onto epoch 0's root.
        let r1 = [rec("chain", 1, 0, "resumed"), rec("chain", 1, 1, "finished")];
        let e1 = build_segment(&r1);
        let i1 = SegmentInput { session: &session, epoch: Epoch(1), prior: root0, entries: &e1 };
        let root1 = segment_root(&i1).unwrap();
        let signer = TraceSigner::generate();
        let sig1 = signer.sign_root(&root1);

        // Verifies with the correct prior...
        verify_segment(&i1, &root1, &sig1, &signer.verifying_key()).expect("chained segment verifies");

        // ...but a broken link (wrong prior) is rejected as a root mismatch.
        let broken = SegmentInput { session: &session, epoch: Epoch(1), prior: GENESIS_ROOT, entries: &e1 };
        assert_eq!(
            verify_segment(&broken, &root1, &sig1, &signer.verifying_key()).unwrap_err(),
            VerifyError::RootMismatch
        );
    }

    #[test]
    fn wrong_key_fails_signature() {
        let session = SessionId::new("sig");
        let records = [rec("sig", 0, 0, "started")];
        let entries = build_segment(&records);
        let input = SegmentInput { session: &session, epoch: Epoch(0), prior: GENESIS_ROOT, entries: &entries };
        let root = segment_root(&input).unwrap();
        let signer = TraceSigner::generate();
        let sig = signer.sign_root(&root);

        let other = TraceSigner::generate();
        assert_eq!(
            verify_segment(&input, &root, &sig, &other.verifying_key()).unwrap_err(),
            VerifyError::BadSignature
        );
    }
}
