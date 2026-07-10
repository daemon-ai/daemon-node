// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The verifiable journal: one hash-linked, per-segment-signed chain per stream carrying typed
//! entries — coarse management records and coalesced finished chat blocks.
//!
//! Each entry is stored as its canonical CBOR ([`JournalEntryView`]) so a reader can **replay** it
//! with a plain deserialize, while its tamper-evidence comes from a **Gordian [`Envelope`]** built
//! deterministically from the same view: the envelope's digest is the entry's [`ContentHash`]. A
//! `(stream, segment)` **segment** is itself an envelope whose subject binds the prior segment's
//! root (a rolling hash chain) and which carries one assertion per entry envelope; its digest is
//! the segment [`MerkleRoot`], signed with an ed25519 key. [`verify_segment`] rebuilds every entry
//! envelope from the stored bytes, recomputes the root, and checks the signature and the chain, so
//! any mutation — to an entry, the set of entries, or the chain — is detected.
//!
//! Keyed `(stream, segment)` rather than `(session, epoch)`: a stream is any addressable agent (a
//! durable session, a live session, a fleet/foreign unit), and a segment is a turn (streaming) or
//! an incarnation (durable). The store persists only the opaque view bytes, the content hashes, and
//! the 32-byte roots; all crypto lives here (layout §3 keeps the DAG root crypto-free).

use bc_components::{
    PrivateKeyBase, Signature, Signer, SigningPrivateKey, SigningPublicKey, Verifier,
};
use bc_envelope::prelude::*;
use daemon_common::{ContentHash, JournalStreamId, MerkleRoot};
use serde::{Deserialize, Serialize};

/// The genesis "prior root" for a stream's first segment (segment 0 chains onto zero).
pub const GENESIS_ROOT: MerkleRoot = MerkleRoot::new([0u8; 32]);

/// The typed payload of a journal entry — what distinguishes a management record from a chat block.
/// Opaque to the store (it sees only the encoded entry bytes); a reader routes on it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JournalPayload {
    /// A coarse management lifecycle record (the human/structured `detail` of a `ManageEvent` or a
    /// credential-audit event).
    Management {
        /// Human/structured detail for the event.
        detail: String,
    },
    /// A coalesced finished chat block: opaque CBOR of a `daemon-protocol` `TranscriptBlock`, which
    /// the consuming GUI decodes. Kept opaque here so the crypto/store layers stay protocol-free.
    Block {
        /// The opaque encoded block (CBOR by convention).
        body: Vec<u8>,
    },
    /// One conversation chat message (wire vNEXT): opaque CBOR of a `daemon-api` `ChatMessage`,
    /// the per-message record a messaging adapter's send/inbound obligation appends to a
    /// `conv:<transport>:<conv>` stream. Opaque here for the same reason as `Block`: the
    /// crypto/store layers stay contract-free; the host's history reader decodes it.
    Chat {
        /// The opaque encoded message (CBOR by convention).
        body: Vec<u8>,
    },
}

/// One journal entry: the canonical, replayable record the host appends and a reader decodes. The
/// stored bytes are this value's CBOR; the entry's [`ContentHash`] is the digest of the Gordian
/// envelope built deterministically from it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalEntryView {
    /// The stream the entry belongs to.
    pub stream: JournalStreamId,
    /// The segment (turn / incarnation) the entry belongs to.
    pub segment: u64,
    /// Monotonic per-`(stream, segment)` sequence number.
    pub seq: u64,
    /// The incarnation epoch active when recorded (metadata; 0 for non-durable or first turn).
    pub epoch: u64,
    /// The correlation trace context active when the entry was recorded.
    pub trace: u64,
    /// A short kind label (the envelope subject), e.g. `"mgmt.started"`, `"block.message"`.
    pub kind: String,
    /// Milliseconds since the Unix epoch when the entry was recorded.
    pub timestamp_ms: u64,
    /// Provenance: the node build that wrote this entry (the host stamps `daemon_common::VERSION`).
    /// Append-only — entries are never rewritten. Omitted from the wire when empty so entries
    /// written before this field existed reproduce their original Gordian digest and still verify
    /// (see [`entry_envelope`]).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub writer_version: String,
    /// The typed payload.
    pub payload: JournalPayload,
}

fn digest_to_32(d: &Digest) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(d.data());
    out
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Build the deterministic Gordian Envelope for one entry (the tamper-evidence over every field).
fn entry_envelope(v: &JournalEntryView) -> Envelope {
    let mut env = Envelope::new(v.kind.clone())
        .add_assertion("stream", v.stream.as_str())
        .add_assertion("segment", v.segment)
        .add_assertion("seq", v.seq)
        .add_assertion("epoch", v.epoch)
        .add_assertion("trace", v.trace)
        .add_assertion("timestamp", v.timestamp_ms);
    // Provenance assertion, added only when present so a pre-provenance entry (empty value)
    // reproduces its original digest and still verifies. Never backfilled onto old rows.
    if !v.writer_version.is_empty() {
        env = env.add_assertion("writer", v.writer_version.clone());
    }
    match &v.payload {
        JournalPayload::Management { detail } => env.add_assertion("detail", detail.clone()),
        // The block/chat bodies are hashed (as hex) so any mutation invalidates the digest; the
        // raw bytes ride the stored CBOR for replay. Distinct assertion names keep a Chat entry's
        // digest disjoint from a Block entry carrying identical bytes.
        JournalPayload::Block { body } => env.add_assertion("block", hex(body)),
        JournalPayload::Chat { body } => env.add_assertion("chat", hex(body)),
    }
}

/// Encode an entry to its canonical CBOR bytes (for storage + replay) and its content hash (the
/// Gordian envelope digest). The bytes are what `daemon-store` persists; the [`ContentHash`] is
/// stored alongside so a verifier detects byte-level tampering before recomputing the segment root.
pub fn encode_entry(view: &JournalEntryView) -> (Vec<u8>, ContentHash) {
    let mut bytes = Vec::new();
    ciborium::into_writer(view, &mut bytes).expect("encode journal entry to CBOR");
    let hash = ContentHash::new(digest_to_32(&entry_envelope(view).digest()));
    (bytes, hash)
}

/// Decode a stored entry's bytes back to the typed view (the replay path for reconnect/scroll-back).
pub fn decode_entry(bytes: &[u8]) -> Result<JournalEntryView, VerifyError> {
    ciborium::from_reader(bytes).map_err(|_| VerifyError::Decode)
}

/// Build the segment envelope: prior root in the subject, one assertion per entry envelope.
fn segment_envelope(
    stream: &JournalStreamId,
    segment: u64,
    prior: MerkleRoot,
    entries: &[Envelope],
) -> Envelope {
    let mut env = Envelope::new(format!("journal:{stream}:{segment}"))
        .add_assertion("prior_root", prior.to_hex());
    for entry in entries {
        env = env.add_assertion("entry", entry.clone());
    }
    env
}

/// The inputs needed to (re)compute and verify a `(stream, segment)` segment: the prior segment's
/// root and the per-entry `(seq, CBOR bytes, stored content hash)` as loaded from the store.
#[derive(Clone, Debug)]
pub struct SegmentInput<'a> {
    /// The stream the segment belongs to.
    pub stream: &'a JournalStreamId,
    /// The segment index this covers.
    pub segment: u64,
    /// The prior segment's committed root (or [`GENESIS_ROOT`] for segment 0).
    pub prior: MerkleRoot,
    /// The segment's entries as loaded: `(seq, CBOR bytes, stored content hash)`.
    pub entries: &'a [(u64, Vec<u8>, ContentHash)],
}

/// Recompute the segment [`MerkleRoot`] from rebuilt entry envelopes + the prior root.
fn recompute_root(input: &SegmentInput<'_>) -> Result<MerkleRoot, VerifyError> {
    let mut envs = Vec::with_capacity(input.entries.len());
    for (_seq, bytes, stored_hash) in input.entries {
        let view = decode_entry(bytes)?;
        let env = entry_envelope(&view);
        // Byte/content tamper: the stored content hash must equal the rebuilt envelope's digest.
        let recomputed = ContentHash::new(digest_to_32(&env.digest()));
        if recomputed != *stored_hash {
            return Err(VerifyError::ContentHashMismatch);
        }
        envs.push(env);
    }
    let seg = segment_envelope(input.stream, input.segment, input.prior, &envs);
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

impl Default for TraceSigner {
    fn default() -> Self {
        Self::generate()
    }
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

    /// Derive a deterministic key from a 32-byte seed, so a node's verifying key is stable across
    /// restarts (an auditor keeps verifying old segments). Seeds are a `daemon-node` config concern.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let private = PrivateKeyBase::from_data(*seed).ed25519_signing_private_key();
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
    /// The verifying key as hex-encoded dCBOR, for publishing to auditors (so they can verify a
    /// node's sealed segments without holding the private seed).
    pub fn to_hex(&self) -> String {
        hex(&self.0.to_cbor_data())
    }

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
    /// A stored entry's bytes did not decode as a journal entry.
    #[error("a journal entry did not decode")]
    Decode,
    /// A stored content hash did not match the entry envelope's digest (entry tampered).
    #[error("journal entry content hash mismatch (entry tampered)")]
    ContentHashMismatch,
    /// The recomputed segment root did not match the committed root (entries/chain tampered).
    #[error("recomputed segment root does not match the committed root")]
    RootMismatch,
    /// The signature over the committed root did not verify.
    #[error("segment root signature verification failed")]
    BadSignature,
}

/// Verify a sealed `(stream, segment)` segment end-to-end:
/// 1. every entry's bytes decode and match their stored content hash;
/// 2. the recomputed Merkle root (folding entries + the prior root) equals `committed_root`;
/// 3. `signature` is a valid ed25519 signature over `committed_root` by `key`.
///
/// Because the prior root is folded into the recomputed root, passing the wrong prior (a broken
/// cross-segment link) surfaces as [`VerifyError::RootMismatch`].
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

    fn mgmt(stream: &str, segment: u64, seq: u64, kind: &str) -> JournalEntryView {
        JournalEntryView {
            stream: JournalStreamId::new(stream),
            segment,
            seq,
            epoch: segment,
            trace: 0xC0FFEE,
            kind: kind.into(),
            timestamp_ms: 1_000 + seq,
            writer_version: String::new(),
            payload: JournalPayload::Management {
                detail: format!("detail-{seq}"),
            },
        }
    }

    fn block(stream: &str, segment: u64, seq: u64, body: &[u8]) -> JournalEntryView {
        JournalEntryView {
            stream: JournalStreamId::new(stream),
            segment,
            seq,
            epoch: segment,
            trace: 1,
            kind: "block.message".into(),
            timestamp_ms: 2_000 + seq,
            writer_version: String::new(),
            payload: JournalPayload::Block {
                body: body.to_vec(),
            },
        }
    }

    fn build_segment(records: &[JournalEntryView]) -> Vec<(u64, Vec<u8>, ContentHash)> {
        records
            .iter()
            .map(|r| {
                let (bytes, hash) = encode_entry(r);
                (r.seq, bytes, hash)
            })
            .collect()
    }

    #[test]
    fn encode_decode_round_trips() {
        let v = block("s", 0, 3, b"hello-block");
        let (bytes, _hash) = encode_entry(&v);
        assert_eq!(decode_entry(&bytes).unwrap(), v);
    }

    #[test]
    fn seal_and_verify_round_trip_interleaving_management_and_blocks() {
        let stream = JournalStreamId::new("verifiable");
        let records = [
            mgmt("verifiable", 0, 0, "mgmt.started"),
            block("verifiable", 0, 1, b"assistant says hi"),
            mgmt("verifiable", 0, 2, "mgmt.finished"),
        ];
        let entries = build_segment(&records);
        let input = SegmentInput {
            stream: &stream,
            segment: 0,
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
        let stream = JournalStreamId::new("det");
        let records = [mgmt("det", 0, 0, "a"), block("det", 0, 1, b"b")];
        let e1 = build_segment(&records);
        let e2 = build_segment(&records);
        let i1 = SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &e1,
        };
        let i2 = SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &e2,
        };
        assert_eq!(segment_root(&i1).unwrap(), segment_root(&i2).unwrap());
    }

    #[test]
    fn tampering_a_block_body_is_detected() {
        let stream = JournalStreamId::new("tamper");
        let records = [
            block("tamper", 0, 0, b"original"),
            mgmt("tamper", 0, 1, "mgmt.finished"),
        ];
        let mut entries = build_segment(&records);
        let input = SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &entries,
        };
        let root = segment_root(&input).unwrap();
        let signer = TraceSigner::generate();
        let sig = signer.sign_root(&root);

        // Re-encode entry 0 with a mutated body but keep the OLD stored content hash.
        let tampered_view = block("tamper", 0, 0, b"forged!!");
        let (tampered_bytes, _) = encode_entry(&tampered_view);
        entries[0].1 = tampered_bytes;
        let tampered = SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &entries,
        };
        let err = verify_segment(&tampered, &root, &sig, &signer.verifying_key()).unwrap_err();
        assert!(matches!(
            err,
            VerifyError::ContentHashMismatch | VerifyError::RootMismatch
        ));
    }

    #[test]
    fn cross_segment_chain_links() {
        let stream = JournalStreamId::new("chain");
        let r0 = [
            mgmt("chain", 0, 0, "mgmt.started"),
            mgmt("chain", 0, 1, "mgmt.finished"),
        ];
        let e0 = build_segment(&r0);
        let i0 = SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &e0,
        };
        let root0 = segment_root(&i0).unwrap();

        let r1 = [
            block("chain", 1, 0, b"turn 2"),
            mgmt("chain", 1, 1, "mgmt.finished"),
        ];
        let e1 = build_segment(&r1);
        let i1 = SegmentInput {
            stream: &stream,
            segment: 1,
            prior: root0,
            entries: &e1,
        };
        let root1 = segment_root(&i1).unwrap();
        let signer = TraceSigner::generate();
        let sig1 = signer.sign_root(&root1);

        verify_segment(&i1, &root1, &sig1, &signer.verifying_key())
            .expect("chained segment verifies");

        let broken = SegmentInput {
            stream: &stream,
            segment: 1,
            prior: GENESIS_ROOT,
            entries: &e1,
        };
        assert_eq!(
            verify_segment(&broken, &root1, &sig1, &signer.verifying_key()).unwrap_err(),
            VerifyError::RootMismatch
        );
    }

    #[test]
    fn deterministic_seed_key_is_stable() {
        let seed = [7u8; 32];
        let a = TraceSigner::from_seed(&seed);
        let b = TraceSigner::from_seed(&seed);
        let root = MerkleRoot::new([1; 32]);
        // Same seed -> same key -> each verifies the other's signature.
        let sig = a.sign_root(&root);
        assert!(b.verifying_key().verify_root(&root, &sig));
    }

    #[test]
    fn wrong_key_fails_signature() {
        let stream = JournalStreamId::new("sig");
        let records = [mgmt("sig", 0, 0, "mgmt.started")];
        let entries = build_segment(&records);
        let input = SegmentInput {
            stream: &stream,
            segment: 0,
            prior: GENESIS_ROOT,
            entries: &entries,
        };
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
