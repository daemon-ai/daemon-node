// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The swarm control-plane messages (spec §6.4, §7.3; TDD PROTO-19).
//!
//! The seven round messages — `RoundOpen`, `Commitment`, `Attestation`, `StorageReceipt`,
//! `RoundRecord`, `Digest`, `Straggle` — plus the `Join`/`Heartbeat` envelope messages. Every one
//! travels as **signed CBOR**: the [`SignedMessage`] frame carries the [`SwarmProtoVersion`], the
//! externally-tagged [`SwarmMessage`] payload, the signer's [`PeerId`], and an ed25519
//! [`Signature`] over the canonical CBOR of `(version, payload)`.
//!
//! Attestations and records carry **commitments to sets** ([`SetCommitment`]), not the sets
//! themselves, so the consensus messages are scale-invariant (constant-size at any roster, spec
//! §6.4). The full set may ride alongside as an `inline` list while rosters are small — a transport
//! optimization, never the signed field.

use serde::{Deserialize, Serialize};

use crate::bytes::{Hash, IrohId, PeerId, Seed, Signature, StateDigest};
use crate::capability::CapabilitySet;
use crate::error::SwarmProtoError;
use crate::merkle::SetCommitment;
use crate::sign::{peer_id, sign_canonical, verify_canonical, SigningKey};
use crate::version::SwarmProtoVersion;

/// A measured throughput class (§6.3). Boundaries are `daemon-swarm-proto` constants, versioned
/// with [`SwarmProtoVersion`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThroughputClass {
    /// < 1k tok/s.
    C1,
    /// 1–4k tok/s.
    C2,
    /// 4–16k tok/s.
    C3,
    /// > 16k tok/s.
    C4,
}

impl ThroughputClass {
    /// Classify a measured aggregate tokens/s into its class (§6.3 ladder boundaries).
    #[must_use]
    pub fn classify(tokens_per_s: u64) -> Self {
        match tokens_per_s {
            0..=999 => Self::C1,
            1_000..=3_999 => Self::C2,
            4_000..=15_999 => Self::C3,
            _ => Self::C4,
        }
    }
}

/// Where a committed payload can be fetched (a store key and/or a blob ticket, spec §6.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Locator {
    /// A key in the presigned `r2` payload store.
    StoreKey(String),
    /// An iroh-blobs content ticket.
    BlobTicket(String),
}

/// A contiguous `BatchId` interval over the epoch's data window (spec §6.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchWindow {
    /// First `BatchId` (inclusive).
    pub start: u64,
    /// Last `BatchId` (exclusive).
    pub end: u64,
}

/// A `(peer, payload-hash)` element of a witness's fetch-verified set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestEntry {
    /// The contributing peer's node identity.
    pub peer: PeerId,
    /// blake3 of its payload.
    pub hash: Hash,
}

/// A `(peer, hash, size)` element of a round record / storage receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordEntry {
    /// The contributing peer's node identity.
    pub peer: PeerId,
    /// blake3 of its payload.
    pub hash: Hash,
    /// Payload size in bytes.
    pub size: u64,
}

/// `RoundOpen` — coordinator opens a round (§6.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundOpen {
    /// Round number.
    pub round: u64,
    /// Round seed (drives assignment + digest schedule).
    pub seed: Seed,
    /// blake3 digest of the frozen roster.
    pub roster_digest: Hash,
    /// The round's global batch window.
    pub batch: BatchWindow,
    /// Deadline (unix seconds) for commitments.
    pub deadline_unix_s: u64,
}

/// `Commitment` — a trainer commits its sealed update (§6.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    /// Round number.
    pub round: u64,
    /// blake3 of the payload.
    pub payload: Hash,
    /// Payload size in bytes (checked against `update_mb_max` receive-side, §7.3).
    pub size: u64,
    /// Where the payload can be fetched (one per plane it is on).
    pub locators: Vec<Locator>,
}

/// `Attestation` — a witness commits to its cumulative fetch-verified set (§6.4). The signed field
/// is the [`SetCommitment`]; `inline` is a transport optimization only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Attestation {
    /// Round number.
    pub round: u64,
    /// Set commitment (root + count) over the sorted verified `(peer, hash)` pairs.
    pub set: SetCommitment,
    /// Optional inline set (small rosters only); never the signed/consensus field.
    pub inline: Option<Vec<AttestEntry>>,
}

/// `StorageReceipt` — the coordinator-as-storage-client reports `HEAD`-verified objects as a signed
/// message, so the commit rule stays a pure function of its inputs (§6.4 I6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageReceipt {
    /// Round number.
    pub round: u64,
    /// The `(peer, hash, size)` tuples the coordinator has verified against the payload store.
    pub verified: Vec<RecordEntry>,
}

/// `RoundRecord` — the consensus artifact (§6.4). Signs the committed set's root + count; carries
/// drops, the next seed, and the locator of the full `record-set.cbor` object (inline set optional).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoundRecord {
    /// Round number.
    pub round: u64,
    /// Set commitment (root + count) over the committed set, ordered by node public-key bytes.
    pub set: SetCommitment,
    /// Peers dropped this round.
    pub drops: Vec<PeerId>,
    /// The next round's seed.
    pub next_seed: Seed,
    /// Locator of the full set object (`record-set.cbor`).
    pub set_locator: Locator,
    /// Optional inline set (small rosters only); never the signed/consensus field.
    pub inline: Option<Vec<RecordEntry>>,
}

/// `Digest` — a peer's post-ingest round state digest (§5.6, §6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Digest {
    /// Round number.
    pub round: u64,
    /// xxh3-128 digest over the seed-keyed sampled state blocks.
    pub digest: StateDigest,
}

/// The recovery status a stalled peer reports (§6.4 recovery ladder).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StraggleStatus {
    /// Still fetching a committed payload it missed.
    Fetching,
    /// Skipping training while it catches up.
    Stalled,
    /// Late-ingesting and rejoining.
    CatchingUp,
}

/// `Straggle` — a stalled peer's status, riding the heartbeat (§6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Straggle {
    /// The round being recovered.
    pub round: u64,
    /// Recovery status.
    pub status: StraggleStatus,
}

/// `Join` — a peer requests roster entry, binding its iroh id to its node identity (§6.5, §7.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Join {
    /// The run being joined.
    pub run_id: String,
    /// The peer's iroh `NodeId`.
    pub iroh_id: IrohId,
    /// The peer's declared throughput class.
    pub class: ThroughputClass,
    /// The peer's advertised capability set (pre-screened against the envelope, §6.5).
    pub capabilities: CapabilitySet,
    /// The frozen-envelope hash the peer asserts it is joining under (§6.1/§6.5; TDD PROTO-12).
    /// `Some(h)` lets the coordinator reject a peer that assessed a *different* envelope
    /// (`AdmissionReject::EnvelopeHashMismatch`); `None` skips the check. Additive (Wave 3): omitted
    /// on the wire for legacy joins (`#[serde(default)]`), so the pre-carrier back-compat path is
    /// unchanged for senders that never set it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope_hash: Option<Hash>,
}

/// `Heartbeat` — a peer's liveness ping (WS, ~15 s; §6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Heartbeat {
    /// The peer's current round.
    pub round: u64,
    /// Optional model-readiness signal during `Warmup` (§6.2/§6.5): `Some(true)` means the peer has
    /// built + is ready to train, letting the coordinator exit `Warmup` early once every admitted
    /// member is ready. Additive (Wave 3): omitted on the wire for legacy heartbeats, so the
    /// timeout-only warmup path is unchanged for senders that never set it (back-compat).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready: Option<bool>,
}

/// The externally-tagged union of every control-plane message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmMessage {
    /// Coordinator opens a round.
    RoundOpen(RoundOpen),
    /// Trainer commits an update.
    Commitment(Commitment),
    /// Witness attests a verified set.
    Attestation(Attestation),
    /// Coordinator reports store-verified objects.
    StorageReceipt(StorageReceipt),
    /// Coordinator publishes the round record.
    RoundRecord(RoundRecord),
    /// Peer publishes its state digest.
    Digest(Digest),
    /// Stalled peer reports status.
    Straggle(Straggle),
    /// Peer requests roster entry.
    Join(Join),
    /// Peer liveness ping.
    Heartbeat(Heartbeat),
}

/// The signed preimage: the exact bytes an ed25519 signature covers.
#[derive(Serialize)]
struct Preimage<'a> {
    version: SwarmProtoVersion,
    payload: &'a SwarmMessage,
}

/// A signed control-plane message frame — everything on the wire is one of these (spec §7.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedMessage {
    /// The swarm proto version (exact-match join gate, §16).
    pub version: SwarmProtoVersion,
    /// The message payload.
    pub payload: SwarmMessage,
    /// The signing node's identity.
    pub signer: PeerId,
    /// ed25519 signature over the canonical CBOR of `(version, payload)`.
    pub sig: Signature,
}

impl SignedMessage {
    /// Sign `payload` at `version` with `key`.
    pub fn sign(
        key: &SigningKey,
        version: SwarmProtoVersion,
        payload: SwarmMessage,
    ) -> Result<Self, SwarmProtoError> {
        let sig = sign_canonical(
            key,
            &Preimage {
                version,
                payload: &payload,
            },
        )?;
        Ok(Self {
            version,
            payload,
            signer: peer_id(key),
            sig,
        })
    }

    /// Verify the signature over `(version, payload)` against the embedded signer.
    pub fn verify(&self) -> Result<(), SwarmProtoError> {
        verify_canonical(
            &self.signer,
            &self.sig,
            &Preimage {
                version: self.version,
                payload: &self.payload,
            },
        )
    }

    /// Verify the signature **and** that the version exactly matches the run's pinned `expected`
    /// (§16 join gate — the message is rejected on either failure).
    pub fn verify_for_run(&self, expected: SwarmProtoVersion) -> Result<(), SwarmProtoError> {
        expected.check_join(self.version)?;
        self.verify()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn throughput_class_ladder_boundaries() {
        assert_eq!(ThroughputClass::classify(0), ThroughputClass::C1);
        assert_eq!(ThroughputClass::classify(999), ThroughputClass::C1);
        assert_eq!(ThroughputClass::classify(1_000), ThroughputClass::C2);
        assert_eq!(ThroughputClass::classify(3_999), ThroughputClass::C2);
        assert_eq!(ThroughputClass::classify(4_000), ThroughputClass::C3);
        assert_eq!(ThroughputClass::classify(15_999), ThroughputClass::C3);
        assert_eq!(ThroughputClass::classify(16_000), ThroughputClass::C4);
    }
}
