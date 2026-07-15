// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`ReceiptProducer`] — turns payload-store availability into a signed proto `StorageReceipt`.
//!
//! The commit rule (§6.4 I6) consumes only **signed messages**: a payload enters the round record
//! iff its `Commitment` arrived *and* signed availability evidence exists — either a
//! `StorageReceipt` (the coordinator-as-storage-client `HEAD`s the object and *emits the result as
//! a signed message*) or a witness-quorum `Attestation`. This keeps the coordinator's `tick` a pure
//! function of its inputs (no inline I/O in the rule).
//!
//! Lane R models the *receipt-producer* half: a small component that polls/checks a
//! [`PayloadStore`] via `head` (`stat`) and, when the object is available, produces a signed
//! [`StorageReceipt`](daemon_swarm_proto::messages::StorageReceipt). Merge 1 re-expressed this over
//! proto: the message shape is proto's CDDL `StorageReceipt` (a round + its verified
//! `(peer, hash, size)` [`RecordEntry`] set), and the signature is the real ed25519 node-identity
//! [`SignedMessage`] frame produced by [`SignedMessage::sign`] over canonical CBOR (§7.3). Net
//! never *defines* the message or signature type — it only assembles store evidence into proto's.

use daemon_swarm_proto::messages::{RecordEntry, StorageReceipt};
use daemon_swarm_proto::{SignedMessage, SigningKey, SwarmMessage, SwarmProtoVersion};

use crate::seam::{PayloadKey, RoundId};
use crate::transport::PayloadStore;
use crate::SwarmNetError;

/// Polls a [`PayloadStore`] and emits signed availability evidence for committed payloads as proto
/// [`SignedMessage`] frames carrying a [`StorageReceipt`].
///
/// The producer holds the node's ed25519 [`SigningKey`] and the run's pinned [`SwarmProtoVersion`]
/// so every emitted receipt is a fully-signed control message the commit rule can consume directly.
pub struct ReceiptProducer<S> {
    store: S,
    key: SigningKey,
    version: SwarmProtoVersion,
}

impl<S: PayloadStore> ReceiptProducer<S> {
    /// Build a producer over `store`, signing receipts with the node identity `key` at the run's
    /// pinned proto `version`.
    pub fn new(store: S, key: SigningKey, version: SwarmProtoVersion) -> Self {
        Self {
            store,
            key,
            version,
        }
    }

    /// Check `key` in the store and, if the object is available, produce a signed single-entry
    /// [`StorageReceipt`] message for its round. A missing/expired object surfaces as
    /// [`SwarmNetError::PayloadMiss`] (no evidence is emitted — the §6.4 recovery-ladder rung 1 "no
    /// availability yet" case).
    pub async fn produce(&self, key: &PayloadKey) -> Result<SignedMessage, SwarmNetError> {
        let stat = self.store.head(key).await?;
        let receipt = StorageReceipt {
            round: key.round,
            verified: vec![RecordEntry {
                peer: key.peer,
                hash: stat.hash,
                size: stat.size,
            }],
        };
        self.sign(receipt)
    }

    /// Head-verify every key of `round` and aggregate the available objects into **one** signed
    /// [`StorageReceipt`] (proto's batch shape: one round, many verified `(peer, hash, size)`
    /// entries). Keys the store cannot yet attest are skipped (not failed on) — the "poll a batch of
    /// committed payloads" shape. Returns `Ok(None)` when nothing in `keys` is available yet.
    pub async fn produce_round(
        &self,
        round: RoundId,
        keys: &[PayloadKey],
    ) -> Result<Option<SignedMessage>, SwarmNetError> {
        let mut verified = Vec::new();
        for key in keys {
            if let Ok(stat) = self.store.head(key).await {
                verified.push(RecordEntry {
                    peer: key.peer,
                    hash: stat.hash,
                    size: stat.size,
                });
            }
        }
        if verified.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.sign(StorageReceipt { round, verified })?))
    }

    /// Sign a [`StorageReceipt`] into the proto [`SignedMessage`] control frame (ed25519 over
    /// canonical CBOR of `(version, payload)`).
    fn sign(&self, receipt: StorageReceipt) -> Result<SignedMessage, SwarmNetError> {
        SignedMessage::sign(
            &self.key,
            self.version,
            SwarmMessage::StorageReceipt(receipt),
        )
        .map_err(|e| SwarmNetError::Transport(format!("sign storage receipt: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seam::{PeerId, RunId};
    use crate::store::FsPayloadStore;
    use crate::test_support::temp_root;
    use daemon_swarm_proto::SWARM_PROTO_VERSION;

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[0x42; 32])
    }

    fn key(round: RoundId, peer: u8) -> PayloadKey {
        PayloadKey::new(RunId::new("run-r"), round, PeerId([peer; 32]))
    }

    /// Extract the `StorageReceipt` payload from a signed control message (panics on the wrong
    /// variant — a test helper).
    fn receipt_of(msg: &SignedMessage) -> &StorageReceipt {
        match &msg.payload {
            SwarmMessage::StorageReceipt(r) => r,
            other => panic!("expected StorageReceipt, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn produces_signed_receipt_for_available_object() {
        let dir = temp_root("receipt-ok");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = key(2, 0x01);
        let hash = store.put(&k, b"peer-update").await.unwrap();

        let producer = ReceiptProducer::new(store, signing_key(), SWARM_PROTO_VERSION);
        let signed = producer.produce(&k).await.unwrap();

        // The frame is a real ed25519-signed control message at the pinned version.
        assert!(signed.verify().is_ok());
        assert_eq!(signed.version, SWARM_PROTO_VERSION);

        let receipt = receipt_of(&signed);
        assert_eq!(receipt.round, 2);
        assert_eq!(receipt.verified.len(), 1);
        assert_eq!(receipt.verified[0].peer, k.peer);
        assert_eq!(receipt.verified[0].hash, hash);
        assert_eq!(receipt.verified[0].size, b"peer-update".len() as u64);
    }

    #[tokio::test]
    async fn no_evidence_for_absent_object() {
        let dir = temp_root("receipt-absent");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let producer = ReceiptProducer::new(store, signing_key(), SWARM_PROTO_VERSION);
        let err = producer.produce(&key(9, 0x02)).await.unwrap_err();
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn produce_round_aggregates_available_and_skips_missing() {
        let dir = temp_root("receipt-batch");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let present_a = key(1, 0x03);
        let present_b = key(1, 0x05);
        store.put(&present_a, b"here").await.unwrap();
        store.put(&present_b, b"there").await.unwrap();
        let absent = key(1, 0x04);

        let producer = ReceiptProducer::new(store, signing_key(), SWARM_PROTO_VERSION);
        let signed = producer
            .produce_round(1, &[present_a.clone(), absent, present_b.clone()])
            .await
            .unwrap()
            .expect("at least one object is available");

        assert!(signed.verify().is_ok());
        let receipt = receipt_of(&signed);
        assert_eq!(receipt.round, 1);
        // The missing object is skipped; only the two available ones are attested.
        assert_eq!(receipt.verified.len(), 2);
        let peers: Vec<PeerId> = receipt.verified.iter().map(|e| e.peer).collect();
        assert!(peers.contains(&present_a.peer));
        assert!(peers.contains(&present_b.peer));
    }

    #[tokio::test]
    async fn produce_round_emits_nothing_when_all_absent() {
        let dir = temp_root("receipt-empty");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let producer = ReceiptProducer::new(store, signing_key(), SWARM_PROTO_VERSION);
        let out = producer.produce_round(3, &[key(3, 0x06)]).await.unwrap();
        assert!(out.is_none());
    }
}
