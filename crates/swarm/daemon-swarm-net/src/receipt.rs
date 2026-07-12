// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`ReceiptProducer`] — turns payload-store availability into signed `StorageReceipt` evidence.
//!
//! The commit rule (§6.4 I6) consumes only **signed messages**: a payload enters the round record
//! iff its `Commitment` arrived *and* signed availability evidence exists — either a
//! `StorageReceipt` (the coordinator-as-storage-client `HEAD`s the object and *emits the result as
//! a signed message*) or a witness-quorum `Attestation`. This keeps the coordinator's `tick` a pure
//! function of its inputs (no inline I/O in the rule).
//!
//! Lane R models the *receipt-producer* half: a small component that polls/checks a
//! [`PayloadStore`] via `head` (`stat`) and, when the object is available, produces a signed
//! [`StorageReceipt`]. Signing is injected through the [`ReceiptSigner`] seam — Merge 1 replaces the
//! placeholder receipt + signer with the proto crate's CDDL `StorageReceipt` message and the real
//! ed25519 node-identity signature.

use serde::{Deserialize, Serialize};

use crate::seam::{ContentHash, PayloadKey, PeerId, RoundId, RunId};
use crate::transport::PayloadStore;
use crate::SwarmNetError;

/// The availability-evidence message body: `(run, round, peer, hash, size)` the storage client has
/// `HEAD`-verified against the payload store (spec §6.4 `StorageReceipt`).
///
// MERGE-1: replace with daemon_swarm_proto::StorageReceipt (the CDDL message).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageReceipt {
    /// The run this evidence is for.
    pub run: RunId,
    /// The round this evidence is for.
    pub round: RoundId,
    /// The peer whose payload availability is attested.
    pub peer: PeerId,
    /// The content hash the store holds for `(run, round, peer)`.
    pub hash: ContentHash,
    /// The object's size in bytes.
    pub size: u64,
}

/// A [`StorageReceipt`] plus the signed bytes a consumer of the commit rule verifies.
///
/// `bytes` is the signed message wire form; this wave's [`UnsignedSigner`] simply CBOR-encodes the
/// receipt (no signature), which is a mechanically-swappable placeholder for lane P's ed25519
/// envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignedReceipt {
    /// The decoded receipt body.
    pub receipt: StorageReceipt,
    /// The signed wire bytes (CBOR envelope) — what the commit rule consumes as a signed message.
    pub bytes: Vec<u8>,
}

/// Signs a [`StorageReceipt`] into its wire bytes.
///
// MERGE-1: replace the implementation with lane P's ed25519 node-identity signing over the
// canonical-CBOR receipt (the `StorageReceipt` control message, §7.3).
pub trait ReceiptSigner: Send + Sync {
    /// Produce the signed wire bytes for `receipt`.
    fn sign(&self, receipt: &StorageReceipt) -> Result<Vec<u8>, SwarmNetError>;
}

/// A placeholder signer that CBOR-encodes the receipt without a signature (local mode / tests).
#[derive(Clone, Copy, Debug, Default)]
pub struct UnsignedSigner;

impl ReceiptSigner for UnsignedSigner {
    fn sign(&self, receipt: &StorageReceipt) -> Result<Vec<u8>, SwarmNetError> {
        let mut buf = Vec::new();
        ciborium::into_writer(receipt, &mut buf)
            .map_err(|e| SwarmNetError::Transport(format!("encode receipt: {e}")))?;
        Ok(buf)
    }
}

/// Polls a [`PayloadStore`] and emits signed availability evidence for committed payloads.
pub struct ReceiptProducer<S, Sig> {
    store: S,
    signer: Sig,
}

impl<S: PayloadStore, Sig: ReceiptSigner> ReceiptProducer<S, Sig> {
    /// Build a producer over `store`, signing receipts with `signer`.
    pub fn new(store: S, signer: Sig) -> Self {
        Self { store, signer }
    }

    /// Check `key` in the store and, if the object is available, produce a signed
    /// [`StorageReceipt`]. A missing/expired object surfaces as [`SwarmNetError::PayloadMiss`] (no
    /// evidence is emitted — exactly the §6.4 recovery-ladder rung 1 "no availability yet" case).
    pub async fn produce(&self, key: &PayloadKey) -> Result<SignedReceipt, SwarmNetError> {
        let stat = self.store.head(key).await?;
        let receipt = StorageReceipt {
            run: key.run.clone(),
            round: key.round,
            peer: key.peer,
            hash: stat.hash,
            size: stat.size,
        };
        let bytes = self.signer.sign(&receipt)?;
        Ok(SignedReceipt { receipt, bytes })
    }

    /// Produce receipts for every available key in `keys`, skipping (not failing on) the ones the
    /// store cannot yet attest — the "poll a batch of committed payloads" shape.
    pub async fn produce_available(&self, keys: &[PayloadKey]) -> Vec<SignedReceipt> {
        let mut out = Vec::new();
        for key in keys {
            if let Ok(receipt) = self.produce(key).await {
                out.push(receipt);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::FsPayloadStore;
    use crate::test_support::temp_root;

    fn key(round: RoundId, peer: u8) -> PayloadKey {
        PayloadKey::new(RunId::new("run-r"), round, PeerId([peer; 32]))
    }

    #[tokio::test]
    async fn produces_signed_receipt_for_available_object() {
        let dir = temp_root("receipt-ok");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let k = key(2, 0x01);
        let hash = store.put(&k, b"peer-update").await.unwrap();

        let producer = ReceiptProducer::new(store, UnsignedSigner);
        let signed = producer.produce(&k).await.unwrap();

        assert_eq!(signed.receipt.hash, hash);
        assert_eq!(signed.receipt.size, b"peer-update".len() as u64);
        assert_eq!(signed.receipt.round, 2);
        // The signed bytes decode back to the same receipt (placeholder = CBOR round-trip).
        let decoded: StorageReceipt = ciborium::from_reader(&signed.bytes[..]).unwrap();
        assert_eq!(decoded, signed.receipt);
    }

    #[tokio::test]
    async fn no_evidence_for_absent_object() {
        let dir = temp_root("receipt-absent");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let producer = ReceiptProducer::new(store, UnsignedSigner);
        let err = producer.produce(&key(9, 0x02)).await.unwrap_err();
        assert!(matches!(err, SwarmNetError::PayloadMiss(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn produce_available_skips_missing() {
        let dir = temp_root("receipt-batch");
        let store = FsPayloadStore::open(dir.path(), 8).unwrap();
        let present = key(1, 0x03);
        store.put(&present, b"here").await.unwrap();
        let absent = key(1, 0x04);

        let producer = ReceiptProducer::new(store, UnsignedSigner);
        let receipts = producer.produce_available(&[present.clone(), absent]).await;
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0].receipt.peer, present.peer);
    }
}
