// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `VhcTransport` seam: one control plane + one payload-plane trait (spec §7.1).
//!
//! The control plane is not tiered — every peer publishes/subscribes **already-signed** opaque
//! control-message bytes (the seven §6.4 round messages plus join/heartbeat). Signing and
//! verification are lane P's envelope surface; the transport only disseminates and de-duplicates
//! (gossip is dissemination, never arbitration — §7.1).
//!
//! Bulk **payloads** move on whichever plane the envelope's `payload_store` names. Both the `r2`
//! store and `iroh-blobs` implement one [`PayloadStore`]: PUT your update object, GET committed
//! objects (hash-verified), HEAD for availability. This wave ships the filesystem implementation
//! ([`FsPayloadStore`](crate::store::FsPayloadStore)); the network planes slot in behind the same
//! trait later.

use async_trait::async_trait;

use crate::seam::{ContentHash, PayloadKey};
use crate::VhcNetError;

/// The control plane: publish/subscribe of already-signed, opaque control-message bytes (§7.1).
///
/// A message is an opaque `&[u8]` — a signed CBOR envelope produced by lane P. Implementations
/// disseminate it to every subscriber and de-duplicate re-deliveries (the same message arriving via
/// both WS and gossip is delivered once — NET-6).
#[async_trait]
pub trait ControlPlane: Send + Sync {
    /// Publish one already-signed control message to all peers. Re-publishing identical bytes is a
    /// no-op (content-hash dedupe), so a WS+gossip double-send fans out once.
    async fn publish(&self, message: &[u8]) -> Result<(), VhcNetError>;

    /// Open a subscription to inbound control messages. Each distinct message is delivered at most
    /// once per subscriber.
    fn subscribe(&self) -> ControlSubscription;
}

/// A control-plane subscription: an inbox of inbound control-message bytes.
///
/// Thin wrapper over an mpsc receiver so the concrete channel type is not part of the frozen seam
/// (Merge 1 / later waves can change the carrier without touching consumers).
pub struct ControlSubscription {
    rx: tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>,
}

impl ControlSubscription {
    /// Wrap a receiver as a subscription.
    pub(crate) fn new(rx: tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>) -> Self {
        Self { rx }
    }

    /// Await the next inbound message, or `None` once the plane is dropped.
    pub async fn recv(&mut self) -> Option<Vec<u8>> {
        self.rx.recv().await
    }

    /// Try to take a ready message without awaiting.
    pub fn try_recv(&mut self) -> Option<Vec<u8>> {
        self.rx.try_recv().ok()
    }
}

/// Availability metadata for one payload object — the HEAD/`stat()` result (§7.1).
///
/// This is what a coordinator-seat receipt producer folds into a signed
/// `StorageReceipt`: the object's content hash + size, verified against the store (§6.4 I6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadStat {
    /// The object's content hash (blake3).
    pub hash: ContentHash,
    /// The object's size in bytes.
    pub size: u64,
}

/// A **content-addressed** payload plane: opaque objects keyed by their blake3 alone
/// (`payload/<blake3>`), no run/round/peer coordinates anywhere in the key.
///
/// This is the plane the role session services module `payload_put`/`payload_get` and
/// `data.fetch` operations against: the module names CONTENT, the store moves bytes, and the run
/// pump re-verifies every fetched object against the requested hash before delivery — so the
/// store is untrusted by construction. The coordinate-keyed [`PayloadStore`] below predates this
/// seam and remains for the engine-era consumers; new production bindings are content-addressed.
#[async_trait]
pub trait ContentStore: Send + Sync {
    /// PUT an opaque object, returning its content hash (blake3). Idempotent: re-putting
    /// identical bytes stores/returns the same address.
    async fn put_content(&self, bytes: &[u8]) -> Result<ContentHash, VhcNetError>;

    /// GET an object by content hash. A missing object is [`VhcNetError::PayloadMiss`]; a store
    /// returning bytes that do not hash to `hash` is [`VhcNetError::HashMismatch`] (the caller's
    /// pump re-verifies regardless — defense in depth, not trust).
    async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError>;
}

/// An in-memory [`ContentStore`]: the in-process seat for single-host smoke runs and tests.
#[derive(Default)]
pub struct MemoryContentStore {
    objects: std::sync::Mutex<std::collections::HashMap<ContentHash, Vec<u8>>>,
}

impl MemoryContentStore {
    /// A fresh, empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Seed an object directly (test/genesis staging helper). Returns its content hash.
    pub fn seed(&self, bytes: &[u8]) -> ContentHash {
        let hash = daemon_vhc_proto::blake3_hash(bytes);
        self.seed_under(hash, bytes);
        hash
    }

    /// Seed an object under an EXPLICIT key — the chunk-addressed corpus seat: a shard's
    /// artifact identity is its chunk FOLD (`daemon_vhc_proto::shard_fold`), not the plain
    /// blake3 of its bytes, so staging one requires naming the key. The consumer pump verifies
    /// the covering chunks on every fetch regardless (the store stays untrusted); `get_content`
    /// serves whatever was seeded here verbatim.
    pub fn seed_under(&self, hash: ContentHash, bytes: &[u8]) {
        self.objects
            .lock()
            .expect("content store lock")
            .insert(hash, bytes.to_vec());
    }
}

#[async_trait]
impl ContentStore for MemoryContentStore {
    async fn put_content(&self, bytes: &[u8]) -> Result<ContentHash, VhcNetError> {
        Ok(self.seed(bytes))
    }

    async fn get_content(&self, hash: &ContentHash) -> Result<Vec<u8>, VhcNetError> {
        self.objects
            .lock()
            .expect("content store lock")
            .get(hash)
            .cloned()
            .ok_or_else(|| VhcNetError::PayloadMiss(hash.to_hex()))
    }
}

/// A payload plane: opaque payload objects keyed by `(run, round, peer)` + content hash (§7.1).
///
/// PUT your sealed update object; GET a committed object (verified against the hash the commitment
/// carried); HEAD (`stat`) to attest availability without transferring bytes. A payload is opaque —
/// the transport moves, hashes, and (on GET) verifies it, but never parses it (§7.3).
#[async_trait]
pub trait PayloadStore: Send + Sync {
    /// PUT an opaque payload object, returning its content hash (blake3).
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<ContentHash, VhcNetError>;

    /// GET a payload object, verifying its content hash equals `expected`. A hash mismatch is a
    /// typed [`VhcNetError::HashMismatch`]; a missing/expired object is [`VhcNetError::PayloadMiss`].
    async fn get(&self, key: &PayloadKey, expected: &ContentHash) -> Result<Vec<u8>, VhcNetError>;

    /// HEAD-equivalent availability check (`stat`): the object's size + content hash, without
    /// transferring the bytes to the caller. A missing/expired object is
    /// [`VhcNetError::PayloadMiss`].
    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, VhcNetError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn content_store_round_trips_by_hash_and_misses_typed() {
        let store = MemoryContentStore::new();
        let hash = store.put_content(b"sealed-object").await.unwrap();
        assert_eq!(hash, daemon_vhc_proto::blake3_hash(b"sealed-object"));
        assert_eq!(store.get_content(&hash).await.unwrap(), b"sealed-object");
        // Idempotent re-put returns the same address.
        assert_eq!(store.put_content(b"sealed-object").await.unwrap(), hash);
        // A miss is typed, never empty bytes.
        let absent = ContentHash([0xEE; 32]);
        assert!(matches!(
            store.get_content(&absent).await,
            Err(VhcNetError::PayloadMiss(_))
        ));
    }
}
