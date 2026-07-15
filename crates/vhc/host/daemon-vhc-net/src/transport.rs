// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The `SwarmTransport` seam: one control plane + one payload-plane trait (spec §7.1).
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
use crate::SwarmNetError;

/// The control plane: publish/subscribe of already-signed, opaque control-message bytes (§7.1).
///
/// A message is an opaque `&[u8]` — a signed CBOR envelope produced by lane P. Implementations
/// disseminate it to every subscriber and de-duplicate re-deliveries (the same message arriving via
/// both WS and gossip is delivered once — NET-6).
#[async_trait]
pub trait ControlPlane: Send + Sync {
    /// Publish one already-signed control message to all peers. Re-publishing identical bytes is a
    /// no-op (content-hash dedupe), so a WS+gossip double-send fans out once.
    async fn publish(&self, message: &[u8]) -> Result<(), SwarmNetError>;

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
/// This is what a [`ReceiptProducer`](crate::receipt::ReceiptProducer) folds into a signed
/// `StorageReceipt`: the object's content hash + size, verified against the store (§6.4 I6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadStat {
    /// The object's content hash (blake3).
    pub hash: ContentHash,
    /// The object's size in bytes.
    pub size: u64,
}

/// A payload plane: opaque payload objects keyed by `(run, round, peer)` + content hash (§7.1).
///
/// PUT your sealed update object; GET a committed object (verified against the hash the commitment
/// carried); HEAD (`stat`) to attest availability without transferring bytes. A payload is opaque —
/// the transport moves, hashes, and (on GET) verifies it, but never parses it (§7.3).
#[async_trait]
pub trait PayloadStore: Send + Sync {
    /// PUT an opaque payload object, returning its content hash (blake3).
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<ContentHash, SwarmNetError>;

    /// GET a payload object, verifying its content hash equals `expected`. A hash mismatch is a
    /// typed [`SwarmNetError::HashMismatch`]; a missing/expired object is [`SwarmNetError::PayloadMiss`].
    async fn get(&self, key: &PayloadKey, expected: &ContentHash)
        -> Result<Vec<u8>, SwarmNetError>;

    /// HEAD-equivalent availability check (`stat`): the object's size + content hash, without
    /// transferring the bytes to the caller. A missing/expired object is
    /// [`SwarmNetError::PayloadMiss`].
    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, SwarmNetError>;
}
