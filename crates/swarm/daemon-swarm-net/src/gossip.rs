// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`LoopbackGossip`] — an in-process [`ControlPlane`] for local mode + tests.
//!
//! It stands in for the iroh gossip mesh (§7.1) when every peer lives in one process: a `publish`
//! fans a message out to every subscriber, and re-publishing identical bytes is a no-op
//! (content-hash dedupe), which is exactly the NET-6 property "the same message via WS and gossip
//! dedupes". No network, no signing — the bytes are already-signed opaque envelopes.

use std::sync::Mutex;

use async_trait::async_trait;

use daemon_swarm_proto::blake3_hash;

use crate::seam::ContentHash;
use crate::transport::{ControlPlane, ControlSubscription};
use crate::SwarmNetError;

/// An in-process broadcast control plane (fanout + content-hash dedupe).
#[derive(Default)]
pub struct LoopbackGossip {
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    subscribers: Vec<tokio::sync::mpsc::UnboundedSender<Vec<u8>>>,
    /// Content hashes already published, so a WS+gossip double-send fans out once.
    seen: std::collections::HashSet<[u8; 32]>,
}

impl LoopbackGossip {
    /// A fresh loopback gossip mesh with no subscribers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of live subscribers (test/observability helper).
    #[must_use]
    pub fn subscriber_count(&self) -> usize {
        self.inner.lock().expect("gossip lock").subscribers.len()
    }
}

#[async_trait]
impl ControlPlane for LoopbackGossip {
    async fn publish(&self, message: &[u8]) -> Result<(), SwarmNetError> {
        let hash = *blake3::hash(message).as_bytes();
        let mut inner = self.inner.lock().expect("gossip lock");
        // Dedupe: a message already disseminated (e.g. seen on WS then gossip) fans out once.
        if !inner.seen.insert(hash) {
            return Ok(());
        }
        // Fan out to every live subscriber, dropping any whose receiver has been closed.
        inner
            .subscribers
            .retain(|tx| tx.send(message.to_vec()).is_ok());
        Ok(())
    }

    fn subscribe(&self) -> ControlSubscription {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.inner.lock().expect("gossip lock").subscribers.push(tx);
        ControlSubscription::new(rx)
    }
}

/// A convenience: the blake3 message id used for dedupe (proto's [`blake3_hash`]).
#[must_use]
pub fn message_id(message: &[u8]) -> ContentHash {
    blake3_hash(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn fans_out_to_all_subscribers() {
        let gossip = LoopbackGossip::new();
        let mut a = gossip.subscribe();
        let mut b = gossip.subscribe();
        assert_eq!(gossip.subscriber_count(), 2);

        gossip.publish(b"round-open").await.unwrap();

        assert_eq!(a.recv().await.unwrap(), b"round-open");
        assert_eq!(b.recv().await.unwrap(), b"round-open");
    }

    #[tokio::test]
    async fn dedupes_identical_republish() {
        let gossip = LoopbackGossip::new();
        let mut sub = gossip.subscribe();

        // Same message arriving twice (WS then gossip) is disseminated once.
        gossip.publish(b"commitment").await.unwrap();
        gossip.publish(b"commitment").await.unwrap();
        // A distinct message still fans out.
        gossip.publish(b"attestation").await.unwrap();

        assert_eq!(sub.recv().await.unwrap(), b"commitment");
        assert_eq!(sub.recv().await.unwrap(), b"attestation");
        assert!(
            sub.try_recv().is_none(),
            "the duplicate must not be delivered"
        );
    }

    #[tokio::test]
    async fn late_subscriber_misses_prior_but_gets_future() {
        let gossip = LoopbackGossip::new();
        gossip.publish(b"early").await.unwrap();
        let mut late = gossip.subscribe();
        gossip.publish(b"late").await.unwrap();
        assert_eq!(late.recv().await.unwrap(), b"late");
        assert!(late.try_recv().is_none());
    }
}
