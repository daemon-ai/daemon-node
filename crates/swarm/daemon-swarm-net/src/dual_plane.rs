// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`DualPlane`] — run two (or more) [`ControlPlane`]s at once, deduping across them (spec §7.1; A1).
//!
//! The coordinator WS plane ([`WsControlPlane`](crate::ws_client::WsControlPlane)) and the iroh
//! gossip mesh ([`IrohGossip`](crate::iroh_gossip::IrohGossip)) carry the **same** signed
//! `SignedMessage` frames, so the run survives one plane degrading (spec: "coordinator WS carries
//! the same messages if gossip degrades"). [`DualPlane`] fans a `publish` out on **every** inner
//! plane and merges their subscriptions behind one [`Deduper`] per subscription, so a frame that
//! arrives on both WS and gossip is delivered to each subscriber exactly **once** (NET-6).
//!
//! It is a thin composition over `Arc<dyn ControlPlane>` — it needs neither the `ws` nor the `iroh`
//! feature itself (the caller supplies the concrete planes), so the same type composes any mix:
//! WS + gossip in production, or WS + loopback / two loopbacks in tests.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::dedupe::Deduper;
use crate::transport::{ControlPlane, ControlSubscription};
use crate::SwarmNetError;

/// A control plane composed of several inner planes with cross-plane content-hash dedupe.
pub struct DualPlane {
    planes: Vec<Arc<dyn ControlPlane>>,
}

impl DualPlane {
    /// Compose the given planes (order only affects publish fan-out order, never delivery).
    #[must_use]
    pub fn new(planes: Vec<Arc<dyn ControlPlane>>) -> Self {
        Self { planes }
    }

    /// The common case: a WS plane + a gossip plane (both carry the same frames).
    #[must_use]
    pub fn pair(a: Arc<dyn ControlPlane>, b: Arc<dyn ControlPlane>) -> Self {
        Self::new(vec![a, b])
    }

    /// The number of composed planes (test / observability helper).
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }
}

#[async_trait]
impl ControlPlane for DualPlane {
    async fn publish(&self, message: &[u8]) -> Result<(), SwarmNetError> {
        // Fan out on every plane (a WS + gossip double-send is exactly the redundancy the dual plane
        // is for). Succeed if any plane accepted it; surface the last error only if all failed, so a
        // single degraded plane never fails the publish.
        let mut any_ok = false;
        let mut last_err = None;
        for plane in &self.planes {
            match plane.publish(message).await {
                Ok(()) => any_ok = true,
                Err(e) => last_err = Some(e),
            }
        }
        if any_ok {
            Ok(())
        } else {
            Err(last_err.unwrap_or_else(|| {
                SwarmNetError::Transport("dual plane has no inner planes".into())
            }))
        }
    }

    fn subscribe(&self) -> ControlSubscription {
        let (out_tx, out_rx) = tokio::sync::mpsc::unbounded_channel();
        // One dedupe set per subscription: each distinct frame is delivered once across ALL planes
        // (the same frame arriving on WS and gossip collapses to one delivery — NET-6).
        let dedupe = Arc::new(Mutex::new(Deduper::new()));
        for plane in &self.planes {
            let mut sub = plane.subscribe();
            let out_tx = out_tx.clone();
            let dedupe = dedupe.clone();
            tokio::spawn(async move {
                while let Some(msg) = sub.recv().await {
                    let fresh = dedupe.lock().expect("dual dedupe lock").observe(&msg);
                    if fresh && out_tx.send(msg).is_err() {
                        break; // the subscriber was dropped
                    }
                }
            });
        }
        ControlSubscription::new(out_rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gossip::LoopbackGossip;

    /// Publishing through the dual plane fans out to both inner buses; the merged subscription
    /// dedupes the two self-deliveries to a single delivery.
    #[tokio::test]
    async fn dual_publish_delivers_once_across_planes() {
        let a = Arc::new(LoopbackGossip::new());
        let b = Arc::new(LoopbackGossip::new());
        let dual = DualPlane::pair(a.clone(), b.clone());
        let mut sub = dual.subscribe();

        dual.publish(b"round-open").await.unwrap();

        assert_eq!(sub.recv().await.as_deref(), Some(&b"round-open"[..]));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), sub.recv())
                .await
                .is_err(),
            "the WS + gossip double-arrival must dedupe to one delivery"
        );
    }

    /// The same frame injected independently into BOTH inner buses (the "arrived via WS AND via
    /// gossip" case) still delivers once.
    #[tokio::test]
    async fn same_frame_via_two_planes_dedupes() {
        let a = Arc::new(LoopbackGossip::new());
        let b = Arc::new(LoopbackGossip::new());
        let dual = DualPlane::pair(
            a.clone() as Arc<dyn ControlPlane>,
            b.clone() as Arc<dyn ControlPlane>,
        );
        let mut sub = dual.subscribe();
        // Give the forwarder tasks a tick to register their inner subscriptions.
        tokio::task::yield_now().await;

        a.publish(b"commitment").await.unwrap();
        b.publish(b"commitment").await.unwrap();

        assert_eq!(sub.recv().await.as_deref(), Some(&b"commitment"[..]));
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(200), sub.recv())
                .await
                .is_err(),
            "one delivery for the same frame on two planes"
        );
    }
}
