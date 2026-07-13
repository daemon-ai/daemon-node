// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared harness for the `ControlPlane` conformance + iroh gossip integration suites (B2).
//!
//! The parametric conformance suite ([`conformance_fanout`] / [`conformance_dedupe`]) runs over any
//! [`Mesh`] — a set of N connected control planes each with one subscriber — so the *same* behavior
//! tests exercise both [`LoopbackGossip`](daemon_swarm_net::LoopbackGossip) (a shared in-process bus)
//! and [`IrohGossip`](daemon_swarm_net::IrohGossip) (N real iroh endpoints on loopback). The signed-
//! message helpers build canonical-CBOR `daemon_swarm_proto::SignedMessage` bytes — the opaque
//! already-signed payloads the plane carries (NET-6).

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use daemon_swarm_net::{ControlPlane, ControlSubscription};
use daemon_swarm_proto::messages::Heartbeat;
use daemon_swarm_proto::{
    from_canonical_slice, to_canonical_vec, SignedMessage, SigningKey, SwarmMessage,
    SWARM_PROTO_VERSION,
};

/// How long to wait for a message to be delivered (iroh mesh formation + flood).
pub const DELIVER: Duration = Duration::from_secs(10);
/// A short grace window to assert a duplicate is NOT delivered.
pub const GRACE: Duration = Duration::from_millis(400);

/// A mesh of N connected control planes, each with one subscriber inbox.
///
/// - Loopback: all `planes` are clones of one `Arc<LoopbackGossip>` (a shared bus); `subs` are N
///   subscriptions on it.
/// - Iroh: `planes` are N distinct `Arc<IrohGossip>` wired into a mesh; `subs` are one per node.
///
/// The shared observable property both satisfy: publishing from any plane delivers to every
/// subscriber exactly once.
pub struct Mesh {
    pub planes: Vec<Arc<dyn ControlPlane>>,
    pub subs: Vec<ControlSubscription>,
}

/// Await the next message with a timeout (`None` on timeout or close).
pub async fn recv_timeout(sub: &mut ControlSubscription, dur: Duration) -> Option<Vec<u8>> {
    tokio::time::timeout(dur, sub.recv()).await.ok().flatten()
}

/// Conformance: a message published from plane 0 reaches every subscriber exactly once.
pub async fn conformance_fanout(mesh: &mut Mesh, msg: &[u8]) {
    mesh.planes[0].publish(msg).await.expect("publish");
    for (i, sub) in mesh.subs.iter_mut().enumerate() {
        let got = recv_timeout(sub, DELIVER).await;
        assert_eq!(
            got.as_deref(),
            Some(msg),
            "subscriber {i} must receive the fanned-out message"
        );
        assert!(
            recv_timeout(sub, GRACE).await.is_none(),
            "subscriber {i} must not receive a duplicate"
        );
    }
}

/// Conformance: the same bytes published via two planes (the WS path and the gossip path) still
/// deliver exactly once to every subscriber (content-hash dedupe — NET-6).
pub async fn conformance_dedupe(mesh: &mut Mesh, msg: &[u8]) {
    mesh.planes[0]
        .publish(msg)
        .await
        .expect("publish via path 0");
    if mesh.planes.len() > 1 {
        mesh.planes[1]
            .publish(msg)
            .await
            .expect("publish via path 1");
    }
    for (i, sub) in mesh.subs.iter_mut().enumerate() {
        let got = recv_timeout(sub, DELIVER).await;
        assert_eq!(
            got.as_deref(),
            Some(msg),
            "subscriber {i} must receive the message once"
        );
        assert!(
            recv_timeout(sub, GRACE).await.is_none(),
            "subscriber {i} must dedupe the second path to one delivery"
        );
    }
}

/// A deterministic ed25519 signing key for tests.
pub fn signing_key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

/// Canonical-CBOR bytes of a signed `Heartbeat` — a valid opaque control-plane payload.
pub fn signed_heartbeat_bytes(key: &SigningKey, round: u64) -> Vec<u8> {
    let payload = SwarmMessage::Heartbeat(Heartbeat { round, ready: None });
    let signed = SignedMessage::sign(key, SWARM_PROTO_VERSION, payload).expect("sign");
    to_canonical_vec(&signed).expect("encode")
}

/// Tamper a signed message's signature and re-encode — a payload whose signature no longer verifies
/// (distinct bytes from the valid message, so it is not deduped against it).
pub fn tampered_bytes(valid: &[u8]) -> Vec<u8> {
    let mut signed: SignedMessage = from_canonical_slice(valid).expect("decode");
    signed.sig.0[0] ^= 0xff;
    to_canonical_vec(&signed).expect("re-encode")
}

/// The in-process iroh multi-node harness (feature-gated): N endpoints on loopback with static
/// discovery, no relay. Ports the iroh-gossip 0.101 `net.rs::gossip_net_smoke` / Psyche `router.rs`
/// spawn pattern to explicit-roster addressing (no external discovery).
#[cfg(feature = "iroh")]
pub mod iroh_harness {
    use super::*;
    use std::net::SocketAddr;

    use daemon_swarm_net::{IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig};

    /// A fresh loopback bind address (OS-assigned port).
    pub fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().expect("loopback addr")
    }

    /// Rebroadcast disabled — deterministic single-flood delivery for most tests.
    pub fn no_rebroadcast() -> RebroadcastConfig {
        RebroadcastConfig {
            enabled: false,
            ..RebroadcastConfig::default()
        }
    }

    /// Connect one iroh node on loopback with no relay and an empty initial roster.
    pub async fn connect_node(seed: u8, rebroadcast: RebroadcastConfig) -> Arc<IrohGossip> {
        let config = IrohGossipConfig {
            secret_key: [seed; 32],
            relay_urls: vec![],
            roster: vec![],
            topic_input: [0x42; 32],
            rebroadcast,
            bind_addr: Some(loopback()),
        };
        Arc::new(
            IrohGossip::connect(config)
                .await
                .expect("connect iroh node"),
        )
    }

    /// Connect one iroh node pointed at a relay (relay-only reachability path).
    pub async fn connect_relay_node(seed: u8, relay_url: &str) -> Arc<IrohGossip> {
        let config = IrohGossipConfig {
            secret_key: [seed; 32],
            relay_urls: vec![relay_url.to_string()],
            roster: vec![],
            topic_input: [0x42; 32],
            rebroadcast: no_rebroadcast(),
            bind_addr: Some(loopback()),
        };
        Arc::new(
            IrohGossip::connect(config)
                .await
                .expect("connect relay iroh node"),
        )
    }

    /// Distribute the full roster (each node's dialable `local_peer`) to every node (the
    /// admission/ensure_gossip_connected step).
    pub async fn wire_roster(nodes: &[Arc<IrohGossip>]) {
        let roster: Vec<IrohPeer> = nodes.iter().map(|n| n.local_peer()).collect();
        for node in nodes {
            node.update_roster(roster.clone())
                .await
                .expect("update_roster");
        }
    }

    /// Poll until every node has at least `min_neighbors` gossip neighbors (mesh formed).
    pub async fn wait_for_mesh(nodes: &[Arc<IrohGossip>], min_neighbors: usize) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            if nodes.iter().all(|n| n.neighbor_count() >= min_neighbors) {
                return;
            }
            if std::time::Instant::now() > deadline {
                let counts: Vec<_> = nodes.iter().map(|n| n.neighbor_count()).collect();
                panic!("iroh mesh did not form within 20s: neighbor counts {counts:?}");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Build an N-node loopback iroh mesh (connect all, wire the roster, wait for formation).
    pub async fn build_mesh(n: usize, rebroadcast: RebroadcastConfig) -> Vec<Arc<IrohGossip>> {
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(connect_node(i as u8 + 1, rebroadcast.clone()).await);
        }
        wire_roster(&nodes).await;
        wait_for_mesh(&nodes, 1).await;
        nodes
    }

    /// Wrap live nodes as a [`Mesh`] (subscribe each node once).
    pub fn mesh_from(nodes: &[Arc<IrohGossip>]) -> Mesh {
        let planes = nodes
            .iter()
            .map(|n| n.clone() as Arc<dyn ControlPlane>)
            .collect();
        let subs = nodes.iter().map(|n| n.subscribe()).collect();
        Mesh { planes, subs }
    }
}
