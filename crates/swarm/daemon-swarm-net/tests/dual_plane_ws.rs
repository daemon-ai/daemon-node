// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![cfg(feature = "ws")]

//! Dual-plane dedupe (A1): the same signed `SignedMessage` arriving over BOTH the coordinator WS
//! plane and the (loopback) gossip plane is delivered to a peer exactly once (spec §7.1 — "the
//! coordinator WS carries the same messages if gossip degrades", deduped by the shared [`Deduper`]).

mod common;

use std::sync::Arc;

use common::ws_harness::{no_reconnect, MockWsCoordinator};
use common::{recv_timeout, signed_heartbeat_bytes, signing_key, DELIVER, GRACE};
use daemon_swarm_net::{ControlPlane, DualPlane, LoopbackGossip, WsAuth};

/// Two dual-plane peers share a loopback gossip bus and a mock coordinator. A publish from peer 1
/// reaches peer 2 over BOTH planes (WS relay + gossip fanout); the merged subscription dedupes them
/// to a single delivery.
#[tokio::test(flavor = "multi_thread")]
async fn same_frame_over_ws_and_gossip_delivers_once() {
    let coord = MockWsCoordinator::start().await;
    let gossip = Arc::new(LoopbackGossip::new());

    let ws1 = Arc::new(coord.client("run-1", WsAuth::None, no_reconnect()).await);
    let ws2 = Arc::new(coord.client("run-1", WsAuth::None, no_reconnect()).await);
    coord.wait_peers(2).await;

    let peer1 = DualPlane::pair(ws1.clone(), gossip.clone());
    let peer2 = DualPlane::pair(ws2.clone(), gossip.clone());

    // Subscribing registers peer 2's WS + gossip inner subscriptions synchronously (no race).
    let mut sub2 = peer2.subscribe();

    let frame = signed_heartbeat_bytes(&signing_key(4), 2);
    peer1.publish(&frame).await.expect("dual publish");

    assert_eq!(
        recv_timeout(&mut sub2, DELIVER).await.as_deref(),
        Some(frame.as_slice()),
        "peer 2 receives the frame once (via whichever plane arrived first)"
    );
    assert!(
        recv_timeout(&mut sub2, GRACE).await.is_none(),
        "the second plane's copy is deduped to a single delivery"
    );
}
