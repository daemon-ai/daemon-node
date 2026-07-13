// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Parametric `ControlPlane` conformance suite (B2): the same behavior tests run over
//! [`LoopbackGossip`] (always) and [`IrohGossip`] (behind the `iroh` feature + an in-process
//! multi-node harness). The shared property: publishing from any plane delivers to every subscriber
//! exactly once, and a duplicate publish (WS + gossip paths) still delivers once (content-hash
//! dedupe). Loopback-only timing semantics (late-subscriber) stay in `gossip.rs`'s unit tests.
//!
//! [`LoopbackGossip`]: daemon_swarm_net::LoopbackGossip
//! [`IrohGossip`]: daemon_swarm_net::IrohGossip

mod common;

use std::sync::Arc;

use daemon_swarm_net::{ControlPlane, LoopbackGossip};

use common::{conformance_dedupe, conformance_fanout, Mesh};

/// A loopback "mesh" of N nodes: one shared bus, N subscriptions, N plane handles to the same bus.
fn loopback_mesh(n: usize) -> Mesh {
    let plane = Arc::new(LoopbackGossip::new());
    let subs = (0..n).map(|_| plane.subscribe()).collect();
    let planes = (0..n)
        .map(|_| plane.clone() as Arc<dyn ControlPlane>)
        .collect();
    Mesh { planes, subs }
}

#[tokio::test]
async fn loopback_conformance_fanout() {
    let mut mesh = loopback_mesh(3);
    conformance_fanout(&mut mesh, b"round-open").await;
}

#[tokio::test]
async fn loopback_conformance_dedupe() {
    let mut mesh = loopback_mesh(3);
    conformance_dedupe(&mut mesh, b"commitment").await;
}

#[cfg(feature = "iroh")]
#[tokio::test(flavor = "multi_thread")]
async fn iroh_conformance_fanout() {
    use common::iroh_harness::{build_mesh, mesh_from, no_rebroadcast};
    let nodes = build_mesh(3, no_rebroadcast()).await;
    let mut mesh = mesh_from(&nodes);
    conformance_fanout(&mut mesh, b"round-open").await;
}

#[cfg(feature = "iroh")]
#[tokio::test(flavor = "multi_thread")]
async fn iroh_conformance_dedupe() {
    use common::iroh_harness::{build_mesh, mesh_from, no_rebroadcast};
    let nodes = build_mesh(3, no_rebroadcast()).await;
    let mut mesh = mesh_from(&nodes);
    conformance_dedupe(&mut mesh, b"commitment").await;
}

// The parametric suite over the WS plane: N `WsControlPlane`s against one in-process mock
// `RunCoordinatorDO` implementing the DO's dissemination framing (relay-to-others, no self-echo).
#[cfg(feature = "ws")]
#[tokio::test(flavor = "multi_thread")]
async fn ws_conformance_fanout() {
    use common::ws_harness::{build_ws_mesh, ws_mesh_from, MockWsCoordinator};
    let coord = MockWsCoordinator::start().await;
    let nodes = build_ws_mesh(&coord, 3).await;
    let mut mesh = ws_mesh_from(&nodes);
    conformance_fanout(&mut mesh, b"round-open").await;
}

#[cfg(feature = "ws")]
#[tokio::test(flavor = "multi_thread")]
async fn ws_conformance_dedupe() {
    use common::ws_harness::{build_ws_mesh, ws_mesh_from, MockWsCoordinator};
    let coord = MockWsCoordinator::start().await;
    let nodes = build_ws_mesh(&coord, 3).await;
    let mut mesh = ws_mesh_from(&nodes);
    conformance_dedupe(&mut mesh, b"commitment").await;
}
