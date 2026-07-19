// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! A focused smoke of the registry fixture's WS relay: two `WsControlPlane` clients connect to
//! `{base}/runs/:id/ws`, and a frame published by one is disseminated to the other (the relay
//! contract the coordinator + trainers depend on). Isolates the fixture from the full 3-process
//! run so a relay regression fails here, cheaply.

mod harness;

use std::time::Duration;

use daemon_vhc_net::{ControlPlane, ReconnectConfig, WsAuth, WsConfig, WsControlPlane};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fixture_ws_relay_disseminates() {
    // Author a throwaway genesis just to stand the fixture up (the WS relay ignores run state).
    let bases = vec![daemon_vhc_proto::PeerId([0x01; 32])];
    let port = harness::free_port();
    let cluster = harness::start_cluster_on(port, "ws-smoke", &bases, 0, 2).await;
    let base = cluster.base_url.clone();

    let a = WsControlPlane::connect(WsConfig {
        base_url: base.clone(),
        run_id: "ws-smoke".into(),
        auth: WsAuth::None,
        reconnect: ReconnectConfig::default(),
    })
    .await
    .expect("client A connects to the fixture relay");
    let b = WsControlPlane::connect(WsConfig {
        base_url: base,
        run_id: "ws-smoke".into(),
        auth: WsAuth::None,
        reconnect: ReconnectConfig::default(),
    })
    .await
    .expect("client B connects to the fixture relay");

    let mut sub_b = b.subscribe();
    // Give the relay a moment to register both peers.
    tokio::time::sleep(Duration::from_millis(200)).await;

    let msg = b"acceptance-ws-relay-frame".to_vec();
    a.publish(&msg).await.expect("publish from A");

    let got = tokio::time::timeout(Duration::from_secs(5), sub_b.recv())
        .await
        .expect("B receives within 5s")
        .expect("a frame");
    assert_eq!(got, msg, "the relay disseminates A's frame to B");

    drop(cluster);
}
