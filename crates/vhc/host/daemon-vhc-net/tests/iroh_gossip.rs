// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Live iroh-gossip integration suite (B2), behind the `iroh` feature: NET-6 (signed-gossip
//! accept/reject + WS/gossip dedupe), multi-node fanout, roster-update reconnect, partition/rejoin,
//! the rebroadcast-frame no-duplicate property, and a relay-path smoke through a self-hosted
//! `iroh-relay --dev` (skipped cleanly when the binary is absent).

#![cfg(feature = "iroh")]

mod common;

use std::time::Duration;

use daemon_swarm_net::{ControlPlane, IrohPeer, RebroadcastConfig};
use daemon_swarm_proto::{from_canonical_slice, SignedMessage};

use common::iroh_harness::{
    build_mesh, connect_node, connect_relay_node, no_rebroadcast, wait_for_mesh, wire_roster,
};
use common::{recv_timeout, signed_heartbeat_bytes, signing_key, tampered_bytes, DELIVER, GRACE};

/// NET-6 (accept/reject half). The plane carries **opaque already-signed bytes** and never
/// verifies (§7.1: gossip is dissemination, never arbitration) — verification is the **consumer's**
/// gate. So the plane delivers both a valid and a tampered message unchanged; proto's
/// `SignedMessage::verify` is what accepts the good one and rejects the tampered one.
#[tokio::test(flavor = "multi_thread")]
async fn signed_gossip_bad_sig_rejected() {
    let nodes = build_mesh(2, no_rebroadcast()).await;
    let mut sub1 = nodes[1].subscribe();
    let key = signing_key(9);

    // A valid signed message: delivered, and the consumer's verify accepts it.
    let good = signed_heartbeat_bytes(&key, 1);
    nodes[0].publish(&good).await.expect("publish good");
    let got = recv_timeout(&mut sub1, DELIVER)
        .await
        .expect("valid message delivered");
    let decoded: SignedMessage = from_canonical_slice(&got).expect("decode");
    assert!(
        decoded.verify().is_ok(),
        "the consumer's verify accepts a valid signature"
    );

    // A tampered signed message: the plane still delivers the opaque bytes unchanged; the
    // consumer's verify is what rejects it (the engine-side contract).
    let bad = tampered_bytes(&good);
    nodes[0].publish(&bad).await.expect("publish bad");
    let got_bad = recv_timeout(&mut sub1, DELIVER)
        .await
        .expect("the plane delivers opaque bytes regardless of signature validity");
    assert_eq!(
        got_bad, bad,
        "the plane delivers the bytes unchanged (no plane-side verify)"
    );
    let decoded_bad: SignedMessage = from_canonical_slice(&got_bad).expect("decode");
    assert!(
        decoded_bad.verify().is_err(),
        "the consumer's verify rejects the tampered signature"
    );
}

/// NET-6 (dedupe half). The same signed payload arriving via two paths (WS = node 0, gossip =
/// node 1) is delivered exactly once at a third node (content-hash dedupe).
#[tokio::test(flavor = "multi_thread")]
async fn ws_gossip_duplicate_message_dedupes() {
    let nodes = build_mesh(3, no_rebroadcast()).await;
    let mut sub2 = nodes[2].subscribe();
    let msg = signed_heartbeat_bytes(&signing_key(3), 5);

    nodes[0].publish(&msg).await.expect("publish via node 0");
    nodes[1].publish(&msg).await.expect("publish via node 1");

    let got = recv_timeout(&mut sub2, DELIVER)
        .await
        .expect("one delivery");
    assert_eq!(got, msg);
    assert!(
        recv_timeout(&mut sub2, GRACE).await.is_none(),
        "the same payload via two paths dedupes to a single delivery"
    );
}

/// Fanout: a message published from one node reaches every node in the mesh.
#[tokio::test(flavor = "multi_thread")]
async fn iroh_fanout_reaches_all_nodes() {
    let nodes = build_mesh(3, no_rebroadcast()).await;
    let mut subs: Vec<_> = nodes.iter().map(|n| n.subscribe()).collect();
    let msg = signed_heartbeat_bytes(&signing_key(1), 7);

    nodes[0].publish(&msg).await.expect("publish");
    for (i, sub) in subs.iter_mut().enumerate() {
        assert_eq!(
            recv_timeout(sub, DELIVER).await.as_deref(),
            Some(&msg[..]),
            "node {i} receives the fanned-out message"
        );
    }
}

/// Roster-update reconnect: a node that joins after the mesh formed is reached once every node runs
/// `update_roster` (the admission / ensure_gossip_connected path).
#[tokio::test(flavor = "multi_thread")]
async fn iroh_roster_update_reconnects_new_peer() {
    let mut nodes = vec![
        connect_node(1, no_rebroadcast()).await,
        connect_node(2, no_rebroadcast()).await,
    ];
    wire_roster(&nodes).await;
    wait_for_mesh(&nodes, 1).await;

    // A third node joins; re-distribute the roster so the mesh re-forms around it.
    nodes.push(connect_node(3, no_rebroadcast()).await);
    wire_roster(&nodes).await;
    wait_for_mesh(&nodes, 1).await;

    let mut sub3 = nodes[2].subscribe();
    let msg = signed_heartbeat_bytes(&signing_key(1), 9);
    nodes[0].publish(&msg).await.expect("publish");
    assert_eq!(
        recv_timeout(&mut sub3, DELIVER).await.as_deref(),
        Some(&msg[..]),
        "the roster-added node receives after update_roster"
    );
}

/// Partition / rejoin smoke: drop a node from the mesh, confirm the rest still deliver, then bring a
/// fresh node in and re-wire the roster — messages flow to it again.
#[tokio::test(flavor = "multi_thread")]
async fn iroh_partition_rejoin_smoke() {
    let nodes = build_mesh(3, no_rebroadcast()).await;
    let mut sub1 = nodes[1].subscribe();

    // Baseline: node 0 -> node 1 delivers.
    let m1 = signed_heartbeat_bytes(&signing_key(1), 1);
    nodes[0].publish(&m1).await.expect("publish m1");
    assert_eq!(
        recv_timeout(&mut sub1, DELIVER).await.as_deref(),
        Some(&m1[..]),
        "baseline delivery"
    );

    // Partition: shut node 2 down. The rest still deliver.
    nodes[2].shutdown().await;
    let m2 = signed_heartbeat_bytes(&signing_key(1), 2);
    nodes[0].publish(&m2).await.expect("publish m2");
    assert_eq!(
        recv_timeout(&mut sub1, DELIVER).await.as_deref(),
        Some(&m2[..]),
        "surviving nodes still deliver after a partition"
    );

    // Rejoin: a fresh node joins and the roster is re-wired — messages flow to it again.
    let rejoined = connect_node(22, no_rebroadcast()).await;
    let live = vec![nodes[0].clone(), nodes[1].clone(), rejoined.clone()];
    wire_roster(&live).await;
    wait_for_mesh(&live, 1).await;
    let mut sub_rejoined = rejoined.subscribe();
    let m3 = signed_heartbeat_bytes(&signing_key(1), 3);
    nodes[0].publish(&m3).await.expect("publish m3");
    assert_eq!(
        recv_timeout(&mut sub_rejoined, DELIVER).await.as_deref(),
        Some(&m3[..]),
        "messages flow again to the rejoined node"
    );
}

/// The rebroadcast frame re-floods at the gossip layer (a bumped nonce -> a new `MessageId`) but the
/// app-layer content-hash dedupe still delivers exactly once — the delivery-assurance design's
/// correctness guarantee.
#[tokio::test(flavor = "multi_thread")]
async fn rebroadcast_refloods_without_duplicate_delivery() {
    let rebroadcast = RebroadcastConfig {
        enabled: true,
        interval: Duration::from_millis(150),
        ring_capacity: 8,
    };
    let nodes = build_mesh(2, rebroadcast).await;
    let mut sub1 = nodes[1].subscribe();

    let msg = signed_heartbeat_bytes(&signing_key(1), 1);
    nodes[0].publish(&msg).await.expect("publish");
    assert_eq!(
        recv_timeout(&mut sub1, DELIVER).await.as_deref(),
        Some(&msg[..]),
        "first delivery"
    );
    // Over several rebroadcast intervals (~4 x 150 ms) the receiver must NOT see a duplicate.
    assert!(
        recv_timeout(&mut sub1, Duration::from_millis(700))
            .await
            .is_none(),
        "rebroadcasts re-flood the gossip layer but the app Deduper drops duplicate deliveries"
    );
}

/// Relay-path smoke: prove gossip forms + delivers when reachability is via a **self-hosted
/// `iroh-relay --dev`** (plain HTTP, port 3340) and the roster carries relay-only addresses (no
/// direct IPs). Skips cleanly when the relay binary is not on PATH (standalone checkout without the
/// devShell) or its default dev port is busy.
#[tokio::test(flavor = "multi_thread")]
async fn relay_path_delivers_through_self_hosted_relay() {
    let Some(mut relay) = spawn_dev_relay() else {
        eprintln!(
            "SKIP relay_path: iroh-relay --dev not available (binary absent or port 3340 busy)"
        );
        return;
    };
    if !wait_for_tcp(RELAY_PORT, Duration::from_secs(8)).await {
        let _ = relay.kill();
        eprintln!("SKIP relay_path: iroh-relay did not come up on port {RELAY_PORT}");
        return;
    }
    let relay_url = format!("http://localhost:{RELAY_PORT}");

    let n0 = connect_relay_node(1, &relay_url).await;
    let n1 = connect_relay_node(2, &relay_url).await;
    let nodes = [n0.clone(), n1.clone()];

    // Relay-only roster: no direct IPs, so initial reachability must go through the relay.
    let roster: Vec<IrohPeer> = nodes
        .iter()
        .map(|n| IrohPeer {
            endpoint_id: n.node_id(),
            direct_addrs: vec![],
            relay_url: Some(relay_url.clone()),
        })
        .collect();
    for node in &nodes {
        node.update_roster(roster.clone())
            .await
            .expect("update_roster over relay");
    }
    wait_for_mesh(&nodes, 1).await;

    let mut sub1 = n1.subscribe();
    let msg = signed_heartbeat_bytes(&signing_key(1), 1);
    n0.publish(&msg).await.expect("publish over relay");
    let delivered = recv_timeout(&mut sub1, DELIVER).await;

    n0.shutdown().await;
    n1.shutdown().await;
    let _ = relay.kill();

    assert_eq!(
        delivered.as_deref(),
        Some(&msg[..]),
        "message delivered through the self-hosted relay"
    );
}

/// The `iroh-relay --dev` default HTTP port (see `dev/README.md`).
const RELAY_PORT: u16 = 3340;

/// Spawn the devShell `iroh-relay --dev` (plain-HTTP dev relay). Returns `None` if the port is
/// already bound (another relay/test owns it) or the binary is not on PATH — the caller then skips.
#[allow(clippy::disallowed_methods)] // test-only: spawn the known devShell `iroh-relay --dev` dev tool
fn spawn_dev_relay() -> Option<std::process::Child> {
    // If the dev port is not free, do not fight for it — skip.
    if std::net::TcpListener::bind(("127.0.0.1", RELAY_PORT)).is_err() {
        return None;
    }
    std::process::Command::new("iroh-relay")
        .arg("--dev")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()
}

/// Poll until a TCP connection to `127.0.0.1:port` succeeds, or the deadline elapses.
async fn wait_for_tcp(port: u16, timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .is_ok()
        {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    false
}
