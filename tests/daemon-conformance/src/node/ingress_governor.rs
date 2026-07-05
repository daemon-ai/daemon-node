// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE CLUSTER-F INGRESS-GOVERNOR GATE (Phase 4): the central ingress governor wired into a real
//! network carrier. The WebSocket mux carrier ([`serve_mux_ws`]) is the lightest injectable
//! transport (a plain `TcpListener`, no PKI), so these tests drive it with *tiny-limit* governors to
//! prove the four fail-closed limits actually bite at the carrier boundary:
//!
//! * an oversize **decoded** payload (a `BlobPut` over the decoded cap) is refused at ingress,
//!   before dispatch;
//! * the **connection-concurrency** cap refuses an excess connection;
//! * the **per-peer connection rate** refuses a burst beyond the bucket;
//! * every refusal is **fail-closed** (the connection is dropped / the request errors).
//!
//! The governor's decision *logic* (token bucket, semaphore, and the local-trust exemption where
//! [`daemon_common::PeerKey::Local`] is never rate/concurrency-limited) is unit-tested in
//! `daemon-common::ingress`; these tests prove the *wiring*.

use super::ws_transport::WsMuxClient;
use daemon_api::{ApiError, ApiRequest, ApiResponse};
use daemon_auth::{AuthStore, Role};
use daemon_common::{IngressGovernor, IngressLimits, RateSpec};
use daemon_host::{serve_mux_ws, Authenticator};
use std::sync::Arc;
use tokio::net::TcpListener;

use super::harness::assemble;

/// Serve the WS mux carrier on an ephemeral loopback port under an explicit (tiny-limit) governor,
/// with a seeded operator. Returns the bound address, the server task, and the resident handle.
async fn serve_ws_governed(
    governor: Arc<IngressGovernor>,
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
    daemon_host::SupervisorHandle,
) {
    let (node, handle) = assemble();
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .create_user("operator", "op-pw", &[Role::Operator])
        .expect("create operator");
    let auth = Arc::new(Authenticator::new(store));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(serve_mux_ws(listener, node, auth, Vec::new(), governor));
    (addr, server, handle)
}

/// Connect + `Hello`-handshake a WS mux client against `addr`.
async fn connect(addr: std::net::SocketAddr) -> Result<WsMuxClient, String> {
    WsMuxClient::connect_url(&format!("ws://{addr}/"), None)
        .await
        .map_err(|e| e.to_string())
}

/// An oversize decoded payload (a `BlobPut` whose decoded bytes exceed the governor's decoded cap)
/// is refused at ingress — BEFORE dispatch — with the ingress error, while a same-op payload under
/// the cap is NOT rejected by the ingress check. Repro: pre-governor, both would reach the blob
/// store (which only rejects at 256 MiB), so a 64-byte blob under a 16-byte decoded cap would be
/// accepted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn oversize_decoded_blob_put_is_rejected_at_ingress() {
    let governor = IngressGovernor::new(IngressLimits {
        max_decoded_bytes: 16,
        ..IngressLimits::unlimited()
    });
    let (addr, server, handle) = serve_ws_governed(governor).await;

    let mut client = connect(addr).await.expect("ws connect + hello");
    client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram over ws");

    // 64 decoded bytes > the 16-byte decoded cap → rejected at ingress (the reject rides the
    // correlated `Reply` as an `ApiResponse::Error`, not a transport error).
    let over = client
        .call(ApiRequest::BlobPut {
            bytes: vec![0u8; 64],
        })
        .await;
    match over {
        Ok(ApiResponse::Error(ApiError::Other(msg))) => assert!(
            msg.contains("decoded payload too large"),
            "an oversize BlobPut must be refused by the ingress decoded cap, got {msg:?}"
        ),
        other => panic!("expected the ingress decoded-cap error, got {other:?}"),
    }

    // 8 decoded bytes ≤ the cap → NOT rejected by the ingress check (it reaches dispatch; whatever
    // the store returns, it is never the ingress decoded-cap error).
    let under = client
        .call(ApiRequest::BlobPut {
            bytes: vec![0u8; 8],
        })
        .await;
    if let Ok(ApiResponse::Error(ApiError::Other(msg))) = &under {
        assert!(
            !msg.contains("decoded payload too large"),
            "an under-cap BlobPut must pass the ingress decoded check, got {msg:?}"
        );
    }

    server.abort();
    handle.shutdown().await;
}

/// The connection-concurrency cap refuses an excess connection (fail-closed): with `max_connections
/// = 1`, a second connection is dropped at accept, and freeing the first lets a later one in. Repro:
/// pre-governor, the accept loop spawned unbounded connections, so the 2nd would connect fine.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connection_concurrency_cap_refuses_excess() {
    let governor = IngressGovernor::new(IngressLimits {
        max_connections: Some(1),
        ..IngressLimits::unlimited()
    });
    let (addr, server, handle) = serve_ws_governed(governor).await;

    // The first connection is admitted and holds the single slot for its lifetime.
    let first = connect(addr).await.expect("1st connection admitted");

    // The second is refused: the server drops the accepted stream, so the upgrade never completes.
    assert!(
        connect(addr).await.is_err(),
        "a connection beyond the concurrency cap must be refused (fail closed)"
    );

    // Free the slot; a subsequent connection is admitted again.
    drop(first);
    // Give the dropped connection's task a moment to release the permit.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        connect(addr).await.is_ok(),
        "freeing a connection must free a slot for a new one"
    );

    server.abort();
    handle.shutdown().await;
}

/// The per-peer connection rate refuses a burst beyond the token bucket (fail-closed): with `burst =
/// 1, refill = 0`, the peer's second connection is refused. Repro: pre-governor there was no
/// per-peer rate limit at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn per_peer_connection_rate_refuses_burst() {
    let governor = IngressGovernor::new(IngressLimits {
        peer_conn_rate: Some(RateSpec {
            burst: 1.0,
            refill_per_sec: 0.0,
        }),
        max_tracked_peers: 16,
        ..IngressLimits::unlimited()
    });
    let (addr, server, handle) = serve_ws_governed(governor).await;

    // The first connection from this peer consumes the single token.
    let _first = connect(addr).await.expect("1st connection within burst");

    // The second (same loopback peer, no refill) is refused at accept.
    assert!(
        connect(addr).await.is_err(),
        "a per-peer connection beyond the burst must be refused (fail closed)"
    );

    server.abort();
    handle.shutdown().await;
}

/// A sanity check that the SECURE-DEFAULT governor (the production posture) does not perturb a normal
/// connect + authenticate + Call over the WS carrier — the proof the generous defaults are wide
/// enough for legitimate traffic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn secure_default_governor_serves_normal_traffic() {
    let (addr, server, handle) = serve_ws_governed(IngressGovernor::secure_default()).await;
    let mut client = connect(addr).await.expect("ws connect + hello");
    client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram over ws");
    let res = client.call(ApiRequest::Health).await.expect("health call");
    assert!(
        !matches!(res, ApiResponse::Error(_)),
        "the secure-default governor must serve normal traffic, got {res:?}"
    );
    server.abort();
    handle.shutdown().await;
}
