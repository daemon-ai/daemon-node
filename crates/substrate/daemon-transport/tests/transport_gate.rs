//! THE PHASE-6c GATE: a managed unit driven over a real socket, trace-in-envelope across the node
//! boundary, and a stale remote fence rejected by the authority (`daemon-workspace-layout.md` §7
//! phase-6: "cross-node lease/fence when fleets-of-fleets is real" — the single process-pair proof).

use std::sync::Arc;

use daemon_common::{Epoch, PartitionId, SessionId, SnapshotBlob, TraceId, UnitId};
use daemon_core::{Engine, MockProvider, Provider, SystemPrompt, ToolRegistry};
use daemon_host::EngineUnit;
use daemon_store::{InMemoryStore, SessionStatus, SessionStore, StoreErrorWire};
use daemon_supervision::ManagedUnit;
use daemon_telemetry::{current_trace, with_trace};
use daemon_transport::{RemoteClient, RemoteHost, TransportError};
use tokio::net::TcpListener;

async fn start_server() -> (Arc<InMemoryStore>, String) {
    let store = Arc::new(InMemoryStore::new());

    let provider: Arc<dyn Provider> = Arc::new(MockProvider::completing("remote done"));
    let engine = Engine::fresh(
        SessionId::new("u1"),
        SystemPrompt::new("remote-hosted unit"),
        provider,
        Arc::new(ToolRegistry::new()),
    );
    let unit: Arc<dyn ManagedUnit> = Arc::new(EngineUnit::spawn(UnitId::new("u1"), engine));

    let server = Arc::new(RemoteHost::new(
        store.clone() as Arc<dyn SessionStore>,
        unit,
    ));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        let _ = server.serve(listener).await;
    });
    (store, addr)
}

/// A managed unit is driven end-to-end over a TCP socket, and the client's trace rides the wire and
/// is restored on the server (and echoed back).
#[tokio::test]
async fn unit_driven_over_socket_with_trace_propagation() {
    let (_store, addr) = start_server().await;
    let mut client = RemoteClient::connect(&addr).await.unwrap();
    assert!(client.hello().await.unwrap(), "version handshake");

    let trace = TraceId::generate();
    assert!(!trace.is_none());

    with_trace(trace, async {
        let outcome = client.drive("do the remote work").await.unwrap();
        assert!(outcome.ok, "the remote unit should complete: {outcome:?}");
        assert_eq!(outcome.end_reason, "Completed");
        // The server restored the client's trace from the request envelope...
        assert_eq!(
            outcome.observed_trace, trace,
            "the trace must ride the socket and be restored on the server"
        );
        // ...and the reply envelope carried it back (restored into this task's scope).
        assert_eq!(
            current_trace(),
            trace,
            "the peer trace must round-trip back"
        );
    })
    .await;
}

/// A stale remote owner's commit is rejected by the authoritative store across the socket — fencing
/// holds across the node boundary (acceptance test #6 over the network).
#[tokio::test]
async fn stale_remote_fence_is_rejected() {
    let (store, addr) = start_server().await;
    let session = SessionId::new("remote-fenced");
    store
        .create_session(
            session.clone(),
            PartitionId::DEFAULT,
            SnapshotBlob::default(),
        )
        .await
        .unwrap();

    let mut client = RemoteClient::connect(&addr).await.unwrap();

    // Two cross-node lease acquisitions: the second supersedes the first.
    let stale = client.acquire_fence(&session).await.unwrap();
    let current = client.acquire_fence(&session).await.unwrap();
    assert!(current > stale, "the second lease must supersede the first");

    // The stale owner's commit is fenced by the remote authority.
    let err = client
        .commit(&session, Epoch(1), stale)
        .await
        .expect_err("a stale remote fence must be rejected");
    assert!(
        matches!(err, TransportError::Store(StoreErrorWire::Fenced { .. })),
        "expected a Fenced rejection across the socket, got {err:?}"
    );
    assert_ne!(
        store.status(&session).await,
        Some(SessionStatus::Completed),
        "a fenced remote commit must not have landed"
    );

    // The current owner commits successfully over the socket.
    client
        .commit(&session, Epoch(1), current)
        .await
        .expect("the current remote owner commits");
    assert_eq!(
        client.status(&session).await.unwrap(),
        Some(SessionStatus::Completed)
    );
}
