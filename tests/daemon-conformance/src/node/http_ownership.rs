// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Auth 4 / HTTP-transport regression: the HTTP session-log routes must keep working under
//! `[api].local_trust` after the None-principal flip. HTTP carries no per-user SASL yet, so the
//! whole surface is all-or-nothing local trust; the routes previously called the node with NO bound
//! principal (fine while `None ⇒ allow`), which after the flip would deny every read. The fix wraps
//! each route in a `RequestContext::system()` scope; this test proves `GET /sessions/{id}/log`
//! still returns a `LogPage` (not an auth error) end-to-end over a real TCP listener.

use super::harness::*;
use daemon_api::{ApiResponse, SessionApi};
use daemon_common::ReqId;
use daemon_protocol::{AgentCommand, UserMsg};

/// `GET /sessions/{id}/log` under local trust returns the merged-log page after the flip (the
/// `system()` wrap keeps the route functional even though it has no per-request principal).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn http_log_route_serves_under_local_trust_after_flip() {
    let (node, handle) = assemble();
    let session = SessionId::new("http-log");

    // Populate the session's live log as the trusted local embedder (`system`), exactly as the HTTP
    // route will read it. `system` owns the session (stamped owner = "system"); the HTTP read below
    // reaches it via the same `system()` scope the route now wraps its call in.
    as_system(node.submit(
        session.clone(),
        AgentCommand::StartTurn {
            input: UserMsg::new("over http"),
            request_id: ReqId(1),
        },
    ))
    .await
    .expect("submit populates the live log");
    // Wait until an in-process read sees entries, so the HTTP read below is deterministic.
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let page = as_system(node.log_after(session.clone(), 0, 64))
            .await
            .expect("in-process log_after");
        if !page.entries.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "session produced no log entries");
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    // Serve the HTTP adapter under local trust on an ephemeral port.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind http");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(daemon_http::serve_http(listener, node.clone(), true));

    let url = format!("http://{addr}/sessions/{}/log?after_seq=0&max=64", session);
    let resp = reqwest::get(&url).await.expect("GET /log");
    assert_eq!(
        resp.status(),
        reqwest::StatusCode::OK,
        "route must be mounted"
    );
    let body: ApiResponse = resp.json().await.expect("decode ApiResponse");
    match body {
        ApiResponse::LogPage(page) => assert!(
            !page.entries.is_empty(),
            "the HTTP log route must return the merged-log page, got an empty page"
        ),
        other => panic!("expected a LogPage over HTTP, got {other:?}"),
    }

    handle.shutdown().await;
    server.abort();
}
