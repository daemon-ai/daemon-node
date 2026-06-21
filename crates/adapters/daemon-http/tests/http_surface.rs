//! End-to-end tests for the HTTP/WS adapter over a mock [`NodeApi`]: JSON dispatch (`POST /api`),
//! the non-destructive merged-log cursor read (`GET …/log`), and the live SSE subscription
//! (`GET …/subscribe`). The adapter is a thin transport over the shared interface, so a mock surface
//! exercises the full request/response + streaming path without standing up the substrate.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{
    ApiError, ApiResponse, ControlApi, FleetReport, HealthReport, LogPageView, LogStream, NodeApi,
    Outbound, SessionApi, SessionInfo, StatsReport,
};
use daemon_common::{SessionId, UnitId};
use daemon_protocol::{
    AgentCommand, Direction, Disposition, HostResponse, Origin, OriginScope, SessionLogEntry,
    SessionPayload,
};
use futures::StreamExt;

fn entry(seq: u64) -> SessionLogEntry {
    SessionLogEntry {
        seq,
        direction: Direction::Inbound,
        origin: Origin::new("http", OriginScope::Api { key: "k1".into() }),
        disposition: Disposition::Context,
        payload: SessionPayload::Command(AgentCommand::Shutdown),
    }
}

struct MockApi;

#[async_trait]
impl SessionApi for MockApi {
    async fn submit(&self, _: SessionId, _: AgentCommand) -> Result<(), ApiError> {
        Ok(())
    }
    async fn poll(&self, _: SessionId, _: u32) -> Result<Vec<Outbound>, ApiError> {
        Ok(Vec::new())
    }
    async fn respond(&self, _: SessionId, _: HostResponse) -> Result<(), ApiError> {
        Ok(())
    }
    async fn log_after(
        &self,
        _: SessionId,
        after_seq: u64,
        _: u32,
    ) -> Result<LogPageView, ApiError> {
        Ok(LogPageView {
            entries: vec![entry(1), entry(2)]
                .into_iter()
                .filter(|e| e.seq > after_seq)
                .collect(),
            next_seq: 2,
            head_seq: 2,
        })
    }
    async fn subscribe(&self, _: SessionId, _: u64) -> Result<LogStream, ApiError> {
        Ok(futures::stream::iter(vec![entry(1), entry(2)]).boxed())
    }
}

#[async_trait]
impl ControlApi for MockApi {
    async fn health(&self) -> HealthReport {
        HealthReport {
            all_ok: true,
            services: Vec::new(),
        }
    }
    async fn stats(&self) -> StatsReport {
        StatsReport::default()
    }
    async fn sessions(&self) -> Vec<SessionInfo> {
        Vec::new()
    }
    async fn assign(&self, _: SessionId) -> Result<(), ApiError> {
        Ok(())
    }
    async fn cancel(&self, _: SessionId) -> Result<(), ApiError> {
        Ok(())
    }
    async fn fleet(&self) -> FleetReport {
        FleetReport::default()
    }
    async fn unit(&self, _: UnitId) -> Option<daemon_api::UnitNode> {
        None
    }
}

async fn spawn_server() -> String {
    let api: Arc<dyn NodeApi> = Arc::new(MockApi);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = daemon_http::serve_http(listener, api).await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn post_api_dispatches_json() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    let resp: ApiResponse = client
        .post(format!("{base}/api"))
        .json(&serde_json::json!("Health"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    match resp {
        ApiResponse::Health(h) => assert!(h.all_ok),
        other => panic!("expected Health, got {other:?}"),
    }
}

#[tokio::test]
async fn get_log_returns_a_cursor_page() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    let resp: ApiResponse = client
        .get(format!("{base}/sessions/s1/log?after_seq=0&max=0"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    match resp {
        ApiResponse::LogPage(page) => {
            assert_eq!(page.entries.len(), 2);
            assert_eq!(page.head_seq, 2);
            assert_eq!(page.entries[0].seq, 1);
        }
        other => panic!("expected LogPage, got {other:?}"),
    }

    // A cursor past the first entry returns only the tail (non-destructive paging).
    let tail: ApiResponse = client
        .get(format!("{base}/sessions/s1/log?after_seq=1"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    match tail {
        ApiResponse::LogPage(page) => {
            assert_eq!(page.entries.len(), 1);
            assert_eq!(page.entries[0].seq, 2);
        }
        other => panic!("expected LogPage, got {other:?}"),
    }
}

#[tokio::test]
async fn sse_subscribe_streams_entries() {
    let base = spawn_server().await;
    let client = reqwest::Client::new();

    let resp = client
        .get(format!("{base}/sessions/s1/subscribe?after_seq=0"))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success());

    // Read the first streamed chunk and confirm it carries an SSE data frame for a log entry.
    let mut stream = resp.bytes_stream();
    let chunk = tokio::time::timeout(std::time::Duration::from_secs(5), stream.next())
        .await
        .expect("timed out waiting for sse chunk")
        .expect("stream ended")
        .expect("chunk error");
    let text = String::from_utf8_lossy(&chunk);
    assert!(
        text.contains("data:"),
        "expected an SSE data frame, got: {text}"
    );
    assert!(
        text.contains("\"seq\""),
        "expected a serialized log entry: {text}"
    );
}
