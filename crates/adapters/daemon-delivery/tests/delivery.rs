// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Unit tests for the reusable pull subscriber over a mock [`NodeApi`]: it discovers a transport
//! instance's owned sessions, projects their merged-log entries, and halts a session once the
//! transport is handed over (demoted from `Primary` to `Spectator`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use daemon_api::{
    ApiError, ControlApi, FleetReport, HealthReport, LogStream, ModelApi, NodeApi, Outbound,
    SessionApi, SessionInfo, StatsReport,
};
use daemon_common::{SessionId, UnitId};
use daemon_delivery::{serve_delivery, Projector};
use daemon_protocol::{
    AgentCommand, DeliveryTarget, Direction, Disposition, HostResponse, Origin, OriginScope,
    SessionLogEntry, SessionPayload, SinkKind, TransportId,
};
use futures::StreamExt;

fn entry(seq: u64) -> SessionLogEntry {
    SessionLogEntry {
        seq,
        direction: Direction::Outbound,
        origin: Origin::new("engine", OriginScope::Api { key: "k".into() }),
        disposition: Disposition::Context,
        payload: SessionPayload::Command(AgentCommand::Shutdown),
    }
}

/// A mock node exposing just the two delivery primitives the pull helper uses. `demote_after`
/// flips the owned session's `Primary` to a `Spectator` once `delivery_targets` has been polled that
/// many times, modeling a handover mid-stream.
struct MockApi {
    transport: TransportId,
    sessions: Vec<SessionId>,
    entries: Vec<SessionLogEntry>,
    demote_after: Option<usize>,
    targets_calls: AtomicUsize,
}

impl MockApi {
    fn new(transport: &str, sessions: &[&str], entries: usize) -> Self {
        Self {
            transport: TransportId::new(transport),
            sessions: sessions.iter().map(|s| SessionId::new(*s)).collect(),
            entries: (1..=entries as u64).map(entry).collect(),
            demote_after: None,
            targets_calls: AtomicUsize::new(0),
        }
    }

    fn demote_after(mut self, n: usize) -> Self {
        self.demote_after = Some(n);
        self
    }
}

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
    async fn subscribe(&self, _: SessionId, _: u64) -> Result<LogStream, ApiError> {
        Ok(futures::stream::iter(self.entries.clone())
            .map(daemon_api::LogStreamItem::Entry)
            .boxed())
    }
    async fn delivery_sessions(
        &self,
        transport: TransportId,
        after: Option<String>,
    ) -> daemon_api::WirePage<SessionId> {
        let sessions = if transport == self.transport {
            self.sessions.clone()
        } else {
            Vec::new()
        };
        daemon_api::paginate(sessions, after.as_deref(), daemon_api::WIRE_PAGE_MAX, |s| {
            s.as_str().to_string()
        })
    }
    async fn delivery_targets(&self, _: SessionId) -> Vec<DeliveryTarget> {
        let n = self.targets_calls.fetch_add(1, Ordering::SeqCst);
        let kind = match self.demote_after {
            Some(limit) if n >= limit => SinkKind::Spectator,
            _ => SinkKind::Primary,
        };
        vec![DeliveryTarget::new(self.transport.as_str(), "route", kind)]
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

impl ModelApi for MockApi {}
impl daemon_api::ProfileApi for MockApi {}
impl daemon_api::CredentialApi for MockApi {}
impl daemon_api::AuthApi for MockApi {}
impl daemon_api::AccessControlApi for MockApi {}

/// A projector that records every `(session, seq)` it is handed.
#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<(SessionId, u64)>>,
}

#[async_trait]
impl Projector for Recorder {
    async fn project(&self, session: SessionId, entry: SessionLogEntry) {
        self.seen.lock().unwrap().push((session, entry.seq));
    }
}

#[tokio::test]
async fn discovers_owned_sessions_and_projects_entries() {
    let api: Arc<dyn NodeApi> = Arc::new(MockApi::new("mock/inst", &["s1", "s2"], 3));
    let recorder = Arc::new(Recorder::default());

    let sub = serve_delivery(api, TransportId::new("mock/inst"), recorder.clone()).await;
    assert_eq!(sub.len(), 2, "one delivery task per owned session");
    sub.join().await;

    let seen = recorder.seen.lock().unwrap();
    // Both owned sessions projected all three entries (still the Primary throughout).
    assert_eq!(seen.len(), 6, "2 sessions x 3 entries, got {seen:?}");
    for s in ["s1", "s2"] {
        let seqs: Vec<u64> = seen
            .iter()
            .filter(|(sess, _)| sess.as_str() == s)
            .map(|(_, seq)| *seq)
            .collect();
        assert_eq!(seqs, vec![1, 2, 3], "session {s} projected all entries");
    }
}

#[tokio::test]
async fn ignores_unowned_transport() {
    let api: Arc<dyn NodeApi> = Arc::new(MockApi::new("mock/inst", &["s1"], 2));
    let recorder = Arc::new(Recorder::default());

    // A transport that owns no sessions gets an empty subscription (nothing to deliver).
    let sub = serve_delivery(api, TransportId::new("other/inst"), recorder.clone()).await;
    assert!(sub.is_empty(), "no owned sessions -> no delivery tasks");
    sub.join().await;
    assert!(recorder.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn halts_a_session_on_handover_demotion() {
    // The owned session is the Primary for the first ownership check, then demoted: the helper
    // re-checks before each projected entry, so it projects exactly the first entry and then stops.
    let api: Arc<dyn NodeApi> = Arc::new(MockApi::new("mock/inst", &["s1"], 5).demote_after(1));
    let recorder = Arc::new(Recorder::default());

    let sub = serve_delivery(api, TransportId::new("mock/inst"), recorder.clone()).await;
    sub.join().await;

    let seen = recorder.seen.lock().unwrap();
    assert_eq!(
        seen.len(),
        1,
        "delivery halts on demotion after the first entry, got {seen:?}"
    );
    assert_eq!(seen[0].1, 1);
}
