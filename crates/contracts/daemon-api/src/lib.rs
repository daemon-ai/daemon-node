//! `daemon-api` — the node's one canonical interface/surface.
//!
//! A node exposes a single operable surface; *how* a caller reaches it (in-process call, a Unix
//! socket, TCP, JSON-RPC, the C FFI buffer pump) is a transport detail. This crate defines that
//! surface **once** and is the invariant every transport binds to:
//!
//! - the **interface** — [`SessionApi`] (the §17 per-session surface: submit a command, drain
//!   outbound items, answer a host request) and [`ControlApi`] (node/operator ops: health, stats,
//!   sessions, assign, cancel, fleet), composed as [`NodeApi`];
//! - the **serializable mirror** — [`ApiRequest`]/[`ApiResponse`], a 1:1 reflection of the
//!   interface that every *non*-in-process transport marshals (CBOR, `wire_version`-governed; see
//!   `daemon-api.cddl`);
//! - the shared **[`dispatch`]** — decode a request, call the interface, encode the response. The
//!   in-process transport skips the mirror and calls the trait directly; the socket and the FFI run
//!   the *same* `dispatch`, differing only in how the bytes arrive.
//!
//! It is a pure contracts crate (no runtime, no substrate): it depends only on `daemon-common` and
//! `daemon-protocol`, and uses decoupled DTOs ([`HealthReport`], [`StatsReport`], [`SessionInfo`],
//! [`FleetReport`]) so the surface never drags the substrate's concrete types into the contract.

#![forbid(unsafe_code)]

use async_trait::async_trait;
use daemon_common::{SessionId, UnitId, UsageDelta, WireVersion};
pub use daemon_protocol::Outbound;
use daemon_protocol::{AgentCommand, HostResponse, TranscriptBlock};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// The wire version of the api mirror (rides every framed request/response; governs evolution).
pub const API_WIRE_VERSION: WireVersion = WireVersion::CURRENT;

// ---------------------------------------------------------------------------
// The interface (two sub-surfaces, one node surface)
// ---------------------------------------------------------------------------

/// The §17 per-session surface: drive one interactive engine session.
#[async_trait]
pub trait SessionApi: Send + Sync {
    /// Submit a §17 command to `session` (opening it on the first `StartTurn`).
    async fn submit(&self, session: SessionId, command: AgentCommand) -> Result<(), ApiError>;

    /// Drain up to `max` outbound items (events + raised host requests) for `session`.
    async fn poll(&self, session: SessionId, max: u32) -> Result<Vec<Outbound>, ApiError>;

    /// Answer a host request the session raised (matched by `response.request_id`).
    async fn respond(&self, session: SessionId, response: HostResponse) -> Result<(), ApiError>;

    /// Non-destructive, cursor-paged read of a session's **durable** verifiable history (decoded
    /// finished chat blocks + lifecycle records, with a per-entry `verified` flag) for reconnect /
    /// scroll-back. Complements [`Self::poll`] (which destructively drains the *live* delta stream):
    /// repeated reads from the same `after_cursor` return the same page. Default: an empty page (a
    /// transport with no durable journal, e.g. an ephemeral session-only FFI).
    async fn session_history(
        &self,
        _session: SessionId,
        _after_cursor: u64,
        _max: u32,
    ) -> JournalPageView {
        JournalPageView::default()
    }
}

/// The node/operator control surface: inspect and steer the running node.
#[async_trait]
pub trait ControlApi: Send + Sync {
    /// The resident-service tree health.
    async fn health(&self) -> HealthReport;

    /// Durable queue depths + session/active counts.
    async fn stats(&self) -> StatsReport;

    /// The known durable sessions and their statuses.
    async fn sessions(&self) -> Vec<SessionInfo>;

    /// Ensure a durable session exists and wake it (start/resume work).
    async fn assign(&self, session: SessionId) -> Result<(), ApiError>;

    /// Cancel in-flight work for a session.
    async fn cancel(&self, session: SessionId) -> Result<(), ApiError>;

    /// The orchestration fleet roster + folded usage.
    async fn fleet(&self) -> FleetReport;

    /// The orchestration tree as the GUI/TUI drives it: every unit (single agent through
    /// fleets-of-fleets) with its parent/child structure, state, work, and folded usage. The default
    /// is an empty tree (a transport with no fleet projection, e.g. the session-only FFI).
    async fn tree(&self) -> TreeReport {
        TreeReport::default()
    }

    /// One unit's node view (`None` if unknown). Default: not available.
    async fn unit(&self, _id: UnitId) -> Option<UnitNode> {
        None
    }

    /// Drain up to `max` recent management events for one unit (GUI drill-down). Default: empty.
    async fn unit_events(&self, _id: UnitId, _max: u32) -> Vec<ManageEventView> {
        Vec::new()
    }

    /// Drain up to `max` recent §17 [`Outbound`] items (streamed events + raised host requests) for
    /// one unit — the rich, transcript-fidelity drill-down a GUI reads to render a full transcript
    /// for *any* unit in the tree (not just a top-level interactive session). The coarse
    /// [`Self::unit_events`] is the fleet-dashboard view; this is the drill-down-to-transcript view,
    /// carrying the full §17 vocabulary (text, reasoning, tool I/O with opaque structured `detail`,
    /// opaque `ContentDelta`, usage, errors) plus blocking host requests, untouched. A destructive
    /// drain like [`Self::poll`] (each call consumes what it returns; `max == 0` drains all).
    /// Default: empty (a transport with no fleet projection).
    async fn unit_outbound(&self, _id: UnitId, _max: u32) -> Vec<Outbound> {
        Vec::new()
    }

    /// Non-destructive, cursor-paged read of *any* unit's **durable** verifiable history (decoded
    /// finished chat blocks + management lifecycle records, each carrying the `verified` flag of its
    /// sealed segment) — the reconnect / scroll-back read for a GUI rendering a transcript for any
    /// node in the tree, and the auditor's one-chain verify pass. Complements (does not replace) the
    /// destructive live [`Self::unit_outbound`] drain. Default: an empty page.
    async fn unit_history(&self, _id: UnitId, _after_cursor: u64, _max: u32) -> JournalPageView {
        JournalPageView::default()
    }

    /// Pause a unit (lifecycle `ManageCommand`). Default: unsupported.
    async fn pause(&self, _id: UnitId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("pause".into()))
    }

    /// Resume a unit. Default: unsupported.
    async fn resume(&self, _id: UnitId) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("resume".into()))
    }

    /// Scale a unit (an orchestrator sub-fleet) to `n` members. Default: unsupported.
    async fn scale(&self, _id: UnitId, _n: u32) -> Result<(), ApiError> {
        Err(ApiError::Unsupported("scale".into()))
    }

    /// The node's journal **verifying** key (hex-encoded dCBOR), so an auditor can independently
    /// verify the sealed segments returned by the history reads. `None` when the node exposes no
    /// journal signer. Default: `None`.
    async fn verifying_key(&self) -> Option<String> {
        None
    }
}

/// The whole node surface: the session sub-surface plus the control sub-surface.
pub trait NodeApi: SessionApi + ControlApi {}
impl<T: SessionApi + ControlApi> NodeApi for T {}

// ---------------------------------------------------------------------------
// Outbound drain item (§17 events + raised host requests share one queue)
// ---------------------------------------------------------------------------
//
// The drain item is the canonical `daemon_protocol::Outbound` union, re-exported above. Events and
// blocking host requests ride the **same** drain queue (daemon-ffi-spec §3.3), so a poll-based
// embedder (see [`SessionApi::poll`]) sees both in order.

// ---------------------------------------------------------------------------
// Report DTOs (decoupled from the substrate's concrete types)
// ---------------------------------------------------------------------------

/// The resident-service tree health (a projection of the host supervisor handle).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthReport {
    /// Whether every resident service is currently `Ok`.
    pub all_ok: bool,
    /// Per-service health.
    pub services: Vec<ServiceHealth>,
}

/// One resident service's health line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// The service name (e.g. `JobOutboxDispatcher`).
    pub name: String,
    /// Whether the service is currently healthy.
    pub ok: bool,
    /// How many times it has been restarted.
    pub restarts: u32,
    /// A human-readable detail when not `ok`.
    pub detail: Option<String>,
}

/// Durable queue depths and live counts (a projection of `StoreStats` + active sessions).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatsReport {
    /// Pending background jobs on the durable job outbox.
    pub pending_jobs: u64,
    /// Pending wake hints on the durable wake outbox.
    pub pending_wakes: u64,
    /// Total durable session records.
    pub sessions: u64,
    /// Currently-active (in-memory) incarnations.
    pub active: u64,
}

/// A durable session's identity + lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionInfo {
    /// The session id.
    pub session: SessionId,
    /// Its durable lifecycle state.
    pub state: SessionState,
}

/// A transport-stable mirror of the durable session lifecycle (decoupled from `daemon-store`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionState {
    /// A live incarnation is (or was) running.
    Active,
    /// Suspended awaiting a background job.
    Suspended {
        /// The job this session is waiting on.
        job_id: String,
    },
    /// A completion is recorded; resumable.
    Ready,
    /// Terminal.
    Completed,
    /// State could not be resolved.
    Unknown,
}

/// The orchestration fleet roster + folded usage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetReport {
    /// The ids of all registered children.
    pub children: Vec<UnitId>,
    /// The folded fleet usage total.
    pub usage: UsageDelta,
}

// `UnitKind`, `UnitState`, `UnitNode`, and `TreeReport` are defined in `daemon-protocol` (next to
// `Outbound`) so the management contract can carry the projection seam without depending on this
// surface crate; they are re-exported here so every transport and the cddl mirror are unchanged.
pub use daemon_protocol::{TreeReport, UnitKind, UnitNode, UnitState};

// `ManageEventView` is defined in `daemon-protocol` (so the `ManagedUnit` projection seam can carry
// it without a surface-crate edge) and re-exported here unchanged.
pub use daemon_protocol::ManageEventView;

// ---------------------------------------------------------------------------
// Verifiable journal read DTOs (the non-destructive reconnect / scroll-back surface)
// ---------------------------------------------------------------------------

/// The decoded payload of one journal entry: a coarse management lifecycle record or a coalesced
/// finished chat block (a `daemon-protocol` [`TranscriptBlock`], already decoded for the consumer —
/// the GUI renders it directly, an auditor reads one timeline). Streaming deltas never appear here;
/// they stay on the ephemeral live drains.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum JournalRecordPayload {
    /// A management lifecycle / credential-audit record (its human/structured detail).
    Management {
        /// The record detail.
        detail: String,
    },
    /// A finished chat block, decoded from the entry's opaque body.
    Block {
        /// The decoded transcript block.
        block: TranscriptBlock,
    },
}

/// One decoded + verified journal entry, as returned by a history read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalRecord {
    /// The stream-monotonic pagination cursor (pass as the next `after_cursor`).
    pub cursor: u64,
    /// The segment (turn / incarnation) this entry belongs to.
    pub segment: u64,
    /// Monotonic per-`(stream, segment)` sequence number.
    pub seq: u64,
    /// The incarnation epoch active when recorded (0 for non-durable / first turn).
    pub epoch: u64,
    /// The correlation trace context active when recorded.
    pub trace: u64,
    /// The entry's kind label (e.g. `"mgmt.started"`, `"block.message"`).
    pub kind: String,
    /// Milliseconds since the Unix epoch when the entry was recorded.
    pub timestamp_ms: u64,
    /// Whether the sealed segment carrying this entry verified end-to-end (root recomputed +
    /// signature checked + chain linked). `false` for an as-yet-unsealed (open) segment or when the
    /// node exposes no verifying key.
    pub verified: bool,
    /// The decoded entry payload.
    pub payload: JournalRecordPayload,
}

/// A page of a unit/session's verifiable journal: decoded entries past a cursor plus the pagination
/// cursors. Non-destructive — repeated reads from the same `after_cursor` return the same page.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JournalPageView {
    /// The decoded entries in cursor order.
    pub entries: Vec<JournalRecord>,
    /// The cursor to pass as `after_cursor` on the next read (the last entry's cursor, or the input
    /// `after_cursor` when the page is empty).
    pub next_cursor: u64,
    /// The highest cursor currently stored for the stream (how far a reader can scroll).
    pub head_cursor: u64,
}

// ---------------------------------------------------------------------------
// The serializable mirror (1:1 with the interface methods)
// ---------------------------------------------------------------------------

/// The serializable reflection of a call into the interface — what every non-in-process transport
/// marshals onto the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiRequest {
    /// [`SessionApi::submit`].
    Submit {
        /// Target session.
        session: SessionId,
        /// The §17 command.
        command: AgentCommand,
    },
    /// [`SessionApi::poll`].
    Poll {
        /// Target session.
        session: SessionId,
        /// Maximum items to drain.
        max: u32,
    },
    /// [`SessionApi::respond`].
    Respond {
        /// Target session.
        session: SessionId,
        /// The correlated host response.
        response: HostResponse,
    },
    /// [`ControlApi::health`].
    Health,
    /// [`ControlApi::stats`].
    Stats,
    /// [`ControlApi::sessions`].
    Sessions,
    /// [`ControlApi::assign`].
    Assign {
        /// Session to assign/wake.
        session: SessionId,
    },
    /// [`ControlApi::cancel`].
    Cancel {
        /// Session to cancel.
        session: SessionId,
    },
    /// [`ControlApi::fleet`].
    Fleet,
    /// [`ControlApi::tree`].
    Tree,
    /// [`ControlApi::unit`].
    Unit {
        /// The unit to view.
        unit: UnitId,
    },
    /// [`ControlApi::unit_events`].
    UnitEvents {
        /// The unit to drain events for.
        unit: UnitId,
        /// Maximum events to drain.
        max: u32,
    },
    /// [`ControlApi::unit_outbound`].
    UnitOutbound {
        /// The unit to drain §17 outbound items for.
        unit: UnitId,
        /// Maximum items to drain.
        max: u32,
    },
    /// [`SessionApi::session_history`].
    SessionHistory {
        /// The session whose durable history to read.
        session: SessionId,
        /// The exclusive lower-bound cursor (0 from the start).
        after_cursor: u64,
        /// Maximum entries to return (0 = all available).
        max: u32,
    },
    /// [`ControlApi::unit_history`].
    UnitHistory {
        /// The unit whose durable history to read.
        unit: UnitId,
        /// The exclusive lower-bound cursor (0 from the start).
        after_cursor: u64,
        /// Maximum entries to return (0 = all available).
        max: u32,
    },
    /// [`ControlApi::pause`].
    Pause {
        /// The unit to pause.
        unit: UnitId,
    },
    /// [`ControlApi::resume`].
    Resume {
        /// The unit to resume.
        unit: UnitId,
    },
    /// [`ControlApi::scale`].
    Scale {
        /// The unit (sub-fleet) to scale.
        unit: UnitId,
        /// The target member count.
        n: u32,
    },
    /// [`ControlApi::verifying_key`].
    VerifyingKey,
}

/// The serializable reflection of an interface result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiResponse {
    /// A successful unit reply (submit/respond/assign/cancel).
    Ok,
    /// Drained outbound items (poll).
    Drained(Vec<Outbound>),
    /// A health report.
    Health(HealthReport),
    /// A stats report.
    Stats(StatsReport),
    /// A session list.
    Sessions(Vec<SessionInfo>),
    /// A fleet report.
    Fleet(FleetReport),
    /// A tree report.
    Tree(TreeReport),
    /// One unit's node view (`None` rendered as the absent variant).
    Unit(Option<UnitNode>),
    /// Drained per-unit management events.
    UnitEvents(Vec<ManageEventView>),
    /// A page of decoded + verified journal history (session/unit history).
    Journal(JournalPageView),
    /// The node's journal verifying key (hex dCBOR), or `None` if it exposes no signer.
    VerifyingKey(Option<String>),
    /// A failure (the interface's `ApiError`, round-tripped faithfully).
    Error(ApiError),
}

/// Why an api call failed (serializable so it round-trips over any transport).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
pub enum ApiError {
    /// No such live/durable session.
    #[error("unknown session: {0}")]
    UnknownSession(String),
    /// The operation is not available (e.g. a control op over a session-only FFI transport, or an
    /// unsupported §17 command).
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// The session is already owned by the *other* lifecycle: a `SessionId` is durable-managed
    /// (control surface, `assign`) **or** live-interactive (session surface, `submit`), never both.
    #[error("conflict: {0}")]
    Conflict(String),
    /// Any other failure.
    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// Dispatch — the shared core every non-in-process transport calls
// ---------------------------------------------------------------------------

fn unit_or_err(r: Result<(), ApiError>) -> ApiResponse {
    match r {
        Ok(()) => ApiResponse::Ok,
        Err(e) => ApiResponse::Error(e),
    }
}

async fn serve_session(api: &dyn SessionApi, req: ApiRequest) -> Option<ApiResponse> {
    Some(match req {
        ApiRequest::Submit { session, command } => unit_or_err(api.submit(session, command).await),
        ApiRequest::Poll { session, max } => match api.poll(session, max).await {
            Ok(items) => ApiResponse::Drained(items),
            Err(e) => ApiResponse::Error(e),
        },
        ApiRequest::Respond { session, response } => {
            unit_or_err(api.respond(session, response).await)
        }
        ApiRequest::SessionHistory {
            session,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.session_history(session, after_cursor, max).await),
        _ => return None,
    })
}

/// Dispatch a request against a full [`NodeApi`] — the entry point the socket/TCP/JSON-RPC node
/// transports call.
pub async fn dispatch(api: &dyn NodeApi, req: ApiRequest) -> ApiResponse {
    if let Some(resp) = serve_session(api, req.clone()).await {
        return resp;
    }
    match req {
        ApiRequest::Health => ApiResponse::Health(api.health().await),
        ApiRequest::Stats => ApiResponse::Stats(api.stats().await),
        ApiRequest::Sessions => ApiResponse::Sessions(api.sessions().await),
        ApiRequest::Assign { session } => unit_or_err(api.assign(session).await),
        ApiRequest::Cancel { session } => unit_or_err(api.cancel(session).await),
        ApiRequest::Fleet => ApiResponse::Fleet(api.fleet().await),
        ApiRequest::Tree => ApiResponse::Tree(api.tree().await),
        ApiRequest::Unit { unit } => ApiResponse::Unit(api.unit(unit).await),
        ApiRequest::UnitEvents { unit, max } => {
            ApiResponse::UnitEvents(api.unit_events(unit, max).await)
        }
        ApiRequest::UnitOutbound { unit, max } => {
            // Reuses the `Drained(Vec<Outbound>)` response — the same rich §17 drain shape as `poll`.
            ApiResponse::Drained(api.unit_outbound(unit, max).await)
        }
        ApiRequest::UnitHistory {
            unit,
            after_cursor,
            max,
        } => ApiResponse::Journal(api.unit_history(unit, after_cursor, max).await),
        ApiRequest::Pause { unit } => unit_or_err(api.pause(unit).await),
        ApiRequest::Resume { unit } => unit_or_err(api.resume(unit).await),
        ApiRequest::Scale { unit, n } => unit_or_err(api.scale(unit, n).await),
        ApiRequest::VerifyingKey => ApiResponse::VerifyingKey(api.verifying_key().await),
        // Session variants were handled by `serve_session`.
        ApiRequest::Submit { .. }
        | ApiRequest::Poll { .. }
        | ApiRequest::Respond { .. }
        | ApiRequest::SessionHistory { .. } => {
            unreachable!("session variants handled above")
        }
    }
}

/// Dispatch against a **session-only** surface — the entry point the `daemon-core-ffi` transport
/// calls. Control-surface requests resolve to [`ApiError::Unsupported`] (this transport is the §17
/// brain seam, not the node control plane).
pub async fn dispatch_session(api: &dyn SessionApi, req: ApiRequest) -> ApiResponse {
    match serve_session(api, req).await {
        Some(resp) => resp,
        None => ApiResponse::Error(ApiError::Unsupported(
            "control surface is not available on this transport".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// CBOR codec helpers (the one encoding shared by every framed transport)
// ---------------------------------------------------------------------------

/// CBOR-encode a mirror value (request or response).
pub fn to_cbor<T: Serialize>(value: &T) -> Vec<u8> {
    let mut buf = Vec::new();
    // Mirror types are always serializable; a failure here is a programming error.
    ciborium::into_writer(value, &mut buf).expect("encode api mirror as CBOR");
    buf
}

/// CBOR-decode a mirror value, mapping a decode failure to [`ApiError::Other`].
pub fn from_cbor<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ApiError> {
    ciborium::from_reader(bytes).map_err(|e| ApiError::Other(format!("CBOR decode: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::ReqId;
    use daemon_protocol::{AgentEvent, EndReason, TurnSummary, UserMsg};

    #[test]
    fn request_cbor_round_trips() {
        let reqs = vec![
            ApiRequest::Health,
            ApiRequest::Assign {
                session: SessionId::new("s1"),
            },
            ApiRequest::Submit {
                session: SessionId::new("s1"),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hi"),
                    request_id: ReqId(7),
                },
            },
            ApiRequest::Poll {
                session: SessionId::new("s1"),
                max: 16,
            },
        ];
        for req in reqs {
            let bytes = to_cbor(&req);
            let back: ApiRequest = from_cbor(&bytes).unwrap();
            assert_eq!(req, back);
        }
    }

    #[test]
    fn response_cbor_round_trips() {
        let resp = ApiResponse::Drained(vec![Outbound::Event(AgentEvent::TurnFinished {
            seq: 3,
            summary: TurnSummary::ended(EndReason::Completed),
        })]);
        let bytes = to_cbor(&resp);
        let back: ApiResponse = from_cbor(&bytes).unwrap();
        assert_eq!(resp, back);
    }
}
