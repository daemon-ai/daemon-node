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
use daemon_protocol::{AgentCommand, HostResponse};
pub use daemon_protocol::Outbound;
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

/// What kind of unit a tree node is (a transport-stable mirror of the supervision `UnitKind`). A
/// foreign agent and a `daemon-core` engine are both `Engine` — the GUI cannot, and need not, tell
/// them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitKind {
    /// A leaf brain (a `daemon-core` engine or a foreign agent over a §17 cut).
    Engine,
    /// A host running a unit.
    Host,
    /// An orchestrator running a sub-fleet.
    Orchestrator,
}

/// A tree node's lifecycle state (decoupled from the orchestration runtime's `ChildStatus`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitState {
    /// Attached, no terminal outcome yet (working or idle).
    Running,
    /// Reached a terminal outcome (`end_reason` is the supervision end reason, rendered).
    Finished {
        /// The terminal end reason (e.g. `Completed`, `Interrupted`, `Failed`).
        end_reason: String,
    },
    /// State could not be resolved.
    Unknown,
}

/// One node in the orchestration tree projection (the GUI's per-unit view). The tree is a flat node
/// list plus per-node `children` ids, so deeper / cross-node nesting can fill in later without a DTO
/// change.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitNode {
    /// The unit id.
    pub id: UnitId,
    /// What kind of unit this is.
    pub kind: UnitKind,
    /// Its lifecycle state.
    pub state: UnitState,
    /// A short description of the unit's current work, when known.
    pub work: Option<String>,
    /// The unit's folded usage.
    pub usage: UsageDelta,
    /// The ids of this unit's direct children.
    pub children: Vec<UnitId>,
}

/// The orchestration tree as the GUI/TUI sees it: a flat node list rooted at `root`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TreeReport {
    /// The root unit id (the node itself), when there is one.
    pub root: Option<UnitId>,
    /// Every node in the tree.
    pub nodes: Vec<UnitNode>,
}

/// A transport-stable projection of a unit's management event, for GUI drill-down (decoupled from
/// the supervision `ManageEvent`). Mirrors the per-session poll model: a bounded drain of recent
/// events for one unit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ManageEventView {
    /// The unit started a unit of work.
    Started {
        /// Monotonic per-unit sequence.
        seq: u64,
    },
    /// Streamed progress (text/reasoning/tool activity rendered to a line).
    Progress {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// A rendered progress line, when textual.
        text: Option<String>,
    },
    /// A usage delta the unit reported.
    Usage {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// The reported delta.
        delta: UsageDelta,
    },
    /// The unit reached a terminal outcome.
    Finished {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// The terminal end reason, rendered.
        end_reason: String,
        /// A final summary, when present.
        summary: Option<String>,
    },
    /// The unit raised an error.
    Error {
        /// Monotonic per-unit sequence.
        seq: u64,
        /// A rendered error message.
        message: String,
    },
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
        ApiRequest::Respond { session, response } => unit_or_err(api.respond(session, response).await),
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
        ApiRequest::Pause { unit } => unit_or_err(api.pause(unit).await),
        ApiRequest::Resume { unit } => unit_or_err(api.resume(unit).await),
        ApiRequest::Scale { unit, n } => unit_or_err(api.scale(unit, n).await),
        // Session variants were handled by `serve_session`.
        ApiRequest::Submit { .. } | ApiRequest::Poll { .. } | ApiRequest::Respond { .. } => {
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
