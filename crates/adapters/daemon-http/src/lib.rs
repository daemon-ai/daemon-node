//! `daemon-http` — the optional in-process HTTP/WS transport adapter for the [`daemon_api`] surface.
//!
//! Like the Unix socket and the C FFI, this is a thin shell over the shared
//! [`daemon_api::dispatch`]: the node speaks one canonical interface, and *how* a client reaches it
//! is a transport detail. This adapter adds two HTTP-native capabilities on top of the one-shot
//! request/response surface:
//!
//! - **JSON dispatch** (`POST /api`) — a JSON [`ApiRequest`] in, a JSON [`ApiResponse`] out, run
//!   through the exact same `dispatch` every transport calls (JSON on the HTTP surface, per the
//!   event-io spec's §7 decision; CBOR stays the socket/FFI encoding).
//! - **Live merged-log streaming** (`GET …/subscribe` over SSE, `…/ws` over WebSocket) — a push
//!   delivery over the non-destructive merged session event log ([`daemon_api::SessionApi::subscribe`]).
//!   The one-shot/long-poll cursor read (`GET …/log`, the wire `Subscribe` op) is also exposed for
//!   lowest-common-denominator clients.
//!
//! Streaming is a *transport capability* here, not a new protocol variant: the socket/FFI long-poll
//! the same cursor (`log_after`); this adapter holds the connection open and pushes frames backed by
//! the live actor's broadcast. It is isolated to this crate so axum / tower-http never leak into
//! `daemon-host` or `daemon-core`; the binary toggles it on at launch like the MCP surface.

#![forbid(unsafe_code)]

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use daemon_api::{
    dispatch, ApiRequest, ApiResponse, FsRootId, LogFilter, LogLineStream, LogStream, NodeApi,
    SessionLogEntry, TreeStream, TreeSubFilter,
};
use daemon_common::SessionId;
use daemon_delivery::{serve_delivery, Projector};
use daemon_protocol::{AgentCommand, Origin, OriginScope, TransportId};
use daemon_telemetry::{fields, ingress_trace, with_trace_span, SpanKind};
use futures::{Stream, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

/// The shared adapter state: the one node surface every request reaches.
#[derive(Clone)]
struct AppState {
    api: Arc<dyn NodeApi>,
}

/// Query params for the cursor reads (`?after_seq=N&max=M`). Both default to 0 (from the start; all).
#[derive(Debug, Default, Deserialize)]
struct CursorQuery {
    #[serde(default)]
    after_seq: u64,
    #[serde(default)]
    max: u32,
}

/// Build the adapter [`Router`] over a node surface (handy for embedding or testing without binding a
/// socket). Routes:
/// - `POST /api` — JSON [`ApiRequest`] → JSON [`ApiResponse`] via [`dispatch`].
/// - `GET /sessions/{session}/log` — non-destructive cursor page of the merged log.
/// - `GET /sessions/{session}/subscribe` — Server-Sent Events stream of the merged log.
/// - `GET /sessions/{session}/ws` — WebSocket stream of the merged log.
/// - `GET /tree/subscribe` — Server-Sent Events stream of orchestration-tree events
///   ([`daemon_api::TreeEvent`]), churn-filtered by `?include_ephemeral=&coalesce_ms=`.
/// - `GET /logs` — Server-Sent Events stream of node log lines, filtered by `?min_level=&target=`.
/// - `POST /tenants/{tenant}/submit` — a routed submit: the adapter maps the path tenant onto an
///   `Origin` (`transport: http/{tenant}`, `scope: Api{ key: tenant }`) and lets the host's §5.9
///   routing registry pick the session + profile + delivery. Demonstrates an external transport
///   routing by its own principal without deriving the `SessionId` itself.
/// - `GET /tenants/{tenant}/delivery` — the *outbound* counterpart (§5.9.3): on (re)connect the
///   adapter enumerates the `http/{tenant}` instance's owned sessions via `daemon-delivery` and
///   multiplexes their merged-log subscriptions into one SSE stream. This is the reconnect-safe pull
///   path — a reconnecting tenant rediscovers and resumes every session it owns without tracking ids.
pub fn router(api: Arc<dyn NodeApi>) -> Router {
    Router::new()
        .route("/api", post(api_dispatch))
        .route("/tenants/{tenant}/submit", post(submit_routed_tenant))
        .route("/tenants/{tenant}/delivery", get(tenant_delivery_sse))
        .route("/sessions/{session}/log", get(log_after))
        .route("/sessions/{session}/subscribe", get(subscribe_sse))
        .route("/sessions/{session}/ws", get(subscribe_ws))
        .route("/tree/subscribe", get(tree_subscribe_sse))
        .route("/fs/watch", get(fs_watch_sse))
        .route("/logs", get(logs_sse))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(AppState { api })
}

/// Serve the adapter over a bound [`tokio::net::TcpListener`] until it errors. Spawn it as a
/// background task alongside the other transports after the host starts.
pub async fn serve_http(
    listener: tokio::net::TcpListener,
    api: Arc<dyn NodeApi>,
) -> std::io::Result<()> {
    axum::serve(listener, router(api)).await
}

/// `POST /api` — the JSON reflection of [`dispatch`]: decode an [`ApiRequest`], call the interface,
/// encode the [`ApiResponse`].
async fn api_dispatch(
    State(state): State<AppState>,
    Json(req): Json<ApiRequest>,
) -> Json<ApiResponse> {
    let trace = ingress_trace(None);
    let response = with_trace_span(
        trace,
        fields::span::API_HTTP_REQUEST,
        SpanKind::Boundary,
        async {
            tracing::debug!(
                trace_id = %trace,
                api_variant = ?std::mem::discriminant(&req),
                event = fields::event::API_REQUEST,
                "api request received over http"
            );
            dispatch(state.api.as_ref(), req).await
        },
    )
    .await;
    Json(response)
}

/// `POST /tenants/{tenant}/submit` — a routed submit. The adapter derives the `Origin` from its own
/// principal (the path `tenant` as an `Api` key on the `http/{tenant}` transport instance) and hands
/// it to [`daemon_api::SessionApi::submit_routed`]; the host resolves the session + profile +
/// delivery. Returns [`ApiResponse::Routed`] with the derived session, so the client can then open
/// `…/subscribe` on the same surface to read the reply.
async fn submit_routed_tenant(
    State(state): State<AppState>,
    Path(tenant): Path<String>,
    Json(command): Json<AgentCommand>,
) -> Json<ApiResponse> {
    let trace = ingress_trace(None);
    let response = with_trace_span(
        trace,
        fields::span::API_HTTP_REQUEST,
        SpanKind::Boundary,
        async {
            let origin = Origin::new(
                TransportId::new(format!("http/{tenant}")),
                OriginScope::Api { key: tenant },
            );
            tracing::debug!(
                trace_id = %trace,
                event = fields::event::API_REQUEST,
                operation = "submit_routed",
                "tenant routed submit received over http"
            );
            match state.api.submit_routed(origin, command).await {
                Ok(session) => ApiResponse::Routed { session },
                Err(e) => ApiResponse::Error(e),
            }
        },
    )
    .await;
    Json(response)
}

/// A [`Projector`] that forwards every projected `(session, entry)` into an mpsc channel — the
/// fan-in side of the tenant SSE multiplex. Projection policy here is the trivial "emit each merged
/// entry as-is"; a real transport would render/coalesce instead.
struct ChannelProjector {
    tx: mpsc::Sender<(SessionId, SessionLogEntry)>,
}

#[async_trait::async_trait]
impl Projector for ChannelProjector {
    async fn project(&self, session: SessionId, entry: SessionLogEntry) {
        // A closed receiver (client disconnected) just drops the entry; the subscription is torn down
        // when the response stream — which owns it — is dropped.
        let _ = self.tx.send((session, entry)).await;
    }
}

/// `GET /tenants/{tenant}/delivery` — the reconnect-safe outbound pull path (§5.9.3). Uses
/// [`serve_delivery`] to enumerate the `http/{tenant}` instance's owned sessions, subscribe each
/// merged log, and multiplex them into a single SSE stream (one event per `(session, entry)`). The
/// returned [`daemon_delivery::DeliverySubscription`] is owned by the response stream, so it lives
/// exactly as long as the client stays connected and each session falls off on handover demotion.
async fn tenant_delivery_sse(
    State(state): State<AppState>,
    Path(tenant): Path<String>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let transport = TransportId::new(format!("http/{tenant}"));
    let (tx, rx) = mpsc::channel(256);
    let projector = Arc::new(ChannelProjector { tx });
    let subscription = serve_delivery(state.api.clone(), transport, projector).await;
    // Carry the subscription in the stream state so it is dropped (aborting the per-session tasks)
    // only when the SSE response is dropped — i.e. when the tenant disconnects.
    let events = futures::stream::unfold(
        (ReceiverStream::new(rx), subscription),
        |(mut rx, subscription)| async move {
            rx.next()
                .await
                .map(|(session, entry)| ((session, entry), (rx, subscription)))
        },
    )
    .map(|(session, entry)| {
        let event = Event::default()
            .event("delivery")
            .id(session.as_str())
            .json_data(&entry)
            .unwrap_or_else(|_| Event::default().data("serialize error"));
        Ok::<_, Infallible>(event)
    });
    Sse::new(events).keep_alive(KeepAlive::default())
}

/// `GET /sessions/{session}/log` — the one-shot/long-poll cursor read of the merged session event
/// log (the HTTP form of the wire `Subscribe` op). Returns an [`ApiResponse`] (`LogPage` or `Error`)
/// so the JSON shape matches `POST /api`.
async fn log_after(
    State(state): State<AppState>,
    Path(session): Path<String>,
    Query(q): Query<CursorQuery>,
) -> Json<ApiResponse> {
    let session = SessionId::new(session);
    match state.api.log_after(session, q.after_seq, q.max).await {
        Ok(page) => Json(ApiResponse::LogPage(page)),
        Err(e) => Json(ApiResponse::Error(e)),
    }
}

/// `GET /sessions/{session}/subscribe` — a Server-Sent Events stream of merged-log entries with
/// `seq > after_seq` (backfilled from history, then live).
async fn subscribe_sse(
    State(state): State<AppState>,
    Path(session): Path<String>,
    Query(q): Query<CursorQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let log = open_log(&state, session, q.after_seq).await;
    let events = log.map(|entry| {
        let event = Event::default()
            .json_data(&entry)
            .unwrap_or_else(|_| Event::default().data("serialize error"));
        Ok::<_, Infallible>(event)
    });
    Sse::new(events).keep_alive(KeepAlive::default())
}

/// `GET /sessions/{session}/ws` — a WebSocket stream of merged-log entries (one JSON text frame per
/// entry).
async fn subscribe_ws(
    State(state): State<AppState>,
    Path(session): Path<String>,
    Query(q): Query<CursorQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| pump_ws(socket, state, session, q.after_seq))
}

async fn pump_ws(mut socket: WebSocket, state: AppState, session: String, after_seq: u64) {
    let mut log = open_log(&state, session, after_seq).await;
    while let Some(entry) = log.next().await {
        let Ok(text) = serde_json::to_string(&entry) else {
            continue;
        };
        if socket.send(Message::Text(text.into())).await.is_err() {
            break;
        }
    }
}

/// Open the merged-log push stream for a session, degrading to an empty stream on error (an unknown
/// session, or a transport with no live log).
async fn open_log(state: &AppState, session: String, after_seq: u64) -> LogStream {
    state
        .api
        .subscribe(SessionId::new(session), after_seq)
        .await
        .unwrap_or_else(|_| futures::stream::empty().boxed())
}

/// Query params for the tree-subscribe push stream (`?include_ephemeral=bool&coalesce_ms=N`). Both
/// optional; absent → the [`TreeSubFilter`] defaults (deliver every change, include ephemerals).
#[derive(Debug, Default, Deserialize)]
struct TreeSubQuery {
    #[serde(default)]
    include_ephemeral: Option<bool>,
    #[serde(default)]
    coalesce_ms: Option<u64>,
}

/// `GET /tree/subscribe` — a Server-Sent Events stream of [`daemon_api::TreeEvent`]s over the
/// orchestration tree, churn-filtered by the query params. Mirrors [`subscribe_sse`]: the push
/// delivery a streaming transport holds open (one-shot transports poll the wire `Tree` op instead).
async fn tree_subscribe_sse(
    State(state): State<AppState>,
    Query(q): Query<TreeSubQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let filter = TreeSubFilter {
        include_ephemeral: q.include_ephemeral.unwrap_or(true),
        coalesce_ms: q.coalesce_ms,
    };
    let stream = open_tree(&state, filter).await;
    let events = stream.map(|event| {
        let event = Event::default()
            .json_data(&event)
            .unwrap_or_else(|_| Event::default().data("serialize error"));
        Ok::<_, Infallible>(event)
    });
    Sse::new(events).keep_alive(KeepAlive::default())
}

/// Open the tree push stream, degrading to an empty stream when the node exposes no live tree.
async fn open_tree(state: &AppState, filter: TreeSubFilter) -> TreeStream {
    state
        .api
        .tree_subscribe(filter)
        .await
        .unwrap_or_else(|_| futures::stream::empty().boxed())
}

/// Query params for the filesystem change stream (`?root=workspace&dir=src&poll_ms=750`). `root` is
/// `workspace` (default), `session:<id>`, or `host:<id>`; `dir` is the root-relative directory.
#[derive(Debug, Deserialize)]
struct FsWatchQuery {
    #[serde(default = "default_fs_root")]
    root: String,
    #[serde(default)]
    dir: String,
    #[serde(default)]
    poll_ms: u64,
}

fn default_fs_root() -> String {
    "workspace".to_string()
}

/// Parse the `root` query param into an [`FsRootId`].
fn parse_fs_root(s: &str) -> FsRootId {
    if let Some(id) = s.strip_prefix("session:") {
        FsRootId::Session(SessionId::new(id))
    } else if let Some(id) = s.strip_prefix("host:") {
        FsRootId::Host(id.to_string())
    } else {
        FsRootId::Workspace
    }
}

/// `GET /fs/watch` — a Server-Sent Events stream of filesystem change events under a watched
/// directory (daemon-fs-surface-spec.md). Mirrors [`subscribe_sse`] as a transport capability: it
/// holds the connection open and polls the wire cursor (`fs_watch_after`) on an interval, pushing a
/// JSON array of [`daemon_api::FsChange`]s whenever the directory changes (one-shot clients poll the
/// `FsWatchPoll` op directly instead).
async fn fs_watch_sse(
    State(state): State<AppState>,
    Query(q): Query<FsWatchQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let root = parse_fs_root(&q.root);
    let dir = q.dir;
    let poll = std::time::Duration::from_millis(if q.poll_ms == 0 { 750 } else { q.poll_ms });
    let api = state.api.clone();
    let stream = futures::stream::unfold(0u64, move |after_seq| {
        let api = api.clone();
        let root = root.clone();
        let dir = dir.clone();
        async move {
            loop {
                tokio::time::sleep(poll).await;
                match api
                    .fs_watch_after(root.clone(), dir.clone(), after_seq, 0)
                    .await
                {
                    Ok(page) if !page.events.is_empty() => {
                        let event = Event::default()
                            .json_data(&page.events)
                            .unwrap_or_else(|_| Event::default().data("serialize error"));
                        return Some((Ok::<_, Infallible>(event), page.next_seq));
                    }
                    // Primed / no changes: keep polling on the same cursor.
                    Ok(_) => continue,
                    // The surface is unbound (no workspace) — end the stream.
                    Err(_) => return None,
                }
            }
        }
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Query params for the node log-tail stream (`?min_level=info&target=substr`). Both optional.
#[derive(Debug, Default, Deserialize)]
struct LogQuery {
    #[serde(default)]
    min_level: Option<String>,
    #[serde(default)]
    target: Option<String>,
}

/// `GET /logs` — a Server-Sent Events stream of node [`daemon_api::LogLine`]s (resident-service /
/// dashboard view), filtered by the query params. Mirrors [`subscribe_sse`].
async fn logs_sse(
    State(state): State<AppState>,
    Query(q): Query<LogQuery>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let filter = LogFilter {
        min_level: q.min_level,
        target: q.target,
    };
    let stream = open_logs(&state, filter).await;
    let events = stream.map(|line| {
        let event = Event::default()
            .json_data(&line)
            .unwrap_or_else(|_| Event::default().data("serialize error"));
        Ok::<_, Infallible>(event)
    });
    Sse::new(events).keep_alive(KeepAlive::default())
}

/// Open the node log-tail push stream, degrading to an empty stream when the node exposes no tail.
async fn open_logs(state: &AppState, filter: LogFilter) -> LogLineStream {
    state
        .api
        .logs(filter)
        .await
        .unwrap_or_else(|_| futures::stream::empty().boxed())
}

/// Convenience: the response a 404 handler would give (kept so a caller embedding the [`router`] can
/// compose a fallback).
pub fn not_found() -> Response {
    (axum::http::StatusCode::NOT_FOUND, "not found").into_response()
}
