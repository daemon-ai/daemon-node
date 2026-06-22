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
use daemon_api::{dispatch, ApiRequest, ApiResponse, LogStream, NodeApi, SessionLogEntry};
use daemon_common::SessionId;
use daemon_delivery::{serve_delivery, Projector};
use daemon_protocol::{AgentCommand, Origin, OriginScope, TransportId};
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
    Json(dispatch(state.api.as_ref(), req).await)
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
    let origin = Origin::new(
        TransportId::new(format!("http/{tenant}")),
        OriginScope::Api { key: tenant },
    );
    match state.api.submit_routed(origin, command).await {
        Ok(session) => Json(ApiResponse::Routed { session }),
        Err(e) => Json(ApiResponse::Error(e)),
    }
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

/// Convenience: the response a 404 handler would give (kept so a caller embedding the [`router`] can
/// compose a fallback).
pub fn not_found() -> Response {
    (axum::http::StatusCode::NOT_FOUND, "not found").into_response()
}
