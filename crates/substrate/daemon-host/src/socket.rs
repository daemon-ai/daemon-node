// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The Unix-domain-socket transport adapter for the [`daemon_api`] surface.
//!
//! Two modes share one length-framed (4-byte big-endian length + CBOR payload) byte stream:
//!
//! * **Legacy one-shot** — a connection whose first frame decodes as a bare [`ApiRequest`] is served
//!   read-one / `dispatch` / write-one, exactly as before. The C FFI and operator CLI use this.
//! * **Multiplexed** ([`WireC2S`]/[`WireS2C`], daemon-sync-protocol-spec.md §2) — a connection that
//!   opens with a `Hello` (or any `WireC2S`) frame carries many correlated exchanges: each `Call`
//!   spawns its own task, a single writer task multiplexes `Reply`/`Item`/`End` frames back, and
//!   streaming requests (`Subscribe`) push `Item`s until `End`/`Cancel`. The disjoint tag sets make
//!   the first-frame mode select unambiguous.

use crate::auth_audit::AuthAudit;
use crate::authn::{AuthExchange, AuthSuccess, Authenticator, BeginOutcome, StepOutcome, TlsState};
use crate::authz::authorize;
use crate::request_context::{with_request_context, AuthMethod, RequestContext};
use daemon_api::{
    dispatch, from_cbor, is_streaming, to_cbor, wire_feature_api, ApiError, ApiRequest,
    ApiResponse, EventsPage, LogPageView, LogStreamItem, NodeApi, WireC2S, WireS2C,
    WIRE_FEATURE_AUTH, WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM, WIRE_FEATURE_VERSIONING,
    WIRE_VERSION,
};
use daemon_auth::Principal;
use daemon_telemetry::{fields, ingress_trace, with_trace_span, SpanKind};
use futures::StreamExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
// tokio has no AF_UNIX support on windows; the unix-socket entry points below are unix-only, while
// the shared mux/legacy loops stay portable (they serve the TLS/WS carriers on every platform).
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

/// Bound on buffered server -> client frames per connection before the writer applies backpressure.
pub(crate) const WRITER_QUEUE: usize = 256;
/// Idle keepalive cadence on a live subscription so a silently dead socket is noticed without the
/// connection-level Health probe. An empty `LogPage` doubles as the keepalive.
const STREAM_KEEPALIVE: Duration = Duration::from_secs(20);

/// Server-assigned per-connection id (audit/telemetry correlation; threaded into the request
/// context as `conn_id`).
static NEXT_CONN_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_conn_id() -> u64 {
    NEXT_CONN_ID.fetch_add(1, Ordering::Relaxed)
}

/// How a connection establishes the [`RequestContext`] every dispatch runs under.
pub(crate) enum AuthMode {
    /// Local trust ([`api].local_trust`): bind [`RequestContext::system`] (full-trust admin) for
    /// every request, advertise NO mechanisms, and run no SASL exchange. The Unix socket / Windows
    /// named pipe / FFI / CLI default — the deliberate, audited unauthenticated-but-trusted path.
    // The unix-socket and windows named-pipe entry points construct this; the TLS/WS carriers are
    // always Required.
    LocalSystem,
    /// Require a completed SASL exchange before any `Call`/`Open`. Advertises the authenticator's
    /// permitted mechanisms for the transport; binds the authenticated principal post-`AuthOk`.
    /// The only mode used on TCP/TLS (and on the Unix socket when `local_trust` is disabled).
    Required {
        /// The SASL authenticator driving the handshake.
        auth: Arc<Authenticator>,
        /// Transport security facts (gates PLAIN/EXTERNAL, carries the client-cert fingerprint).
        tls_state: TlsState,
    },
}

impl AuthMode {
    fn is_local_system(&self) -> bool {
        matches!(self, AuthMode::LocalSystem)
    }
}

/// Serve the node surface over a bound [`UnixListener`] until it errors, under **local trust**: the
/// Unix socket is the deployment-trusted local path, so every request runs as
/// [`RequestContext::system`] and no SASL exchange is offered. This is the default + the
/// FFI/CLI/conformance entry point. For an authenticated Unix socket (operator disabled
/// `local_trust`) use [`serve_api_unix_authenticated`]. Runs forever; spawn it as a background task.
#[cfg(unix)]
pub async fn serve_api_unix(listener: UnixListener, api: Arc<dyn NodeApi>) {
    accept_unix(listener, api, Arc::new(AuthMode::LocalSystem)).await;
}

/// As [`serve_api_unix`], but the Unix socket **requires** a SASL exchange (no local trust): the
/// connection must authenticate via the [`Authenticator`] before any `Call`/`Open`. Used when
/// `[api].local_trust` is disabled.
#[cfg(unix)]
pub async fn serve_api_unix_authenticated(
    listener: UnixListener,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
) {
    let mode = Arc::new(AuthMode::Required {
        auth,
        tls_state: TlsState::plaintext(),
    });
    accept_unix(listener, api, mode).await;
}

#[cfg(unix)]
async fn accept_unix(listener: UnixListener, api: Arc<dyn NodeApi>, mode: Arc<AuthMode>) {
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let api = api.clone();
                let mode = mode.clone();
                tokio::spawn(async move {
                    if let Err(e) = serve_conn(stream, api, mode).await {
                        tracing::debug!("api socket connection ended: {e}");
                    }
                });
            }
            Err(e) => {
                tracing::warn!("api socket accept failed: {e}");
                return;
            }
        }
    }
}

#[cfg(unix)]
async fn serve_conn(
    stream: UnixStream,
    api: Arc<dyn NodeApi>,
    mode: Arc<AuthMode>,
) -> std::io::Result<()> {
    let (rd, wr) = stream.into_split();
    serve_conn_split(rd, wr, api, mode, next_conn_id()).await
}

/// First-frame mode-select over any split byte stream — the shared per-connection entry point for
/// the Unix socket and the Windows named pipe (both are local-trust carriers of the bare + mux
/// protocols). A multiplexed client opens with a `WireC2S` frame; a legacy client sends a bare
/// `ApiRequest`, whose externally-tagged variants are disjoint from `WireC2S`
/// (`Hello`/`Call`/`Cancel`), so it never decodes as one.
#[cfg(any(unix, windows))]
async fn serve_conn_split<R, W>(
    mut rd: R,
    wr: W,
    api: Arc<dyn NodeApi>,
    mode: Arc<AuthMode>,
    conn_id: u64,
) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let first = match read_frame(&mut rd).await? {
        Some(bytes) => bytes,
        None => return Ok(()),
    };
    match from_cbor::<WireC2S>(&first) {
        Ok(frame) => serve_mux(rd, wr, api, mode, Some(frame), conn_id).await,
        Err(_) => serve_legacy(rd, wr, api, mode, first, conn_id).await,
    }
}

/// Serve the node surface over a Windows named pipe at `pipe_name` under **local trust** — the
/// windows analog of [`serve_api_unix`]: every request runs as [`RequestContext::system`] and no
/// SASL exchange is offered. This is the managed-daemon / FFI / CLI local path on windows (tokio
/// has no AF_UNIX there). For an authenticated pipe use [`serve_api_windows_pipe_authenticated`].
/// Runs forever; spawn it as a background task.
#[cfg(windows)]
pub async fn serve_api_windows_pipe(pipe_name: String, api: Arc<dyn NodeApi>) {
    accept_windows_pipe(pipe_name, api, Arc::new(AuthMode::LocalSystem)).await;
}

/// As [`serve_api_windows_pipe`], but the pipe **requires** a SASL exchange (no local trust): a
/// connection must authenticate via the [`Authenticator`] before any `Call`/`Open`. Used when
/// `[api].local_trust` is disabled.
#[cfg(windows)]
pub async fn serve_api_windows_pipe_authenticated(
    pipe_name: String,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
) {
    let mode = Arc::new(AuthMode::Required {
        auth,
        tls_state: TlsState::plaintext(),
    });
    accept_windows_pipe(pipe_name, api, mode).await;
}

/// The windows named-pipe accept loop (peer of [`accept_unix`]). Uses the standard tokio pattern:
/// the first instance claims the name, and each accepted client is handed to a per-connection task
/// while the next instance is created ahead of it, so there is never a window where a connecting
/// client finds no server listening.
#[cfg(windows)]
async fn accept_windows_pipe(pipe_name: String, api: Arc<dyn NodeApi>, mode: Arc<AuthMode>) {
    use tokio::net::windows::named_pipe::ServerOptions;
    let mut server = match ServerOptions::new()
        .first_pipe_instance(true)
        .create(&pipe_name)
    {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(pipe = %pipe_name, "failed to create named pipe: {e}");
            return;
        }
    };
    loop {
        if let Err(e) = server.connect().await {
            tracing::warn!(pipe = %pipe_name, "named pipe connect failed: {e}");
        }
        // Take the connected instance and stand up the next one before serving this one.
        let connected = server;
        server = match ServerOptions::new().create(&pipe_name) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(pipe = %pipe_name, "failed to recreate named pipe: {e}");
                return;
            }
        };
        let api = api.clone();
        let mode = mode.clone();
        tokio::spawn(async move {
            let conn_id = next_conn_id();
            let (rd, wr) = tokio::io::split(connected);
            if let Err(e) = serve_conn_split(rd, wr, api, mode, conn_id).await {
                tracing::debug!("named pipe connection ended: {e}");
            }
        });
    }
}

/// Legacy one-shot: bare `ApiRequest` -> `dispatch` -> bare `ApiResponse`, sequential per connection.
/// The bare protocol carries no SASL handshake, so it is served only under [`AuthMode::LocalSystem`]
/// (the FFI/CLI local-trust path); when auth is required this transport refuses with
/// [`ApiError::Unauthenticated`] (a networked client must use the multiplexed SASL path).
// Served by `serve_conn_split` over the local-trust carriers (Unix socket + Windows named pipe);
// the TLS/WS carriers are mux-only.
#[cfg(any(unix, windows))]
async fn serve_legacy<R, W>(
    mut rd: R,
    mut wr: W,
    api: Arc<dyn NodeApi>,
    mode: Arc<AuthMode>,
    first: Vec<u8>,
    conn_id: u64,
) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut bytes = first;
    loop {
        let trace = ingress_trace(None);
        let response = with_trace_span(
            trace,
            fields::span::API_UNIX_REQUEST,
            SpanKind::Boundary,
            async {
                match from_cbor::<ApiRequest>(&bytes) {
                    Ok(request) => {
                        tracing::debug!(
                            trace_id = %trace,
                            api_variant = ?std::mem::discriminant(&request),
                            event = fields::event::API_REQUEST,
                            "api request received over unix socket"
                        );
                        if mode.is_local_system() {
                            let ctx = RequestContext::system().with_conn_id(conn_id);
                            // Local-trust system principal holds every capability, so no denial can
                            // occur here; the bare one-shot path carries no audit sink.
                            authorize_and_dispatch(api.as_ref(), request, ctx, None).await
                        } else {
                            // No SASL handshake on the bare protocol: fail closed.
                            ApiResponse::Error(ApiError::Unauthenticated(
                                "authentication required; use the multiplexed SASL protocol".into(),
                            ))
                        }
                    }
                    Err(e) => ApiResponse::Error(e),
                }
            },
        )
        .await;
        write_frame(&mut wr, &to_cbor(&response)).await?;
        match read_frame(&mut rd).await? {
            Some(b) => bytes = b,
            None => return Ok(()),
        }
    }
}

/// The per-connection authentication state.
enum ConnAuth {
    /// No successful auth yet; `Call`/`Open` are refused with `Unauthenticated`.
    Unauthenticated,
    /// A multi-step mechanism (SCRAM) is mid-exchange; carries the method for the eventual success.
    InProgress(AuthExchange, AuthMethod),
    /// Authenticated (or local-trust): every dispatch binds this principal + method.
    Authenticated {
        principal: Principal,
        method: AuthMethod,
    },
}

/// Map a SASL mechanism name to its [`AuthMethod`] (audit/telemetry tag).
fn auth_method_for(mechanism: &str) -> AuthMethod {
    use crate::authn::{MECH_EXTERNAL, MECH_PLAIN, MECH_SCRAM_SHA_256};
    match mechanism {
        m if m == MECH_SCRAM_SHA_256 => AuthMethod::Scram,
        m if m == MECH_PLAIN => AuthMethod::Plain,
        m if m == MECH_EXTERNAL => AuthMethod::External,
        _ => AuthMethod::Scram,
    }
}

/// The stable audit label for an [`AuthMethod`] (how the principal proved its identity).
fn method_label(method: AuthMethod) -> &'static str {
    match method {
        AuthMethod::LocalTrust => "local_trust",
        AuthMethod::Scram => "scram",
        AuthMethod::Plain => "plain",
        AuthMethod::External => "external",
        AuthMethod::Token => "token",
    }
}

/// Run `req` through the capability gate then `dispatch`, all inside `ctx`'s task-local scope.
/// `tokio::spawn`ed tasks do NOT inherit the caller's task-local, so EVERY spawned per-`Call` task
/// re-establishes the scope here (the Auth 2 caveat); `current_principal` is otherwise `None`
/// (fail-closed) and the gate denies.
async fn authorize_and_dispatch(
    api: &dyn NodeApi,
    req: ApiRequest,
    ctx: RequestContext,
    audit: Option<Arc<AuthAudit>>,
) -> ApiResponse {
    let conn_id = ctx.conn_id;
    // Request-level dispatch log (both the mux per-`Call` tasks and the legacy path funnel
    // through here): the payload-free op tag only, never the body (it may carry a credential).
    if tracing::enabled!(tracing::Level::DEBUG) {
        tracing::debug!(
            op = %op_tag(&req),
            conn_id,
            event = fields::event::API_REQUEST,
            "api request dispatched"
        );
    }
    with_request_context(ctx, async {
        match authorize(&req) {
            Ok(()) => dispatch(api, req).await,
            Err(e) => {
                if let Some(a) = &audit {
                    // Payload-free op tag (NEVER the request body — it may carry a password).
                    a.permission_denied(&op_tag(&req), conn_id, &e.to_string())
                        .await;
                }
                ApiResponse::Error(e)
            }
        }
    })
    .await
}

/// The externally-tagged variant name of a request (the CBOR/JSON tag), with **no** payload — safe
/// to put in an audit record even when the request body carries a credential (e.g. `UserCreate`).
fn op_tag(req: &ApiRequest) -> String {
    match serde_json::to_value(req) {
        Ok(serde_json::Value::String(s)) => s,
        Ok(serde_json::Value::Object(map)) => map.keys().next().cloned().unwrap_or_default(),
        _ => String::new(),
    }
}

/// Deliver a completed authentication to the client: any trailing mechanism bytes (the SCRAM
/// server-final message) ride a final `AuthChallenge` before `AuthOk` (the frozen `AuthOk` carries
/// no mechanism bytes), then `AuthOk { token, principal }`. Returns the bound principal.
async fn complete_auth(
    tx: &mpsc::Sender<WireS2C>,
    final_data: Option<Vec<u8>>,
    success: Box<AuthSuccess>,
) -> Principal {
    if let Some(data) = final_data {
        let _ = tx.send(WireS2C::AuthChallenge { data }).await;
    }
    let AuthSuccess {
        principal,
        token,
        principal_view,
    } = *success;
    let _ = tx
        .send(WireS2C::AuthOk {
            token,
            principal: principal_view,
        })
        .await;
    principal
}

/// The capability strings a server `Hello` ack advertises: the always-on envelope features
/// (`mux`, `stream`) plus the API contract version (`api/<N>`, so a client can refuse or replace
/// a stale daemon whose contract it cannot decode), then `versioning` / `auth` when the node /
/// transport supports them. Pure, so the advertisement is unit-testable without a socket.
fn hello_features(supports_versioning: bool, auth_required: bool) -> Vec<String> {
    let mut features = vec![
        WIRE_FEATURE_MUX.to_string(),
        WIRE_FEATURE_STREAM.to_string(),
        wire_feature_api(),
    ];
    if supports_versioning {
        features.push(WIRE_FEATURE_VERSIONING.to_string());
    }
    if auth_required {
        features.push(WIRE_FEATURE_AUTH.to_string());
    }
    features
}

/// Multiplexed serve loop, generic over the byte stream halves so it backs both the Unix socket and
/// the TLS/TCP transport. Decodes each `WireC2S`, runs the SASL handshake (when [`AuthMode::Required`]),
/// and — only once authenticated (or under local trust) — spawns a task per `Call` that
/// re-establishes the request context, runs the capability gate, and dispatches. A single writer
/// task multiplexes `Reply`/`Item`/`End` frames back.
pub(crate) async fn serve_mux<R, W>(
    mut rd: R,
    wr: W,
    api: Arc<dyn NodeApi>,
    mode: Arc<AuthMode>,
    first: Option<WireC2S>,
    conn_id: u64,
) -> std::io::Result<()>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let (tx, mut rx) = mpsc::channel::<WireS2C>(WRITER_QUEUE);
    let writer = tokio::spawn(async move {
        let mut wr = wr;
        while let Some(frame) = rx.recv().await {
            if write_frame(&mut wr, &to_cbor(&frame)).await.is_err() {
                break;
            }
        }
    });

    // Local trust pre-binds the system principal; an auth-required transport starts unelevated.
    let mut conn = if mode.is_local_system() {
        ConnAuth::Authenticated {
            principal: RequestContext::system().principal,
            method: AuthMethod::LocalTrust,
        }
    } else {
        ConnAuth::Unauthenticated
    };

    // The shared auth-audit sink (login success/failure + permission denials), if the authenticator
    // carries one. `None` under local trust / a node without journaling -> audit is a no-op.
    let audit: Option<Arc<AuthAudit>> = match mode.as_ref() {
        AuthMode::Required { auth, .. } => auth.audit().cloned(),
        AuthMode::LocalSystem => None,
    };

    // Streaming `Open`s register an abort handle so a later `Cancel { id }` can tear them down.
    let mut streams: HashMap<u64, AbortHandle> = HashMap::new();
    let mut pending = first;
    loop {
        let frame = match pending.take() {
            Some(f) => f,
            None => match read_frame(&mut rd).await? {
                Some(bytes) => match from_cbor::<WireC2S>(&bytes) {
                    Ok(f) => f,
                    // A frame we cannot decode is dropped rather than killing the connection.
                    Err(_) => continue,
                },
                None => break,
            },
        };
        match frame {
            WireC2S::Hello { .. } => {
                let features = hello_features(api.supports_versioning(), !mode.is_local_system());
                // Advertise mechanisms only when an exchange is required; local trust offers none.
                let auth_mechanisms = match mode.as_ref() {
                    AuthMode::Required { auth, tls_state } => auth.advertised_mechanisms(tls_state),
                    AuthMode::LocalSystem => Vec::new(),
                };
                let _ = tx
                    .send(WireS2C::Hello {
                        wire_version: WIRE_VERSION,
                        features,
                        auth_mechanisms,
                    })
                    .await;
            }
            WireC2S::AuthStart { mechanism, initial } => match mode.as_ref() {
                AuthMode::Required { auth, tls_state } => {
                    if matches!(conn, ConnAuth::Authenticated { .. }) {
                        let _ = tx
                            .send(WireS2C::AuthError {
                                reason: "already authenticated".into(),
                            })
                            .await;
                    } else {
                        let method = auth_method_for(&mechanism);
                        conn = match auth.begin(&mechanism, &initial, tls_state.clone()) {
                            BeginOutcome::Challenge { data, exchange } => {
                                let _ = tx.send(WireS2C::AuthChallenge { data }).await;
                                ConnAuth::InProgress(exchange, method)
                            }
                            BeginOutcome::Success {
                                final_data,
                                success,
                            } => {
                                let principal = complete_auth(&tx, final_data, success).await;
                                auth.audit_login_ok(&principal.user_id, method_label(method))
                                    .await;
                                ConnAuth::Authenticated { principal, method }
                            }
                            BeginOutcome::Failed(reject) => {
                                auth.audit_login_fail(&mechanism, None).await;
                                let _ = tx
                                    .send(WireS2C::AuthError {
                                        reason: reject.reason,
                                    })
                                    .await;
                                ConnAuth::Unauthenticated
                            }
                        };
                    }
                }
                // Local trust offers no mechanisms; an auth frame here is a protocol error by the
                // client, refused without disturbing the (already trusted) connection.
                AuthMode::LocalSystem => {
                    let _ = tx
                        .send(WireS2C::AuthError {
                            reason: "authentication is not offered on this transport".into(),
                        })
                        .await;
                }
            },
            WireC2S::AuthStep { data } => {
                match std::mem::replace(&mut conn, ConnAuth::Unauthenticated) {
                    ConnAuth::InProgress(mut exchange, method) => {
                        conn = match exchange.step(&data) {
                            StepOutcome::Challenge(challenge) => {
                                let _ = tx.send(WireS2C::AuthChallenge { data: challenge }).await;
                                ConnAuth::InProgress(exchange, method)
                            }
                            StepOutcome::Success {
                                final_data,
                                success,
                            } => {
                                let principal = complete_auth(&tx, final_data, success).await;
                                if let Some(a) = &audit {
                                    a.login_ok(&principal.user_id, method_label(method)).await;
                                }
                                ConnAuth::Authenticated { principal, method }
                            }
                            StepOutcome::Failed(reject) => {
                                if let Some(a) = &audit {
                                    a.login_fail(method_label(method), None).await;
                                }
                                let _ = tx
                                    .send(WireS2C::AuthError {
                                        reason: reject.reason,
                                    })
                                    .await;
                                ConnAuth::Unauthenticated
                            }
                        };
                    }
                    // No exchange in progress: refuse, restoring the prior state.
                    other => {
                        conn = other;
                        let _ = tx
                            .send(WireS2C::AuthError {
                                reason: "no authentication in progress".into(),
                            })
                            .await;
                    }
                }
            }
            WireC2S::AuthResume { token } => match mode.as_ref() {
                AuthMode::Required { auth, .. } => {
                    if matches!(conn, ConnAuth::Authenticated { .. }) {
                        let _ = tx
                            .send(WireS2C::AuthError {
                                reason: "already authenticated".into(),
                            })
                            .await;
                    } else {
                        conn = match auth.resume(&token) {
                            BeginOutcome::Success {
                                final_data,
                                success,
                            } => {
                                let principal = complete_auth(&tx, final_data, success).await;
                                auth.audit_login_ok(
                                    &principal.user_id,
                                    method_label(AuthMethod::Token),
                                )
                                .await;
                                ConnAuth::Authenticated {
                                    principal,
                                    method: AuthMethod::Token,
                                }
                            }
                            BeginOutcome::Failed(reject) => {
                                auth.audit_login_fail("token", None).await;
                                let _ = tx
                                    .send(WireS2C::AuthError {
                                        reason: reject.reason,
                                    })
                                    .await;
                                ConnAuth::Unauthenticated
                            }
                            BeginOutcome::Challenge { .. } => ConnAuth::Unauthenticated,
                        };
                    }
                }
                AuthMode::LocalSystem => {
                    let _ = tx
                        .send(WireS2C::AuthError {
                            reason: "authentication is not offered on this transport".into(),
                        })
                        .await;
                }
            },
            WireC2S::Call { id, req } => {
                if let ConnAuth::Authenticated { principal, method } = &conn {
                    let api = api.clone();
                    let tx = tx.clone();
                    let audit = audit.clone();
                    let ctx = RequestContext::authenticated(principal.clone(), None)
                        .with_conn_id(conn_id)
                        .with_auth_method(*method);
                    // The spawned task does NOT inherit the task-local; `authorize_and_dispatch`
                    // re-enters the scope before the gate + dispatch (Auth 2 caveat).
                    tokio::spawn(async move {
                        let trace = ingress_trace(None);
                        let res = with_trace_span(
                            trace,
                            fields::span::API_UNIX_REQUEST,
                            SpanKind::Boundary,
                            authorize_and_dispatch(api.as_ref(), req, ctx, audit),
                        )
                        .await;
                        let _ = tx.send(WireS2C::Reply { id, res }).await;
                    });
                } else {
                    let _ = tx
                        .send(WireS2C::Reply {
                            id,
                            res: ApiResponse::Error(ApiError::Unauthenticated(
                                "authenticate before issuing requests".into(),
                            )),
                        })
                        .await;
                }
            }
            WireC2S::Open { id, req } => {
                if let ConnAuth::Authenticated { principal, method } = &conn {
                    if is_streaming(&req) {
                        // Gate the stream at open time under the request context (the pump itself
                        // does not re-dispatch). On allow, spawn the stream; else End{error}.
                        let ctx = RequestContext::authenticated(principal.clone(), None)
                            .with_conn_id(conn_id)
                            .with_auth_method(*method);
                        let allowed = with_request_context(ctx, async { authorize(&req) }).await;
                        match allowed {
                            Ok(()) => {
                                streams.insert(id, spawn_stream(api.clone(), tx.clone(), id, req));
                            }
                            Err(e) => {
                                if let Some(a) = &audit {
                                    a.permission_denied(
                                        &op_tag(&req),
                                        Some(conn_id),
                                        &e.to_string(),
                                    )
                                    .await;
                                }
                                let _ = tx.send(WireS2C::End { id, error: Some(e) }).await;
                            }
                        }
                    } else {
                        let _ = tx
                            .send(WireS2C::End {
                                id,
                                error: Some(ApiError::Unsupported(
                                    "request is not streamable; use Call".into(),
                                )),
                            })
                            .await;
                    }
                } else {
                    let _ = tx
                        .send(WireS2C::End {
                            id,
                            error: Some(ApiError::Unauthenticated(
                                "authenticate before opening a stream".into(),
                            )),
                        })
                        .await;
                }
            }
            WireC2S::Cancel { id } => {
                if let Some(handle) = streams.remove(&id) {
                    handle.abort();
                    let _ = tx.send(WireS2C::End { id, error: None }).await;
                }
            }
        }
    }
    drop(tx);
    let _ = writer.await;
    Ok(())
}

/// Pump a streaming `Subscribe` `Call`: backlog + live merged-log entries become `Item { LogPage }`
/// frames (one entry per page in L0), each stamped with the session-activation `epoch` (L2). A lossy
/// broadcast lag becomes a `Reset { epoch, head_seq }` so the client re-baselines. Idle ticks send an
/// empty keepalive page; the stream ends with `End`.
pub(crate) fn spawn_stream(
    api: Arc<dyn NodeApi>,
    tx: mpsc::Sender<WireS2C>,
    id: u64,
    req: ApiRequest,
) -> AbortHandle {
    let task = tokio::spawn(async move {
        match req {
            ApiRequest::Subscribe {
                session, after_seq, ..
            } => pump_session_log(api, tx, id, session, after_seq).await,
            ApiRequest::EventsSince { cursor, .. } => pump_node_events(api, tx, id, cursor).await,
            _ => {
                let _ = tx
                    .send(WireS2C::End {
                        id,
                        error: Some(ApiError::Unsupported(
                            "non-streaming request on the stream path".into(),
                        )),
                    })
                    .await;
            }
        }
    });
    task.abort_handle()
}

/// Pump a streaming `Subscribe`: backlog + live merged-log entries become `Item { LogPage }` frames
/// (one entry per page in L0), each stamped with the session-activation `epoch` (L2). A lossy
/// broadcast lag becomes a `Reset { epoch, head_seq }`; idle ticks send an empty keepalive page; the
/// stream ends with `End`.
async fn pump_session_log(
    api: Arc<dyn NodeApi>,
    tx: mpsc::Sender<WireS2C>,
    id: u64,
    session: daemon_common::SessionId,
    after_seq: u64,
) {
    // The activation epoch is constant for this log generation; read it once and stamp every
    // page + any Reset with it.
    let epoch = api.log_epoch(session.clone()).await;
    let mut stream = match api.subscribe(session, after_seq).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(WireS2C::End { id, error: Some(e) }).await;
            return;
        }
    };
    let mut keepalive = tokio::time::interval(STREAM_KEEPALIVE);
    keepalive.tick().await; // consume the immediate first tick
    let mut last_seq = after_seq;
    loop {
        tokio::select! {
            item = stream.next() => match item {
                Some(LogStreamItem::Entry(entry)) => {
                    last_seq = entry.seq;
                    let page = LogPageView {
                        entries: vec![entry],
                        next_seq: last_seq,
                        head_seq: last_seq,
                        epoch,
                    };
                    if tx.send(WireS2C::Item { id, res: ApiResponse::LogPage(page) }).await.is_err() {
                        break;
                    }
                }
                Some(LogStreamItem::Lagged) => {
                    // The live broadcast dropped entries for this consumer; tell the client to
                    // re-baseline from the durable journal rather than silently miss them.
                    if tx.send(WireS2C::Reset { id, epoch, head_seq: last_seq }).await.is_err() {
                        break;
                    }
                }
                None => {
                    let _ = tx.send(WireS2C::End { id, error: None }).await;
                    break;
                }
            },
            _ = keepalive.tick() => {
                let page = LogPageView { entries: Vec::new(), next_seq: last_seq, head_seq: last_seq, epoch };
                if tx.send(WireS2C::Item { id, res: ApiResponse::LogPage(page) }).await.is_err() {
                    break;
                }
            }
        }
    }
}

/// Pump a streaming `EventsSince` (L3): backlog + live node-event pages become `Item { EventsPage }`
/// frames. The feed itself surfaces an aged-out cursor / a broadcast lag as a `ResyncNeeded` event
/// *inside* a page (the client re-baselines), so this pump just forwards pages; idle ticks send an
/// empty keepalive page; the stream ends with `End`.
async fn pump_node_events(api: Arc<dyn NodeApi>, tx: mpsc::Sender<WireS2C>, id: u64, cursor: u64) {
    let mut stream = match api.events_subscribe(cursor).await {
        Ok(s) => s,
        Err(e) => {
            let _ = tx.send(WireS2C::End { id, error: Some(e) }).await;
            return;
        }
    };
    let mut keepalive = tokio::time::interval(STREAM_KEEPALIVE);
    keepalive.tick().await; // consume the immediate first tick
    let mut last_cursor = cursor;
    loop {
        tokio::select! {
            item = stream.next() => match item {
                Some(page) => {
                    if page.next_cursor > last_cursor {
                        last_cursor = page.next_cursor;
                    }
                    if tx.send(WireS2C::Item { id, res: ApiResponse::EventsPage(page) }).await.is_err() {
                        break;
                    }
                }
                None => {
                    let _ = tx.send(WireS2C::End { id, error: None }).await;
                    break;
                }
            },
            _ = keepalive.tick() => {
                let page = EventsPage { events: Vec::new(), next_cursor: last_cursor, head_cursor: last_cursor };
                if tx.send(WireS2C::Item { id, res: ApiResponse::EventsPage(page) }).await.is_err() {
                    break;
                }
            }
        }
    }
}

/// A one-shot legacy client over the Unix-socket adapter: connect, send one request, read one
/// response. Cheap to clone (it only holds the socket path); each [`ApiClient::call`] opens a fresh
/// connection — the model an operator CLI wants. Speaks the bare (non-multiplexed) protocol.
#[derive(Clone)]
pub struct ApiClient {
    path: PathBuf,
}

impl ApiClient {
    /// A client targeting the socket at `path`.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    /// Connect, send `request`, and await the single framed response.
    #[cfg(unix)]
    pub async fn call(&self, request: ApiRequest) -> Result<ApiResponse, ApiError> {
        let mut stream = UnixStream::connect(&self.path)
            .await
            .map_err(|e| ApiError::Other(format!("connect {}: {e}", self.path.display())))?;
        write_frame(&mut stream, &to_cbor(&request))
            .await
            .map_err(|e| ApiError::Other(format!("send: {e}")))?;
        let bytes = read_frame(&mut stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv: {e}")))?
            .ok_or_else(|| ApiError::Other("connection closed before a response".into()))?;
        from_cbor::<ApiResponse>(&bytes)
    }

    /// Windows: connect over the named pipe derived from `path` ([`windows_pipe_path`]) and run the
    /// same one-shot bare protocol the unix socket does — the operator CLI / managed-daemon local
    /// path (tokio has no AF_UNIX on windows). Uses the pipe-name contract the daemon binds.
    #[cfg(windows)]
    pub async fn call(&self, request: ApiRequest) -> Result<ApiResponse, ApiError> {
        use tokio::net::windows::named_pipe::ClientOptions;
        // ERROR_PIPE_BUSY: every server instance is momentarily busy serving another client. The
        // accept loop creates the next instance right after each connect, so a short retry wins.
        const ERROR_PIPE_BUSY: i32 = 231;
        let name = windows_pipe_path(&self.path.to_string_lossy());
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match ClientOptions::new().open(&name) {
                Ok(c) => break c,
                Err(e)
                    if e.raw_os_error() == Some(ERROR_PIPE_BUSY)
                        && std::time::Instant::now() < deadline =>
                {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(e) => return Err(ApiError::Other(format!("connect {name}: {e}"))),
            }
        };
        write_frame(&mut stream, &to_cbor(&request))
            .await
            .map_err(|e| ApiError::Other(format!("send: {e}")))?;
        let bytes = read_frame(&mut stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv: {e}")))?
            .ok_or_else(|| ApiError::Other("connection closed before a response".into()))?;
        from_cbor::<ApiResponse>(&bytes)
    }

    /// Neither unix nor windows: there is no local-trust transport, so a call fails explicitly
    /// instead of pretending — connect over the node's TLS/WebSocket/HTTP surface instead.
    #[cfg(not(any(unix, windows)))]
    pub async fn call(&self, _request: ApiRequest) -> Result<ApiResponse, ApiError> {
        Err(ApiError::Other(format!(
            "no local transport ({}) on this platform; connect over the node's \
             TLS/WebSocket/HTTP surface instead",
            self.path.display()
        )))
    }
}

/// The machine-global pipe **name component** for a `socket_path` (no `\\.\pipe\` prefix): the
/// shared contract with the Qt launcher, which passes this to `QLocalSocket::connectToServer` and
/// lets Qt prepend `\\.\pipe\`. **This is the authority for the pipe-name contract** — the Qt side
/// mirrors this rule byte-for-byte in `daemon-app/src/core/daemon/windows_pipe_name.h` (keep the two
/// in lockstep). Pipe names share a single machine-global namespace, so the full socket path is
/// folded in to keep distinct sockets on distinct pipes. Rule: the fixed prefix `daemon-api-`, then
/// each UTF-8 byte of `socket_path` that is an ASCII alphanumeric or one of `.`/`_`/`-` is kept
/// verbatim; every other byte maps to `_`.
pub fn windows_pipe_component(socket_path: &str) -> String {
    let mut name = String::from("daemon-api-");
    for b in socket_path.bytes() {
        let keep = b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-');
        name.push(if keep { b as char } else { '_' });
    }
    name
}

/// The full `\\.\pipe\<component>` path the daemon binds and `daemon-cli` connects. See
/// [`windows_pipe_component`] for the shared name-component contract with the Qt side.
pub fn windows_pipe_path(socket_path: &str) -> String {
    format!(r"\\.\pipe\{}", windows_pipe_component(socket_path))
}

/// A multiplexed client over the Unix-socket adapter: one connection, a `Hello` handshake, then
/// many correlated `Call`s (one-shot via [`MuxApiClient::call`]) and streaming subscriptions (via
/// [`MuxApiClient::open`] + [`MuxApiClient::next`]). Single-stream-at-a-time in shape (the reader is
/// `&mut self`), which suits drivers and conformance; a fully concurrent client would demux on a
/// background task.
// unix-only: it dials the unix socket (drivers/conformance); networked clients mux over TLS/WS.
#[cfg(unix)]
pub struct MuxApiClient {
    stream: UnixStream,
    next_id: u64,
}

#[cfg(unix)]
impl MuxApiClient {
    /// Connect and complete the `Hello` handshake (sends `WireC2S::Hello`, awaits `WireS2C::Hello`).
    pub async fn connect(path: impl Into<PathBuf>) -> Result<Self, ApiError> {
        let path = path.into();
        let mut stream = UnixStream::connect(&path)
            .await
            .map_err(|e| ApiError::Other(format!("connect {}: {e}", path.display())))?;
        let hello = WireC2S::Hello {
            wire_version: WIRE_VERSION,
            features: vec![
                WIRE_FEATURE_MUX.to_string(),
                WIRE_FEATURE_STREAM.to_string(),
            ],
        };
        write_frame(&mut stream, &to_cbor(&hello))
            .await
            .map_err(|e| ApiError::Other(format!("send hello: {e}")))?;
        let bytes = read_frame(&mut stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv hello: {e}")))?
            .ok_or_else(|| ApiError::Other("closed before hello ack".into()))?;
        match from_cbor::<WireS2C>(&bytes)? {
            WireS2C::Hello { .. } => Ok(Self { stream, next_id: 1 }),
            other => Err(ApiError::Other(format!(
                "expected Hello ack, got {other:?}"
            ))),
        }
    }

    /// Authenticate this connection with `SCRAM-SHA-256` (post-`Hello`), driving the SASL exchange
    /// with an `rsasl` client and returning the authenticated [`daemon_api::PrincipalView`] from
    /// `AuthOk`. The reusable client side of the handshake (also the shape a real GUI/TUI client
    /// uses); used by the auth-transport integration tests.
    pub async fn authenticate_scram(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<daemon_api::PrincipalView, ApiError> {
        use rsasl::prelude::{Mechname, SASLClient, SASLConfig, State as ClientState};

        let config = SASLConfig::with_credentials(None, username.into(), password.into())
            .map_err(|e| ApiError::Other(format!("sasl client config: {e}")))?;
        let mechname = Mechname::parse(crate::authn::MECH_SCRAM_SHA_256.as_bytes())
            .map_err(|e| ApiError::Other(format!("mechname: {e}")))?;
        let mut session = SASLClient::new(config)
            .start_suggested_iter([mechname])
            .map_err(|e| ApiError::Other(format!("sasl client start: {e}")))?;

        // SCRAM is client-first: produce the client-first message with no input.
        let mut out = Vec::new();
        session
            .step(None, &mut out)
            .map_err(|e| ApiError::Other(format!("sasl client step: {e}")))?;
        self.send(WireC2S::AuthStart {
            mechanism: crate::authn::MECH_SCRAM_SHA_256.to_string(),
            initial: out.clone(),
        })
        .await?;

        loop {
            match self.next().await? {
                WireS2C::AuthChallenge { data } => {
                    out.clear();
                    let state = session
                        .step(Some(&data), &mut out)
                        .map_err(|e| ApiError::Unauthenticated(format!("sasl: {e}")))?;
                    // Only respond when the mechanism produced output; the final server message
                    // (server-final) leaves the client `Finished` with nothing to send.
                    if !out.is_empty() {
                        self.send(WireC2S::AuthStep { data: out.clone() }).await?;
                    } else if state == ClientState::Running {
                        // Running with no output is unexpected for SCRAM; keep waiting for AuthOk.
                    }
                }
                WireS2C::AuthOk { principal, .. } => return Ok(principal),
                WireS2C::AuthError { reason } => return Err(ApiError::Unauthenticated(reason)),
                other => {
                    return Err(ApiError::Other(format!(
                        "unexpected frame during authentication: {other:?}"
                    )))
                }
            }
        }
    }

    fn take_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Send a one-shot `Call` and await its `Reply` (skipping unrelated frames).
    pub async fn call(&mut self, req: ApiRequest) -> Result<ApiResponse, ApiError> {
        let id = self.take_id();
        self.send(WireC2S::Call { id, req }).await?;
        loop {
            match self.next().await? {
                WireS2C::Reply { id: rid, res } if rid == id => return Ok(res),
                WireS2C::End { id: rid, error } if rid == id => {
                    return Err(error.unwrap_or_else(|| {
                        ApiError::Other("stream ended without a reply".into())
                    }));
                }
                _ => continue,
            }
        }
    }

    /// Open a server-stream for a streaming request (e.g. `Subscribe`); read its `Item`s with
    /// [`MuxApiClient::next`].
    pub async fn open(&mut self, req: ApiRequest) -> Result<u64, ApiError> {
        let id = self.take_id();
        self.send(WireC2S::Open { id, req }).await?;
        Ok(id)
    }

    /// Cancel a streaming exchange.
    pub async fn cancel(&mut self, id: u64) -> Result<(), ApiError> {
        self.send(WireC2S::Cancel { id }).await
    }

    /// Read the next server frame.
    pub async fn next(&mut self) -> Result<WireS2C, ApiError> {
        let bytes = read_frame(&mut self.stream)
            .await
            .map_err(|e| ApiError::Other(format!("recv: {e}")))?
            .ok_or_else(|| ApiError::Other("connection closed".into()))?;
        from_cbor::<WireS2C>(&bytes)
    }

    async fn send(&mut self, frame: WireC2S) -> Result<(), ApiError> {
        write_frame(&mut self.stream, &to_cbor(&frame))
            .await
            .map_err(|e| ApiError::Other(format!("send: {e}")))
    }
}

/// Read one length-framed message. Returns `Ok(None)` on a clean EOF at a frame boundary.
pub(crate) async fn read_frame<R: tokio::io::AsyncRead + Unpin>(
    stream: &mut R,
) -> std::io::Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(Some(buf))
}

/// Write one length-framed message.
pub(crate) async fn write_frame<W: tokio::io::AsyncWrite + Unpin>(
    stream: &mut W,
    bytes: &[u8],
) -> std::io::Result<()> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "frame too large"))?;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(bytes).await?;
    stream.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every server `Hello` advertises the envelope features AND the API contract version — the
    /// `api/<N>` string a client uses to detect a stale daemon (the orphaned-daemon incident).
    #[test]
    fn hello_always_advertises_api_contract_version() {
        for (versioning, auth) in [(false, false), (true, false), (false, true), (true, true)] {
            let features = hello_features(versioning, auth);
            assert!(features.contains(&WIRE_FEATURE_MUX.to_string()));
            assert!(features.contains(&WIRE_FEATURE_STREAM.to_string()));
            assert!(
                features.contains(&wire_feature_api()),
                "hello must advertise {} (got {features:?})",
                wire_feature_api()
            );
        }
    }

    /// `versioning` / `auth` ride the Hello only when the node / transport actually offers them.
    #[test]
    fn hello_gates_versioning_and_auth_features() {
        let bare = hello_features(false, false);
        assert!(!bare.contains(&WIRE_FEATURE_VERSIONING.to_string()));
        assert!(!bare.contains(&WIRE_FEATURE_AUTH.to_string()));

        let versioned = hello_features(true, false);
        assert!(versioned.contains(&WIRE_FEATURE_VERSIONING.to_string()));
        assert!(!versioned.contains(&WIRE_FEATURE_AUTH.to_string()));

        let authed = hello_features(false, true);
        assert!(!authed.contains(&WIRE_FEATURE_VERSIONING.to_string()));
        assert!(authed.contains(&WIRE_FEATURE_AUTH.to_string()));
    }

    /// The Windows pipe-name contract is deterministic, path-scoped (distinct sockets => distinct
    /// pipes), and folds every non-`[A-Za-z0-9._-]` byte to `_`. The Qt launcher mirrors this rule
    /// byte-for-byte, so this pins the shared behavior on every platform (not just windows).
    #[test]
    fn windows_pipe_contract_is_deterministic_and_path_scoped() {
        let a = windows_pipe_component("/tmp/daemon-api.sock");
        assert_eq!(a, "daemon-api-_tmp_daemon-api.sock");
        assert_eq!(a, windows_pipe_component("/tmp/daemon-api.sock")); // deterministic
        assert_ne!(a, windows_pipe_component("/tmp/other/daemon-api.sock")); // path-scoped
                                                                             // Separators / spaces / colons all collapse to '_'; the allowed classes survive verbatim.
        assert_eq!(windows_pipe_component("/a b:c"), "daemon-api-_a_b_c");
        assert_eq!(
            windows_pipe_component(r"C:\Temp\daemon.sock"),
            "daemon-api-C__Temp_daemon.sock"
        );
        // The full path is the component with the fixed `\\.\pipe\` prefix.
        assert_eq!(
            windows_pipe_path("/tmp/daemon-api.sock"),
            r"\\.\pipe\daemon-api-_tmp_daemon-api.sock"
        );
    }
}
