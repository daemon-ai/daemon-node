// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The axum surface: `POST /v1/chat/completions` (stream + non-stream) and `GET /v1/models`, a
//! bearer-auth middleware on `/v1/*`, and the SSE `[DONE]` scaffolding. The route handlers do the
//! wire<->engine mapping ([`crate::mapping`]) around the injected [`GatewayBackend`].

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Json, Router};
use daemon_core::StreamEvent;
use futures::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::backend::{Completion, EventStream, GatewayBackend, GatewayError, GatewayPrincipal};
use crate::mapping::{self, ChunkCtx};
use crate::wire::ChatCompletionRequest;

/// The shared handler state: the injected backend.
#[derive(Clone)]
struct AppState {
    backend: Arc<dyn GatewayBackend>,
}

/// Build the gateway [`Router`] over an injected [`GatewayBackend`]. Handy for embedding/testing
/// without binding a socket. Auth is delegated to the backend ([`GatewayBackend::authorize`]) — the
/// backend owns the admin token and the per-session token registry — so the server holds no token.
/// The resolved [`GatewayPrincipal`] is threaded into the route handlers as a request extension. An
/// unmatched path is a plain `404` (no token needed to learn a route does not exist).
pub fn router(backend: Arc<dyn GatewayBackend>) -> Router {
    let state = AppState { backend };
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route_layer(from_fn_with_state(state.clone(), bearer_auth))
        .with_state(state)
}

/// Serve the gateway over a bound listener until it errors. Spawn it as a background task; the
/// binary registers its abort in the shutdown block.
pub async fn serve(
    listener: tokio::net::TcpListener,
    backend: Arc<dyn GatewayBackend>,
) -> std::io::Result<()> {
    axum::serve(listener, router(backend)).await
}

/// Bearer middleware for `/v1/*`: require `Authorization: Bearer <token>` that the backend resolves
/// to a [`GatewayPrincipal`] (the admin token or a registered per-session token), else `401`. The
/// resolved principal is stashed as a request extension for the route handler.
async fn bearer_auth(
    State(state): State<AppState>,
    mut req: axum::extract::Request,
    next: Next,
) -> Response {
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_string);
    let principal = match presented {
        Some(tok) => state.backend.authorize(&tok).await,
        None => None,
    };
    match principal {
        Some(principal) => {
            req.extensions_mut().insert(principal);
            next.run(req).await
        }
        None => error_response(&GatewayError::bad_auth()),
    }
}

impl GatewayError {
    /// A synthetic unauthenticated error (the bearer middleware's `401`).
    fn bad_auth() -> Self {
        GatewayError::BadRequest("missing or invalid bearer token".into())
    }

    /// The HTTP status this error maps to.
    fn status(&self) -> StatusCode {
        match self {
            GatewayError::UnknownModel(_) => StatusCode::NOT_FOUND,
            GatewayError::BadRequest(_) => StatusCode::BAD_REQUEST,
            GatewayError::Credential(_) | GatewayError::Provider(_) => StatusCode::BAD_GATEWAY,
        }
    }

    /// The OpenAI-style error `type` tag.
    fn kind(&self) -> &'static str {
        match self {
            GatewayError::UnknownModel(_) => "invalid_request_error",
            GatewayError::BadRequest(_) => "invalid_request_error",
            GatewayError::Credential(_) => "authentication_error",
            GatewayError::Provider(_) => "upstream_error",
        }
    }
}

/// Render a [`GatewayError`] as an OpenAI-style error body with the mapped status. The bearer `401`
/// path overrides the status (a `BadRequest` variant carries the message, but auth is `401`).
fn error_response(err: &GatewayError) -> Response {
    // The auth failure is the one BadRequest we surface as 401; everything else uses `status()`.
    let status = if matches!(err, GatewayError::BadRequest(m) if m.contains("bearer")) {
        StatusCode::UNAUTHORIZED
    } else {
        err.status()
    };
    let body = serde_json::json!({
        "error": {
            "message": err.to_string(),
            "type": err.kind(),
        }
    });
    (status, Json(body)).into_response()
}

/// `GET /v1/models` — the node catalog as an OpenAI model listing.
async fn list_models(State(state): State<AppState>) -> Response {
    let catalog = state.backend.catalog().await;
    Json(mapping::catalog_to_models(&catalog)).into_response()
}

/// `POST /v1/chat/completions` — stream + non-stream. The [`GatewayPrincipal`] resolved by the
/// bearer middleware is passed to the backend so a per-session token enforces its binding.
async fn chat_completions(
    State(state): State<AppState>,
    Extension(principal): Extension<GatewayPrincipal>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(false);
    let model = req.model.clone();
    let core = match mapping::request_to_core(&req) {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };
    match state
        .backend
        .complete(&principal, &model, core, stream)
        .await
    {
        Ok(Completion::Once(out)) if !stream => {
            Json(mapping::output_to_response(&model, &out)).into_response()
        }
        // Streaming requested but the backend returned an assembled output: emit it as a single
        // content + terminal chunk so the client still gets an SSE stream.
        Ok(Completion::Once(out)) => once_as_sse(model, out),
        Ok(Completion::Stream(s)) if stream => sse_response(model, s),
        // Non-stream requested but the backend streamed: drain to an assembled output.
        Ok(Completion::Stream(s)) => match drain_stream(s).await {
            Ok(out) => Json(mapping::output_to_response(&model, &out)).into_response(),
            Err(e) => error_response(&e),
        },
        Err(e) => error_response(&e),
    }
}

/// Send one chunk as an SSE data event, returning `false` if the receiver has gone away.
async fn emit_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    chunk: &crate::wire::ChatCompletionChunkResponse,
) -> bool {
    let event = match Event::default().json_data(chunk) {
        Ok(e) => e,
        Err(_) => return true, // skip an unserializable chunk rather than tearing the stream down
    };
    tx.send(Ok(event)).await.is_ok()
}

/// Build the SSE response draining a live provider [`EventStream`], emitting OpenAI chunks and the
/// terminal `data: [DONE]`.
fn sse_response(model: String, mut stream: EventStream) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let ctx = ChunkCtx::new(&model);
        let mut role_sent = false;
        while let Some(ev) = stream.next().await {
            let sent = match ev {
                Ok(StreamEvent::TextDelta(t)) if !t.is_empty() => {
                    let role = (!role_sent).then_some("assistant");
                    role_sent = true;
                    emit_chunk(&tx, &ctx.content_chunk(t, role)).await
                }
                Ok(StreamEvent::ReasoningDelta(t)) if !t.is_empty() => {
                    let role = (!role_sent).then_some("assistant");
                    role_sent = true;
                    emit_chunk(&tx, &ctx.reasoning_chunk(t, role)).await
                }
                // Per-chunk usage is folded into the terminal Done; empty deltas are dropped.
                Ok(StreamEvent::TextDelta(_))
                | Ok(StreamEvent::ReasoningDelta(_))
                | Ok(StreamEvent::Usage(_)) => true,
                Ok(StreamEvent::Done(out)) => {
                    let role = (!role_sent).then_some("assistant");
                    role_sent = true;
                    emit_chunk(&tx, &ctx.final_chunk(&out, role)).await
                }
                Err(e) => {
                    tracing::warn!(error = %e, "gateway: provider stream failed mid-flight");
                    false
                }
            };
            if !sent {
                return;
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Emit an already-assembled [`daemon_core::ModelOutput`] as a minimal SSE stream (role+content
/// chunk, terminal chunk, `[DONE]`) — the fallback when streaming was requested but the backend
/// returned a one-shot completion.
fn once_as_sse(model: String, out: daemon_core::ModelOutput) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(4);
    tokio::spawn(async move {
        let ctx = ChunkCtx::new(&model);
        if !out.text.is_empty()
            && !emit_chunk(&tx, &ctx.content_chunk(out.text.clone(), Some("assistant"))).await
        {
            return;
        }
        let role = out.text.is_empty().then_some("assistant");
        if !emit_chunk(&tx, &ctx.final_chunk(&out, role)).await {
            return;
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Drain a provider [`EventStream`] to its assembled [`daemon_core::ModelOutput`] (the terminal
/// `Done`), folding any streamed usage — used when a non-stream request met a streaming backend.
async fn drain_stream(mut stream: EventStream) -> Result<daemon_core::ModelOutput, GatewayError> {
    use daemon_common::UsageDelta;
    let mut out = daemon_core::ModelOutput::default();
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut streamed_usage = UsageDelta::default();
    let mut got_done = false;
    while let Some(ev) = stream.next().await {
        match ev.map_err(|e| GatewayError::Provider(e.to_string()))? {
            StreamEvent::TextDelta(t) => text.push_str(&t),
            StreamEvent::ReasoningDelta(t) => reasoning.push_str(&t),
            StreamEvent::Usage(d) => streamed_usage.add(&d),
            StreamEvent::Done(done) => {
                out = done;
                got_done = true;
            }
        }
    }
    if !got_done {
        // No terminal Done: assemble from the accumulated deltas.
        out.text = text;
        if !reasoning.is_empty() {
            out.reasoning = Some(reasoning);
        }
        if out.usage == UsageDelta::default() {
            out.usage = streamed_usage;
        }
    }
    Ok(out)
}
