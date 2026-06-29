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

use daemon_api::{
    dispatch, from_cbor, is_streaming, to_cbor, ApiError, ApiRequest, ApiResponse, EventsPage,
    LogPageView, LogStreamItem, NodeApi, WireC2S, WireS2C, WIRE_FEATURE_MUX, WIRE_FEATURE_STREAM,
    WIRE_FEATURE_VERSIONING, WIRE_VERSION,
};
use daemon_telemetry::{fields, ingress_trace, with_trace_span, SpanKind};
use futures::StreamExt;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

/// Bound on buffered server -> client frames per connection before the writer applies backpressure.
const WRITER_QUEUE: usize = 256;
/// Idle keepalive cadence on a live subscription so a silently dead socket is noticed without the
/// connection-level Health probe. An empty `LogPage` doubles as the keepalive.
const STREAM_KEEPALIVE: Duration = Duration::from_secs(20);

/// Serve the node surface over a bound [`UnixListener`] until it errors. Each accepted connection
/// runs independently; within a connection the mode (legacy vs multiplexed) is chosen by its first
/// frame. Runs forever; spawn it as a background task after `host.start()`.
pub async fn serve_api_unix(listener: UnixListener, api: Arc<dyn NodeApi>) {
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let api = api.clone();
                tokio::spawn(async move {
                    if let Err(e) = serve_conn(stream, api).await {
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

async fn serve_conn(stream: UnixStream, api: Arc<dyn NodeApi>) -> std::io::Result<()> {
    let (mut rd, wr) = stream.into_split();
    let first = match read_frame(&mut rd).await? {
        Some(bytes) => bytes,
        None => return Ok(()),
    };
    // Mode select by the first frame: a multiplexed client opens with a `WireC2S` frame; a legacy
    // client sends a bare `ApiRequest`, whose externally-tagged variants are disjoint from
    // `WireC2S` (`Hello`/`Call`/`Cancel`), so it never decodes as one.
    match from_cbor::<WireC2S>(&first) {
        Ok(frame) => serve_mux(rd, wr, api, frame).await,
        Err(_) => serve_legacy(rd, wr, api, first).await,
    }
}

/// Legacy one-shot: bare `ApiRequest` -> `dispatch` -> bare `ApiResponse`, sequential per connection.
async fn serve_legacy(
    mut rd: OwnedReadHalf,
    mut wr: OwnedWriteHalf,
    api: Arc<dyn NodeApi>,
    first: Vec<u8>,
) -> std::io::Result<()> {
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
                        dispatch(api.as_ref(), request).await
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

/// Multiplexed: decode each `WireC2S`, spawning a task per `Call` and multiplexing results back over
/// a single writer task. One slow handler no longer blocks others on the same connection.
async fn serve_mux(
    mut rd: OwnedReadHalf,
    wr: OwnedWriteHalf,
    api: Arc<dyn NodeApi>,
    first: WireC2S,
) -> std::io::Result<()> {
    let (tx, mut rx) = mpsc::channel::<WireS2C>(WRITER_QUEUE);
    let writer = tokio::spawn(async move {
        let mut wr = wr;
        while let Some(frame) = rx.recv().await {
            if write_frame(&mut wr, &to_cbor(&frame)).await.is_err() {
                break;
            }
        }
    });

    // Streaming `Call`s register an abort handle so a later `Cancel { id }` can tear them down. The
    // map is owned solely by this reader loop, so no lock is needed.
    let mut streams: HashMap<u64, AbortHandle> = HashMap::new();
    let mut pending = Some(first);
    loop {
        let frame = match pending.take() {
            Some(f) => f,
            None => match read_frame(&mut rd).await? {
                Some(bytes) => match from_cbor::<WireC2S>(&bytes) {
                    Ok(f) => f,
                    // A frame we cannot decode is dropped rather than killing the connection; a
                    // well-behaved client never sends one.
                    Err(_) => continue,
                },
                None => break,
            },
        };
        match frame {
            WireC2S::Hello { .. } => {
                // Advertise the always-on envelope capabilities plus any optional surface the node
                // actually hosts (versioning needs a bound revision log), so the client can hide
                // unavailable affordances up front.
                let mut features = vec![
                    WIRE_FEATURE_MUX.to_string(),
                    WIRE_FEATURE_STREAM.to_string(),
                ];
                if api.supports_versioning() {
                    features.push(WIRE_FEATURE_VERSIONING.to_string());
                }
                let _ = tx
                    .send(WireS2C::Hello {
                        wire_version: WIRE_VERSION,
                        features,
                    })
                    .await;
            }
            WireC2S::Call { id, req } => {
                // One-shot: dispatch (Subscribe here is the non-destructive cursor read) -> Reply.
                let api = api.clone();
                let tx = tx.clone();
                tokio::spawn(async move {
                    let trace = ingress_trace(None);
                    let res = with_trace_span(
                        trace,
                        fields::span::API_UNIX_REQUEST,
                        SpanKind::Boundary,
                        async {
                            tracing::debug!(
                                trace_id = %trace,
                                api_variant = ?std::mem::discriminant(&req),
                                event = fields::event::API_REQUEST,
                                "api request received over unix socket (mux)"
                            );
                            dispatch(api.as_ref(), req).await
                        },
                    )
                    .await;
                    let _ = tx.send(WireS2C::Reply { id, res }).await;
                });
            }
            WireC2S::Open { id, req } => {
                // Server-stream a streaming-capable request; reject anything else with End{error}.
                if is_streaming(&req) {
                    streams.insert(id, spawn_stream(api.clone(), tx.clone(), id, req));
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
fn spawn_stream(
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
}

/// A multiplexed client over the Unix-socket adapter: one connection, a `Hello` handshake, then
/// many correlated `Call`s (one-shot via [`MuxApiClient::call`]) and streaming subscriptions (via
/// [`MuxApiClient::open`] + [`MuxApiClient::next`]). Single-stream-at-a-time in shape (the reader is
/// `&mut self`), which suits drivers and conformance; a fully concurrent client would demux on a
/// background task.
pub struct MuxApiClient {
    stream: UnixStream,
    next_id: u64,
}

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
async fn read_frame<R: tokio::io::AsyncRead + Unpin>(
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
async fn write_frame<W: tokio::io::AsyncWrite + Unpin>(
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
