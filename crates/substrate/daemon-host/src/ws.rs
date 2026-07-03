// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The plain-WebSocket carrier for the multiplexed api wire protocol (the browser / Qt WASM path).
//!
//! Browsers cannot open Unix sockets or raw TLS TCP, so this listener serves the SAME mux
//! ([`WireC2S`](daemon_api::WireC2S)/[`WireS2C`](daemon_api::WireS2C): `Hello` handshake, SASL auth
//! frames, `Call`/`Open`/`Cancel` correlation) over WebSocket. It is a new *carrier* only — the
//! CBOR contract and `WIRE_VERSION` are untouched:
//!
//! * **Framing** — one WS **binary** message == exactly one mux CBOR frame, with NO u32 length
//!   prefix (WebSocket is already message-oriented, so the byte-stream framing is redundant).
//!   Text frames are ignored with a log; ping/pong ride the WS library's automatic handling.
//! * **Subprotocol** — clients SHOULD request [`WS_SUBPROTOCOL`] (`daemon-mux`), which the
//!   handshake echoes; a client requesting only foreign subprotocols is refused. A client
//!   requesting none is accepted (tolerant bring-up).
//! * **Origin policy** — a browser stamps the upgrade request with an `Origin` header the page
//!   cannot forge: when present it MUST be on the configured allow-list or the upgrade is refused
//!   with 403 *before any mux traffic*. Absent `Origin` (non-browser clients) is allowed — those
//!   clients still authenticate like everyone else.
//! * **Authentication** — ALWAYS [`AuthMode::Required`], regardless of `[api].local_trust`: a
//!   browser-reachable listener is never local-trusted. Plain WS is a plaintext transport, so
//!   [`TlsState::plaintext`] gates PLAIN/EXTERNAL off (SCRAM-SHA-256 only) exactly like the
//!   auth-required Unix socket. `wss://` is expected to terminate at a reverse proxy for now.
//!
//! The adaptation into the unchanged [`serve_mux`] loop is a pair of [`AsyncRead`]/[`AsyncWrite`]
//! shims: reads re-add the u32 big-endian length prefix over each binary message, writes strip it
//! and send one binary message per frame. `socket.rs`/`tls.rs` are untouched.

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use daemon_api::NodeApi;
use futures::stream::{SplitSink, SplitStream};
use futures::{Sink, Stream, StreamExt};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::http::header::{ORIGIN, SEC_WEBSOCKET_PROTOCOL};
use tokio_tungstenite::tungstenite::http::{HeaderValue, StatusCode};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use crate::authn::{Authenticator, TlsState};
use crate::socket::{next_conn_id, serve_mux, AuthMode};

/// The WebSocket subprotocol name of the mux carrier, negotiated during the upgrade handshake.
pub const WS_SUBPROTOCOL: &str = "daemon-mux";

/// Serve the node api surface over plain WebSocket until the listener errors. Every connection is
/// mux-only and **must authenticate** (a browser-reachable listener is never local-trusted): after
/// the upgrade gate (Origin allow-list + subprotocol negotiation) the connection is handed to the
/// shared [`serve_mux`] in [`AuthMode::Required`] mode with a plaintext [`TlsState`] (SCRAM only).
/// Spawn it as a background task alongside the Unix/TLS listeners.
pub async fn serve_mux_ws(
    listener: TcpListener,
    api: Arc<dyn NodeApi>,
    auth: Arc<Authenticator>,
    allowed_origins: Vec<String>,
) {
    let allowed = Arc::new(allowed_origins);
    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let api = api.clone();
                let auth = auth.clone();
                let allowed = allowed.clone();
                tokio::spawn(async move {
                    match accept_mux_upgrade(stream, &allowed).await {
                        Ok(ws) => {
                            let mode = Arc::new(AuthMode::Required {
                                auth,
                                tls_state: TlsState::plaintext(),
                            });
                            let (wr, rd) = split_frames(ws);
                            if let Err(e) = serve_mux(rd, wr, api, mode, None, next_conn_id()).await
                            {
                                tracing::debug!("ws api connection ended: {e}");
                            }
                        }
                        // A refused upgrade (disallowed Origin, foreign subprotocol, a non-WS
                        // probe) is dropped cleanly — never panics the accept loop.
                        Err(e) => tracing::debug!("ws upgrade rejected: {e}"),
                    }
                });
            }
            Err(e) => {
                tracing::warn!("ws api accept failed: {e}");
                return;
            }
        }
    }
}

/// Run the WebSocket server handshake with the mux upgrade gate applied: the policy callback
/// refuses (with an HTTP error response, before any mux traffic) or negotiates the subprotocol.
// tungstenite's `Callback` trait dictates the `Result<Response, ErrorResponse>` shape; not ours to shrink.
#[allow(clippy::result_large_err)]
async fn accept_mux_upgrade(
    stream: TcpStream,
    allowed_origins: &[String],
) -> Result<WebSocketStream<TcpStream>, tokio_tungstenite::tungstenite::Error> {
    tokio_tungstenite::accept_hdr_async(stream, |req: &Request, resp: Response| {
        apply_upgrade_policy(req, resp, allowed_origins)
    })
    .await
}

/// The handshake callback: extract `Origin` + the requested subprotocols from the upgrade request,
/// run the pure [`negotiate_upgrade`] policy, and either echo the negotiated subprotocol on the
/// 101 response or refuse with the policy's HTTP status.
// tungstenite's `Callback` trait dictates the `Result<Response, ErrorResponse>` shape; not ours to shrink.
#[allow(clippy::result_large_err)]
fn apply_upgrade_policy(
    req: &Request,
    mut resp: Response,
    allowed_origins: &[String],
) -> Result<Response, ErrorResponse> {
    let origin = match req.headers().get(ORIGIN) {
        Some(value) => match value.to_str() {
            Ok(o) => Some(o),
            // A non-UTF8 Origin can never be on the allow-list: fail closed.
            Err(_) => return Err(refusal(StatusCode::FORBIDDEN, "origin not allowed")),
        },
        None => None,
    };
    let requested: Vec<&str> = req
        .headers()
        .get_all(SEC_WEBSOCKET_PROTOCOL)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    match negotiate_upgrade(origin, &requested, allowed_origins) {
        Ok(Some(subprotocol)) => {
            resp.headers_mut().insert(
                SEC_WEBSOCKET_PROTOCOL,
                HeaderValue::from_static(subprotocol),
            );
            Ok(resp)
        }
        Ok(None) => Ok(resp),
        Err((status, reason)) => {
            tracing::debug!(%status, reason, origin, "refusing websocket upgrade");
            Err(refusal(status, reason))
        }
    }
}

/// An HTTP refusal response for the upgrade handshake.
fn refusal(status: StatusCode, reason: &str) -> ErrorResponse {
    let mut resp = ErrorResponse::new(Some(reason.to_string()));
    *resp.status_mut() = status;
    resp
}

/// The pure upgrade policy (unit-testable without a socket):
///
/// * An `Origin` header, when present, must match the allow-list or the upgrade is refused with
///   403 — with the default empty list every browser origin is refused. Absent `Origin`
///   (non-browser clients) passes.
/// * A client requesting subprotocols must include [`WS_SUBPROTOCOL`] (echoed back); requesting
///   only foreign subprotocols is refused. Requesting none is tolerated (bring-up clients).
fn negotiate_upgrade(
    origin: Option<&str>,
    requested_subprotocols: &[&str],
    allowed_origins: &[String],
) -> Result<Option<&'static str>, (StatusCode, &'static str)> {
    if let Some(origin) = origin {
        if !origin_allowed(origin, allowed_origins) {
            return Err((StatusCode::FORBIDDEN, "origin not allowed"));
        }
    }
    if requested_subprotocols.is_empty() {
        return Ok(None);
    }
    if requested_subprotocols.contains(&WS_SUBPROTOCOL) {
        Ok(Some(WS_SUBPROTOCOL))
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            "unsupported websocket subprotocol (expected daemon-mux)",
        ))
    }
}

/// Whether `origin` matches an allow-list entry. Serialized origins are compared ASCII
/// case-insensitively with any trailing `/` dropped (scheme/host serialize lower-case, but a
/// hand-written config entry may not be).
fn origin_allowed(origin: &str, allowed: &[String]) -> bool {
    let norm = |s: &str| s.trim().trim_end_matches('/').to_ascii_lowercase();
    let origin = norm(origin);
    allowed.iter().any(|a| norm(a) == origin)
}

/// Split a WebSocket into the byte-stream halves [`serve_mux`] consumes: reads re-add the u32
/// big-endian length prefix over each binary message, writes strip it and send one binary message
/// per mux frame.
fn split_frames<S>(ws: WebSocketStream<S>) -> (WsFrameWriter<S>, WsFrameReader<S>)
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (sink, stream) = ws.split();
    (
        WsFrameWriter {
            inner: sink,
            buf: Vec::new(),
        },
        WsFrameReader {
            inner: stream,
            buf: Vec::new(),
            pos: 0,
        },
    )
}

/// The read half: each binary WS message surfaces as one length-prefixed frame (so the unchanged
/// `read_frame` sees the exact Unix/TLS byte shape). Text frames are ignored with a log (the
/// protocol is binary-only); ping/pong/close are handled by tungstenite (the close reply is
/// flushed while draining the stream to its end, which then reads as a clean EOF).
struct WsFrameReader<S> {
    inner: SplitStream<WebSocketStream<S>>,
    /// The current frame (length prefix + payload), partially consumed.
    buf: Vec<u8>,
    pos: usize,
}

impl<S> AsyncRead for WsFrameReader<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        loop {
            // Serve buffered frame bytes first.
            if this.pos < this.buf.len() {
                let n = out.remaining().min(this.buf.len() - this.pos);
                out.put_slice(&this.buf[this.pos..this.pos + n]);
                this.pos += n;
                if this.pos == this.buf.len() {
                    this.buf.clear();
                    this.pos = 0;
                }
                return Poll::Ready(Ok(()));
            }
            match ready!(Pin::new(&mut this.inner).poll_next(cx)) {
                Some(Ok(Message::Binary(payload))) => {
                    let Ok(len) = u32::try_from(payload.len()) else {
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "ws message exceeds the frame size limit",
                        )));
                    };
                    this.buf.clear();
                    this.buf.extend_from_slice(&len.to_be_bytes());
                    this.buf.extend_from_slice(&payload);
                    this.pos = 0;
                }
                Some(Ok(Message::Text(_))) => {
                    tracing::debug!("ignoring text frame on the binary-only mux websocket");
                }
                // Ping/pong are answered by tungstenite; a Close begins the close handshake, which
                // completes while draining the stream (None below) — always a frame-boundary EOF.
                Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Close(_))) => {}
                // Raw frames never surface on read; ignore defensively.
                Some(Ok(Message::Frame(_))) => {}
                Some(Err(e)) => return Poll::Ready(Err(io::Error::other(e))),
                None => return Poll::Ready(Ok(())), // clean EOF
            }
        }
    }
}

/// The write half: buffers the length-prefixed bytes `write_frame` produces and, per complete
/// frame, strips the prefix and sends the payload as one binary WS message. `write_frame` flushes
/// after every frame, so `poll_flush` always fully drains.
struct WsFrameWriter<S> {
    inner: SplitSink<WebSocketStream<S>, Message>,
    /// Accumulated length-prefixed bytes (at most a frame plus a partial prefix in practice).
    buf: Vec<u8>,
}

impl<S> WsFrameWriter<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    /// Send every complete length-prefixed frame in `buf` as one binary WS message.
    fn drain_frames(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        loop {
            let Some(total) = complete_frame_len(&self.buf) else {
                return Poll::Ready(Ok(()));
            };
            ready!(Pin::new(&mut self.inner).poll_ready(cx)).map_err(io::Error::other)?;
            let payload = self.buf[4..total].to_vec();
            Pin::new(&mut self.inner)
                .start_send(Message::binary(payload))
                .map_err(io::Error::other)?;
            self.buf.drain(..total);
        }
    }
}

/// The total length (prefix + payload) of the first frame in `buf`, once fully buffered.
fn complete_frame_len(buf: &[u8]) -> Option<usize> {
    let len_bytes: [u8; 4] = buf.get(..4)?.try_into().ok()?;
    let total = 4 + u32::from_be_bytes(len_bytes) as usize;
    (buf.len() >= total).then_some(total)
}

impl<S> AsyncWrite for WsFrameWriter<S>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        this.buf.extend_from_slice(data);
        // Opportunistic drain: the bytes are accepted either way (a Pending sink only defers the
        // send to the flush every `write_frame` ends with), but surface a sink error now.
        match this.drain_frames(cx) {
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            _ => Poll::Ready(Ok(data.len())),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        ready!(this.drain_frames(cx))?;
        Pin::new(&mut this.inner)
            .poll_flush(cx)
            .map_err(io::Error::other)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        ready!(this.drain_frames(cx))?;
        Pin::new(&mut this.inner)
            .poll_close(cx)
            .map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::socket::{read_frame, write_frame};
    use futures::SinkExt;
    use tokio_tungstenite::tungstenite::protocol::Role;

    /// A connected server/client WebSocket pair over an in-memory duplex pipe (no TCP, no
    /// handshake — the framing shims are what is under test).
    async fn ws_pair() -> (
        WebSocketStream<tokio::io::DuplexStream>,
        WebSocketStream<tokio::io::DuplexStream>,
    ) {
        let (a, b) = tokio::io::duplex(64 * 1024);
        let server = WebSocketStream::from_raw_socket(a, Role::Server, None).await;
        let client = WebSocketStream::from_raw_socket(b, Role::Client, None).await;
        (server, client)
    }

    /// One binary WS message == one length-prefixed mux frame, both directions, byte-exact.
    #[tokio::test]
    async fn binary_messages_round_trip_as_mux_frames() {
        let (server, mut client) = ws_pair().await;
        let (mut wr, mut rd) = split_frames(server);

        client
            .send(Message::binary(b"client-frame".to_vec()))
            .await
            .expect("client send");
        let got = read_frame(&mut rd).await.expect("read").expect("frame");
        assert_eq!(got, b"client-frame", "reads must re-add exactly one prefix");

        write_frame(&mut wr, b"server-frame").await.expect("write");
        match client.next().await.expect("client recv").expect("message") {
            Message::Binary(payload) => assert_eq!(
                &payload[..],
                b"server-frame",
                "writes must strip the prefix and send the bare CBOR payload"
            ),
            other => panic!("expected one binary message per frame, got {other:?}"),
        }
    }

    /// Text frames are ignored (binary-only protocol); the following binary frame still arrives.
    #[tokio::test]
    async fn text_frames_are_ignored() {
        let (server, mut client) = ws_pair().await;
        let (_wr, mut rd) = split_frames(server);

        client
            .send(Message::text("not-cbor"))
            .await
            .expect("send text");
        client
            .send(Message::binary(b"after-text".to_vec()))
            .await
            .expect("send binary");
        let got = read_frame(&mut rd).await.expect("read").expect("frame");
        assert_eq!(got, b"after-text");
    }

    /// A client close reads as a clean EOF at a frame boundary (`read_frame` -> `Ok(None)`), the
    /// same shape as a Unix-socket disconnect.
    #[tokio::test]
    async fn client_close_is_a_clean_eof() {
        let (server, mut client) = ws_pair().await;
        let (_wr, mut rd) = split_frames(server);

        client.close(None).await.expect("client close");
        assert!(
            read_frame(&mut rd).await.expect("read").is_none(),
            "a websocket close must surface as a clean frame-boundary EOF"
        );
    }

    /// A multi-frame burst written through the shim arrives as one binary message per frame.
    #[tokio::test]
    async fn each_frame_becomes_its_own_message() {
        let (server, mut client) = ws_pair().await;
        let (mut wr, _rd) = split_frames(server);

        write_frame(&mut wr, b"one").await.expect("write one");
        write_frame(&mut wr, b"two").await.expect("write two");
        for want in [&b"one"[..], &b"two"[..]] {
            match client.next().await.expect("recv").expect("message") {
                Message::Binary(payload) => assert_eq!(&payload[..], want),
                other => panic!("expected a binary message, got {other:?}"),
            }
        }
    }

    /// The Origin policy: absent passes, allow-listed passes (case/trailing-slash tolerant),
    /// anything else — including everything under the default empty list — is refused with 403.
    #[test]
    fn origin_policy_fails_closed() {
        let allowed = vec!["https://app.example.com".to_string()];

        assert_eq!(negotiate_upgrade(None, &[], &allowed), Ok(None));
        assert_eq!(
            negotiate_upgrade(Some("https://app.example.com"), &[], &allowed),
            Ok(None)
        );
        assert_eq!(
            negotiate_upgrade(Some("HTTPS://APP.Example.COM/"), &[], &allowed),
            Ok(None),
            "origin matching must tolerate case + a trailing slash"
        );
        assert_eq!(
            negotiate_upgrade(Some("https://evil.example.com"), &[], &allowed),
            Err((StatusCode::FORBIDDEN, "origin not allowed"))
        );
        // The default empty allow-list refuses every browser origin (fail closed) but keeps
        // non-browser clients (no Origin header) connectable.
        assert_eq!(
            negotiate_upgrade(Some("https://app.example.com"), &[], &[]),
            Err((StatusCode::FORBIDDEN, "origin not allowed"))
        );
        assert_eq!(negotiate_upgrade(None, &[], &[]), Ok(None));
    }

    /// Subprotocol negotiation: `daemon-mux` is echoed, none is tolerated, foreign-only refused.
    #[test]
    fn subprotocol_negotiation() {
        assert_eq!(
            negotiate_upgrade(None, &[WS_SUBPROTOCOL], &[]),
            Ok(Some(WS_SUBPROTOCOL))
        );
        assert_eq!(
            negotiate_upgrade(None, &["graphql-ws", WS_SUBPROTOCOL], &[]),
            Ok(Some(WS_SUBPROTOCOL)),
            "daemon-mux must be selected from a multi-protocol offer"
        );
        assert_eq!(negotiate_upgrade(None, &[], &[]), Ok(None));
        assert!(
            matches!(
                negotiate_upgrade(None, &["graphql-ws"], &[]),
                Err((StatusCode::BAD_REQUEST, _))
            ),
            "a foreign-only subprotocol offer must be refused"
        );
    }
}
