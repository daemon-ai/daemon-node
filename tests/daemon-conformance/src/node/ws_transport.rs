// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! THE WEBSOCKET CARRIER GATE: the same CBOR wire mux (`Hello` handshake, SASL exchange,
//! correlated `Call`s) served over plain WebSocket for browser (Qt WASM) clients — proven over a
//! real TCP listener with a real `tokio-tungstenite` client speaking the pinned client contract:
//!
//! - one WS **binary** message == one mux CBOR frame, with NO u32 length prefix;
//! - the `daemon-mux` subprotocol is negotiated (echoed on the 101 response);
//! - the listener ALWAYS requires authentication (plaintext transport => SCRAM only, no PLAIN),
//!   pre-auth `Call`s are `Unauthenticated`, a full SCRAM-SHA-256 exchange unlocks dispatch;
//! - a browser-style `Origin` header must be on the allow-list (403 before any mux traffic;
//!   the default empty list refuses every origin); absent `Origin` (non-browser) connects;
//! - text frames are ignored (binary-only protocol), and a client close is answered (clean
//!   close handshake), mirroring a Unix-socket disconnect.

use super::harness::*;
use daemon_api::{
    from_cbor, to_cbor, ApiError, PrincipalView, WireC2S, WireS2C, WIRE_FEATURE_MUX,
    WIRE_FEATURE_STREAM, WIRE_VERSION,
};
use daemon_auth::{AuthStore, Role};
use daemon_host::{serve_mux_ws, Authenticator, MECH_SCRAM_SHA_256, WS_SUBPROTOCOL};
use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{HeaderValue, ORIGIN, SEC_WEBSOCKET_PROTOCOL};
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

/// A node + authenticator over one identity store with a seeded operator, serving the WebSocket
/// mux carrier on an ephemeral port with `allowed_origins`. Returns the bound address, the server
/// task, and the resident-service handle.
async fn serve_ws(
    allowed_origins: &[&str],
) -> (
    std::net::SocketAddr,
    tokio::task::JoinHandle<()>,
    daemon_host::SupervisorHandle,
) {
    let (node, handle) = assemble();
    let store = Arc::new(AuthStore::open_in_memory().expect("auth store"));
    store
        .create_user("operator", "op-pw", &[Role::Operator])
        .expect("create operator");
    let auth = Arc::new(Authenticator::new(store));
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let server = tokio::spawn(serve_mux_ws(
        listener,
        node,
        auth,
        allowed_origins.iter().map(|s| s.to_string()).collect(),
    ));
    (addr, server, handle)
}

/// A minimal client of the **pinned WS contract** (what the Qt WASM client implements): one binary
/// message per CBOR mux frame, no length prefix, subprotocol `daemon-mux`. Not a byte-stream
/// client on purpose — it proves the message-per-frame shape on the wire.
struct WsMuxClient {
    ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    next_id: u64,
    /// The mechanisms the server advertised on its `Hello`.
    mechanisms: Vec<String>,
}

impl WsMuxClient {
    /// Connect with the `daemon-mux` subprotocol (+ an optional browser-style `Origin`) and
    /// complete the mux `Hello` handshake. Asserts the subprotocol is echoed.
    async fn connect(
        addr: std::net::SocketAddr,
        origin: Option<&str>,
    ) -> Result<Self, Box<WsError>> {
        let mut request = format!("ws://{addr}/")
            .into_client_request()
            .expect("client request");
        request.headers_mut().insert(
            SEC_WEBSOCKET_PROTOCOL,
            HeaderValue::from_static(WS_SUBPROTOCOL),
        );
        if let Some(origin) = origin {
            request
                .headers_mut()
                .insert(ORIGIN, HeaderValue::from_str(origin).expect("origin value"));
        }
        let (ws, response) = connect_async(request).await.map_err(Box::new)?;
        assert_eq!(
            response
                .headers()
                .get(SEC_WEBSOCKET_PROTOCOL)
                .and_then(|v| v.to_str().ok()),
            Some(WS_SUBPROTOCOL),
            "the server must negotiate (echo) the daemon-mux subprotocol"
        );
        let mut client = Self {
            ws,
            next_id: 1,
            mechanisms: Vec::new(),
        };
        client
            .send(WireC2S::Hello {
                wire_version: WIRE_VERSION,
                features: vec![
                    WIRE_FEATURE_MUX.to_string(),
                    WIRE_FEATURE_STREAM.to_string(),
                ],
            })
            .await
            .expect("send hello");
        match client.next().await.expect("hello ack") {
            WireS2C::Hello {
                auth_mechanisms, ..
            } => client.mechanisms = auth_mechanisms,
            other => panic!("expected Hello ack, got {other:?}"),
        }
        Ok(client)
    }

    /// Send one mux frame as one binary WS message (bare CBOR, no length prefix).
    async fn send(&mut self, frame: WireC2S) -> Result<(), ApiError> {
        self.ws
            .send(Message::binary(to_cbor(&frame)))
            .await
            .map_err(|e| ApiError::Other(format!("ws send: {e}")))
    }

    /// Read the next server mux frame (one binary message == one frame; ping/pong skipped).
    async fn next(&mut self) -> Result<WireS2C, ApiError> {
        loop {
            match self.ws.next().await {
                Some(Ok(Message::Binary(payload))) => return from_cbor::<WireS2C>(&payload),
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Ok(other)) => {
                    return Err(ApiError::Other(format!(
                        "unexpected non-binary server message: {other:?}"
                    )))
                }
                Some(Err(e)) => return Err(ApiError::Other(format!("ws recv: {e}"))),
                None => return Err(ApiError::Other("connection closed".into())),
            }
        }
    }

    /// Send a one-shot `Call` and await its correlated `Reply`.
    async fn call(&mut self, req: ApiRequest) -> Result<ApiResponse, ApiError> {
        let id = self.next_id;
        self.next_id += 1;
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

    /// Drive a full `SCRAM-SHA-256` exchange with a real `rsasl` client (the same client shape the
    /// GUI uses), returning the authenticated principal from `AuthOk`.
    async fn authenticate_scram(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<PrincipalView, ApiError> {
        use rsasl::prelude::{Mechname, SASLClient, SASLConfig};

        let config = SASLConfig::with_credentials(None, username.into(), password.into())
            .map_err(|e| ApiError::Other(format!("sasl client config: {e}")))?;
        let mechname = Mechname::parse(MECH_SCRAM_SHA_256.as_bytes())
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
            mechanism: MECH_SCRAM_SHA_256.to_string(),
            initial: out.clone(),
        })
        .await?;

        loop {
            match self.next().await? {
                WireS2C::AuthChallenge { data } => {
                    out.clear();
                    session
                        .step(Some(&data), &mut out)
                        .map_err(|e| ApiError::Unauthenticated(format!("sasl: {e}")))?;
                    // The final server message (server-final) leaves nothing to send.
                    if !out.is_empty() {
                        self.send(WireC2S::AuthStep { data: out.clone() }).await?;
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
}

/// The full pinned client flow: subprotocol negotiation, Hello, SCRAM-only mechanisms (plaintext
/// transport), fail-closed pre-auth, SCRAM-SHA-256 unlock, a Health `Call`, tolerance of a stray
/// text frame, and a clean close handshake.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_negotiates_subprotocol_requires_scram_then_serves_and_closes() {
    let (addr, server, handle) = serve_ws(&[]).await;

    // No Origin header (a non-browser client): the empty allow-list does not apply.
    let mut client = WsMuxClient::connect(addr, None)
        .await
        .expect("ws connect + hello");

    // Plain WS is a plaintext transport: SCRAM only, PLAIN/EXTERNAL are not advertised.
    assert_eq!(
        client.mechanisms,
        vec![MECH_SCRAM_SHA_256.to_string()],
        "the WS listener must advertise SCRAM only (plaintext transport)"
    );

    // The listener is NEVER local-trusted: a pre-auth Call is refused and stays refused.
    let pre = client
        .call(ApiRequest::Health)
        .await
        .expect("pre-auth call");
    assert!(
        matches!(pre, ApiResponse::Error(ApiError::Unauthenticated(_))),
        "a pre-auth Call over WS must be Unauthenticated, got {pre:?}"
    );

    // A stray text frame is ignored (binary-only protocol), not a connection kill.
    client
        .ws
        .send(Message::text("not-cbor"))
        .await
        .expect("send text frame");

    // A full SCRAM-SHA-256 exchange unlocks dispatch.
    let view = client
        .authenticate_scram("operator", "op-pw")
        .await
        .expect("scram over ws");
    assert_eq!(view.username, "operator");
    let post = client
        .call(ApiRequest::Health)
        .await
        .expect("post-auth call");
    assert!(
        !matches!(post, ApiResponse::Error(_)),
        "a Call after AuthOk over WS must succeed, got {post:?}"
    );

    // Clean close: the server answers the close handshake (the stream ends without an error).
    client.ws.close(None).await.expect("client close");
    loop {
        match client.ws.next().await {
            Some(Ok(_)) => continue, // the server's Close ack (and anything in flight)
            None => break,
            Some(Err(e)) => panic!("the close handshake must complete cleanly, got {e}"),
        }
    }

    server.abort();
    handle.shutdown().await;
}

/// The browser Origin policy: a disallowed `Origin` is refused with 403 **at the upgrade** (before
/// any mux traffic), an allow-listed `Origin` completes the handshake and serves, and the default
/// empty allow-list refuses every browser origin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_origin_allow_list_gates_browser_upgrades() {
    let (addr, server, handle) = serve_ws(&["https://app.example.com"]).await;

    // A disallowed browser origin: refused at the HTTP upgrade with 403.
    match WsMuxClient::connect(addr, Some("https://evil.example.com")).await {
        Err(e) => match *e {
            WsError::Http(response) => assert_eq!(
                response.status(),
                403,
                "a disallowed Origin must be refused with 403"
            ),
            other => panic!("expected an HTTP 403 refusal, got {other}"),
        },
        Ok(_) => panic!("a disallowed Origin must not complete the upgrade"),
    }

    // The allow-listed origin passes the gate and the mux serves after SCRAM.
    let mut ok = WsMuxClient::connect(addr, Some("https://app.example.com"))
        .await
        .expect("allow-listed origin connects");
    ok.authenticate_scram("operator", "op-pw")
        .await
        .expect("scram over ws");
    let res = ok.call(ApiRequest::Health).await.expect("health call");
    assert!(
        !matches!(res, ApiResponse::Error(_)),
        "an allow-listed browser client must be served, got {res:?}"
    );

    server.abort();
    handle.shutdown().await;

    // The default (empty) allow-list refuses every browser origin.
    let (addr, server, handle) = serve_ws(&[]).await;
    match WsMuxClient::connect(addr, Some("https://app.example.com")).await {
        Err(e) => match *e {
            WsError::Http(response) => assert_eq!(response.status(), 403),
            other => panic!("expected an HTTP 403 refusal, got {other}"),
        },
        Ok(_) => panic!("the default empty allow-list must refuse every browser origin"),
    }
    server.abort();
    handle.shutdown().await;
}
