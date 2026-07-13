// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`WsControlPlane`] — the node WS coordinator client (spec §11.2; A1).
//!
//! The node dials **out** to the cloud `RunCoordinatorDO`'s WebSocket surface —
//! `GET {base}/runs/:id/ws` (daemon-cloud `apps/swarm`, `coordinator/do.ts`) — over `wss://`,
//! speaking canonical-CBOR [`SignedMessage`](daemon_swarm_proto::SignedMessage) frames **both
//! ways**. It presents as a [`ControlPlane`]: `publish` sends a frame up the socket, `subscribe`
//! delivers inbound frames, so the frozen `RoundEngine` runs over it unchanged.
//!
//! ## Framing (matches the DO verbatim)
//!
//! One WS **binary** message == exactly one canonical-CBOR `SignedMessage` frame, with **no** u32
//! length prefix (WebSocket is message-oriented) — identical to the in-tree
//! [`daemon_host::ws`](../../daemon-host/src/ws.rs) mux carrier convention. The DO's
//! `webSocketMessage` **disseminates** every inbound frame to the *other* connected peers (never
//! echoes it to the sender — `broadcast([bytes], ws)` with `except: ws`) and broadcasts the
//! coordinator's own emissions (`RoundOpen` / `StorageReceipt` / `RoundRecord`) to **all** peers.
//! So the delivery contract matches [`LoopbackGossip`](crate::gossip::LoopbackGossip) and
//! [`IrohGossip`](crate::iroh_gossip::IrohGossip): a plane **self-delivers** its own publish
//! locally (the DO won't echo it back), and every frame is deduped by content hash ([`Deduper`]) so
//! a WS + gossip double-arrival delivers once (NET-6).
//!
//! ## Reconnect + resubscribe
//!
//! The connection runs in a background task that reconnects with exponential backoff on any drop
//! and, on every (re)connect, re-sends the registered **resubscribe frames** (the peer's signed
//! `Join`, per [`WsControlPlane::add_resubscribe_frame`]) so a DO eviction / socket reset
//! re-establishes roster membership without app involvement. Frames published while disconnected
//! buffer in the outbound channel and flush on reconnect.
//!
//! ## Auth
//!
//! Credentials come from the run's `JoinRun.credentials` / node config — **never hardcoded**. The
//! gateway path carries a `swarm:join`-scoped API key as `Authorization: Bearer <token>`
//! (`apps/gateway/src/routes/swarm.ts`); the direct-to-worker dev path carries the internal
//! identity headers `x-daemon-org-id` / `x-daemon-actor` (`apps/swarm/src/middleware/internalAuth`).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt as _, StreamExt as _};
use tokio::net::TcpStream;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::header::{
    HeaderMap, HeaderName, HeaderValue, AUTHORIZATION,
};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::dedupe::Deduper;
use crate::transport::{ControlPlane, ControlSubscription};
use crate::SwarmNetError;

/// The connected client WebSocket stream (TLS for `wss://`, plain for `ws://`).
type WsStream = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// How the node authenticates to the coordinator surface (credentials come from
/// `JoinRun.credentials` / node config — never hardcoded).
#[derive(Clone, Debug, Default)]
pub enum WsAuth {
    /// No auth headers (a bare mock server / a listener that trusts the network position).
    #[default]
    None,
    /// `Authorization: Bearer <token>` — the gateway path (`swarm:join`-scoped API key).
    Bearer(String),
    /// The internal identity headers `x-daemon-org-id` / `x-daemon-actor` — the direct-to-worker
    /// dev path (`apps/swarm` trusts the gateway-forwarded headers).
    Internal {
        /// `x-daemon-org-id`.
        org_id: String,
        /// `x-daemon-actor`.
        actor: String,
    },
}

/// Reconnect + backoff policy for a dropped coordinator socket.
#[derive(Clone, Debug)]
pub struct ReconnectConfig {
    /// Whether to reconnect after a drop (off = one-shot).
    pub enabled: bool,
    /// The first backoff delay; doubles per consecutive failed attempt.
    pub initial_backoff: Duration,
    /// The backoff ceiling.
    pub max_backoff: Duration,
    /// Consecutive failed dial attempts before giving up (`None` = retry forever).
    pub max_attempts: Option<u32>,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            initial_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            max_attempts: None,
        }
    }
}

/// Construction surface for [`WsControlPlane`] (frozen at Merge 1).
#[derive(Clone, Debug)]
pub struct WsConfig {
    /// The swarm coordinator base URL (spec §11.1), e.g. `https://api.daemon.ai/api/v1/swarm` (the
    /// gateway) or `http://127.0.0.1:8795/api/v1/swarm` (a bare `wrangler dev`). The `base_url` swap
    /// (gateway ↔ wrangler-dev ↔ mock) is trivial by design — only this field changes.
    pub base_url: String,
    /// The run whose coordinator DO to attach to (`{base}/runs/:id/ws`).
    pub run_id: String,
    /// How to authenticate the upgrade request.
    pub auth: WsAuth,
    /// Reconnect + backoff policy.
    pub reconnect: ReconnectConfig,
}

impl WsConfig {
    /// The resolved `wss://…/runs/:id/ws` (or `ws://…` for a plaintext base) endpoint.
    #[must_use]
    pub fn endpoint(&self) -> String {
        ws_endpoint(&self.base_url, &self.run_id)
    }
}

/// State shared between the plane handle and the background connection task.
struct Shared {
    /// Live local subscriber inboxes.
    subscribers: Vec<UnboundedSender<Vec<u8>>>,
    /// Content-hash dedupe over frame bytes (the reusable NET-6 rule, [`Deduper`]) — shared with
    /// the WS + gossip dual plane so the same frame on both paths delivers once.
    dedupe: Deduper,
    /// Successful (re)connections so far (test / observability signal for the reconnect drill).
    connects: u64,
    /// Whether the socket is currently up.
    connected: bool,
}

/// The node WS coordinator client, presented as a [`ControlPlane`].
///
/// `publish` sends an already-signed frame up the socket (and self-delivers + records it); inbound
/// frames (peer disseminations + coordinator emissions) are deduped and fanned out to subscribers.
pub struct WsControlPlane {
    endpoint: String,
    shared: Arc<Mutex<Shared>>,
    outbound_tx: UnboundedSender<Vec<u8>>,
    /// Frames re-sent on every (re)connect (the peer's `Join` — resubscription).
    resubscribe: Arc<Mutex<Vec<Vec<u8>>>>,
    task: Mutex<Option<JoinHandle<()>>>,
}

impl WsControlPlane {
    /// Dial the coordinator and spawn the background connection task. The **first** connection is
    /// established eagerly so a bad endpoint / rejected auth fails fast; subsequent drops are
    /// covered by the reconnect loop.
    pub async fn connect(config: WsConfig) -> Result<Self, SwarmNetError> {
        let endpoint = config.endpoint();
        let initial = dial(&endpoint, &config.auth).await?;
        // The initial connection is live before the task spawns; count it synchronously so
        // `connect_count`/`is_connected` are accurate the instant `connect` returns (no task race).
        let shared = Arc::new(Mutex::new(Shared {
            subscribers: Vec::new(),
            dedupe: Deduper::new(),
            connects: 1,
            connected: true,
        }));
        let resubscribe = Arc::new(Mutex::new(Vec::new()));
        let (outbound_tx, outbound_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(conn_loop(
            endpoint.clone(),
            config.auth,
            config.reconnect,
            initial,
            outbound_rx,
            shared.clone(),
            resubscribe.clone(),
        ));
        Ok(Self {
            endpoint,
            shared,
            outbound_tx,
            resubscribe,
            task: Mutex::new(Some(task)),
        })
    }

    /// The resolved coordinator WS endpoint (`wss://…/runs/:id/ws`).
    #[must_use]
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Register a frame to (re)send on every connection — the peer's signed `Join`, so a reconnect
    /// re-establishes roster membership (resubscription). It is also sent once immediately on the
    /// current connection.
    pub fn add_resubscribe_frame(&self, frame: Vec<u8>) {
        self.resubscribe
            .lock()
            .expect("ws resubscribe lock")
            .push(frame.clone());
        // Send it now too (best-effort; the plane may be mid-reconnect, in which case it flushes on
        // the next connect from the resubscribe list).
        let _ = self.outbound_tx.send(frame);
    }

    /// Successful (re)connections so far — the reconnect drill asserts this grows after a sever.
    #[must_use]
    pub fn connect_count(&self) -> u64 {
        self.shared.lock().expect("ws shared lock").connects
    }

    /// Whether the coordinator socket is currently up.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.shared.lock().expect("ws shared lock").connected
    }

    /// Abort the background connection task (graceful shutdown; also runs on `Drop`).
    pub async fn shutdown(&self) {
        if let Some(handle) = self.task.lock().expect("ws task lock").take() {
            handle.abort();
        }
    }
}

impl Drop for WsControlPlane {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.task.lock() {
            if let Some(handle) = guard.take() {
                handle.abort();
            }
        }
    }
}

#[async_trait]
impl ControlPlane for WsControlPlane {
    async fn publish(&self, message: &[u8]) -> Result<(), SwarmNetError> {
        // Dedupe + self-deliver under the lock: a frame already seen (e.g. arrived via gossip, or a
        // WS+gossip double-send) is a no-op; the DO never echoes our own frame back, so self-deliver
        // keeps the contract identical to Loopback/Iroh (publish reaches our own subscriber once).
        {
            let mut sh = self.shared.lock().expect("ws shared lock");
            if !sh.dedupe.observe(message) {
                return Ok(());
            }
            sh.subscribers
                .retain(|tx| tx.send(message.to_vec()).is_ok());
        }
        self.outbound_tx
            .send(message.to_vec())
            .map_err(|_| SwarmNetError::Transport("ws control plane closed".into()))?;
        Ok(())
    }

    fn subscribe(&self) -> ControlSubscription {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.shared
            .lock()
            .expect("ws shared lock")
            .subscribers
            .push(tx);
        ControlSubscription::new(rx)
    }
}

/// Why the per-connection serve loop returned.
enum ServeOutcome {
    /// The socket dropped/errored — reconnect (if enabled).
    Disconnected,
    /// The plane handle was dropped (outbound closed) — stop.
    Closed,
}

/// The background connection lifecycle: serve the initial stream, then reconnect with backoff on
/// each drop, re-sending the resubscribe frames on every (re)connect.
async fn conn_loop(
    endpoint: String,
    auth: WsAuth,
    reconnect: ReconnectConfig,
    initial: WsStream,
    mut outbound_rx: UnboundedReceiver<Vec<u8>>,
    shared: Arc<Mutex<Shared>>,
    resubscribe: Arc<Mutex<Vec<Vec<u8>>>>,
) {
    let mut current = Some(initial);
    let mut failed_attempts: u32 = 0;
    loop {
        // The initial stream is pre-counted in `connect`; only a re-dial bumps `connects`/`connected`.
        let stream = match current.take() {
            Some(s) => s,
            None => {
                if !reconnect.enabled {
                    return;
                }
                tokio::time::sleep(backoff(failed_attempts, &reconnect)).await;
                match dial(&endpoint, &auth).await {
                    Ok(s) => {
                        failed_attempts = 0;
                        let mut sh = shared.lock().expect("ws shared lock");
                        sh.connected = true;
                        sh.connects = sh.connects.saturating_add(1);
                        s
                    }
                    Err(_) => {
                        failed_attempts = failed_attempts.saturating_add(1);
                        if let Some(max) = reconnect.max_attempts {
                            if failed_attempts >= max {
                                return;
                            }
                        }
                        continue;
                    }
                }
            }
        };
        let outcome = serve(stream, &mut outbound_rx, &shared, &resubscribe).await;
        shared.lock().expect("ws shared lock").connected = false;
        match outcome {
            ServeOutcome::Closed => return,
            ServeOutcome::Disconnected if reconnect.enabled => continue,
            ServeOutcome::Disconnected => return,
        }
    }
}

/// Serve one connected socket: resend resubscribe frames, then pump outbound publishes up and
/// inbound frames down until the socket drops or the plane is closed.
async fn serve(
    stream: WsStream,
    outbound_rx: &mut UnboundedReceiver<Vec<u8>>,
    shared: &Arc<Mutex<Shared>>,
    resubscribe: &Arc<Mutex<Vec<Vec<u8>>>>,
) -> ServeOutcome {
    let (mut write, mut read) = stream.split();

    // Resubscription: re-send the registered frames (the peer's Join) so a reconnect re-establishes
    // roster membership (spec §6.5) without app involvement.
    let frames = resubscribe.lock().expect("ws resubscribe lock").clone();
    for frame in frames {
        if write.send(Message::binary(frame)).await.is_err() {
            return ServeOutcome::Disconnected;
        }
    }

    loop {
        tokio::select! {
            outbound = outbound_rx.recv() => match outbound {
                Some(bytes) => {
                    if write.send(Message::binary(bytes)).await.is_err() {
                        return ServeOutcome::Disconnected;
                    }
                }
                // The plane handle was dropped: close the socket cleanly and stop.
                None => {
                    let _ = write.close().await;
                    return ServeOutcome::Closed;
                }
            },
            inbound = read.next() => match inbound {
                Some(Ok(Message::Binary(payload))) => deliver(shared, payload.to_vec()),
                // Text frames are ignored (the protocol is binary-only); ping/pong ride tungstenite's
                // automatic handling; a Close begins teardown.
                Some(Ok(Message::Text(_) | Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => {}
                Some(Ok(Message::Close(_))) | None => return ServeOutcome::Disconnected,
                Some(Err(_)) => return ServeOutcome::Disconnected,
            },
        }
    }
}

/// Dedupe an inbound frame and fan it out to every live subscriber (one delivery — NET-6).
fn deliver(shared: &Arc<Mutex<Shared>>, payload: Vec<u8>) {
    let mut sh = shared.lock().expect("ws shared lock");
    if sh.dedupe.observe(&payload) {
        sh.subscribers.retain(|tx| tx.send(payload.clone()).is_ok());
    }
}

/// Dial the coordinator WS endpoint with the auth headers applied (TLS auto-selected for `wss://`).
async fn dial(endpoint: &str, auth: &WsAuth) -> Result<WsStream, SwarmNetError> {
    let mut request = endpoint
        .into_client_request()
        .map_err(|e| SwarmNetError::Transport(format!("bad ws url {endpoint}: {e}")))?;
    apply_auth(request.headers_mut(), auth);
    let (stream, _resp) = connect_async(request)
        .await
        .map_err(|e| SwarmNetError::Transport(format!("ws connect {endpoint}: {e}")))?;
    Ok(stream)
}

/// Stamp the upgrade request with the configured auth headers.
fn apply_auth(headers: &mut HeaderMap, auth: &WsAuth) {
    match auth {
        WsAuth::None => {}
        WsAuth::Bearer(token) => {
            if let Ok(value) = HeaderValue::from_str(&format!("Bearer {token}")) {
                headers.insert(AUTHORIZATION, value);
            }
        }
        WsAuth::Internal { org_id, actor } => {
            if let Ok(value) = HeaderValue::from_str(org_id) {
                headers.insert(HeaderName::from_static("x-daemon-org-id"), value);
            }
            if let Ok(value) = HeaderValue::from_str(actor) {
                headers.insert(HeaderName::from_static("x-daemon-actor"), value);
            }
        }
    }
}

/// Exponential backoff for `attempt` consecutive failures, capped at `max_backoff`.
fn backoff(attempt: u32, cfg: &ReconnectConfig) -> Duration {
    let base_ms = cfg.initial_backoff.as_millis() as u64;
    let ms = base_ms.saturating_mul(2u64.saturating_pow(attempt.min(16)));
    Duration::from_millis(ms).min(cfg.max_backoff)
}

/// Build the coordinator WS endpoint: swap the base's HTTP scheme to WS (`https`→`wss`,
/// `http`→`ws`; an already-`ws`/`wss` base is kept) and append `/runs/:id/ws`.
fn ws_endpoint(base: &str, run_id: &str) -> String {
    let base = base.trim_end_matches('/');
    let swapped = if let Some(rest) = base.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = base.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        base.to_string()
    };
    format!("{swapped}/runs/{run_id}/ws")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_swaps_scheme_and_appends_ws_path() {
        assert_eq!(
            ws_endpoint("https://api.daemon.ai/api/v1/swarm", "run-1"),
            "wss://api.daemon.ai/api/v1/swarm/runs/run-1/ws"
        );
        assert_eq!(
            ws_endpoint("http://127.0.0.1:8795/api/v1/swarm/", "r"),
            "ws://127.0.0.1:8795/api/v1/swarm/runs/r/ws"
        );
        // An already-ws base is preserved (only the path is appended).
        assert_eq!(
            ws_endpoint("wss://coord.example/swarm", "x"),
            "wss://coord.example/swarm/runs/x/ws"
        );
    }

    #[test]
    fn backoff_is_monotonic_and_capped() {
        let cfg = ReconnectConfig {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(2),
            ..ReconnectConfig::default()
        };
        assert_eq!(backoff(0, &cfg), Duration::from_millis(100));
        assert_eq!(backoff(1, &cfg), Duration::from_millis(200));
        assert_eq!(backoff(2, &cfg), Duration::from_millis(400));
        // Grows until the cap, then stays there (no overflow at large attempts).
        assert_eq!(backoff(50, &cfg), Duration::from_secs(2));
    }

    #[test]
    fn auth_headers_are_stamped() {
        let mut h = HeaderMap::new();
        apply_auth(&mut h, &WsAuth::Bearer("tok".into()));
        assert_eq!(h.get(AUTHORIZATION).unwrap(), "Bearer tok");

        let mut h = HeaderMap::new();
        apply_auth(
            &mut h,
            &WsAuth::Internal {
                org_id: "org-1".into(),
                actor: "key:k1".into(),
            },
        );
        assert_eq!(h.get("x-daemon-org-id").unwrap(), "org-1");
        assert_eq!(h.get("x-daemon-actor").unwrap(), "key:k1");

        let mut h = HeaderMap::new();
        apply_auth(&mut h, &WsAuth::None);
        assert!(h.is_empty());
    }
}
