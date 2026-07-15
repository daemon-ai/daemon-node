// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Shared harness for the `ControlPlane` conformance + iroh gossip integration suites (B2).
//!
//! The parametric conformance suite ([`conformance_fanout`] / [`conformance_dedupe`]) runs over any
//! [`Mesh`] — a set of N connected control planes each with one subscriber — so the *same* behavior
//! tests exercise both [`LoopbackGossip`](daemon_swarm_net::LoopbackGossip) (a shared in-process bus)
//! and [`IrohGossip`](daemon_swarm_net::IrohGossip) (N real iroh endpoints on loopback). The signed-
//! message helpers build canonical-CBOR `daemon_swarm_proto::SignedMessage` bytes — the opaque
//! already-signed payloads the plane carries (NET-6).

#![allow(dead_code)]

use std::sync::Arc;
use std::time::Duration;

use daemon_swarm_net::{ControlPlane, ControlSubscription};
use daemon_swarm_proto::messages::Heartbeat;
use daemon_swarm_proto::{
    from_canonical_slice, to_canonical_vec, SignedMessage, SigningKey, SwarmMessage,
    SWARM_PROTO_VERSION,
};

/// How long to wait for a message to be delivered (iroh mesh formation + flood).
pub const DELIVER: Duration = Duration::from_secs(10);
/// A short grace window to assert a duplicate is NOT delivered.
pub const GRACE: Duration = Duration::from_millis(400);

/// A mesh of N connected control planes, each with one subscriber inbox.
///
/// - Loopback: all `planes` are clones of one `Arc<LoopbackGossip>` (a shared bus); `subs` are N
///   subscriptions on it.
/// - Iroh: `planes` are N distinct `Arc<IrohGossip>` wired into a mesh; `subs` are one per node.
///
/// The shared observable property both satisfy: publishing from any plane delivers to every
/// subscriber exactly once.
pub struct Mesh {
    pub planes: Vec<Arc<dyn ControlPlane>>,
    pub subs: Vec<ControlSubscription>,
}

/// Await the next message with a timeout (`None` on timeout or close).
pub async fn recv_timeout(sub: &mut ControlSubscription, dur: Duration) -> Option<Vec<u8>> {
    tokio::time::timeout(dur, sub.recv()).await.ok().flatten()
}

/// Conformance: a message published from plane 0 reaches every subscriber exactly once.
pub async fn conformance_fanout(mesh: &mut Mesh, msg: &[u8]) {
    mesh.planes[0].publish(msg).await.expect("publish");
    for (i, sub) in mesh.subs.iter_mut().enumerate() {
        let got = recv_timeout(sub, DELIVER).await;
        assert_eq!(
            got.as_deref(),
            Some(msg),
            "subscriber {i} must receive the fanned-out message"
        );
        assert!(
            recv_timeout(sub, GRACE).await.is_none(),
            "subscriber {i} must not receive a duplicate"
        );
    }
}

/// Conformance: the same bytes published via two planes (the WS path and the gossip path) still
/// deliver exactly once to every subscriber (content-hash dedupe — NET-6).
pub async fn conformance_dedupe(mesh: &mut Mesh, msg: &[u8]) {
    mesh.planes[0]
        .publish(msg)
        .await
        .expect("publish via path 0");
    if mesh.planes.len() > 1 {
        mesh.planes[1]
            .publish(msg)
            .await
            .expect("publish via path 1");
    }
    for (i, sub) in mesh.subs.iter_mut().enumerate() {
        let got = recv_timeout(sub, DELIVER).await;
        assert_eq!(
            got.as_deref(),
            Some(msg),
            "subscriber {i} must receive the message once"
        );
        assert!(
            recv_timeout(sub, GRACE).await.is_none(),
            "subscriber {i} must dedupe the second path to one delivery"
        );
    }
}

/// A deterministic ed25519 signing key for tests.
pub fn signing_key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

/// Canonical-CBOR bytes of a signed `Heartbeat` — a valid opaque control-plane payload.
pub fn signed_heartbeat_bytes(key: &SigningKey, round: u64) -> Vec<u8> {
    let payload = SwarmMessage::Heartbeat(Heartbeat { round, ready: None });
    let signed = SignedMessage::sign(key, SWARM_PROTO_VERSION, payload).expect("sign");
    to_canonical_vec(&signed).expect("encode")
}

/// Tamper a signed message's signature and re-encode — a payload whose signature no longer verifies
/// (distinct bytes from the valid message, so it is not deduped against it).
pub fn tampered_bytes(valid: &[u8]) -> Vec<u8> {
    let mut signed: SignedMessage = from_canonical_slice(valid).expect("decode");
    signed.sig.0[0] ^= 0xff;
    to_canonical_vec(&signed).expect("re-encode")
}

/// The in-process iroh multi-node harness (feature-gated): N endpoints on loopback with static
/// discovery, no relay. Ports the iroh-gossip 0.101 `net.rs::gossip_net_smoke` / Psyche `router.rs`
/// spawn pattern to explicit-roster addressing (no external discovery).
#[cfg(feature = "iroh")]
pub mod iroh_harness {
    use super::*;
    use std::net::SocketAddr;

    use daemon_swarm_net::{IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig};

    /// A fresh loopback bind address (OS-assigned port).
    pub fn loopback() -> SocketAddr {
        "127.0.0.1:0".parse().expect("loopback addr")
    }

    /// Rebroadcast disabled — deterministic single-flood delivery for most tests.
    pub fn no_rebroadcast() -> RebroadcastConfig {
        RebroadcastConfig {
            enabled: false,
            ..RebroadcastConfig::default()
        }
    }

    /// Connect one iroh node on loopback with no relay and an empty initial roster.
    pub async fn connect_node(seed: u8, rebroadcast: RebroadcastConfig) -> Arc<IrohGossip> {
        let config = IrohGossipConfig {
            secret_key: [seed; 32],
            relay_urls: vec![],
            roster: vec![],
            topic_input: [0x42; 32],
            rebroadcast,
            bind_addr: Some(loopback()),
        };
        Arc::new(
            IrohGossip::connect(config)
                .await
                .expect("connect iroh node"),
        )
    }

    /// Connect one iroh node pointed at a relay (relay-only reachability path).
    pub async fn connect_relay_node(seed: u8, relay_url: &str) -> Arc<IrohGossip> {
        let config = IrohGossipConfig {
            secret_key: [seed; 32],
            relay_urls: vec![relay_url.to_string()],
            roster: vec![],
            topic_input: [0x42; 32],
            rebroadcast: no_rebroadcast(),
            bind_addr: Some(loopback()),
        };
        Arc::new(
            IrohGossip::connect(config)
                .await
                .expect("connect relay iroh node"),
        )
    }

    /// Distribute the full roster (each node's dialable `local_peer`) to every node (the
    /// admission/ensure_gossip_connected step).
    pub async fn wire_roster(nodes: &[Arc<IrohGossip>]) {
        let roster: Vec<IrohPeer> = nodes.iter().map(|n| n.local_peer()).collect();
        for node in nodes {
            node.update_roster(roster.clone())
                .await
                .expect("update_roster");
        }
    }

    /// Poll until every node has at least `min_neighbors` gossip neighbors (mesh formed).
    pub async fn wait_for_mesh(nodes: &[Arc<IrohGossip>], min_neighbors: usize) {
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            if nodes.iter().all(|n| n.neighbor_count() >= min_neighbors) {
                return;
            }
            if std::time::Instant::now() > deadline {
                let counts: Vec<_> = nodes.iter().map(|n| n.neighbor_count()).collect();
                panic!("iroh mesh did not form within 20s: neighbor counts {counts:?}");
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// Build an N-node loopback iroh mesh (connect all, wire the roster, wait for formation).
    pub async fn build_mesh(n: usize, rebroadcast: RebroadcastConfig) -> Vec<Arc<IrohGossip>> {
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(connect_node(i as u8 + 1, rebroadcast.clone()).await);
        }
        wire_roster(&nodes).await;
        wait_for_mesh(&nodes, 1).await;
        nodes
    }

    /// Wrap live nodes as a [`Mesh`] (subscribe each node once).
    pub fn mesh_from(nodes: &[Arc<IrohGossip>]) -> Mesh {
        let planes = nodes
            .iter()
            .map(|n| n.clone() as Arc<dyn ControlPlane>)
            .collect();
        let subs = nodes.iter().map(|n| n.subscribe()).collect();
        Mesh { planes, subs }
    }
}

/// An in-process mock of the cloud `RunCoordinatorDO` WS surface (feature-gated): it accepts peer
/// WebSocket upgrades on loopback and **disseminates** every inbound binary frame to the *other*
/// connected peers (never echoes the sender) — the `webSocketMessage` `broadcast([bytes], ws)`
/// contract from `apps/swarm/src/coordinator/do.ts`. It can also `broadcast` a coordinator emission
/// to ALL peers, `sever` every live socket (force reconnect), capture the upgrade headers (auth
/// assertion), and count received frames. Enough of the DO framing for the parametric `ControlPlane`
/// conformance suite + the reconnect/resubscribe + dual-plane dedupe drills — NOT a full coordinator
/// (no signature verify / round-state; C1 owns the cloud side, Merge 1 does the live cross-lane check).
#[cfg(feature = "ws")]
pub mod ws_harness {
    use super::*;
    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use futures::{SinkExt as _, StreamExt as _};
    use tokio::net::{TcpListener, TcpStream};
    use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
    use tokio::sync::Notify;
    use tokio::task::JoinHandle;
    use tokio_tungstenite::accept_hdr_async;
    use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
    use tokio_tungstenite::tungstenite::Message;

    use daemon_swarm_net::{ReconnectConfig, WsAuth, WsConfig, WsControlPlane};

    struct Inner {
        peers: Mutex<HashMap<u64, UnboundedSender<Message>>>,
        next_id: AtomicU64,
        received: AtomicU64,
        headers: Mutex<Vec<(String, String)>>,
        sever: Notify,
        relay: AtomicBool,
    }

    /// A running mock coordinator WS server.
    pub struct MockWsCoordinator {
        addr: SocketAddr,
        inner: Arc<Inner>,
        accept_task: JoinHandle<()>,
    }

    impl Drop for MockWsCoordinator {
        fn drop(&mut self) {
            self.accept_task.abort();
        }
    }

    impl MockWsCoordinator {
        /// Bind on loopback and start accepting connections.
        pub async fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind mock ws");
            let addr = listener.local_addr().expect("local addr");
            let inner = Arc::new(Inner {
                peers: Mutex::new(HashMap::new()),
                next_id: AtomicU64::new(0),
                received: AtomicU64::new(0),
                headers: Mutex::new(Vec::new()),
                sever: Notify::new(),
                relay: AtomicBool::new(true),
            });
            let inner2 = inner.clone();
            let accept_task = tokio::spawn(async move {
                while let Ok((stream, _)) = listener.accept().await {
                    tokio::spawn(handle_conn(stream, inner2.clone()));
                }
            });
            Self {
                addr,
                inner,
                accept_task,
            }
        }

        /// The coordinator base URL a [`WsControlPlane`] dials (`{addr}/api/v1/swarm`).
        pub fn base_url(&self) -> String {
            format!("http://{}/api/v1/swarm", self.addr)
        }

        /// Connect a [`WsControlPlane`] client to this coordinator for `run_id`.
        pub async fn client(
            &self,
            run_id: &str,
            auth: WsAuth,
            reconnect: ReconnectConfig,
        ) -> WsControlPlane {
            WsControlPlane::connect(WsConfig {
                base_url: self.base_url(),
                run_id: run_id.to_string(),
                auth,
                reconnect,
            })
            .await
            .expect("connect ws control plane")
        }

        /// A coordinator emission to ALL connected peers (RoundOpen / StorageReceipt / RoundRecord).
        pub fn broadcast(&self, frame: Vec<u8>) {
            let peers = self.inner.peers.lock().expect("peers lock");
            for tx in peers.values() {
                let _ = tx.send(Message::binary(frame.clone()));
            }
        }

        /// Close every currently-connected socket (force the clients to reconnect).
        pub fn sever(&self) {
            self.inner.sever.notify_waiters();
        }

        /// Whether inbound frames are relayed to the other peers (default true).
        pub fn set_relay(&self, on: bool) {
            self.inner.relay.store(on, Ordering::Relaxed);
        }

        /// Frames received from all peers so far.
        pub fn received(&self) -> u64 {
            self.inner.received.load(Ordering::Relaxed)
        }

        /// Currently-connected peer count.
        pub fn peer_count(&self) -> usize {
            self.inner.peers.lock().expect("peers lock").len()
        }

        /// The captured upgrade headers of the most recent connection (auth assertion).
        pub fn last_headers(&self) -> Vec<(String, String)> {
            self.inner.headers.lock().expect("headers lock").clone()
        }

        /// Block until at least `n` peers are connected (mesh formed), or panic after 10 s.
        pub async fn wait_peers(&self, n: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while self.peer_count() < n {
                if Instant::now() > deadline {
                    panic!("only {} of {n} ws peers connected", self.peer_count());
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
    }

    // tungstenite's handshake `Callback` returns `Result<Response, ErrorResponse>`; the `Err` variant
    // is large but is the library's fixed shape (mirrors `daemon_host::ws`).
    #[allow(clippy::result_large_err)]
    async fn handle_conn(stream: TcpStream, inner: Arc<Inner>) {
        let hdr_slot = inner.clone();
        let callback = move |req: &Request, resp: Response| {
            let mut hs = hdr_slot.headers.lock().expect("headers lock");
            hs.clear();
            for (name, value) in req.headers() {
                if let Ok(v) = value.to_str() {
                    hs.push((name.as_str().to_string(), v.to_string()));
                }
            }
            Ok::<Response, ErrorResponse>(resp)
        };
        let ws = match accept_hdr_async(stream, callback).await {
            Ok(ws) => ws,
            Err(_) => return,
        };
        let (mut write, mut read) = ws.split();
        let (tx, mut rx) = unbounded_channel::<Message>();
        let id = inner.next_id.fetch_add(1, Ordering::Relaxed);
        inner.peers.lock().expect("peers lock").insert(id, tx);

        loop {
            tokio::select! {
                biased;
                () = inner.sever.notified() => {
                    let _ = write.close().await;
                    break;
                }
                out = rx.recv() => match out {
                    Some(msg) => {
                        if write.send(msg).await.is_err() {
                            break;
                        }
                    }
                    None => break,
                },
                inbound = read.next() => match inbound {
                    Some(Ok(Message::Binary(bytes))) => {
                        inner.received.fetch_add(1, Ordering::Relaxed);
                        if inner.relay.load(Ordering::Relaxed) {
                            let peers = inner.peers.lock().expect("peers lock");
                            for (pid, ptx) in peers.iter() {
                                if *pid != id {
                                    let _ = ptx.send(Message::binary(bytes.clone()));
                                }
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                    Some(Ok(_)) => {}
                },
            }
        }
        inner.peers.lock().expect("peers lock").remove(&id);
    }

    /// Build an N-client WS mesh against `coord` (all subscribed, all connected).
    pub async fn build_ws_mesh(coord: &MockWsCoordinator, n: usize) -> Vec<Arc<WsControlPlane>> {
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(Arc::new(
                coord.client("run-conf", WsAuth::None, no_reconnect()).await,
            ));
        }
        coord.wait_peers(n).await;
        nodes
    }

    /// Wrap live WS clients as a [`Mesh`] (subscribe each once).
    pub fn ws_mesh_from(nodes: &[Arc<WsControlPlane>]) -> Mesh {
        let planes = nodes
            .iter()
            .map(|n| n.clone() as Arc<dyn ControlPlane>)
            .collect();
        let subs = nodes.iter().map(|n| n.subscribe()).collect();
        Mesh { planes, subs }
    }

    /// A one-shot (no-reconnect) policy for conformance meshes.
    pub fn no_reconnect() -> ReconnectConfig {
        ReconnectConfig {
            enabled: false,
            ..ReconnectConfig::default()
        }
    }

    /// A fast-reconnect policy for the reconnect drill.
    pub fn fast_reconnect() -> ReconnectConfig {
        ReconnectConfig {
            enabled: true,
            initial_backoff: Duration::from_millis(20),
            max_backoff: Duration::from_millis(200),
            max_attempts: None,
        }
    }
}
