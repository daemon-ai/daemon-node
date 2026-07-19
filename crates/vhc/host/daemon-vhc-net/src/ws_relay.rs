// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! A loopback **relay-grade WS coordinator** — the reusable harness fixture behind the `harness`
//! feature (never a production plane).
//!
//! Byte-opaque: it accepts [`WsControlPlane`] connections, counts inbound binary frames, and
//! relays each to every OTHER connected peer (the publisher self-delivers locally, exactly the
//! deployed relay's contract the live suites pin). Knobs for the drills: `broadcast` (a
//! coordinator-emission stand-in), `sever` (force reconnects), `set_relay(false)` (a black-holing
//! relay), and captured upgrade headers (auth assertions). Promoted from this crate's WS test
//! harness so the worker live-attach suites and the multi-process acceptance smoke run against
//! one relay implementation.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::{SinkExt as _, StreamExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_hdr_async;
use tokio_tungstenite::tungstenite::handshake::server::{ErrorResponse, Request, Response};
use tokio_tungstenite::tungstenite::Message;

use crate::ws_client::{ReconnectConfig, WsAuth, WsConfig, WsControlPlane};

struct Inner {
    peers: Mutex<HashMap<u64, UnboundedSender<Message>>>,
    next_id: AtomicU64,
    received: AtomicU64,
    headers: Mutex<Vec<(String, String)>>,
    sever: Notify,
    relay: AtomicBool,
}

/// A running loopback relay-grade WS coordinator.
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
            .expect("bind relay ws listener");
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

    /// The coordinator base URL a [`WsControlPlane`] dials (`{addr}/api/v1/vhc`).
    #[must_use]
    pub fn base_url(&self) -> String {
        format!("http://{}/api/v1/vhc", self.addr)
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

    /// A coordinator emission to ALL connected peers.
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
    #[must_use]
    pub fn received(&self) -> u64 {
        self.inner.received.load(Ordering::Relaxed)
    }

    /// Currently-connected peer count.
    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.inner.peers.lock().expect("peers lock").len()
    }

    /// The captured upgrade headers of the most recent connection (auth assertion).
    #[must_use]
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
