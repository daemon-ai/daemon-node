// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
#![allow(clippy::disallowed_methods, clippy::disallowed_types, dead_code)]

//! The local run-registry + seat-CAS + checkpoint-pointer fixture the acceptance nodes discover
//! runs and claim the coordinator seat against — the untrusted-storage counterpart of the cloud
//! `apps/vhc` worker, served over real loopback HTTP so the nodes' production `RegistryClient`
//! drives it unchanged.
//!
//! State is a [`FakeSeatRegistry`] (the normative seat CAS fold) plus a run descriptor, a
//! checkpoint pointer, and an object store for presigned GET/PUT (the R2 payload tier and the
//! envelope object). It never verifies a signature or judges authority — peers do that; this is
//! storage.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use daemon_vhc_net::FakeSeatRegistry;
use daemon_vhc_proto::{
    blake3_hash, from_canonical_slice, to_canonical_vec, SeatDecision, SeatLease, SeatRelease,
    SeatState,
};
use daemon_vhc_testkit::LiveGenesis;
use futures::{SinkExt as _, StreamExt as _};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio_tungstenite::tungstenite::Message;

/// A published checkpoint pointer (`round → content hash`).
#[derive(Clone)]
pub struct Checkpoint {
    pub round: u64,
    pub hash: String,
    pub size: u64,
}

/// The registry fixture's mutable state.
pub struct FixtureRegistry {
    run_label: String,
    envelope_hash: String,
    envelope_bytes: Vec<u8>,
    proto_version: u32,
    seats: FakeSeatRegistry,
    checkpoint: Mutex<Option<Checkpoint>>,
    /// Presign object store: object key → bytes (the R2 payload tier + the envelope object).
    objects: Mutex<HashMap<String, Vec<u8>>>,
    base: Mutex<String>,
    /// The WS relay's connected control-plane peers (id → outbound sender): a byte-opaque
    /// dissemination relay (a frame from one peer reaches every OTHER peer), the deployed
    /// coordinator relay's contract the seat holder + trainers speak.
    ws_peers: Mutex<HashMap<u64, UnboundedSender<Message>>>,
    ws_next: Mutex<u64>,
}

impl FixtureRegistry {
    /// The seat registry (for direct CAS drives in the seat/negative gates).
    pub fn seats(&self) -> &FakeSeatRegistry {
        &self.seats
    }

    /// Read a seat slot.
    pub fn read_seat(&self, role: &str) -> SeatState {
        self.seats.read(&self.run_label, role)
    }

    /// Publish a checkpoint pointer directly (test seeding).
    pub fn set_checkpoint(&self, ckpt: Checkpoint) {
        *self.checkpoint.lock().unwrap() = Some(ckpt);
    }

    /// Seed a presign object directly (the R2 payload tier / envelope object).
    pub fn put_object(&self, key: &str, bytes: Vec<u8>) {
        self.objects.lock().unwrap().insert(key.to_string(), bytes);
    }

    /// Read a presign object.
    pub fn get_object(&self, key: &str) -> Option<Vec<u8>> {
        self.objects.lock().unwrap().get(key).cloned()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Author the descriptor JSON for the served run (the `{ "data": … }` shape the client decodes).
fn descriptor_json(reg: &FixtureRegistry) -> serde_json::Value {
    serde_json::json!({
        "run_id": reg.run_label,
        "schema": 2,
        "proto_version": reg.proto_version,
        "envelope_hash": reg.envelope_hash,
        "author_pubkey": "00",
        "artifacts": [{
            "path": "envelope.cbor",
            "blake3": reg.envelope_hash,
            "size": reg.envelope_bytes.len(),
        }],
        "update_max_bytes": 1_048_576u64,
        "min_peers": 2,
        "max_peers": 4,
        "rounds": serde_json::Value::Null,
        "created_at": 0,
        "envelope_key": format!("runs/{}/envelope.cbor", reg.run_label),
    })
}

/// Start the fixture over loopback HTTP; returns the shared state, the base URL the nodes point
/// `[vhc.registry] base` at, and the serving task handle.
pub async fn serve(
    genesis: &LiveGenesis,
    run_label: &str,
    port: u16,
) -> (Arc<FixtureRegistry>, String, tokio::task::JoinHandle<()>) {
    let envelope_hash = blake3_hash(&genesis.wire).to_hex().to_string();
    let listener = TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind registry");
    let addr = listener.local_addr().expect("registry addr");
    let base = format!("http://{addr}/api/v1/vhc");

    let reg = Arc::new(FixtureRegistry {
        run_label: run_label.to_string(),
        envelope_hash,
        envelope_bytes: genesis.wire.clone(),
        proto_version: u32::from(daemon_vhc_proto::VHC_PROTO_VERSION.0),
        seats: FakeSeatRegistry::new(),
        checkpoint: Mutex::new(None),
        objects: Mutex::new(HashMap::new()),
        base: Mutex::new(base.clone()),
        ws_peers: Mutex::new(HashMap::new()),
        ws_next: Mutex::new(0),
    });
    // The envelope object lives under its §11.3 key for the presigned GET.
    reg.put_object(
        &format!("runs/{run_label}/envelope.cbor"),
        genesis.wire.clone(),
    );
    *reg.base.lock().unwrap() = format!("http://{addr}");

    let state = reg.clone();
    let task = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                return;
            };
            let st = state.clone();
            tokio::spawn(async move {
                let _ = handle_conn(stream, st).await;
            });
        }
    });
    (reg, base, task)
}

/// A minimal HTTP/1.1 request line + headers + body reader (loopback, single request per conn is
/// what `EgressClient` issues; we read one and reply, then close).
async fn handle_conn(mut stream: TcpStream, reg: Arc<FixtureRegistry>) -> Result<(), Infallible> {
    // Peek the request head WITHOUT consuming it: a WebSocket upgrade is handed to the relay
    // (tungstenite performs its own handshake read); everything else is the REST path below.
    // Peek can return before the whole header has arrived, so retry until the header terminator
    // is visible (or a bounded number of attempts elapses).
    let mut peek = [0u8; 4096];
    let mut is_ws = false;
    for _ in 0..200 {
        let n = match stream.peek(&mut peek).await {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(_) => return Ok(()),
        };
        if find_subslice(&peek[..n], b"\r\n\r\n").is_some() || n == peek.len() {
            is_ws = String::from_utf8_lossy(&peek[..n])
                .to_ascii_lowercase()
                .contains("upgrade: websocket");
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    if is_ws {
        ws_relay(stream, reg).await;
        return Ok(());
    }

    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    // Read headers (until CRLFCRLF), then the declared body length.
    let header_end = loop {
        let n = match stream.read(&mut tmp).await {
            Ok(0) => return Ok(()),
            Ok(n) => n,
            Err(_) => return Ok(()),
        };
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 1 << 20 {
            return Ok(());
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = head.lines();
    let request_line = lines.next().unwrap_or_default().to_string();
    let mut content_length = 0usize;
    for line in lines {
        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    while buf.len() < header_end + content_length {
        let n = match stream.read(&mut tmp).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = buf[header_end..(header_end + content_length).min(buf.len())].to_vec();

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    let path = target.split('?').next().unwrap_or_default().to_string();

    let resp = route(&reg, &method, &path, &body);
    let _ = stream.write_all(&resp).await;
    let _ = stream.flush().await;
    Ok(())
}

fn http(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    out.extend_from_slice(body);
    out
}

fn json_ok(value: &serde_json::Value) -> Vec<u8> {
    http("200 OK", "application/json", value.to_string().as_bytes())
}

/// The route table (the subset of the `apps/vhc` surface the acceptance path exercises).
fn route(reg: &FixtureRegistry, method: &str, path: &str, body: &[u8]) -> Vec<u8> {
    let p = path.strip_prefix("/api/v1/vhc").unwrap_or(path);
    let run = &reg.run_label;

    // GET /runs — the run list.
    if method == "GET" && p == "/runs" {
        return json_ok(&serde_json::json!({ "data": [descriptor_json(reg)] }));
    }
    // GET /runs/:id — one descriptor (404 for an unknown run).
    if method == "GET" && p == format!("/runs/{run}") {
        return json_ok(&serde_json::json!({ "data": descriptor_json(reg) }));
    }
    // GET /runs/:id/state — the checkpoint pointer projection.
    if method == "GET" && p == format!("/runs/{run}/state") {
        let ckpt = reg.checkpoint.lock().unwrap().clone();
        let checkpoint = ckpt.map(|c| {
            serde_json::json!({ "round": c.round, "hash": c.hash, "size": c.size, "cross_checked": true })
        });
        return json_ok(&serde_json::json!({
            "data": { "phase": "round_train", "round": 0, "epoch": 0, "finished": false, "checkpoint": checkpoint }
        }));
    }
    // POST /runs/:id/checkpoint — record a pointer.
    if method == "POST" && p == format!("/runs/{run}/checkpoint") {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) {
            let round = v
                .get("round")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let hash = v
                .get("hash")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string();
            let size = v
                .get("size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            reg.set_checkpoint(Checkpoint { round, hash, size });
        }
        return json_ok(&serde_json::json!({ "data": { "ok": true } }));
    }
    // POST /runs/:id/presign — mint a loopback object URL at the requested key.
    if method == "POST" && p == format!("/runs/{run}/presign") {
        return presign(reg, body);
    }
    // Object store PUT/GET (the presigned URLs point here).
    if let Some(key) = p.strip_prefix("/obj/") {
        if method == "PUT" {
            reg.objects
                .lock()
                .unwrap()
                .insert(key.to_string(), body.to_vec());
            return http("200 OK", "application/octet-stream", b"");
        }
        if method == "GET" {
            return match reg.get_object(key) {
                Some(bytes) => http("200 OK", "application/octet-stream", &bytes),
                None => http("404 Not Found", "text/plain", b"missing object"),
            };
        }
    }
    // Seat CAS: GET/PUT/POST(heartbeat)/DELETE {base}/runs/:id/seat/:role.
    if let Some(role) = p.strip_prefix(&format!("/runs/{run}/seat/")) {
        return seat(reg, method, role, body);
    }

    http("404 Not Found", "text/plain", b"no route")
}

/// The presign responder: parse the request, compute the §11.3 object key with the SAME
/// `r2_object_key` mapping the production `R2Store` uses (so PUT and GET land on one key), and
/// return a loopback `/obj/<key>` URL (healthy expiry).
fn presign(reg: &FixtureRegistry, body: &[u8]) -> Vec<u8> {
    let Ok(req) = serde_json::from_slice::<daemon_vhc_net::PresignRequest>(body) else {
        return http("400 Bad Request", "text/plain", b"bad presign");
    };
    let key = match daemon_vhc_net::r2_object_key(&daemon_vhc_net::RunId::new(&reg.run_label), &req)
    {
        Ok(k) => k,
        Err(e) => return http("400 Bad Request", "text/plain", e.to_string().as_bytes()),
    };
    let base = reg.base.lock().unwrap().clone();
    let resp = serde_json::json!({
        "url": format!("{base}/obj/{key}"),
        "expires_at": now_ms() / 1000 + 900,
        "headers": {},
    });
    json_ok(&resp)
}

/// The seat CAS handler over the normative fold (untrusted storage — structural only).
fn seat(reg: &FixtureRegistry, method: &str, role_and_tail: &str, body: &[u8]) -> Vec<u8> {
    let (role, heartbeat) = match role_and_tail.strip_suffix("/heartbeat") {
        Some(r) => (r, true),
        None => (role_and_tail, false),
    };
    let now = now_ms();
    match method {
        "GET" => {
            let state = reg.seats.read(&reg.run_label, role);
            http(
                "200 OK",
                "application/cbor",
                &to_canonical_vec(&state).expect("encode seat state"),
            )
        }
        "PUT" | "POST" => {
            let Ok(lease) = from_canonical_slice::<SeatLease>(body) else {
                return http("400 Bad Request", "text/plain", b"bad lease");
            };
            let resp = if heartbeat {
                reg.seats.renew(&reg.run_label, &lease, now)
            } else {
                reg.seats.claim(&reg.run_label, &lease, now)
            };
            let bytes = to_canonical_vec(&resp).expect("encode seat resp");
            if resp.decision == SeatDecision::Accepted {
                http("200 OK", "application/cbor", &bytes)
            } else {
                http("409 Conflict", "application/cbor", &bytes)
            }
        }
        "DELETE" => {
            let Ok(release) = from_canonical_slice::<SeatRelease>(body) else {
                return http("400 Bad Request", "text/plain", b"bad release");
            };
            let resp = reg.seats.release(&reg.run_label, role, &release, now);
            let bytes = to_canonical_vec(&resp).expect("encode seat resp");
            if resp.decision == SeatDecision::Accepted {
                http("200 OK", "application/cbor", &bytes)
            } else {
                http("409 Conflict", "application/cbor", &bytes)
            }
        }
        _ => http("405 Method Not Allowed", "text/plain", b"method"),
    }
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// One WS control-plane connection: accept the upgrade, then relay every inbound binary frame to
/// every OTHER connected peer (the publisher self-delivers locally, exactly the deployed relay's
/// contract the live suites pin). Byte-opaque — no signature check, no round state.
async fn ws_relay(stream: TcpStream, reg: Arc<FixtureRegistry>) {
    let ws = match tokio_tungstenite::accept_async(stream).await {
        Ok(ws) => ws,
        Err(_) => return,
    };
    let (mut write, mut read) = ws.split();
    let (tx, mut rx) = unbounded_channel::<Message>();
    let id = {
        let mut next = reg.ws_next.lock().unwrap();
        let id = *next;
        *next += 1;
        reg.ws_peers.lock().unwrap().insert(id, tx);
        id
    };
    loop {
        tokio::select! {
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
                    let peers = reg.ws_peers.lock().unwrap();
                    for (pid, ptx) in peers.iter() {
                        if *pid != id {
                            let _ = ptx.send(Message::Binary(bytes.clone()));
                        }
                    }
                }
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                Some(Ok(_)) => {}
            },
        }
    }
    reg.ws_peers.lock().unwrap().remove(&id);
}
