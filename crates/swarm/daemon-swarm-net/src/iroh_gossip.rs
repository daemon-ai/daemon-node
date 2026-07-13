// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`IrohGossip`] — the real iroh-gossip [`ControlPlane`] (spec §7.1; TDD NET-6).
//!
//! This ports Psyche's verified gossip stack (reference pack, workspace
//! `/home/j/experiments/decentralised-llm-training/psyche`) to the **iroh 1.0.2 / iroh-gossip
//! 0.101.0** API the tree resolved (the plan pinned 0.97; iroh 0.97/0.98 are unresolvable against
//! our frozen `sha2 0.11` tree — see `swarm-ledger-b2.md` for the full delta table). Every ported
//! shape cites its Psyche `file:line` anchor and records the 0.97 -> 1.0 delta inline.
//!
//! The plane carries **already-signed opaque bytes** — signing/verification is proto's surface
//! (`daemon_swarm_proto::SignedMessage`, canonical CBOR + ed25519), not the transport's (§7.1:
//! gossip is dissemination, never arbitration). Delivery matches [`LoopbackGossip`]
//! (crate::gossip::LoopbackGossip), the conformance twin: publish -> every subscriber, once.
//!
//! ## Rebroadcast frame + two-layer dedupe (the round-critical delivery-assurance knob)
//!
//! iroh-gossip dedupes internally by `MessageId = blake3(broadcast content)` for
//! `message_id_retention` (`iroh_gossip::proto::plumtree` — id is the blake3 of the content and is
//! *validated* on receive), so re-broadcasting identical bytes is a gossip-layer no-op. Psyche's
//! delivery-assurance rebroadcast bumps a **nonce** (`shared/client/src/client.rs:490-505`). We
//! reconcile that with our content-hash app dedupe ([`Deduper`]) by framing every broadcast as
//! `[nonce: u64 LE][payload: already-signed bytes]`: bumping the nonce changes the outer bytes (a
//! new gossip `MessageId` -> forced re-flood) while the receiver strips the frame and dedupes by the
//! **inner** payload hash (one delivery). The ed25519 signature is over the inner payload only, so a
//! nonce bump never invalidates it.

use std::collections::{HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt as _;
use iroh::address_lookup::memory::MemoryLookup;
use iroh::endpoint::{presets, QuicTransportConfig};
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMap, RelayMode, RelayUrl, SecretKey};
use iroh_gossip::api::{Event, GossipReceiver, GossipSender};
use iroh_gossip::net::Gossip;
use iroh_gossip::proto::{HyparviewConfig, PlumtreeConfig, TopicId};
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

use daemon_swarm_proto::blake3_hash;

use crate::dedupe::Deduper;
use crate::transport::{ControlPlane, ControlSubscription};
use crate::SwarmNetError;

/// Gossip max message size (Psyche `lib.rs:461`). Our frame prepends an 8-byte nonce, so a
/// control-message payload must be at most `MAX_MESSAGE_SIZE - NONCE_LEN` (signed messages are
/// sub-4 KB by design, §7.1).
const MAX_MESSAGE_SIZE: usize = 4096;
/// The rebroadcast-frame nonce width (little-endian `u64`).
const NONCE_LEN: usize = 8;
/// Bootstrap-neighbor cap adopted from Psyche's `ensure_gossip_connected`
/// (`shared/client/src/client.rs:773`): only add enough peers to reach this many gossip neighbors.
const MAX_BOOTSTRAP_PEERS: usize = 3;

/// Rebroadcast tuning for round-critical delivery-assurance (Psyche `client.rs:490-505`).
#[derive(Clone, Debug)]
pub struct RebroadcastConfig {
    /// Whether the origin periodically re-floods recent messages with a bumped nonce. Default on.
    pub enabled: bool,
    /// The rebroadcast cadence (Psyche re-broadcasts every 10 s).
    pub interval: Duration,
    /// How many recent messages to keep in the rebroadcast ring.
    pub ring_capacity: usize,
}

impl Default for RebroadcastConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(10),
            ring_capacity: 32,
        }
    }
}

/// One roster member's iroh addressing: the iroh `EndpointId` plus how to reach it (direct IP
/// addresses and/or a relay URL). Built from the admission roster — the `Join.iroh_id` binding
/// (§7.2) supplies `endpoint_id`; the addresses come from the run's transport config / relay.
///
/// This is the explicit-addressing path: the program prefers dialing roster addrs over any public
/// discovery service (no DNS/pkarr), so a peer with only a `relay_url` is dialable via the relay and
/// a peer with `direct_addrs` is dialable directly (LAN/loopback).
#[derive(Clone, Debug)]
pub struct IrohPeer {
    /// The peer's iroh `EndpointId` (32 raw bytes — proto `IrohId`).
    pub endpoint_id: [u8; 32],
    /// Direct socket addresses (LAN/loopback); may be empty for relay-only reachability.
    pub direct_addrs: Vec<SocketAddr>,
    /// Home relay URL (NAT-proof reachability); `None` for direct-only.
    pub relay_url: Option<String>,
}

/// Construction surface for [`IrohGossip`] (frozen at Merge 2).
#[derive(Clone, Debug)]
pub struct IrohGossipConfig {
    /// The iroh secret key (32 bytes) — **separate** from the node ed25519 identity (§7.2). The
    /// derived `EndpointId` is exposed via [`IrohGossip::node_id`] so the Join flow carries the
    /// node-key <-> iroh-key binding (`Join.iroh_id`).
    pub secret_key: [u8; 32],
    /// Relay URLs from the envelope / run transport config (NOT hardcoded — the run author pins
    /// them, §7.4). Empty -> `RelayMode::Disabled` (direct-only, e.g. loopback tests).
    pub relay_urls: Vec<String>,
    /// The peer roster to bootstrap the gossip mesh from (admission/Join flow).
    pub roster: Vec<IrohPeer>,
    /// The topic-derivation input: the frozen envelope hash (`FrozenEnvelope::hash`). The topic is
    /// `blake3(envelope hash)` (delta from Psyche's `sha256("psyche gossip" ++ run_id)`).
    pub topic_input: [u8; 32],
    /// Delivery-assurance rebroadcast knob.
    pub rebroadcast: RebroadcastConfig,
    /// Optional bind address (default `0.0.0.0:0`); tests bind `127.0.0.1:0` for direct loopback.
    pub bind_addr: Option<SocketAddr>,
}

/// State shared between the plane, the receive loop, and the rebroadcast loop.
struct Shared {
    /// Live local subscribers (each a distinct [`ControlSubscription`] inbox).
    subscribers: Vec<UnboundedSender<Vec<u8>>>,
    /// Content-hash dedupe over the **inner** payload bytes (the reusable NET-6 rule, [`Deduper`]).
    dedupe: Deduper,
    /// Recently-published inner payloads with their current nonce, for delivery-assurance rebroadcast.
    ring: VecDeque<(Vec<u8>, u64)>,
    /// Cap on `ring`.
    ring_capacity: usize,
    /// Current direct gossip neighbors (tracked from `NeighborUp`/`NeighborDown` for the roster cap).
    neighbors: HashSet<EndpointId>,
    /// Our own endpoint id (never dial/self-count).
    self_id: EndpointId,
}

/// The real iroh-gossip control plane: publish/subscribe of already-signed opaque bytes over an
/// iroh QUIC gossip mesh, with the [`Deduper`] delivery contract of [`LoopbackGossip`].
///
/// [`LoopbackGossip`]: crate::gossip::LoopbackGossip
pub struct IrohGossip {
    endpoint: Endpoint,
    router: Router,
    sender: GossipSender,
    node_id: [u8; 32],
    relay_urls: Vec<String>,
    lookup: MemoryLookup,
    shared: Arc<Mutex<Shared>>,
    tasks: Mutex<Vec<JoinHandle<()>>>,
}

impl IrohGossip {
    /// Build the endpoint + gossip + router, seed the roster into static discovery, subscribe to the
    /// topic, and spawn the receive + rebroadcast loops.
    ///
    /// Endpoint build ports Psyche `shared/network/src/lib.rs:343-378`: secret key, relay mode from
    /// config, QUIC transport tuning (120 s idle / 5 s keepalive, `lib.rs:344-348`). Discovery is
    /// **explicit** ([`MemoryLookup`] seeded from the roster) — no public DNS/pkarr service (delta:
    /// Psyche's `N0` discovery path; we prefer roster addrs per the program brief).
    pub async fn connect(config: IrohGossipConfig) -> Result<Self, SwarmNetError> {
        let secret = SecretKey::from_bytes(&config.secret_key);
        let node_id = *secret.public().as_bytes();

        // QUIC transport tuning — Psyche `lib.rs:344-348` (unchanged 0.97 -> 1.0).
        let idle = Duration::from_secs(120);
        let transport = QuicTransportConfig::builder()
            .max_idle_timeout(Some(idle.try_into().map_err(|e| {
                SwarmNetError::Transport(format!("invalid idle timeout: {e}"))
            })?))
            .keep_alive_interval(Duration::from_secs(5))
            .build();

        // Relay mode — Psyche `lib.rs:350-354`. `Custom(map)` from the envelope-pinned URLs, else
        // `Disabled` (direct-only). Delta: Psyche hardcodes a relay map; ours comes from config.
        let relay_mode = match relay_map(&config.relay_urls)? {
            Some(map) => RelayMode::Custom(map),
            None => RelayMode::Disabled,
        };

        // `presets::Minimal` = crypto provider only, no DNS/pkarr discovery (delta from Psyche's
        // `presets::N0`, which adds the public n0 lookup — we dial explicit roster addrs instead).
        let mut builder = Endpoint::builder(presets::Minimal)
            .secret_key(secret)
            .relay_mode(relay_mode)
            .transport_config(transport);
        if let Some(addr) = config.bind_addr {
            builder = builder
                .bind_addr(addr)
                .map_err(|e| SwarmNetError::Transport(format!("invalid bind addr: {e}")))?;
        }
        let endpoint = builder
            .bind()
            .await
            .map_err(|e| SwarmNetError::Transport(format!("endpoint bind failed: {e}")))?;

        // Explicit static discovery seeded from the roster (delta: Psyche's `LocalTestDiscovery` /
        // `MemoryLookup`, `lib.rs:366-368`). Added post-bind, matching the iroh-gossip 1.0 harness.
        let lookup = MemoryLookup::new();
        for peer in &config.roster {
            lookup.add_endpoint_info(peer_to_addr(peer)?);
        }
        if let Ok(services) = endpoint.address_lookup() {
            services.add(lookup.clone());
        }

        // Gossip init — Psyche `lib.rs:459-474` (Hyparview + Plumtree tuning identical in 0.101).
        let gossip = Gossip::builder()
            .max_message_size(MAX_MESSAGE_SIZE)
            .membership_config(HyparviewConfig {
                active_view_capacity: 8,
                shuffle_interval: Duration::from_secs(30),
                neighbor_request_timeout: Duration::from_secs(2),
                ..HyparviewConfig::default()
            })
            .broadcast_config(PlumtreeConfig {
                graft_timeout_2: Duration::from_millis(200),
                message_cache_retention: Duration::from_secs(60),
                message_id_retention: Duration::from_secs(2 * 60),
                ..PlumtreeConfig::default()
            })
            .spawn(endpoint.clone());

        // Router accepts only the gossip ALPN — no blobs (P4), no model-sharing. Psyche
        // `router.rs:32-46` accepts three; we accept one.
        let router = Router::builder(endpoint.clone())
            .accept(iroh_gossip::ALPN, gossip.clone())
            .spawn();

        let self_id = endpoint.id();
        let topic = derive_topic(&config.topic_input);

        // Bootstrap ids = roster minus self (Psyche `lib.rs:337,500-503`).
        let bootstrap: Vec<EndpointId> = config
            .roster
            .iter()
            .filter_map(|p| EndpointId::from_bytes(&p.endpoint_id).ok())
            .filter(|id| *id != self_id)
            .collect();

        let topic_handle = gossip
            .subscribe(topic, bootstrap)
            .await
            .map_err(|e| SwarmNetError::Transport(format!("gossip subscribe failed: {e}")))?;
        let (sender, receiver) = topic_handle.split();

        let shared = Arc::new(Mutex::new(Shared {
            subscribers: Vec::new(),
            dedupe: Deduper::new(),
            ring: VecDeque::new(),
            ring_capacity: config.rebroadcast.ring_capacity.max(1),
            neighbors: HashSet::new(),
            self_id,
        }));

        let mut tasks = vec![tokio::spawn(receive_loop(receiver, shared.clone()))];
        if config.rebroadcast.enabled {
            tasks.push(tokio::spawn(rebroadcast_loop(
                sender.clone(),
                shared.clone(),
                config.rebroadcast.interval,
            )));
        }

        Ok(Self {
            endpoint,
            router,
            sender,
            node_id,
            relay_urls: config.relay_urls,
            lookup,
            shared,
            tasks: Mutex::new(tasks),
        })
    }

    /// The iroh `EndpointId` (32 bytes) for this node — the value the Join flow carries as
    /// `Join.iroh_id` to bind the iroh key to the node identity (§7.2).
    #[must_use]
    pub fn node_id(&self) -> [u8; 32] {
        self.node_id
    }

    /// Our own dialable [`IrohPeer`] (endpoint id + currently-bound sockets + configured relay), for
    /// wiring other peers' rosters (tests + the Join/admission flow).
    #[must_use]
    pub fn local_peer(&self) -> IrohPeer {
        IrohPeer {
            endpoint_id: self.node_id,
            direct_addrs: self.endpoint.bound_sockets(),
            relay_url: self.relay_urls.first().cloned(),
        }
    }

    /// The number of current direct gossip neighbors (observability / test synchronization).
    #[must_use]
    pub fn neighbor_count(&self) -> usize {
        self.shared
            .lock()
            .expect("gossip shared lock")
            .neighbors
            .len()
    }

    /// Re-seed discovery + re-form the mesh from an updated roster (Psyche `ensure_gossip_connected`,
    /// `client.rs:736-799`): add each roster addr to discovery, then `join_peers` up to the
    /// [`MAX_BOOTSTRAP_PEERS`] neighbor cap (only enough to reach the cap, to avoid force-evicting
    /// existing neighbors — Psyche `client.rs:773-782`).
    pub async fn update_roster(&self, roster: Vec<IrohPeer>) -> Result<(), SwarmNetError> {
        let mut ids = Vec::with_capacity(roster.len());
        for peer in &roster {
            self.lookup.add_endpoint_info(peer_to_addr(peer)?);
            if let Ok(id) = EndpointId::from_bytes(&peer.endpoint_id) {
                ids.push(id);
            }
        }

        let (self_id, current) = {
            let sh = self.shared.lock().expect("gossip shared lock");
            (sh.self_id, sh.neighbors.clone())
        };

        let want = MAX_BOOTSTRAP_PEERS.saturating_sub(current.len());
        let to_add: Vec<EndpointId> = ids
            .into_iter()
            .filter(|id| *id != self_id)
            .filter(|id| !current.contains(id))
            .take(want)
            .collect();

        if !to_add.is_empty() {
            self.sender
                .join_peers(to_add)
                .await
                .map_err(|e| SwarmNetError::Transport(format!("gossip join_peers failed: {e}")))?;
        }
        Ok(())
    }

    /// Abort the background loops and shut down the iroh router (graceful teardown for tests / node
    /// shutdown). Also runs on `Drop` (task abort only).
    pub async fn shutdown(&self) {
        if let Ok(mut tasks) = self.tasks.lock() {
            for handle in tasks.drain(..) {
                handle.abort();
            }
        }
        let _ = self.router.shutdown().await;
    }
}

impl Drop for IrohGossip {
    fn drop(&mut self) {
        if let Ok(mut tasks) = self.tasks.lock() {
            for handle in tasks.drain(..) {
                handle.abort();
            }
        }
    }
}

#[async_trait]
impl ControlPlane for IrohGossip {
    async fn publish(&self, message: &[u8]) -> Result<(), SwarmNetError> {
        if message.len() + NONCE_LEN > MAX_MESSAGE_SIZE {
            return Err(SwarmNetError::Transport(format!(
                "control message {} B exceeds gossip max {} B",
                message.len(),
                MAX_MESSAGE_SIZE - NONCE_LEN
            )));
        }

        // Dedupe + self-deliver + record for rebroadcast under one lock. A WS+gossip double-send of
        // identical bytes is a no-op here (content-hash dedupe). Self-delivery keeps the contract
        // identical to `LoopbackGossip` (a single iroh node does not receive its own broadcast back).
        {
            let mut sh = self.shared.lock().expect("gossip shared lock");
            if !sh.dedupe.observe(message) {
                return Ok(());
            }
            sh.subscribers
                .retain(|tx| tx.send(message.to_vec()).is_ok());
            if sh.ring.len() >= sh.ring_capacity {
                sh.ring.pop_front();
            }
            sh.ring.push_back((message.to_vec(), 0));
        }

        // First flood at nonce 0 (Psyche `lib.rs:568-581`, adapted to the nonce frame).
        let frame = frame_bytes(0, message);
        self.sender
            .broadcast(frame.into())
            .await
            .map_err(|e| SwarmNetError::Transport(format!("gossip broadcast failed: {e}")))?;
        Ok(())
    }

    fn subscribe(&self) -> ControlSubscription {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.shared
            .lock()
            .expect("gossip shared lock")
            .subscribers
            .push(tx);
        ControlSubscription::new(rx)
    }
}

/// Derive the gossip topic: `blake3(frozen envelope hash)` -> [`TopicId`]. Delta from Psyche's
/// `sha256("psyche gossip" ++ run_id)` (`util.rs:5-13`): blake3 not sha256 (spec §6.4 content
/// addressing), and the frozen envelope hash not the run id (binds the topic to the exact frozen run).
fn derive_topic(envelope_hash: &[u8; 32]) -> TopicId {
    TopicId::from_bytes(*blake3_hash(envelope_hash).as_bytes())
}

/// Build a [`RelayMap`] from the configured URLs (`None` when empty -> `RelayMode::Disabled`).
fn relay_map(urls: &[String]) -> Result<Option<RelayMap>, SwarmNetError> {
    if urls.is_empty() {
        return Ok(None);
    }
    let mut parsed = Vec::with_capacity(urls.len());
    for url in urls {
        let relay: RelayUrl = url
            .parse()
            .map_err(|e| SwarmNetError::Transport(format!("bad relay url {url}: {e}")))?;
        parsed.push(relay);
    }
    Ok(Some(RelayMap::from_iter(parsed)))
}

/// Convert a roster [`IrohPeer`] to an iroh [`EndpointAddr`] (id + direct addrs + relay url).
fn peer_to_addr(peer: &IrohPeer) -> Result<EndpointAddr, SwarmNetError> {
    let id = EndpointId::from_bytes(&peer.endpoint_id)
        .map_err(|e| SwarmNetError::Transport(format!("bad iroh endpoint id: {e}")))?;
    let mut addr = EndpointAddr::new(id);
    for socket in &peer.direct_addrs {
        addr = addr.with_ip_addr(*socket);
    }
    if let Some(url) = &peer.relay_url {
        let relay: RelayUrl = url
            .parse()
            .map_err(|e| SwarmNetError::Transport(format!("bad relay url {url}: {e}")))?;
        addr = addr.with_relay_url(relay);
    }
    Ok(addr)
}

/// Frame `[nonce: u64 LE][payload]`. Bumping the nonce forces a new gossip `MessageId` (re-flood);
/// the receiver strips the nonce and dedupes by the inner payload hash.
fn frame_bytes(nonce: u64, payload: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(NONCE_LEN + payload.len());
    framed.extend_from_slice(&nonce.to_le_bytes());
    framed.extend_from_slice(payload);
    framed
}

/// Strip the nonce frame, returning the inner payload (or `None` if the frame is too short).
fn unframe(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.len() < NONCE_LEN {
        return None;
    }
    Some(bytes[NONCE_LEN..].to_vec())
}

/// Dedupe the inner payload and fan it out to every live local subscriber (one delivery — NET-6).
fn deliver(shared: &Arc<Mutex<Shared>>, payload: Vec<u8>) {
    let mut sh = shared.lock().expect("gossip shared lock");
    if sh.dedupe.observe(&payload) {
        sh.subscribers.retain(|tx| tx.send(payload.clone()).is_ok());
    }
}

/// Receive loop: strip frames, dedupe, fan out; track neighbors (Psyche `lib.rs:898-946`).
async fn receive_loop(mut receiver: GossipReceiver, shared: Arc<Mutex<Shared>>) {
    while let Some(event) = receiver.next().await {
        match event {
            Ok(Event::Received(msg)) => {
                if let Some(payload) = unframe(&msg.content) {
                    deliver(&shared, payload);
                }
            }
            Ok(Event::NeighborUp(id)) => {
                shared
                    .lock()
                    .expect("gossip shared lock")
                    .neighbors
                    .insert(id);
            }
            Ok(Event::NeighborDown(id)) => {
                shared
                    .lock()
                    .expect("gossip shared lock")
                    .neighbors
                    .remove(&id);
            }
            // `Lagged` means the subscription fell behind; the rebroadcast loop covers the gap.
            Ok(Event::Lagged) => {}
            Err(_) => break,
        }
    }
}

/// Rebroadcast loop: every `interval`, re-flood one recent message with a bumped nonce so a peer
/// that missed the first flood still gets it (Psyche `client.rs:490-505`). The bumped nonce yields a
/// fresh gossip `MessageId` (re-flood); receivers drop it by inner-payload dedupe (one delivery).
async fn rebroadcast_loop(sender: GossipSender, shared: Arc<Mutex<Shared>>, interval: Duration) {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Skip the immediate first tick so we don't re-flood before anything is published.
    ticker.tick().await;
    let mut index: usize = 0;
    loop {
        ticker.tick().await;
        let frame = {
            let mut sh = shared.lock().expect("gossip shared lock");
            if sh.ring.is_empty() {
                continue;
            }
            index = (index + 1) % sh.ring.len();
            let Some((payload, nonce)) = sh.ring.get_mut(index) else {
                continue;
            };
            *nonce = nonce.wrapping_add(1);
            frame_bytes(*nonce, payload)
        };
        // A send error means the topic/endpoint is gone; the loop will be aborted on shutdown.
        if sender.broadcast(frame.into()).await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trips_and_strips_nonce() {
        let payload = b"already-signed-cbor";
        let framed = frame_bytes(7, payload);
        assert_eq!(framed.len(), NONCE_LEN + payload.len());
        assert_eq!(unframe(&framed).as_deref(), Some(&payload[..]));
        // Different nonce -> different outer bytes (new gossip MessageId) but same inner payload.
        let framed2 = frame_bytes(8, payload);
        assert_ne!(framed, framed2);
        assert_eq!(unframe(&framed2), unframe(&framed));
    }

    #[test]
    fn unframe_rejects_short_frames() {
        assert_eq!(unframe(&[0u8; 4]), None);
        assert_eq!(unframe(&[0u8; NONCE_LEN]), Some(Vec::new()));
    }

    #[test]
    fn topic_is_blake3_of_envelope_hash() {
        let envelope_hash = [0x11u8; 32];
        let topic = derive_topic(&envelope_hash);
        // Deterministic, and equal to blake3 of the input (delta from Psyche's sha256).
        assert_eq!(
            topic,
            TopicId::from_bytes(*blake3_hash(&envelope_hash).as_bytes())
        );
    }

    #[test]
    fn empty_relay_urls_disables_relay() {
        assert!(relay_map(&[]).expect("ok").is_none());
        let map = relay_map(&["http://localhost:3340".to_string()])
            .expect("parse")
            .expect("some");
        assert_eq!(map.len(), 1);
    }
}
