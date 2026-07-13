// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Live-transport multi-peer harness — the round engine over **real [`IrohGossip`]** + a real
//! [`FsPayloadStore`] (spec §7.1, §6.4; TDD §3.8 the live-transport e2e gate; B3).
//!
//! Where [`crate::harness`] drives N [`RoundEngine`]s over ONE shared [`LoopbackGossip`], this module
//! drives them over a **real per-node iroh-gossip mesh**: every peer and the coordinator own a
//! distinct [`IrohGossip`] endpoint, and they form a QUIC gossip mesh on the shared topic
//! `blake3(envelope_hash)`. This is the whole point of the frozen-seam design — `RoundEngine` and
//! [`LocalCoordinator`] are generic over `C: ControlPlane`, so swapping `LoopbackGossip → IrohGossip`
//! is a **construction** change, not a protocol change; the round protocol, barrier, stall ladder,
//! late-join, drop, and replay all run unchanged.
//!
//! ## Wiring recipe (B2 ledger finding 7)
//!
//! 1. Build N peer nodes + 1 coordinator node with an **empty** roster (bind `127.0.0.1:0`;
//!    `RelayMode::Disabled` on the loopback variant, or the envelope-pinned dev-relay URL).
//! 2. Once every node is bound, collect each node's [`IrohGossip::local_peer`] (endpoint id + bound
//!    sockets) and `update_roster(full_roster)` on every node → seeds discovery + `join_peers` → the
//!    mesh forms (B2 finding 1: the direct loopback mesh forms with no relay in ~1 s).
//! 3. Wait for `neighbor_count() >= 1` on every node before the coordinator opens round 0, so the
//!    first `RoundOpen` is not lost to a not-yet-formed mesh (gossip is best-effort; B2's rebroadcast
//!    frame covers residual gaps).
//!
//! The mesh here is wired directly from the real bound endpoints — equivalent to admission carrying
//! `IrohPeer.endpoint_id = Join.iroh_id = node_id()` (the worker path sets `Join.iroh_id = node_id()`
//! for real; the coordinator's synthesized `Join`s in the harness carry a constant iroh id because
//! the harness does not route by it — the mesh routes by the real roster). The plane carries
//! already-signed proto `SignedMessage` bytes (the engine signs before `publish`, verifies after
//! `recv`); [`IrohGossip`] never signs/verifies (§7.1).

#![cfg(feature = "iroh")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use daemon_swarm_coordinator::CoordinatorState;
use daemon_swarm_net::{
    DownloadScheduler, FsPayloadStore, IrohGossip, IrohGossipConfig, IrohPeer, RebroadcastConfig,
    RetryConfig,
};
use daemon_swarm_proto::envelope::StopCondition;
use daemon_swarm_proto::messages::{RecordEntry, RoundRecord};
use daemon_swarm_proto::{
    from_canonical_slice, peer_id, PeerId, SignedMessage, SigningKey, SwarmMessage,
    SwarmProtoVersion, SWARM_PROTO_VERSION,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;

use crate::backend::{StubBackend, TrainerBackend};
use crate::checkpoint::CheckpointManifest;
use crate::engine::{EngineConfig, EngineEvent, RoundEngine};
use crate::harness::{
    build_run_config, coordinator_key, peer_key, peer_store, SwarmConfig, SwarmRun,
    EXPERIMENT_CONFIG,
};
use crate::local_coordinator::{LocalCoordinator, LocalCoordinatorConfig};
use crate::seam::{RoundId, RunId};
use crate::SwarmRunError;

/// Live-transport run configuration: the shared drill knobs ([`SwarmConfig`]) plus the two
/// live-only knobs (relay selection + concurrent barrier fetch). The exit-gate harness API.
#[derive(Clone, Debug)]
pub struct LiveSwarmConfig {
    /// The shared multi-peer / drill configuration (peers, rounds, late-join, drop, stall, outage,
    /// restart, …) — reused verbatim from the loopback harness.
    pub base: SwarmConfig,
    /// Envelope-pinned relay URL(s). `None` → `RelayMode::Disabled` (direct loopback mesh, the
    /// default exit-gate variant); `Some(url)` → route through a self-hosted relay (B2's dev runner).
    pub relay_url: Option<String>,
    /// Enable [`RoundEngine::with_download_scheduler`] so the barrier fetches uncached payloads
    /// concurrently over the real plane (the `// MERGE-2` in-peer concurrent-fetch marker).
    pub concurrent_fetch: bool,
    /// How long to wait for the gossip mesh to form (each node reaches ≥1 neighbor) before opening
    /// round 0.
    pub mesh_timeout: Duration,
}

impl LiveSwarmConfig {
    /// A loopback (no-relay) live config over `base`, concurrent fetch on, 30 s mesh timeout.
    #[must_use]
    pub fn new(base: SwarmConfig) -> Self {
        Self {
            base,
            relay_url: None,
            concurrent_fetch: true,
            mesh_timeout: Duration::from_secs(30),
        }
    }

    /// Route the mesh through a self-hosted relay URL (B2's `dev/run-relay.sh`).
    #[must_use]
    pub fn with_relay(mut self, url: impl Into<String>) -> Self {
        self.relay_url = Some(url.into());
        self
    }

    /// Toggle the concurrent barrier fetch (default on).
    #[must_use]
    pub fn with_concurrent_fetch(mut self, on: bool) -> Self {
        self.concurrent_fetch = on;
        self
    }
}

/// The shared topic-derivation input (the run's frozen-envelope hash). The harness `build_run_config`
/// uses `Hash([0; 32])`, so every node derives the same `blake3([0;32])` topic.
const TOPIC_INPUT: [u8; 32] = [0u8; 32];

/// A deterministic iroh secret key per participant (distinct namespace per role so the coordinator
/// and peer endpoint ids never collide). NOT the node ed25519 identity (§7.2) — iroh is transport.
fn iroh_secret(role: u8, i: usize) -> [u8; 32] {
    let mut s = [0u8; 32];
    s[0] = role;
    s[1] = i as u8;
    s[2] = 0x5A;
    s
}

/// Build one [`IrohGossip`] node with an empty roster (roster wired after all nodes bind).
async fn build_node(
    secret: [u8; 32],
    relay_urls: &[String],
) -> Result<Arc<IrohGossip>, SwarmRunError> {
    let node = IrohGossip::connect(IrohGossipConfig {
        secret_key: secret,
        relay_urls: relay_urls.to_vec(),
        roster: Vec::new(),
        topic_input: TOPIC_INPUT,
        // A brisk rebroadcast so a peer that missed the first flood of a round-critical message
        // recovers in ~2 s rather than the 10 s default (keeps the e2e wall time bounded).
        rebroadcast: RebroadcastConfig {
            enabled: true,
            interval: Duration::from_secs(2),
            ring_capacity: 64,
        },
        bind_addr: Some("127.0.0.1:0".parse().expect("loopback bind addr")),
    })
    .await?;
    Ok(Arc::new(node))
}

/// Run a live-transport swarm over a real iroh mesh + a shared FS payload store, using the
/// deterministic [`StubBackend`] (the digest-equality / drill gate).
pub async fn run_live_swarm(cfg: LiveSwarmConfig) -> Result<SwarmRun, SwarmRunError> {
    run_live_swarm_with(cfg, |_i| {
        let mut backend = StubBackend::new();
        backend
            .build(EXPERIMENT_CONFIG)
            .expect("build stub backend");
        backend
    })
    .await
}

/// Like [`run_live_swarm`], but each peer's [`TrainerBackend`] is produced by `make_backend(index)`
/// — so the e2e crate can inject the real tiny-llama `WasmBackend` (which lives outside this crate's
/// dependency tree) for the flagship live run. The factory MUST return an already-`build`-ed backend.
///
/// `Sync` is required by the engine's `&self` async publish path (its future holds `&RoundEngine`).
/// A `Send`-only backend like `WasmBackend` rides in a `Mutex<T>` newtype adapter (`Mutex<T>: Sync`
/// for `T: Send`; the engine's exclusive `&mut` access means the lock is uncontended) — see the
/// tiny-llama flagship in `daemon-swarm-e2e/tests/live_transport.rs`.
#[allow(clippy::too_many_lines)]
pub async fn run_live_swarm_with<B, F>(
    cfg: LiveSwarmConfig,
    make_backend: F,
) -> Result<SwarmRun, SwarmRunError>
where
    B: TrainerBackend + Send + Sync + 'static,
    F: Fn(usize) -> B,
{
    let base = &cfg.base;
    let run = RunId::new("e2e-live-run");
    let version: SwarmProtoVersion = SWARM_PROTO_VERSION;
    let relay_urls: Vec<String> = cfg.relay_url.iter().cloned().collect();

    // Identities (the same deterministic ed25519 keys as the loopback harness).
    let boot_keys: Vec<SigningKey> = (0..base.num_peers).map(peer_key).collect();
    let boot_ids: Vec<PeerId> = boot_keys.iter().map(peer_id).collect();
    let mut boot_roster = boot_ids.clone();
    boot_roster.sort_unstable();

    let late_key = base.late_join.map(|_| peer_key(base.num_peers));
    let mut full_roster = boot_roster.clone();
    if let Some(k) = &late_key {
        full_roster.push(peer_id(k));
    }
    full_roster.sort_unstable();

    let corpus = Arc::new(crate::data::Corpus::synthetic(base.corpus_seed, 4, 256, 8)?);

    let root = std::env::temp_dir().join(format!(
        "daemon-swarm-live-{}-{}",
        std::process::id(),
        crate::harness::fastcounter()
    ));
    let fs = Arc::new(FsPayloadStore::open(&root, base.num_rounds + 8)?);

    // -- build every iroh node (empty roster), then wire the full roster + wait for the mesh -------
    let coord_node = build_node(iroh_secret(0xC0, 0), &relay_urls).await?;
    let mut peer_nodes: Vec<Arc<IrohGossip>> = Vec::with_capacity(base.num_peers);
    for i in 0..base.num_peers {
        peer_nodes.push(build_node(iroh_secret(0x1A, i), &relay_urls).await?);
    }
    let late_node = if late_key.is_some() {
        Some(build_node(iroh_secret(0x1A, base.num_peers), &relay_urls).await?)
    } else {
        None
    };

    let mut all_nodes: Vec<Arc<IrohGossip>> = vec![coord_node.clone()];
    all_nodes.extend(peer_nodes.iter().cloned());
    if let Some(n) = &late_node {
        all_nodes.push(n.clone());
    }
    let all_peers: Vec<IrohPeer> = all_nodes.iter().map(|n| n.local_peer()).collect();
    for node in &all_nodes {
        node.update_roster(all_peers.clone()).await?;
    }
    wait_for_mesh(&all_nodes, cfg.mesh_timeout).await?;

    // -- event collection + peer engines ---------------------------------------------------------
    let (col_tx, mut col_rx) = unbounded_channel::<(PeerId, EngineEvent)>();
    let mut peer_handles: BTreeMap<
        PeerId,
        JoinHandle<Result<crate::engine::RunOutcome, SwarmRunError>>,
    > = BTreeMap::new();
    let mut fwd_handles: Vec<JoinHandle<()>> = Vec::new();

    for (i, key) in boot_keys.iter().enumerate() {
        let store = peer_store(&fs, &run, &boot_ids, base, i);
        let backend = make_backend(i);
        let (peer, engine_h, fwd_h) = launch_live_engine(
            peer_nodes[i].clone(),
            store,
            backend,
            key.clone(),
            &corpus,
            &run,
            version,
            base,
            boot_roster.clone(),
            None,
            cfg.concurrent_fetch,
            &col_tx,
        );
        peer_handles.insert(peer, engine_h);
        fwd_handles.push(fwd_h);
    }

    // Record collector rides the coordinator's own node (its publishes self-deliver).
    let records: Arc<std::sync::Mutex<BTreeMap<RoundId, Vec<RecordEntry>>>> =
        Arc::new(std::sync::Mutex::new(BTreeMap::new()));
    let rec_handle = spawn_record_collector(coord_node.clone(), version, records.clone());

    // -- the real coordinator tick loop over its own iroh node -----------------------------------
    let coord_cfg = LocalCoordinatorConfig {
        run: run.clone(),
        key: coordinator_key(),
        version,
        state: CoordinatorState::new(
            build_run_config(&run, base),
            daemon_swarm_proto::Seed([0xAB; 32]),
            0,
        ),
        bootstrap_keys: boot_keys.clone(),
        late_keys: late_key.iter().cloned().collect(),
        // A generous quiescence: over a real mesh, commits + records take round-trips, so the
        // coordinator must not force-progress a round before its peers' evidence arrives.
        quiescence: Duration::from_secs(6),
        restart_after_round: base.restart_after_round,
    };
    let coordinator = LocalCoordinator::new(coord_node.clone(), fs.clone(), coord_cfg);
    let coord_handle: JoinHandle<
        Result<crate::local_coordinator::CoordinatorReplay, SwarmRunError>,
    > = tokio::spawn(async move { coordinator.drive().await });

    // -- collect events until every expected peer finishes / leaves / is dropped -----------------
    let last_round = base.num_rounds.saturating_sub(1);
    let mut events: Vec<(PeerId, EngineEvent)> = Vec::new();
    let mut done: BTreeSet<PeerId> = BTreeSet::new();
    let mut expected: BTreeSet<PeerId> = boot_ids.iter().copied().collect();
    let mut awaiting_late = base.late_join.is_some();
    let mut killed_death = false;
    let death_target = base.silent_death.map(|d| boot_ids[d.peer_index]);
    // A live run is slower than loopback; give each step a generous idle budget.
    let idle_budget = Duration::from_secs(40);

    loop {
        if !awaiting_late && expected.iter().all(|p| done.contains(p)) {
            break;
        }
        match tokio::time::timeout(idle_budget, col_rx.recv()).await {
            Ok(Some((peer, ev))) => {
                // Late-join: spawn the late peer when a bootstrap peer checkpoints the resume round.
                if awaiting_late {
                    if let (Some(lj), Some(lk), Some(lnode)) =
                        (base.late_join, &late_key, &late_node)
                    {
                        if let EngineEvent::Checkpointed { round, manifest } = &ev {
                            if *round == lj.resume_round {
                                let store =
                                    Arc::new(crate::harness::FaultyStore::transparent(fs.clone()));
                                let backend = make_backend(base.num_peers);
                                let (lpeer, engine_h, fwd_h) = launch_live_engine(
                                    lnode.clone(),
                                    store,
                                    backend,
                                    lk.clone(),
                                    &corpus,
                                    &run,
                                    version,
                                    base,
                                    full_roster.clone(),
                                    Some(*manifest),
                                    cfg.concurrent_fetch,
                                    &col_tx,
                                );
                                peer_handles.insert(lpeer, engine_h);
                                fwd_handles.push(fwd_h);
                                expected.insert(lpeer);
                                awaiting_late = false;
                            }
                        }
                    }
                }

                // Silent death: abort the target's engine once it reports its last live round.
                if let (Some(d), Some(target)) = (base.silent_death, death_target) {
                    if !killed_death && peer == target {
                        let reached = matches!(&ev,
                            EngineEvent::RoundComplete { round, .. } | EngineEvent::CaughtUp { round, .. }
                            if *round >= d.after_round);
                        if reached {
                            if let Some(h) = peer_handles.remove(&target) {
                                h.abort();
                            }
                            expected.remove(&target);
                            killed_death = true;
                        }
                    }
                }

                match &ev {
                    EngineEvent::RoundComplete { round, .. }
                    | EngineEvent::CaughtUp { round, .. }
                        if *round == last_round =>
                    {
                        done.insert(peer);
                    }
                    EngineEvent::Left { .. } => {
                        done.insert(peer);
                    }
                    _ => {}
                }
                events.push((peer, ev));
            }
            Ok(None) => break,
            Err(_) => break, // idle budget exceeded — a silent/left peer cannot hang the harness
        }
    }

    let replay = coord_handle.await.ok().and_then(Result::ok);
    for h in peer_handles.values() {
        h.abort();
    }
    for h in fwd_handles {
        h.abort();
    }
    rec_handle.abort();
    while let Ok((peer, ev)) = col_rx.try_recv() {
        events.push((peer, ev));
    }
    for node in &all_nodes {
        node.shutdown().await;
    }

    let records = records.lock().expect("records lock").clone();
    Ok(SwarmRun {
        roster: boot_roster,
        events,
        replay,
        store: fs,
        run,
        records,
    })
}

/// Construct + spawn one [`RoundEngine`] over an iroh node (optionally concurrent-fetch + resume),
/// plus its event forwarder. Mirrors [`crate::harness`]'s `launch_engine` for the live plane.
#[allow(clippy::too_many_arguments)]
fn launch_live_engine<B>(
    node: Arc<IrohGossip>,
    store: Arc<crate::harness::FaultyStore>,
    backend: B,
    key: SigningKey,
    corpus: &Arc<crate::data::Corpus>,
    run: &RunId,
    version: SwarmProtoVersion,
    base: &SwarmConfig,
    roster: Vec<PeerId>,
    resume: Option<CheckpointManifest>,
    concurrent_fetch: bool,
    col_tx: &UnboundedSender<(PeerId, EngineEvent)>,
) -> (
    PeerId,
    JoinHandle<Result<crate::engine::RunOutcome, SwarmRunError>>,
    JoinHandle<()>,
)
where
    B: TrainerBackend + Send + Sync + 'static,
{
    let peer = peer_id(&key);
    let engine_cfg = EngineConfig {
        run: run.clone(),
        roster: roster.clone(),
        witnesses: roster,
        steps_per_round: base.steps_per_round,
        micro_batch: base.micro_batch,
        stall_rounds_max: base.stall_rounds_max,
        checkpoint_every_rounds: base.checkpoint_every_rounds,
        version,
    };

    let (ev_tx, mut ev_rx) = unbounded_channel::<EngineEvent>();
    let col = col_tx.clone();
    let fwd = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            if col.send((peer, ev)).is_err() {
                break;
            }
        }
    });

    let mut engine = RoundEngine::new(node, store, backend, key, corpus.clone(), engine_cfg, ev_tx);
    if concurrent_fetch {
        // A per-peer bounded-concurrency gate for the barrier fetch (cap = roster size).
        let scheduler = Arc::new(DownloadScheduler::new(
            base.num_peers.max(1) + 1,
            RetryConfig::default(),
        ));
        engine = engine.with_download_scheduler(scheduler);
    }
    let engine_h = tokio::spawn(async move {
        if let Some(manifest) = resume {
            engine.resume_from_checkpoint(&manifest).await?;
        }
        engine.run().await
    });
    (peer, engine_h, fwd)
}

/// A background task that decodes the coordinator's published `RoundRecord`s off its iroh node (they
/// self-deliver) and records their committed sets (for offline resync / assertions).
fn spawn_record_collector(
    node: Arc<IrohGossip>,
    version: SwarmProtoVersion,
    records: Arc<std::sync::Mutex<BTreeMap<RoundId, Vec<RecordEntry>>>>,
) -> JoinHandle<()> {
    use daemon_swarm_net::ControlPlane;
    let mut sub = node.subscribe();
    tokio::spawn(async move {
        while let Some(bytes) = sub.recv().await {
            let Ok(msg) = from_canonical_slice::<SignedMessage>(&bytes) else {
                continue;
            };
            if msg.verify_for_run(version).is_err() {
                continue;
            }
            if let SwarmMessage::RoundRecord(RoundRecord {
                round,
                inline: Some(entries),
                ..
            }) = &msg.payload
            {
                records
                    .lock()
                    .expect("records lock")
                    .entry(*round)
                    .or_insert_with(|| entries.clone());
            }
        }
    })
}

/// Poll every node until it reports ≥1 gossip neighbor, or `timeout` elapses (mesh-formation gate).
async fn wait_for_mesh(nodes: &[Arc<IrohGossip>], timeout: Duration) -> Result<(), SwarmRunError> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if nodes.iter().all(|n| n.neighbor_count() >= 1) {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            let counts: Vec<usize> = nodes.iter().map(|n| n.neighbor_count()).collect();
            return Err(SwarmRunError::Lifecycle(format!(
                "iroh mesh did not form within {timeout:?} (neighbor counts: {counts:?})"
            )));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Whether the run's [`StopCondition`] is round-bounded (the exit-gate default). A helper for the
/// e2e's termination assertion so it reads the same stop the coordinator enforced.
#[must_use]
pub fn is_round_bounded(stop: &StopCondition) -> bool {
    matches!(stop, StopCondition::Rounds(_))
}
