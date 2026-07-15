// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The **live coordinator attach** for the `JoinRun` path (A3, feature `swarm-net`).
//!
//! Where [`crate::transport::join_and_run_round`] self-drives a single representative round in
//! process (the T0 / test / default-gate fallback), this module moves B3's `live_harness` recipe
//! **into the worker subprocess**: it constructs a real [`RoundEngine`] over a
//! `DualPlane(WsControlPlane, IrohGossip)` control surface + an `R2Store`/`FsPayloadStore` payload
//! plane, registers the signed `Join` for resubscribe, mirrors the §7.3 receive-side size-cap
//! (Merge-1 Decision 2), and runs rounds **continuously** until `Leave`/stop — streaming a
//! `RunPhase`/`Metric`/`RoundOutcome`/`Warning`-per-round event pump (plus the additive
//! `MicroBatch`/`OomLadder` telemetry) back over the stdio cut, which the node's `TrainSupervisor`
//! forwards into `SwarmService::handle_worker_event`.
//!
//! Iroh stays runtime-optional even under the feature: absent iroh credentials the worker runs over
//! the bare [`WsControlPlane`] wrapped in a single-plane `DualPlane` (the T0 WS-only baseline).

use std::sync::Arc;
use std::time::Duration;

use std::collections::BTreeMap;

use daemon_provision::CutWriter;
use daemon_swarm_net::{
    ArtifactRef, ArtifactResolver, ContentCache, ContentHash, ControlPlane, DualPlane,
    FsPayloadStore, HttpPresignClient, IrohGossip, IrohGossipConfig, IrohPeer, PayloadKey,
    PayloadStat, PayloadStore, PresignClient, R2Store, RebroadcastConfig, ReconnectConfig,
    RegistryClient, RunId, SwarmNetError, WsAuth, WsConfig, WsControlPlane,
};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StateDigest, StepCtx, TrainerBackend};
use daemon_swarm_run::checkpoint::{
    plan_resync, CheckpointManifest, ReplayStep, ResyncPlan, CHECKPOINT_PEER,
};
use daemon_swarm_run::data::{Corpus, Manifest};
use daemon_swarm_run::engine::{EngineConfig, EngineEvent, RoundEngine};
use daemon_swarm_run::protocol::{ErrorClass, Event, JoinCredentials};
use daemon_swarm_run::seam::RoundId;
use daemon_swarm_run::SwarmRunError;
use daemon_train::{
    EngineConfig as WasmEngineConfig, TrainError, TrapCode, WasmBackend, WasmBackendConfig,
    WasmBackendError,
};
use daemon_vhc_proto::messages::{Join, RecordEntry, ThroughputClass};
use daemon_vhc_proto::{
    peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, SignedMessage, SigningKey,
    SwarmMessage, SwarmProtoVersion, SWARM_PROTO_VERSION,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};
use tokio::task::JoinHandle;

use crate::send;

/// A running live attach: the engine + forwarder + translator tasks and the plane handles, so the
/// command loop can stop cleanly on `Leave`/`Shutdown` (preemption-as-churn, §10.5).
pub(crate) struct LiveHandle {
    engine_task: JoinHandle<Result<daemon_swarm_run::engine::RunOutcome, SwarmRunError>>,
    forwarder_task: JoinHandle<()>,
    translator_task: JoinHandle<()>,
    ws: Arc<WsControlPlane>,
    iroh: Option<Arc<IrohGossip>>,
}

impl LiveHandle {
    /// Stop the run: abort the engine + pump tasks and shut the transport planes down.
    pub(crate) async fn stop(self) {
        self.engine_task.abort();
        self.forwarder_task.abort();
        self.translator_task.abort();
        self.ws.shutdown().await;
        if let Some(iroh) = &self.iroh {
            iroh.shutdown().await;
        }
    }
}

/// Construct the live plane + engine from the parsed [`JoinCredentials`], spawn the continuous round
/// loop + the worker→node event pump, and return the handle. `coordinator` is the WS base URL from
/// `JoinRun.coordinator`; `credentials` is the canonical-CBOR `JoinCredentials` body.
pub(crate) async fn join_and_run_live(
    module: &[u8],
    config: &[u8],
    run_id: &str,
    coordinator: &str,
    creds: &JoinCredentials,
    assessed_micro_batch: u32,
    writer: &CutWriter,
) -> Result<LiveHandle, String> {
    let version: SwarmProtoVersion = SWARM_PROTO_VERSION;
    let key = SigningKey::from_bytes(&creds.node_secret);
    let peer: PeerId = peer_id(&key);

    // -- iroh (optional) — build first so the Join carries the real iroh_id binding (§7.2) ---------
    let iroh = build_iroh(creds).await?;
    let iroh_id = iroh
        .as_ref()
        .map_or(IrohId([0u8; 32]), |n| IrohId(n.node_id()));

    // -- WS control plane (the T0 baseline; always present) ----------------------------------------
    let ws = Arc::new(
        WsControlPlane::connect(WsConfig {
            base_url: coordinator.to_string(),
            run_id: run_id.to_string(),
            auth: to_ws_auth(&creds.ws_auth),
            reconnect: ReconnectConfig::default(),
        })
        .await
        .map_err(|e| format!("ws connect {coordinator}: {e}"))?,
    );

    // Register the peer's signed Join for resubscribe (re-sent on every (re)connect → re-admits).
    let join = SignedMessage::sign(
        &key,
        version,
        SwarmMessage::Join(Join {
            run_id: run_id.to_string(),
            iroh_id,
            class: ThroughputClass::C1,
            capabilities: CapabilitySet::new(),
            envelope_hash: Some(Hash::new(creds.envelope_hash)),
        }),
    )
    .map_err(|e| format!("sign join: {e}"))?;
    let join_bytes = to_canonical_vec(&join).map_err(|e| format!("encode join: {e}"))?;
    ws.add_resubscribe_frame(join_bytes);

    // -- compose the control plane + apply the §7.3 receive-side size cap (Merge-1 Decision 2) -----
    let planes: Vec<Arc<dyn ControlPlane>> = match &iroh {
        Some(node) => vec![ws.clone(), node.clone()],
        None => vec![ws.clone()],
    };
    let control =
        Arc::new(DualPlane::new(planes).with_receive_size_cap(creds.engine.update_max_bytes));

    // -- payload plane: R2 over presign when a base is declared; else the FS fallback --------------
    let store = Arc::new(build_store(run_id, creds)?);

    // -- corpus + engine config (deterministic across peers → agreeing digests) --------------------
    // P3 lane S: a content-addressed corpus (`engine.corpus`) is fetched by hash from the store and
    // windowed to the run's active shards; else the synthetic fallback (CI/test).
    let corpus = Arc::new(build_corpus(run_id, creds).await?);
    let roster: Vec<PeerId> = creds.roster.iter().map(|b| PeerId(*b)).collect();
    let micro_batch = assessed_micro_batch
        .max(1)
        .min(creds.engine.micro_batch.max(1));
    let engine_cfg = EngineConfig {
        run: RunId::new(run_id),
        roster: roster.clone(),
        witnesses: roster,
        steps_per_round: creds.engine.steps_per_round.max(1),
        micro_batch,
        stall_rounds_max: creds.engine.stall_rounds_max,
        checkpoint_every_rounds: creds.engine.checkpoint_every_rounds,
        version,
    };

    // -- the worker→node event pump: one writer-owning translator, one EngineEvent forwarder -------
    let (out_tx, mut out_rx) = unbounded_channel::<Event>();
    let writer_owned = writer.clone();
    let translator_task = tokio::spawn(async move {
        while let Some(ev) = out_rx.recv().await {
            send(&writer_owned, &ev).await;
        }
    });

    // The join preamble: RunPhase{train} (the supervisor's join resolves here) + the consumed
    // autotune verdict as the additive MicroBatch telemetry (§10.5; P1-deferred follow-on 2).
    let _ = out_tx.send(Event::RunPhase {
        run_id: run_id.to_string(),
        phase: "train".to_string(),
        epoch: 0,
        round: 0,
    });
    let _ = out_tx.send(Event::MicroBatch { micro_batch });

    // The backend: a §10.5 OOM-ladder wrapper around the WasmBackend that emits Metric{loss} +
    // OomLadder telemetry through the same pump. Constructs the backend on its host — a dedicated
    // device thread for the CUDA lane (cubecl per-thread stream discipline), in place otherwise.
    let backend = LadderBackend::new(
        module.to_vec(),
        config.to_vec(),
        out_tx.clone(),
        creds.engine.corpus_vocab_clamp,
    )?;

    // EngineEvent → protocol::Event forwarder (per-round RunPhase/RoundOutcome/Warning). It also
    // publishes each `Checkpointed` manifest to the coordinator's checkpoint-pointer surface
    // (spec §9; lane R) so a later rejoiner can resume_from_checkpoint — best-effort, a POST failure
    // is a soft warning (the pointer is advisory).
    let (ev_tx, mut ev_rx) = unbounded_channel::<EngineEvent>();
    let out_for_fwd = out_tx.clone();
    let run_id_for_fwd = run_id.to_string();
    let roster_len = creds.roster.len().max(1) as u32;
    let registry_fwd = build_registry(coordinator, creds)?;
    let run_id_fwd_publish = run_id.to_string();
    let forwarder_task = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            if let EngineEvent::Checkpointed { round, manifest } = &ev {
                if let Err(e) = registry_fwd
                    .publish_checkpoint(
                        &run_id_fwd_publish,
                        *round,
                        &manifest.blake3.to_hex(),
                        manifest.size,
                    )
                    .await
                {
                    let _ = out_for_fwd.send(Event::Warning {
                        class: "checkpoint_publish".to_string(),
                        detail: format!("register checkpoint r{round} pointer failed: {e}"),
                    });
                }
            }
            for out in translate_engine_event(&ev, &run_id_for_fwd, roster_len) {
                if out_for_fwd.send(out).is_err() {
                    return;
                }
            }
        }
    });

    let mut engine = RoundEngine::new(
        control,
        store.clone(),
        backend,
        key,
        corpus,
        engine_cfg,
        ev_tx,
    );

    // LIVE checkpoint-resync (§9; lane R): before running the round loop, learn the latest
    // coordinator-published checkpoint pointer and — if the run is mid-flight — reload it and replay
    // the retained rounds forward, so this (re)join reaches state byte-identical to the survivors.
    // Best-effort: no checkpoint / a too-old gap / a fetch miss all fall back to the fresh-state
    // rejoin (current behavior). The engine already subscribed (above), so frames published during
    // resync buffer and are caught up by `run()`.
    let registry = build_registry(coordinator, creds)?;
    resync_on_join(
        &mut engine,
        &store,
        &registry,
        run_id,
        creds.engine.payload_retention_rounds,
        &out_tx,
    )
    .await;

    let out_for_engine = out_tx.clone();
    let engine_task = tokio::spawn(async move {
        let mut engine = engine;
        let result = engine.run().await;
        // The RoundEngine `run()` error was previously only stored in this JoinHandle and never
        // surfaced — a live-attach failure (e.g. a payload/transport fault mid-round) then looked
        // like a silent stall to the node + the operator. Surface it as a `Warning` through the
        // pump AND on stderr (inherited by the supervisor) so a failed round is diagnosable.
        if let Err(e) = &result {
            let _ = out_for_engine.send(Event::Warning {
                class: "engine_error".to_string(),
                detail: format!("live RoundEngine run() ended: {e}"),
            });
            eprintln!("[daemon-train-worker] live RoundEngine run() ended with error: {e}");
        }
        result
    });

    let _ = peer; // peer id is the Join signer; kept for parity with the harness recipe.
    Ok(LiveHandle {
        engine_task,
        forwarder_task,
        translator_task,
        ws,
        iroh,
    })
}

/// Build the iroh gossip node from the credentials' iroh half, or `None` for the WS-only baseline.
async fn build_iroh(creds: &JoinCredentials) -> Result<Option<Arc<IrohGossip>>, String> {
    let Some(ic) = &creds.iroh else {
        return Ok(None);
    };
    let mut roster = Vec::with_capacity(ic.roster.len());
    for p in &ic.roster {
        let direct_addrs = p
            .direct_addrs
            .iter()
            .map(|a| a.parse())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("bad iroh direct addr: {e}"))?;
        roster.push(IrohPeer {
            endpoint_id: p.endpoint_id,
            direct_addrs,
            relay_url: p.relay_url.clone(),
        });
    }
    let node = IrohGossip::connect(IrohGossipConfig {
        secret_key: ic.secret_key,
        relay_urls: ic.relay_urls.clone(),
        roster,
        topic_input: creds.envelope_hash,
        rebroadcast: RebroadcastConfig {
            enabled: true,
            interval: Duration::from_secs(2),
            ring_capacity: 64,
        },
        bind_addr: None,
    })
    .await
    .map_err(|e| format!("iroh connect: {e}"))?;
    Ok(Some(Arc::new(node)))
}

/// The worker's concrete payload store: R2-over-presign (live) or an FS fallback (tests / LAN). A
/// concrete (Sized) enum so it can be the `RoundEngine`'s `P` type parameter (a `dyn PayloadStore`
/// would be unsized).
enum WorkerStore {
    R2(R2Store<HttpPresignClient>),
    Fs(FsPayloadStore),
}

fn build_store(run_id: &str, creds: &JoinCredentials) -> Result<WorkerStore, String> {
    match &creds.presign_base {
        Some(base) => {
            use daemon_swarm_run::protocol::WsAuthSpec;
            let egress = daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                .map_err(|e| format!("egress client: {e}"))?;
            let presign_egress =
                daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                    .map_err(|e| format!("presign egress client: {e}"))?;
            // The presign requests carry the same swarm credential as the WS/registry surfaces
            // (Bearer on the gateway path, internal identity headers direct-to-apps/swarm).
            let presign = match &creds.ws_auth {
                WsAuthSpec::None => HttpPresignClient::new(presign_egress, base.clone()),
                WsAuthSpec::Bearer(t) => {
                    HttpPresignClient::new(presign_egress, base.clone()).with_bearer(t.clone())
                }
                WsAuthSpec::Internal { org_id, actor } => {
                    HttpPresignClient::new(presign_egress, base.clone())
                        .with_internal(org_id.clone(), actor.clone())
                }
            };
            Ok(WorkerStore::R2(R2Store::new(
                presign,
                egress,
                RunId::new(run_id),
            )))
        }
        None => {
            let root = std::env::temp_dir().join(format!(
                "daemon-train-worker-fs-{}-{run_id}",
                std::process::id()
            ));
            let retention = 64;
            FsPayloadStore::open(&root, retention)
                .map(WorkerStore::Fs)
                .map_err(|e| format!("fs payload store: {e}"))
        }
    }
}

/// Build the peer's [`Corpus`] (P3 lane S). A content-addressed `engine.corpus` reference triggers a
/// fetch-by-hash: the manifest object and the peer's **assigned** shards (the active data window,
/// [`Manifest::shards_covering`]) are pulled from the payload store, blake3-verified, and cached
/// content-addressed on disk, then assembled into a **windowed** corpus (only the run's shards held
/// in memory — the M4 32 GB RAM-budget guard). Absent ⇒ the deterministic synthetic corpus.
async fn build_corpus(run_id: &str, creds: &JoinCredentials) -> Result<Corpus, String> {
    let Some(cref) = &creds.engine.corpus else {
        return Corpus::synthetic(
            creds.engine.corpus_seed,
            creds.engine.corpus_shards,
            creds.engine.corpus_tokens_per_shard,
            creds.engine.corpus_seq_len,
        )
        .map_err(|e| format!("synthetic corpus: {e}"));
    };

    let cache = open_corpus_cache()?;
    let resolver = corpus_resolver(run_id, creds)?;

    // 1. Fetch + verify the manifest object (`corpus/<manifest_blake3>.json`).
    let manifest_hash = ContentHash::new(cref.manifest_blake3);
    let manifest_url = format!("r2://corpus/{}.json", manifest_hash.to_hex());
    let manifest_bytes = fetch_cached(
        &cache,
        &resolver,
        &ArtifactRef::new(manifest_url, manifest_hash),
    )
    .await?;
    let manifest = Manifest::from_json(
        std::str::from_utf8(&manifest_bytes).map_err(|e| format!("manifest utf8: {e}"))?,
    )
    .map_err(|e| format!("parse corpus manifest: {e}"))?;

    // 2. Stage ONLY the shards the active window touches (§8 windowing), verified + cached.
    let indices = manifest.shards_covering(cref.window_start, cref.window_sequences);
    let mut resident: BTreeMap<usize, Vec<u8>> = BTreeMap::new();
    for idx in indices {
        let desc = manifest
            .shards
            .get(idx)
            .ok_or_else(|| format!("shard index {idx} out of manifest range"))?;
        let shard_hash = hash_from_hex(&desc.blake3)
            .ok_or_else(|| format!("shard {idx} has a malformed blake3 hash"))?;
        let url = format!("r2://corpus/{}.bin", desc.blake3);
        let bytes = fetch_cached(&cache, &resolver, &ArtifactRef::new(url, shard_hash)).await?;
        resident.insert(idx, bytes);
    }

    Corpus::windowed(manifest, resident).map_err(|e| format!("windowed corpus: {e}"))
}

/// Open the on-disk content cache for corpus shards (`DAEMON_SWARM_CACHE_DIR` / `_CACHE_GB`).
fn open_corpus_cache() -> Result<ContentCache, String> {
    let dir = std::env::var_os("DAEMON_SWARM_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("daemon-swarm-cache"));
    let gb = std::env::var("DAEMON_SWARM_CACHE_GB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    ContentCache::open_gb(&dir, gb)
        .map_err(|e| format!("open content cache {}: {e}", dir.display()))
}

/// Build an [`ArtifactResolver`] for the corpus fetch from the live credentials (egress + presign for
/// `r2://`). The presign auth mirrors the payload-store auth (bearer / internal identity).
///
/// The presign **scope** is `DAEMON_SWARM_RUN_ID` when set (the shared asset prefix
/// `runs/<scope>/corpus/…` — content-addressed objects are published once and consumed by many runs;
/// the presign endpoint scopes keys, it does not require the id to be a live run), else the joined
/// run id. Mirrors `backend::store_fetch_context` so module + corpus resolve under one scope.
fn corpus_resolver(run_id: &str, creds: &JoinCredentials) -> Result<ArtifactResolver, String> {
    let scope = std::env::var("DAEMON_SWARM_RUN_ID").unwrap_or_else(|_| run_id.to_string());
    let run_id: &str = &scope;
    let egress = daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
        .map_err(|e| format!("egress client: {e}"))?;
    let mut resolver = ArtifactResolver::with_egress(egress);
    if let Some(base) = &creds.presign_base {
        use daemon_swarm_run::protocol::WsAuthSpec;
        let presign_egress =
            daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
                .map_err(|e| format!("presign egress client: {e}"))?;
        let presign = match &creds.ws_auth {
            WsAuthSpec::None => HttpPresignClient::new(presign_egress, base.clone()),
            WsAuthSpec::Bearer(t) => {
                HttpPresignClient::new(presign_egress, base.clone()).with_bearer(t.clone())
            }
            WsAuthSpec::Internal { org_id, actor } => {
                HttpPresignClient::new(presign_egress, base.clone())
                    .with_internal(org_id.clone(), actor.clone())
            }
        };
        let presign: Arc<dyn PresignClient> = Arc::new(presign);
        resolver = resolver.with_presign(presign, RunId::new(run_id));
    }
    Ok(resolver)
}

/// Fetch `art` (content cache first, then a presigned store GET), caching verified bytes on a miss.
async fn fetch_cached(
    cache: &ContentCache,
    resolver: &ArtifactResolver,
    art: &ArtifactRef,
) -> Result<Vec<u8>, String> {
    if let Some(bytes) = cache.get(&art.blake3).await.map_err(|e| e.to_string())? {
        return Ok(bytes);
    }
    let bytes = resolver.fetch(art).await.map_err(|e| e.to_string())?;
    if let Err(e) = cache.insert(&art.blake3, &bytes).await {
        eprintln!("[daemon-train-worker] corpus cache insert failed (continuing): {e}");
    }
    Ok(bytes)
}

/// Decode a 64-char lowercase-hex blake3 into a [`ContentHash`] (shared with the prefetch mode).
use crate::backend::hash_from_hex;

#[async_trait::async_trait]
impl PayloadStore for WorkerStore {
    async fn put(&self, key: &PayloadKey, bytes: &[u8]) -> Result<ContentHash, SwarmNetError> {
        match self {
            WorkerStore::R2(s) => s.put(key, bytes).await,
            WorkerStore::Fs(s) => s.put(key, bytes).await,
        }
    }
    async fn get(
        &self,
        key: &PayloadKey,
        expected: &ContentHash,
    ) -> Result<Vec<u8>, SwarmNetError> {
        match self {
            WorkerStore::R2(s) => s.get(key, expected).await,
            WorkerStore::Fs(s) => s.get(key, expected).await,
        }
    }
    async fn head(&self, key: &PayloadKey) -> Result<PayloadStat, SwarmNetError> {
        match self {
            WorkerStore::R2(s) => s.head(key).await,
            WorkerStore::Fs(s) => s.head(key).await,
        }
    }
}

impl WorkerStore {
    /// Fetch the committed set (`record-set.cbor`) the coordinator wrote for `round` — the
    /// `(peer, hash, size)` membership a rejoining peer stages when replaying that round forward
    /// during a live checkpoint-resync (§9; lane R). Only the R2 plane carries it (the coordinator
    /// wrote it to R2); the FS fallback plane has no coordinator, so resync there degrades to
    /// fresh-state.
    async fn committed_set(&self, round: RoundId) -> Result<Vec<RecordEntry>, SwarmNetError> {
        match self {
            WorkerStore::R2(s) => Ok(s.fetch_record_set_object(round).await?.entries().to_vec()),
            WorkerStore::Fs(_) => Err(SwarmNetError::Fetch(
                "record-set fetch unsupported on the FS payload plane — resync falls back to \
                 fresh-state"
                    .to_string(),
            )),
        }
    }
}

/// Build a [`RegistryClient`] against the coordinator base with the credentials' `swarm:*` auth —
/// used for the checkpoint-pointer surface (fetch on rejoin, publish on checkpoint). The base is the
/// same `{…}/api/v1/swarm` the WS client dials; the client trims a trailing slash.
fn build_registry(coordinator: &str, creds: &JoinCredentials) -> Result<RegistryClient, String> {
    use daemon_swarm_run::protocol::WsAuthSpec;
    let egress = daemon_egress::EgressClient::new(daemon_egress::EgressConfig::default())
        .map_err(|e| format!("registry egress client: {e}"))?;
    let client = RegistryClient::new(egress, coordinator.to_string());
    Ok(match &creds.ws_auth {
        WsAuthSpec::None => client,
        WsAuthSpec::Bearer(t) => client.with_bearer(t.clone()),
        WsAuthSpec::Internal { org_id, actor } => {
            client.with_internal(org_id.clone(), actor.clone())
        }
    })
}

/// **Live checkpoint-resync on (re)join (spec §9; lane R).** Query the coordinator's latest
/// checkpoint pointer; if the run is mid-flight, reload the checkpoint and replay the retained
/// rounds forward through the engine so this peer rejoins byte-identical to the survivors. Every
/// honest edge — no checkpoint yet, a gap wider than the retention floor, a fetch miss — falls back
/// to the fresh-state rejoin (the pre-P3 behavior) with a `Warning`. Progress is surfaced as
/// `Event::ResyncProgress` telemetry by the engine.
async fn resync_on_join(
    engine: &mut RoundEngine<DualPlane, WorkerStore, LadderBackend>,
    store: &Arc<WorkerStore>,
    registry: &RegistryClient,
    run_id: &str,
    retention: u64,
    out_tx: &UnboundedSender<Event>,
) {
    let warn = |detail: String| {
        let _ = out_tx.send(Event::Warning {
            class: "resync".to_string(),
            detail,
        });
    };

    let state = match registry.fetch_state(run_id).await {
        Ok(Some(s)) => s,
        // Run not initialized yet, or no state — a genuinely fresh join. Nothing to resync.
        Ok(None) => {
            warn("coordinator /state 404 (run not initialized); fresh-state rejoin".to_string());
            return;
        }
        Err(e) => {
            warn(format!(
                "coordinator state fetch failed ({e}); fresh-state rejoin"
            ));
            return;
        }
    };

    // No checkpoint published yet (first epoch, §9) → fresh-state (current behavior).
    let Some(ptr) = state.checkpoint else {
        warn(format!(
            "no checkpoint pointer at rejoin (phase={} round={}); fresh-state rejoin",
            state.phase, state.round
        ));
        return;
    };
    warn(format!(
        "resync plan: checkpoint round {} size {}, current round {} phase {}",
        ptr.round, ptr.size, state.round, state.phase
    ));
    // Resolve the checkpoint object actually stored at round `ptr.round` and load THAT (verified by
    // its own content hash), rather than requiring a byte match to the pointer's hash. A checkpoint
    // captures the deterministic CONSENSUS state (params + replicated persistents, §9) PLUS per-peer
    // LOCAL optimizer state (Adam moments) that legitimately differs across peers — so peers write
    // byte-divergent checkpoint objects to the shared key even though their post-round digest agrees.
    // The digest (§5.6) and the replay fold depend only on the consensus half, so loading whichever
    // valid post-`round` checkpoint is stored + replaying reproduces the exact consensus digest; the
    // pointer's role is to name the round (and prove a checkpoint exists). HEAD yields the stored
    // object's real hash for the content-verified load.
    let ckpt_key = PayloadKey::new(RunId::new(run_id), ptr.round, CHECKPOINT_PEER);
    let stat = match store.head(&ckpt_key).await {
        Ok(s) => s,
        Err(e) => {
            warn(format!(
                "checkpoint object for round {} unavailable ({e}); fresh-state rejoin",
                ptr.round
            ));
            return;
        }
    };
    let manifest = CheckpointManifest {
        round: ptr.round,
        blake3: stat.hash,
        size: stat.size,
        digest: StateDigest([0u8; 16]), // load verifies by blake3; the digest field is unused there
    };

    // The newest finalized round (the coordinator's current open round − 1) is the resync target.
    let target = state.round.saturating_sub(1);
    if target <= ptr.round {
        // The checkpoint already covers the newest finalized round: reload it (no replay) and let
        // the engine catch up from there live.
        if let Err(e) = engine.resume_from_checkpoint(&manifest).await {
            warn(format!(
                "checkpoint resume failed ({e}); fresh-state rejoin"
            ));
        }
        return;
    }

    // §9 resync-replay window: replay only if the whole gap is still retained (else wait for epoch).
    let effective_retention = if retention == 0 { u64::MAX } else { retention };
    let from_round = match plan_resync(ptr.round, target, effective_retention) {
        ResyncPlan::ReplayFromCheckpoint { from_round, .. } => from_round,
        ResyncPlan::WaitForEpoch => {
            warn(format!(
                "resync gap {}->{target} exceeds retention {retention}; fresh-state rejoin \
                 (waiting for the next epoch checkpoint, §9)",
                ptr.round
            ));
            return;
        }
    };

    // Assemble the replay steps FIRST (all fetches complete before the backend is touched) so a
    // network miss falls back to fresh-state with the backend still clean.
    let mut steps = Vec::new();
    for round in (from_round + 1)..=target {
        let entries = match store.committed_set(round).await {
            Ok(e) => e,
            Err(e) => {
                warn(format!(
                    "retained committed set for round {round} unavailable ({e}); fresh-state rejoin"
                ));
                return;
            }
        };
        let mut staged = Vec::with_capacity(entries.len());
        for entry in &entries {
            let key = PayloadKey::new(RunId::new(run_id), round, entry.peer);
            match store.get(&key, &entry.hash).await {
                Ok(bytes) => staged.push(StagedPayload {
                    peer: entry.peer,
                    hash: entry.hash,
                    bytes,
                }),
                Err(e) => {
                    warn(format!(
                        "retained payload r{round}/{} unavailable ({e}); fresh-state rejoin",
                        entry.peer.to_hex()
                    ));
                    return;
                }
            }
        }
        steps.push(ReplayStep { round, staged });
    }

    match engine.resync_from_checkpoint(&manifest, &steps).await {
        Ok(_) => {} // per-round `ResyncProgress` telemetry emitted by the engine
        Err(e) => {
            // The checkpoint + replay data are in hand but the fold failed (a real fault): surface a
            // typed Desync so the supervisor's respawn-and-retry loop (§9) handles it.
            let _ = out_tx.send(Event::Error {
                class: ErrorClass::Desync,
                detail: format!("checkpoint resync replay failed: {e}"),
            });
        }
    }
}

/// Translate an [`EngineEvent`] into the worker protocol [`Event`]s the node's `SwarmService`
/// consumes (a run's phase / round outcome / warnings; §10.3/§10.4).
fn translate_engine_event(ev: &EngineEvent, run_id: &str, roster_len: u32) -> Vec<Event> {
    match ev {
        EngineEvent::RoundComplete { round, digest } => vec![
            Event::RunPhase {
                run_id: run_id.to_string(),
                phase: "train".to_string(),
                epoch: 0,
                round: *round,
            },
            Event::RoundOutcome {
                round: *round,
                committed: roster_len,
                ingested: roster_len,
                stalled: false,
                digest: *digest.as_bytes(),
            },
        ],
        EngineEvent::CaughtUp { round, digest } => vec![
            Event::Warning {
                class: "caught_up".to_string(),
                detail: format!("round {round} late-ingested"),
            },
            Event::RoundOutcome {
                round: *round,
                committed: roster_len,
                ingested: roster_len,
                stalled: false,
                digest: *digest.as_bytes(),
            },
        ],
        EngineEvent::Straggling { round, status } => vec![Event::Warning {
            class: "straggling".to_string(),
            detail: format!("round {round}: {status:?}"),
        }],
        EngineEvent::Checkpointed { round, manifest } => vec![Event::CheckpointPublished {
            round: *round,
            hash: manifest.blake3.to_hex(),
            location: format!("runs/{run_id}/rounds/{round}/checkpoint"),
        }],
        EngineEvent::Resynced {
            round,
            from_checkpoint,
            replayed,
            total,
        } => vec![Event::ResyncProgress {
            round: *round,
            from_checkpoint: *from_checkpoint,
            replayed: *replayed,
            total: *total,
        }],
        EngineEvent::Left { round, reason } => vec![Event::Warning {
            class: "left".to_string(),
            detail: format!("round {round}: {reason}"),
        }],
        // Committed / Attested are per-peer intermediate signals; the node renders round outcomes.
        EngineEvent::Committed { .. } | EngineEvent::Attested { .. } => Vec::new(),
    }
}

fn to_ws_auth(spec: &daemon_swarm_run::protocol::WsAuthSpec) -> WsAuth {
    use daemon_swarm_run::protocol::WsAuthSpec;
    match spec {
        WsAuthSpec::None => WsAuth::None,
        WsAuthSpec::Bearer(t) => WsAuth::Bearer(t.clone()),
        WsAuthSpec::Internal { org_id, actor } => WsAuth::Internal {
            org_id: org_id.clone(),
            actor: actor.clone(),
        },
    }
}

/// The host engine profile for the live worker's `WasmBackend` (Merge-2 reconciliation of the G and S
/// lanes): the **backend + GPU index come from the probe ladder** [`crate::backend::select_backend`]
/// (§10.5 verdict path; swarm-ledger-p3-g D4 — CUDA → wgpu → CPU, with the `DAEMON_TRAIN_BACKEND=cpu`
/// escape hatch + the D6 NVRTC-readiness gate), composed **over the roomy 160M sandbox budgets** from
/// [`crate::backend::engine_config_from_env`] (P3 lane S — a 768-wide model's real fp32 steps trip the
/// tiny-model 5 s epoch watchdog; the budgets apply when `DAEMON_TRAIN_BACKEND` is set, tiny-model
/// defaults otherwise). The det lane stays host fp32 on every backend, so a CUDA/wgpu peer's per-round
/// digests remain byte-identical to CPU peers (the consensus invariant is unchanged).
fn worker_engine_config() -> WasmEngineConfig {
    let (backend, gpu_index) = crate::backend::select_backend();
    WasmEngineConfig {
        backend,
        gpu_index,
        ..crate::backend::engine_config_from_env()
    }
}

/// Build + `da_build` a fresh [`WasmBackend`] (also the OOM-churn rebuild).
fn build_wasm_backend(module: &[u8], config: &[u8]) -> Result<WasmBackend, String> {
    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: module.to_vec(),
        engine: worker_engine_config(),
    })
    .map_err(|e| e.to_string())?;
    backend.build(config).map_err(|e| e.to_string())?;
    Ok(backend)
}

/// A [`TrainerBackend`] wrapper around [`WasmBackend`] that (a) surfaces the per-step `loss` as a
/// `Metric` through the event pump, and (b) implements the §10.5 OOM ladder: a real `BudgetMemory`
/// trap during a step churns the instance (a fresh build releases its memory) and retries, emitting
/// the additive `OomLadder` telemetry. Tiny-llama never OOMs, so the ladder is a defensive recovery
/// seam exercised only under real memory pressure.
///
/// One boxed backend operation, executed on the pinned device thread.
#[cfg(feature = "cuda")]
type DeviceJob = Box<dyn FnOnce(&mut WasmBackend) + Send>;

/// A dedicated OS thread that OWNS the `WasmBackend` — construction and every subsequent call
/// happen on this one thread (the daemon-infer GPU-worker pattern, channel-serialized).
///
/// **Why (P3 Merge-2, the CUDA-live fix).** The engine future runs under tokio and migrates
/// across worker threads at `.await` points, so the backend's sync methods execute on whichever
/// thread polls the future. cubecl derives its CUDA stream + memory-pool registry from the
/// *calling thread* (`cubecl_common::StreamId::current()` is a thread-local), so cross-thread
/// polls split the pool bookkeeping across per-thread streams. At small scale the split pools
/// merely waste memory; at 160M (~24 GiB pool, paging active) a handle allocated under one
/// thread's stream is looked up under another's → cubecl-cuda `server.rs:124` "can't allocate
/// buffer" / `stream.rs:101` "Memory page 0 doesn't exist" panics (ceremony5/7/11 + the 160M
/// live bisect; single-host — one thread — is green at the same scale, reduced-live — low
/// pressure — is green too). Pinning every backend call to the context-owning thread restores
/// cubecl's single-stream discipline. Calls block the caller until the device thread replies —
/// byte-identical semantics to the previous in-place call, just on a fixed thread.
#[cfg(feature = "cuda")]
struct DeviceThread {
    tx: std::sync::mpsc::Sender<DeviceJob>,
}

#[cfg(feature = "cuda")]
impl DeviceThread {
    /// Spawn the device thread and construct the backend ON it (CUDA context + streams are then
    /// owned by this thread for the process lifetime).
    fn spawn(module: Vec<u8>, config: Vec<u8>) -> Result<Self, String> {
        let (tx, rx) = std::sync::mpsc::channel::<DeviceJob>();
        let (init_tx, init_rx) = std::sync::mpsc::sync_channel::<Result<(), String>>(1);
        std::thread::Builder::new()
            .name("gpu-device".into())
            .spawn(move || {
                let mut backend = match build_wasm_backend(&module, &config) {
                    Ok(b) => {
                        let _ = init_tx.send(Ok(()));
                        b
                    }
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(job) = rx.recv() {
                    job(&mut backend);
                }
            })
            .map_err(|e| format!("spawn gpu device thread: {e}"))?;
        init_rx
            .recv()
            .map_err(|e| format!("gpu device thread init: {e}"))??;
        Ok(Self { tx })
    }

    /// Run `f` on the device thread, blocking until it replies.
    fn call<R, F>(&self, f: F) -> R
    where
        R: Send + 'static,
        F: FnOnce(&mut WasmBackend) -> R + Send + 'static,
    {
        let (rtx, rrx) = std::sync::mpsc::sync_channel::<R>(1);
        self.tx
            .send(Box::new(move |b| {
                let _ = rtx.send(f(b));
            }))
            .expect("gpu device thread alive");
        rrx.recv().expect("gpu device thread reply")
    }
}

/// Where the `WasmBackend` lives.
///
/// `Direct`: in-place behind a `Mutex` (the B3 live-harness recipe — the `WasmBackend` is `Send`
/// but not `Sync`, the engine future must be `Send`, and the engine owns the backend exclusively
/// so the lock is uncontended). Used for the CPU and wgpu lanes: the CPU lane has no device
/// state, and wgpu's queue submission is internally synchronized (proven green at gate scale) —
/// don't change what's proven.
///
/// `Pinned`: on a [`DeviceThread`] — the CUDA lane (see there for why).
enum BackendHost {
    Direct(Box<std::sync::Mutex<WasmBackend>>),
    #[cfg(feature = "cuda")]
    Pinned(DeviceThread),
}

impl BackendHost {
    fn with<R, F>(&self, f: F) -> R
    where
        R: Send + 'static,
        F: FnOnce(&mut WasmBackend) -> R + Send + 'static,
    {
        match self {
            BackendHost::Direct(m) => f(&mut m.lock().expect("backend lock")),
            #[cfg(feature = "cuda")]
            BackendHost::Pinned(t) => t.call(f),
        }
    }
}

/// The §10.5 OOM-ladder / event-emitting [`TrainerBackend`] wrapper around the [`WasmBackend`]
/// (hosted per [`BackendHost`]).
struct LadderBackend {
    host: BackendHost,
    module: Vec<u8>,
    config: Vec<u8>,
    events: UnboundedSender<Event>,
    /// Clamp corpus token ids into the experiment vocab (`token % clamp`; 0 = off) — the B3 shim
    /// recipe, applied identically by every peer (deterministic, so digests agree).
    vocab_clamp: u32,
    round: RoundId,
    halvings: u32,
}

impl LadderBackend {
    /// Build the backend, pinning the CUDA lane to a dedicated device thread (see
    /// [`DeviceThread`]); every other lane constructs in place (unchanged).
    fn new(
        module: Vec<u8>,
        config: Vec<u8>,
        events: UnboundedSender<Event>,
        vocab_clamp: u32,
    ) -> Result<Self, String> {
        let (kind, _gpu) = crate::backend::select_backend();
        let host = match kind {
            #[cfg(feature = "cuda")]
            daemon_train::BackendKind::Cuda => {
                eprintln!(
                    "daemon-train-worker: pinning the CUDA backend to a dedicated device thread \
                     (cubecl per-thread stream discipline)"
                );
                BackendHost::Pinned(DeviceThread::spawn(module.clone(), config.clone())?)
            }
            _ => BackendHost::Direct(Box::new(std::sync::Mutex::new(build_wasm_backend(
                &module, &config,
            )?))),
        };
        Ok(Self {
            host,
            module,
            config,
            events,
            vocab_clamp,
            round: 0,
            halvings: 0,
        })
    }
}

fn is_oom(e: &WasmBackendError) -> bool {
    matches!(
        e,
        WasmBackendError::Train(TrainError::Trap(t)) if t.code == TrapCode::BudgetMemory
    )
}

impl TrainerBackend for LadderBackend {
    type Error = WasmBackendError;

    fn build(&mut self, config: &[u8]) -> Result<(), Self::Error> {
        let config = config.to_vec();
        self.host.with(move |b| b.build(&config))
    }
    fn assess(
        &self,
        meta: &daemon_swarm_run::backend::AssessMeta,
    ) -> Result<daemon_swarm_run::backend::Assessment, Self::Error> {
        let meta = *meta;
        self.host.with(move |b| b.assess(&meta))
    }
    fn train_step(
        &mut self,
        batch: &BatchRef,
        ctx: StepCtx,
    ) -> Result<daemon_swarm_run::backend::StepStats, Self::Error> {
        let batch = if self.vocab_clamp > 0 {
            BatchRef {
                tokens: batch.tokens.iter().map(|t| t % self.vocab_clamp).collect(),
                seq_len: batch.seq_len,
            }
        } else {
            batch.clone()
        };
        let step_batch = batch.clone();
        let first = self.host.with(move |b| b.train_step(&step_batch, ctx));
        let stats = match first {
            Ok(s) => s,
            Err(e) if is_oom(&e) => {
                // §10.5 churn: a fresh instance releases the OOMing instance's memory; retry once.
                // The rebuild happens ON the backend's host (device thread for CUDA) so the fresh
                // context/streams stay thread-owned.
                self.halvings += 1;
                let _ = self.events.send(Event::OomLadder {
                    round: self.round,
                    from_micro_batch: ctx.step_seqs,
                    to_micro_batch: ctx.step_seqs.max(2) / 2,
                    halvings: self.halvings,
                });
                let module = self.module.clone();
                let config = self.config.clone();
                self.host.with(move |slot| {
                    let mut fresh = WasmBackend::new(WasmBackendConfig {
                        wasm: module,
                        engine: worker_engine_config(),
                    })?;
                    fresh.build(&config)?;
                    let stats = fresh.train_step(&batch, ctx)?;
                    *slot = fresh;
                    Ok::<_, WasmBackendError>(stats)
                })?
            }
            Err(e) => return Err(e),
        };
        let _ = self.events.send(Event::Metric {
            name: "loss".to_string(),
            value: f64::from(stats.loss),
        });
        Ok(stats)
    }
    fn inner_update(&mut self, inner_step: u32) -> Result<(), Self::Error> {
        self.host.with(move |b| b.inner_update(inner_step))
    }
    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error> {
        self.host.with(move |b| b.make_update(round))
    }
    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, Self::Error> {
        self.round = round;
        self.halvings = 0;
        let staged = staged.to_vec();
        self.host.with(move |b| b.ingest(round, &staged))
    }
    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error> {
        self.host.with(|b| b.checkpoint_save())
    }
    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        let bytes = bytes.to_vec();
        self.host.with(move |b| b.checkpoint_load(&bytes))
    }
}
