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

use daemon_provision::CutWriter;
use daemon_swarm_net::{
    ContentHash, ControlPlane, DualPlane, FsPayloadStore, HttpPresignClient, IrohGossip,
    IrohGossipConfig, IrohPeer, PayloadKey, PayloadStat, PayloadStore, R2Store, RebroadcastConfig,
    ReconnectConfig, RunId, SwarmNetError, WsAuth, WsConfig, WsControlPlane,
};
use daemon_swarm_proto::messages::{Join, ThroughputClass};
use daemon_swarm_proto::{
    peer_id, to_canonical_vec, CapabilitySet, Hash, IrohId, PeerId, SignedMessage, SigningKey,
    SwarmMessage, SwarmProtoVersion, SWARM_PROTO_VERSION,
};
use daemon_swarm_run::backend::{BatchRef, StagedPayload, StateDigest, StepCtx, TrainerBackend};
use daemon_swarm_run::data::Corpus;
use daemon_swarm_run::engine::{EngineConfig, EngineEvent, RoundEngine};
use daemon_swarm_run::protocol::{Event, JoinCredentials};
use daemon_swarm_run::seam::RoundId;
use daemon_swarm_run::SwarmRunError;
use daemon_train::{
    EngineConfig as WasmEngineConfig, TrainError, TrapCode, WasmBackend, WasmBackendConfig,
    WasmBackendError,
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
    let corpus = Arc::new(
        Corpus::synthetic(
            creds.engine.corpus_seed,
            creds.engine.corpus_shards,
            creds.engine.corpus_tokens_per_shard,
            creds.engine.corpus_seq_len,
        )
        .map_err(|e| format!("synthetic corpus: {e}"))?,
    );
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
    // OomLadder telemetry through the same pump.
    let backend = LadderBackend::new(
        build_wasm_backend(module, config)?,
        module.to_vec(),
        config.to_vec(),
        out_tx.clone(),
        creds.engine.corpus_vocab_clamp,
    );

    // EngineEvent → protocol::Event forwarder (per-round RunPhase/RoundOutcome/Warning).
    let (ev_tx, mut ev_rx) = unbounded_channel::<EngineEvent>();
    let out_for_fwd = out_tx.clone();
    let run_id_for_fwd = run_id.to_string();
    let roster_len = creds.roster.len().max(1) as u32;
    let forwarder_task = tokio::spawn(async move {
        while let Some(ev) = ev_rx.recv().await {
            for out in translate_engine_event(&ev, &run_id_for_fwd, roster_len) {
                if out_for_fwd.send(out).is_err() {
                    return;
                }
            }
        }
    });

    let engine = RoundEngine::new(control, store, backend, key, corpus, engine_cfg, ev_tx);
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

/// Build + `da_build` a fresh [`WasmBackend`] (also the OOM-churn rebuild).
fn build_wasm_backend(module: &[u8], config: &[u8]) -> Result<WasmBackend, String> {
    let mut backend = WasmBackend::new(WasmBackendConfig {
        wasm: module.to_vec(),
        engine: WasmEngineConfig::default(),
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
/// The `WasmBackend` is `Send` but **not** `Sync` (its `dyn OpBackend`), while the spawned engine
/// future must be `Send` — which requires the backend be `Sync` (the engine holds `&self` across the
/// publish `.await`). So it rides in a `Mutex` (`Mutex<T>: Sync` for `T: Send`); the engine owns the
/// backend exclusively, so the lock is uncontended (B3's live-harness adapter recipe).
struct LadderBackend {
    inner: std::sync::Mutex<WasmBackend>,
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
    fn new(
        inner: WasmBackend,
        module: Vec<u8>,
        config: Vec<u8>,
        events: UnboundedSender<Event>,
        vocab_clamp: u32,
    ) -> Self {
        Self {
            inner: std::sync::Mutex::new(inner),
            module,
            config,
            events,
            vocab_clamp,
            round: 0,
            halvings: 0,
        }
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
        self.inner.get_mut().expect("backend lock").build(config)
    }
    fn assess(
        &self,
        meta: &daemon_swarm_run::backend::AssessMeta,
    ) -> Result<daemon_swarm_run::backend::Assessment, Self::Error> {
        self.inner.lock().expect("backend lock").assess(meta)
    }
    fn train_step(
        &mut self,
        batch: &BatchRef,
        ctx: StepCtx,
    ) -> Result<daemon_swarm_run::backend::StepStats, Self::Error> {
        let clamped;
        let batch = if self.vocab_clamp > 0 {
            clamped = BatchRef {
                tokens: batch.tokens.iter().map(|t| t % self.vocab_clamp).collect(),
                seq_len: batch.seq_len,
            };
            &clamped
        } else {
            batch
        };
        let first = self
            .inner
            .get_mut()
            .expect("backend lock")
            .train_step(batch, ctx);
        let stats = match first {
            Ok(s) => s,
            Err(e) if is_oom(&e) => {
                // §10.5 churn: a fresh instance releases the OOMing instance's memory; retry once.
                self.halvings += 1;
                let _ = self.events.send(Event::OomLadder {
                    round: self.round,
                    from_micro_batch: ctx.step_seqs,
                    to_micro_batch: ctx.step_seqs.max(2) / 2,
                    halvings: self.halvings,
                });
                let mut fresh = WasmBackend::new(WasmBackendConfig {
                    wasm: self.module.clone(),
                    engine: WasmEngineConfig::default(),
                })?;
                fresh.build(&self.config)?;
                let stats = fresh.train_step(batch, ctx)?;
                *self.inner.get_mut().expect("backend lock") = fresh;
                stats
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
        self.inner
            .get_mut()
            .expect("backend lock")
            .inner_update(inner_step)
    }
    fn make_update(&mut self, round: RoundId) -> Result<Vec<u8>, Self::Error> {
        self.inner
            .get_mut()
            .expect("backend lock")
            .make_update(round)
    }
    fn ingest(
        &mut self,
        round: RoundId,
        staged: &[StagedPayload],
    ) -> Result<StateDigest, Self::Error> {
        self.round = round;
        self.halvings = 0;
        self.inner
            .get_mut()
            .expect("backend lock")
            .ingest(round, staged)
    }
    fn checkpoint_save(&self) -> Result<Vec<u8>, Self::Error> {
        self.inner.lock().expect("backend lock").checkpoint_save()
    }
    fn checkpoint_load(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        self.inner
            .get_mut()
            .expect("backend lock")
            .checkpoint_load(bytes)
    }
}
