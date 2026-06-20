//! `daemon-host` — the durable substrate that runs a unit, and the protocol translator.
//!
//! Composes the phase-1 durable substrate (`daemon-store` + `daemon-activation`) into a
//! continuously-running host whose fixed resident-service tree
//! ([`daemon-host-spec.md`](../../../docs/specs/daemon-host-spec.md) §5) runs under a one-for-one
//! restart/backoff/meltdown [`Supervisor`] (phase 2), and adds the host's defining job: the §17 ⇄
//! management protocol translation (§9, phase 3).
//!
//! Two adapters bridge the engine to the substrate and to the supervisor above:
//! - [`CoreIncarnation`] / [`CoreEngineFactory`] drive a real `daemon-core` engine through the
//!   protocol-agnostic activation seam (host-spec §3.1), keeping `daemon-core` free of the durable
//!   substrate.
//! - [`EngineUnit`] presents a running engine as a `UnitKind::Engine`
//!   [`ManagedUnit`](daemon_supervision::ManagedUnit), realizing the supervision §4 mapping table.
//!
//! Phase 5 adds the protocol-aware side of a placement *cut* ([`cut`]): [`PlacedUnit`] presents an
//! out-of-process child as a `ManagedUnit`, brokering the parent's store across the cut so fencing
//! holds out-of-process, and [`run_placed_child`] is the child-side loop.
//!
//! Phase 6 threads a `TraceId` across the cut (stamped on send, restored on receive) and folds a
//! placed unit's `Usage` into a resident [`Metrics`](daemon_telemetry::Metrics) dump.
//!
//! Phase 7 adds the credential broker ([`credentials`]): [`OwnerBroker`]/[`RelayBroker`] realize
//! recursive serve-or-forward of `daemon-credentials` capability leases, [`RemoteCredentialClient`]
//! is the descendant-side client over a credential cut, and [`BrokeredCredentialProvider`] bridges
//! a broker to the engine's §7 port.
//!
//! See `docs/specs/daemon-host-spec.md`.

#![forbid(unsafe_code)]

pub mod config;
pub mod credentials;
pub mod cut;
pub mod engine_incarnation;
pub mod journal;
pub mod node_api;
pub mod services;
pub mod socket;
pub mod supervisor;
pub mod unit;

pub use config::HostConfig;
pub use credentials::{
    BrokeredCredentialProvider, CredentialBroker, FenceGuard, OwnerBroker, RelayBroker,
};
pub use cut::{
    run_placed_child, serve_credentials, CredCall, CredReplyBody, CutFrame, PlacedUnit,
    RemoteCredentialClient, RemoteStoreClient, StoreCall, StoreReplyBody,
};
pub use journal::{journal_stream, JournalSink};
pub use engine_incarnation::{CoreEngineFactory, CoreIncarnation, ProviderBuilder};
pub use node_api::{NodeApiImpl, SessionEngineBuilder};
pub use socket::{serve_api_unix, ApiClient};
pub use supervisor::{
    Backoff, ChildSpec, HealthStatus, MeltdownPolicy, RestartPolicy, ServiceError, Supervisor,
    SupervisorHandle, SupervisorObserver,
};
pub use unit::EngineUnit;

use async_trait::async_trait;
use daemon_activation::{ActivationManager, EngineFactory};
use daemon_api::FleetReport;
use daemon_common::UnitId;
use daemon_store::SessionStore;
use daemon_telemetry::Metrics;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// The `JobOutboxDispatcher`'s per-tick work: drain the durable job outbox once. The default host
/// uses `ActivationManager::run_workers` (the substrate echo worker); the binary injects the real
/// orchestration `FleetRuntime` through this seam (`Host::with_job_worker`) so the node spawns and
/// drives a child per delegation job — *without* `daemon-host` depending on `daemon-orchestration`.
#[async_trait]
pub trait JobWorker: Send + Sync {
    /// Process every currently-pending durable job, returning when the outbox is drained.
    async fn process_jobs_once(&self) -> Result<(), ServiceError>;
}

/// A read-only projection of the running orchestration fleet for the node control surface
/// (`ControlApi::fleet`/`cancel`). Implemented by the binary over its `FleetRuntime`, keeping
/// `daemon-host` free of the orchestration crate.
#[async_trait]
pub trait FleetView: Send + Sync {
    /// The fleet roster + folded usage.
    async fn report(&self) -> FleetReport;
    /// Cancel a registered child by id; returns whether a child was found and cancelled.
    async fn cancel(&self, child: &UnitId) -> bool;
}

/// An in-process host: the durable activation substrate plus its supervised resident-service tree.
#[derive(Clone)]
pub struct Host {
    store: Arc<dyn SessionStore>,
    manager: ActivationManager,
    config: HostConfig,
    metrics: Metrics,
    job_worker: Option<Arc<dyn JobWorker>>,
}

impl Host {
    /// Construct a host over a durable store and an injected engine factory.
    pub fn new(
        store: Arc<dyn SessionStore>,
        factory: Arc<dyn EngineFactory>,
        config: HostConfig,
    ) -> Self {
        let manager = ActivationManager::new(store.clone(), factory, config.partition);
        Self {
            store,
            manager,
            config,
            metrics: Metrics::new(),
            job_worker: None,
        }
    }

    /// Drive the `JobOutboxDispatcher` with an injected [`JobWorker`] (e.g. the orchestration
    /// `FleetRuntime`) instead of the substrate's placeholder echo worker.
    pub fn with_job_worker(mut self, worker: Arc<dyn JobWorker>) -> Self {
        self.job_worker = Some(worker);
        self
    }

    /// The host's resident usage/health aggregator (folded across units reporting to it).
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// The underlying activation manager (e.g. to inspect `active_count`).
    pub fn manager(&self) -> &ActivationManager {
        &self.manager
    }

    /// The durable store.
    pub fn store(&self) -> &Arc<dyn SessionStore> {
        &self.store
    }

    /// The host configuration.
    pub fn config(&self) -> &HostConfig {
        &self.config
    }

    /// Start the supervised resident-service tree with a fresh cancellation token.
    pub fn start(&self) -> SupervisorHandle {
        self.start_with_cancel(CancellationToken::new())
    }

    /// Start the resident tree under a caller-supplied cancellation token.
    pub fn start_with_cancel(&self, cancel: CancellationToken) -> SupervisorHandle {
        let cfg = self.config;
        // The job-outbox dispatcher runs the injected fleet worker if present, else the substrate's
        // built-in echo worker.
        let job_tick = match &self.job_worker {
            Some(worker) => services::job_worker_tick(worker.clone()),
            None => services::job_tick(self.manager.clone()),
        };
        Supervisor::new(cfg.meltdown)
            .child(services::interval_child(
                "WakeOutboxDispatcher",
                cfg.dispatch_interval,
                RestartPolicy::Permanent,
                cfg.backoff,
                services::wake_tick(self.manager.clone()),
            ))
            .child(services::interval_child(
                "JobOutboxDispatcher",
                cfg.dispatch_interval,
                RestartPolicy::Permanent,
                cfg.backoff,
                job_tick,
            ))
            .child(services::interval_child(
                "RecoveryScanner",
                cfg.scan_interval,
                RestartPolicy::Permanent,
                cfg.backoff,
                services::scan_tick(self.manager.clone()),
            ))
            .child(services::interval_child(
                "Metrics",
                cfg.scan_interval,
                RestartPolicy::Permanent,
                cfg.backoff,
                services::metrics_tick(
                    self.store.clone(),
                    self.manager.clone(),
                    self.metrics.clone(),
                ),
            ))
            .start(cancel)
    }
}
