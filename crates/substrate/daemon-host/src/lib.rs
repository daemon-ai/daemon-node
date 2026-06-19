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
//! Deferred to later phases: credential authority and remote (cross-node) transport.
//!
//! See `docs/specs/daemon-host-spec.md`.

#![forbid(unsafe_code)]

pub mod config;
pub mod cut;
pub mod engine_incarnation;
pub mod journal;
pub mod services;
pub mod supervisor;
pub mod unit;

pub use config::HostConfig;
pub use cut::{run_placed_child, CutFrame, PlacedUnit, RemoteStoreClient, StoreCall, StoreReplyBody};
pub use journal::{journal_stream, JournalSink};
pub use engine_incarnation::{CoreEngineFactory, CoreIncarnation, ProviderBuilder};
pub use supervisor::{
    Backoff, ChildSpec, HealthStatus, MeltdownPolicy, RestartPolicy, ServiceError, Supervisor,
    SupervisorHandle,
};
pub use unit::EngineUnit;

use daemon_activation::{ActivationManager, EngineFactory};
use daemon_store::SessionStore;
use daemon_telemetry::Metrics;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// An in-process host: the durable activation substrate plus its supervised resident-service tree.
#[derive(Clone)]
pub struct Host {
    store: Arc<dyn SessionStore>,
    manager: ActivationManager,
    config: HostConfig,
    metrics: Metrics,
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
        }
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
                services::job_tick(self.manager.clone()),
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
