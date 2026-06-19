//! `daemon-host` — the durable substrate that runs a unit.
//!
//! Phase 2 scope: the **resident-service supervision** layer. It composes the phase-1 durable
//! substrate (`daemon-store` + `daemon-activation`) into a continuously-running host whose fixed
//! resident-service tree ([`daemon-host-spec.md`](../../../docs/specs/daemon-host-spec.md) §5) runs
//! under a one-for-one restart/backoff/meltdown [`Supervisor`]. The host is engine-agnostic: it
//! drives whatever [`EngineFactory`](daemon_activation::EngineFactory) is injected (the stub in
//! tests), keeping it free of `daemon-core` until phase 3.
//!
//! Deferred to phase 3 (and beyond): the §17 ⇄ management protocol translation, the real engine,
//! credentials, provisioning, telemetry, and remote transport.
//!
//! See `docs/specs/daemon-host-spec.md`.

#![forbid(unsafe_code)]

pub mod config;
pub mod services;
pub mod supervisor;

pub use config::HostConfig;
pub use supervisor::{
    Backoff, ChildSpec, HealthStatus, MeltdownPolicy, RestartPolicy, ServiceError, Supervisor,
    SupervisorHandle,
};

use daemon_activation::{ActivationManager, EngineFactory};
use daemon_store::SessionStore;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// An in-process host: the durable activation substrate plus its supervised resident-service tree.
#[derive(Clone)]
pub struct Host {
    store: Arc<dyn SessionStore>,
    manager: ActivationManager,
    config: HostConfig,
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
        }
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
                services::metrics_tick(self.store.clone(), self.manager.clone()),
            ))
            .start(cancel)
    }
}
