// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The host's resident services as supervised interval loops.
//!
//! Every inter-service handoff travels through the durable store queues (`wake_outbox`,
//! `job_outbox`, `completion_inbox`) — never in-memory channels — so a restarted service simply
//! resumes its idempotent loop having lost nothing (host-spec §5; the "at-least-once delivery,
//! idempotent consumer" rule).
//!
//! Service-to-spec mapping (host-spec §5): `WakeOutboxDispatcher`, `JobOutboxDispatcher`,
//! `RecoveryScanner`, and `Metrics/health` are the looping services here. `PartitionLeaseManager`
//! is realized by `SessionStore::acquire_activation_lease` (inside `ActivationManager::wake`),
//! `SessionActivator` by `ActivationManager::activate`, and `CompletionConsumer` by
//! `record_completion_and_wake` (called from the job dispatcher's worker step).

use crate::supervisor::{Backoff, ChildSpec, RestartPolicy, ServiceError, ServiceFuture};
use daemon_activation::ActivationManager;
use daemon_store::SessionStore;
use daemon_telemetry::metrics::QueueDepths;
use daemon_telemetry::{current_trace, Metrics};
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// One unit of resident-service work, run once per interval tick.
pub type TickFn =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), ServiceError>> + Send>> + Send + Sync>;

/// Build a permanent, interval-driven supervised child from a per-tick async action.
pub fn interval_child(
    name: impl Into<String>,
    interval: Duration,
    policy: RestartPolicy,
    backoff: Backoff,
    tick: TickFn,
) -> ChildSpec {
    let factory = Arc::new(move |cancel: CancellationToken| -> ServiceFuture {
        let tick = tick.clone();
        Box::pin(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => return Ok(()),
                    _ = ticker.tick() => { tick().await?; }
                }
            }
        })
    });
    ChildSpec {
        name: name.into(),
        policy,
        backoff,
        factory,
    }
}

/// Wrap a tick so it panics on every `panic_every`-th invocation (test chaos injection). A
/// `panic_every` of 0 disables panics. The `counter` persists across restarts so chaos keeps firing.
pub fn chaos_tick(inner: TickFn, panic_every: usize, counter: Arc<AtomicUsize>) -> TickFn {
    Arc::new(move || {
        let inner = inner.clone();
        let counter = counter.clone();
        Box::pin(async move {
            let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
            if panic_every > 0 && n.is_multiple_of(panic_every) {
                panic!("chaos panic on tick {n}");
            }
            inner().await
        })
    })
}

/// `WakeOutboxDispatcher` tick: drain the wake outbox, activating each hinted session.
pub fn wake_tick(manager: ActivationManager) -> TickFn {
    Arc::new(move || {
        let manager = manager.clone();
        Box::pin(async move {
            manager
                .dispatch_wakes()
                .await
                .map(|_| ())
                .map_err(ServiceError::new)
        })
    })
}

/// `JobOutboxDispatcher` tick: drain the job outbox, run workers, record completions + wakes.
pub fn job_tick(manager: ActivationManager) -> TickFn {
    Arc::new(move || {
        let manager = manager.clone();
        Box::pin(async move {
            manager
                .run_workers()
                .await
                .map(|_| ())
                .map_err(ServiceError::new)
        })
    })
}

/// `JobOutboxDispatcher` tick over an injected [`JobWorker`] — the seam the binary uses to drive
/// the job outbox with the real orchestration `FleetRuntime` (spawn + run a child per delegation
/// job) instead of the substrate's placeholder echo worker. Keeps `daemon-host` free of any
/// dependency on `daemon-orchestration`.
pub fn job_worker_tick(worker: Arc<dyn crate::JobWorker>) -> TickFn {
    Arc::new(move || {
        let worker = worker.clone();
        Box::pin(async move { worker.process_jobs_once().await })
    })
}

/// `CronScheduler` tick (I15): fire every due scheduled job once over an injected
/// [`CronScheduler`](crate::CronScheduler) — the seam the binary uses to drive cron with the node's
/// `CronWorker` (materialize an isolated session + enqueue its wake) without `daemon-host` depending
/// on `daemon-node`/`daemon-schedule`.
pub fn schedule_tick(scheduler: Arc<dyn crate::CronScheduler>) -> TickFn {
    Arc::new(move || {
        let scheduler = scheduler.clone();
        Box::pin(async move { scheduler.tick_once().await })
    })
}

/// `RecoveryScanner` tick: re-activate any resumable session whose wake was lost.
pub fn scan_tick(manager: ActivationManager) -> TickFn {
    Arc::new(move || {
        let manager = manager.clone();
        Box::pin(async move {
            manager
                .scan_once()
                .await
                .map(|_| ())
                .map_err(ServiceError::new)
        })
    })
}

/// `Metrics/health` tick: sample durable queue depths + active sessions and render a [`Dump`]
/// (durable depths joined with the folded `Usage` aggregated from units, plus the health bit). The
/// dump is stamped with the current trace context so it correlates with the work it summarizes.
///
/// [`Dump`]: daemon_telemetry::Dump
pub fn metrics_tick(
    store: Arc<dyn SessionStore>,
    active: ActivationManager,
    metrics: Metrics,
) -> TickFn {
    Arc::new(move || {
        let store = store.clone();
        let active = active.clone();
        let metrics = metrics.clone();
        Box::pin(async move {
            let stats = store.stats().await;
            let depths = QueueDepths {
                pending_jobs: stats.pending_jobs,
                pending_wakes: stats.pending_wakes,
                sessions: stats.sessions,
                active: active.active_count(),
            };
            // Resident host is healthy as long as this loop runs (the supervisor restarts it
            // otherwise); richer health derives from folded unit health in later phases.
            metrics.set_healthy(true);
            let dump = metrics.dump(depths);
            tracing::debug!(
                trace_id = %current_trace(),
                pending_jobs = dump.depths.pending_jobs,
                pending_wakes = dump.depths.pending_wakes,
                sessions = dump.depths.sessions,
                active = dump.depths.active,
                usage_input = dump.usage.input_tokens,
                usage_output = dump.usage.output_tokens,
                usage_api_calls = dump.usage.api_calls,
                events = dump.events,
                healthy = dump.healthy,
                "resident metrics dump"
            );
            Ok(())
        })
    })
}
