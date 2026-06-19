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
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// One unit of resident-service work, run once per interval tick.
pub type TickFn = Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), ServiceError>> + Send>> + Send + Sync>;

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
            if panic_every > 0 && n % panic_every == 0 {
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

/// `Metrics/health` tick: sample durable queue depths + active sessions (observability only).
pub fn metrics_tick(store: Arc<dyn SessionStore>, active: ActivationManager) -> TickFn {
    Arc::new(move || {
        let store = store.clone();
        let active = active.clone();
        Box::pin(async move {
            let stats = store.stats().await;
            tracing::debug!(
                pending_jobs = stats.pending_jobs,
                pending_wakes = stats.pending_wakes,
                sessions = stats.sessions,
                active = active.active_count(),
                "resident metrics"
            );
            Ok(())
        })
    })
}
