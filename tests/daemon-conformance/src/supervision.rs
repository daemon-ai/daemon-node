// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Phase-2 gate: resident services restart/backoff/meltdown, and the running host drives the
//! full lifecycle under churn + injected service crashes (`daemon-host-spec.md` §5;
//! `daemon-workspace-layout.md` §7 phase-2 gate).

use super::harness::{seed, status};
use daemon_activation::ActivationManager;
use daemon_common::{PartitionId, SessionId};
use daemon_host::services::{interval_child, job_tick, scan_tick, wake_tick, TickFn};
use daemon_host::supervisor::ServiceFactory;
use daemon_host::CoreEngineFactory;
use daemon_host::{
    Backoff, ChildSpec, HealthStatus, Host, HostConfig, MeltdownPolicy, RestartPolicy,
    ServiceError, Supervisor,
};
use daemon_store::{InMemoryStore, SessionStatus, SessionStore};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

fn fast_backoff() -> Backoff {
    Backoff {
        initial: Duration::from_millis(1),
        max: Duration::from_millis(3),
        factor: 1.0,
    }
}

fn lenient_meltdown() -> MeltdownPolicy {
    MeltdownPolicy {
        max_restarts: 50,
        window: Duration::from_secs(30),
    }
}

/// A factory that returns immediately with the given result (no internal loop).
fn immediate(ok: bool) -> ServiceFactory {
    Arc::new(move |_cancel| {
        Box::pin(async move {
            if ok {
                Ok(())
            } else {
                Err(ServiceError::new("boom"))
            }
        })
    })
}

/// Poll until every session is `Completed`, or the timeout elapses.
async fn poll_until_completed(store: &InMemoryStore, ids: &[SessionId], timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        let mut all = true;
        for id in ids {
            if store.status(id).await != Some(SessionStatus::Completed) {
                all = false;
                break;
            }
        }
        if all {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Poll a predicate until true or timeout.
async fn wait_until(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if cond() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// A tick wrapper that panics on its first `budget` invocations (bounded chaos), then delegates.
fn bounded_panic_tick(inner: TickFn, budget: Arc<AtomicUsize>) -> TickFn {
    Arc::new(move || {
        let inner = inner.clone();
        let budget = budget.clone();
        Box::pin(async move {
            if budget.load(Ordering::SeqCst) > 0 {
                budget.fetch_sub(1, Ordering::SeqCst);
                panic!("injected dispatcher crash");
            }
            inner().await
        })
    })
}

/// Restart policy: `Permanent` restarts always, `Temporary` never, `Transient` only on abnormal
/// termination.
#[tokio::test]
async fn transient_vs_permanent() {
    let cancel = CancellationToken::new();
    let handle = Supervisor::new(lenient_meltdown())
        .child(ChildSpec::permanent("perm", immediate(true)).with_backoff(fast_backoff()))
        .child(
            ChildSpec::permanent("temp", immediate(true))
                .with_policy(RestartPolicy::Temporary)
                .with_backoff(fast_backoff()),
        )
        .child(
            ChildSpec::permanent("trans_ok", immediate(true))
                .with_policy(RestartPolicy::Transient)
                .with_backoff(fast_backoff()),
        )
        .child(
            ChildSpec::permanent("trans_err", immediate(false))
                .with_policy(RestartPolicy::Transient)
                .with_backoff(fast_backoff()),
        )
        .start(cancel.clone());

    // Give the permanent/transient-err children time to restart several times.
    assert!(
        wait_until(Duration::from_secs(2), || handle
            .restarts("perm")
            .unwrap_or(0)
            >= 2)
        .await,
        "permanent child should keep restarting after clean exits"
    );

    assert!(
        handle.restarts("temp").unwrap() == 0,
        "temporary must not restart"
    );
    assert!(
        handle.restarts("trans_ok").unwrap() == 0,
        "transient + clean exit must not restart"
    );
    assert!(
        handle.restarts("trans_err").unwrap() >= 2,
        "transient + error must restart"
    );

    handle.shutdown().await;
}

/// A child that panics a few times is restarted with backoff and then runs healthily.
#[tokio::test]
async fn restart_with_backoff() {
    let counter = Arc::new(AtomicUsize::new(0));
    let factory: ServiceFactory = {
        let counter = counter.clone();
        Arc::new(move |cancel| {
            let counter = counter.clone();
            Box::pin(async move {
                let n = counter.fetch_add(1, Ordering::SeqCst) + 1;
                if n <= 3 {
                    panic!("boom {n}");
                }
                // Recovered: run until shut down (so no further restarts occur).
                cancel.cancelled().await;
                Ok(())
            })
        })
    };

    let backoff = Backoff {
        initial: Duration::from_millis(5),
        max: Duration::from_millis(40),
        factor: 2.0,
    };
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let handle = Supervisor::new(lenient_meltdown())
        .child(ChildSpec::permanent("svc", factory).with_backoff(backoff))
        .start(cancel.clone());

    // Wait until the service recovers past its 3 panics (the 4th invocation).
    assert!(
        wait_until(Duration::from_secs(2), || counter.load(Ordering::SeqCst)
            >= 4)
        .await,
        "service never recovered past its panics"
    );
    let elapsed = started.elapsed();

    // Exactly three restarts, and backoff (5 + 10 + 20 ms) was applied between them.
    assert_eq!(handle.restarts("svc"), Some(3));
    assert!(
        elapsed >= Duration::from_millis(30),
        "backoff not applied: recovered in {elapsed:?}"
    );
    assert_ne!(
        handle.health("svc"),
        Some(HealthStatus::Unhealthy {
            reason: String::new()
        })
    );
    assert!(matches!(
        handle.health("svc"),
        Some(HealthStatus::Ok) | Some(HealthStatus::Degraded { .. })
    ));

    handle.shutdown().await;
}

/// A child that crashes faster than the meltdown threshold is stopped and marked `Unhealthy`.
#[tokio::test]
async fn meltdown() {
    let factory: ServiceFactory = Arc::new(|_cancel| Box::pin(async { panic!("always") }));
    let cancel = CancellationToken::new();
    let handle = Supervisor::new(MeltdownPolicy {
        max_restarts: 3,
        window: Duration::from_secs(60),
    })
    .child(ChildSpec::permanent("doomed", factory).with_backoff(fast_backoff()))
    .start(cancel.clone());

    assert!(
        wait_until(Duration::from_secs(2), || matches!(
            handle.health("doomed"),
            Some(HealthStatus::Unhealthy { .. })
        ))
        .await,
        "meltdown did not trip"
    );

    // It stopped restarting after meltdown (count stays put).
    let after = handle.restarts("doomed").unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        handle.restarts("doomed"),
        Some(after),
        "child kept restarting after meltdown"
    );

    handle.shutdown().await;
}

/// The running host drives seeded sessions through the full lifecycle with no manual pumping.
#[tokio::test]
async fn running_host_drives_lifecycle() {
    let store = Arc::new(InMemoryStore::new());
    let host = Host::new(
        store.clone(),
        Arc::new(CoreEngineFactory::delegating()),
        HostConfig::default(),
    );

    let ids: Vec<SessionId> = (0..25)
        .map(|i| SessionId::new(format!("host-{i}")))
        .collect();
    for id in &ids {
        seed(&store, id).await;
    }

    let handle = host.start();
    let completed = poll_until_completed(&store, &ids, Duration::from_secs(5)).await;
    assert!(completed, "running host failed to complete all sessions");

    handle.shutdown().await;
    assert_eq!(
        host.manager().active_count(),
        0,
        "active directory above baseline"
    );
}

/// THE PHASE-2 GATE: resident dispatchers crash repeatedly during churn, yet the supervisor
/// restarts them (no meltdown) and every session still drains to completion (durable queues
/// lose nothing), with the active directory back to baseline.
#[tokio::test]
async fn residents_survive_churn() {
    let store = Arc::new(InMemoryStore::new());
    let manager = ActivationManager::new(
        store.clone(),
        Arc::new(CoreEngineFactory::delegating()),
        PartitionId::DEFAULT,
    );

    let wake_budget = Arc::new(AtomicUsize::new(3));
    let job_budget = Arc::new(AtomicUsize::new(2));
    let wake = bounded_panic_tick(wake_tick(manager.clone()), wake_budget.clone());
    let job = bounded_panic_tick(job_tick(manager.clone()), job_budget.clone());

    let cancel = CancellationToken::new();
    let handle = Supervisor::new(MeltdownPolicy {
        max_restarts: 20,
        window: Duration::from_secs(30),
    })
    .child(interval_child(
        "WakeOutboxDispatcher",
        Duration::from_millis(1),
        RestartPolicy::Permanent,
        fast_backoff(),
        wake,
    ))
    .child(interval_child(
        "JobOutboxDispatcher",
        Duration::from_millis(1),
        RestartPolicy::Permanent,
        fast_backoff(),
        job,
    ))
    .child(interval_child(
        "RecoveryScanner",
        Duration::from_millis(5),
        RestartPolicy::Permanent,
        fast_backoff(),
        scan_tick(manager.clone()),
    ))
    .start(cancel.clone());

    let ids: Vec<SessionId> = (0..30)
        .map(|i| SessionId::new(format!("churn-{i}")))
        .collect();
    for id in &ids {
        seed(&store, id).await;
    }

    let completed = poll_until_completed(&store, &ids, Duration::from_secs(10)).await;
    assert!(
        completed,
        "sessions did not complete despite supervised restarts"
    );

    // The injected crashes were consumed and caused real restarts...
    assert_eq!(wake_budget.load(Ordering::SeqCst), 0);
    assert_eq!(job_budget.load(Ordering::SeqCst), 0);
    assert!(handle.restarts("WakeOutboxDispatcher").unwrap() >= 3);
    assert!(handle.restarts("JobOutboxDispatcher").unwrap() >= 2);

    // ...but no service melted down.
    for svc in [
        "WakeOutboxDispatcher",
        "JobOutboxDispatcher",
        "RecoveryScanner",
    ] {
        assert!(
            !matches!(handle.health(svc), Some(HealthStatus::Unhealthy { .. })),
            "{svc} melted down"
        );
    }

    handle.shutdown().await;
    assert_eq!(manager.active_count(), 0, "active directory above baseline");

    // ensure status() import is exercised (sanity on one session)
    assert_eq!(
        status(&store, &ids[0]).await,
        Some(SessionStatus::Completed)
    );
}
