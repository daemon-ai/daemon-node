//! A small one-for-one supervisor for the host's fixed resident-service tree.
//!
//! Restart/backoff/meltdown policy is intentionally host-internal: it is **not** part of the
//! `daemon-supervision` management protocol (which supervises the logical `ManagedUnit` tree of
//! engines, phase 3). This supervises only long-lived infrastructure services
//! ([`daemon-host-spec.md`](../../../../docs/specs/daemon-host-spec.md) §5) — never a per-session
//! child. Catch-unwind-based restart relies on the workspace's mandatory `panic = "unwind"`.

use dashmap::DashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

/// The future a resident service runs until it returns (normally or with an error) or panics.
pub type ServiceFuture = Pin<Box<dyn Future<Output = Result<(), ServiceError>> + Send>>;

/// Factory that (re)creates a resident service's running future, given a cancellation token.
pub type ServiceFactory = Arc<dyn Fn(CancellationToken) -> ServiceFuture + Send + Sync>;

/// An error returned by a resident service loop.
#[derive(Debug, thiserror::Error)]
#[error("service error: {0}")]
pub struct ServiceError(pub String);

impl ServiceError {
    /// Construct a service error from anything displayable.
    pub fn new(msg: impl std::fmt::Display) -> Self {
        Self(msg.to_string())
    }
}

/// When a child should be restarted after it stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestartPolicy {
    /// Always restart, whether it exited normally or abnormally.
    Permanent,
    /// Restart only on abnormal termination (panic or `Err`).
    Transient,
    /// Never restart.
    Temporary,
}

/// Exponential, capped restart backoff.
#[derive(Clone, Copy, Debug)]
pub struct Backoff {
    /// Delay before the first restart.
    pub initial: Duration,
    /// Maximum delay between restarts.
    pub max: Duration,
    /// Multiplicative growth factor per consecutive restart.
    pub factor: f64,
}

impl Backoff {
    /// The delay before the `n`-th restart (0-indexed), capped at `max`.
    pub fn delay(&self, n: u32) -> Duration {
        let scaled = self.initial.as_secs_f64() * self.factor.powi(n as i32);
        let capped = scaled.min(self.max.as_secs_f64());
        Duration::from_secs_f64(capped)
    }
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            initial: Duration::from_millis(100),
            max: Duration::from_secs(5),
            factor: 2.0,
        }
    }
}

/// OTP-style meltdown: too many restarts within a sliding window stops the child.
#[derive(Clone, Copy, Debug)]
pub struct MeltdownPolicy {
    /// Maximum restarts permitted within `window` before meltdown.
    pub max_restarts: u32,
    /// The sliding window over which restarts are counted.
    pub window: Duration,
}

impl Default for MeltdownPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            window: Duration::from_secs(10),
        }
    }
}

/// Health of a supervised child.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HealthStatus {
    /// Running (possibly after restarts).
    Ok,
    /// Running but restarting more than usual.
    Degraded {
        /// Why it is degraded.
        reason: String,
    },
    /// Stopped — exceeded the meltdown threshold or terminated and not restartable.
    Unhealthy {
        /// Why it is unhealthy.
        reason: String,
    },
}

/// The specification of one supervised resident service.
#[derive(Clone)]
pub struct ChildSpec {
    /// Stable name (also the health-map key).
    pub name: String,
    /// Restart policy.
    pub policy: RestartPolicy,
    /// Backoff schedule.
    pub backoff: Backoff,
    /// Factory that (re)creates the service future.
    pub factory: ServiceFactory,
}

impl ChildSpec {
    /// A permanent service with default backoff.
    pub fn permanent(name: impl Into<String>, factory: ServiceFactory) -> Self {
        Self {
            name: name.into(),
            policy: RestartPolicy::Permanent,
            backoff: Backoff::default(),
            factory,
        }
    }

    /// Override the backoff schedule.
    pub fn with_backoff(mut self, backoff: Backoff) -> Self {
        self.backoff = backoff;
        self
    }

    /// Override the restart policy.
    pub fn with_policy(mut self, policy: RestartPolicy) -> Self {
        self.policy = policy;
        self
    }
}

#[derive(Debug)]
struct ChildState {
    /// Total restarts since start (observable for tests).
    restarts: u32,
    health: HealthStatus,
}

/// Shared, observable supervisor state.
struct Shared {
    children: DashMap<String, ChildState>,
}

/// A handle to a running supervisor: observe health/restarts and shut it down.
pub struct SupervisorHandle {
    shared: Arc<Shared>,
    cancel: CancellationToken,
    tasks: Vec<JoinHandle<()>>,
}

impl SupervisorHandle {
    /// The health of a named child, if it exists.
    pub fn health(&self, name: &str) -> Option<HealthStatus> {
        self.shared.children.get(name).map(|s| s.health.clone())
    }

    /// The number of times a named child has been restarted.
    pub fn restarts(&self, name: &str) -> Option<u32> {
        self.shared.children.get(name).map(|s| s.restarts)
    }

    /// Whether every child is currently `Ok`.
    pub fn all_ok(&self) -> bool {
        self.shared
            .children
            .iter()
            .all(|e| e.value().health == HealthStatus::Ok)
    }

    /// A cloneable, read-only observer of this supervisor's child health — the projection the node
    /// control surface (`ControlApi::health`) reads without taking ownership of (and the right to
    /// shut down) the handle.
    pub fn observer(&self) -> SupervisorObserver {
        SupervisorObserver {
            shared: self.shared.clone(),
        }
    }

    /// Trip the cancellation token and wait for every supervising task to drain.
    pub async fn shutdown(self) {
        self.cancel.cancel();
        for task in self.tasks {
            let _ = task.await;
        }
    }
}

/// A read-only view of a [`SupervisorHandle`]'s child health, cheap to clone and safe to hold
/// alongside the owning handle (the handle alone may `shutdown`).
#[derive(Clone)]
pub struct SupervisorObserver {
    shared: Arc<Shared>,
}

impl SupervisorObserver {
    /// The names of every supervised child.
    pub fn service_names(&self) -> Vec<String> {
        self.shared
            .children
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// The health of a named child, if it exists.
    pub fn health(&self, name: &str) -> Option<HealthStatus> {
        self.shared.children.get(name).map(|s| s.health.clone())
    }

    /// The number of times a named child has been restarted.
    pub fn restarts(&self, name: &str) -> Option<u32> {
        self.shared.children.get(name).map(|s| s.restarts)
    }

    /// Whether every child is currently `Ok`.
    pub fn all_ok(&self) -> bool {
        self.shared
            .children
            .iter()
            .all(|e| e.value().health == HealthStatus::Ok)
    }
}

/// The one-for-one supervisor over a fixed child set.
pub struct Supervisor {
    meltdown: MeltdownPolicy,
    children: Vec<ChildSpec>,
}

impl Supervisor {
    /// Construct a supervisor with the given meltdown policy.
    pub fn new(meltdown: MeltdownPolicy) -> Self {
        Self {
            meltdown,
            children: Vec::new(),
        }
    }

    /// Add a child specification.
    pub fn child(mut self, spec: ChildSpec) -> Self {
        self.children.push(spec);
        self
    }

    /// Start every child under its own supervising task, returning a handle. The supplied
    /// `cancel` token (typically the host's) stops all children when tripped.
    pub fn start(self, cancel: CancellationToken) -> SupervisorHandle {
        let shared = Arc::new(Shared {
            children: DashMap::new(),
        });
        let mut tasks = Vec::with_capacity(self.children.len());

        for spec in self.children {
            shared.children.insert(
                spec.name.clone(),
                ChildState {
                    restarts: 0,
                    health: HealthStatus::Ok,
                },
            );
            let shared = shared.clone();
            let cancel = cancel.clone();
            let meltdown = self.meltdown;
            tasks.push(tokio::spawn(async move {
                supervise_child(spec, shared, cancel, meltdown).await;
            }));
        }

        SupervisorHandle {
            shared,
            cancel,
            tasks,
        }
    }
}

/// The supervising loop for one child: (re)spawn the service future, observe its termination,
/// apply restart policy with backoff, and trip meltdown if it churns too fast.
async fn supervise_child(
    spec: ChildSpec,
    shared: Arc<Shared>,
    cancel: CancellationToken,
    meltdown: MeltdownPolicy,
) {
    let mut restart_times: Vec<Instant> = Vec::new();

    loop {
        if cancel.is_cancelled() {
            return;
        }

        // Run the service to termination, isolating panics (requires panic = "unwind").
        let fut = (spec.factory)(cancel.clone());
        let outcome = tokio::spawn(fut).await;

        let abnormal = match &outcome {
            Ok(Ok(())) => false,                  // clean return
            Ok(Err(_)) => true,                   // service returned Err
            Err(join_err) => join_err.is_panic(), // task panicked
        };

        if cancel.is_cancelled() {
            return;
        }

        let should_restart = match spec.policy {
            RestartPolicy::Permanent => true,
            RestartPolicy::Transient => abnormal,
            RestartPolicy::Temporary => false,
        };

        if !should_restart {
            // Terminal: mark unhealthy only if it died abnormally; otherwise it simply finished.
            if abnormal {
                set_health(
                    &shared,
                    &spec.name,
                    HealthStatus::Unhealthy {
                        reason: "terminated abnormally; not restartable".into(),
                    },
                );
            }
            return;
        }

        // Meltdown accounting over the sliding window.
        let now = Instant::now();
        restart_times.push(now);
        restart_times.retain(|t| now.duration_since(*t) <= meltdown.window);

        let restarts = bump_restart(&shared, &spec.name);

        if restart_times.len() as u32 > meltdown.max_restarts {
            set_health(
                &shared,
                &spec.name,
                HealthStatus::Unhealthy {
                    reason: format!(
                        "meltdown: {} restarts within {:?}",
                        restart_times.len(),
                        meltdown.window
                    ),
                },
            );
            return;
        }

        // Mild churn -> Degraded; otherwise stay Ok.
        if restart_times.len() > 1 {
            set_health(
                &shared,
                &spec.name,
                HealthStatus::Degraded {
                    reason: format!("{} recent restarts", restart_times.len()),
                },
            );
        }

        // Backoff before the next restart, but wake early on cancellation.
        let delay = spec.backoff.delay(restarts.saturating_sub(1));
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = tokio::time::sleep(delay) => {}
        }
    }
}

fn set_health(shared: &Arc<Shared>, name: &str, health: HealthStatus) {
    if let Some(mut s) = shared.children.get_mut(name) {
        s.health = health;
    }
}

fn bump_restart(shared: &Arc<Shared>, name: &str) -> u32 {
    if let Some(mut s) = shared.children.get_mut(name) {
        s.restarts += 1;
        s.restarts
    } else {
        0
    }
}
