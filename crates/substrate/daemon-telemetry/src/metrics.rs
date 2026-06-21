//! A lightweight, in-tree metrics aggregator + serializable dump.
//!
//! `UsageDelta` aggregates up the supervision tree by construction (supervision spec §4): an
//! orchestrator's usage is the fold of its children's. [`Metrics`] is that fold made resident —
//! it accumulates usage and an event count, tracks a health bit, and renders a point-in-time
//! [`Dump`] (folded usage + durable queue depths + health) for the resident `Metrics/health`
//! service. This is deliberately small; the external `metrics` facade can slot behind it later.

use daemon_common::UsageDelta;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Durable queue depths sampled from the store, folded into a [`Dump`] alongside usage/health.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct QueueDepths {
    /// Jobs waiting in the durable job outbox.
    pub pending_jobs: usize,
    /// Wakes waiting in the durable wake outbox.
    pub pending_wakes: usize,
    /// Sessions known to the store.
    pub sessions: usize,
    /// Sessions currently activated in this host.
    pub active: usize,
}

/// A point-in-time observability snapshot (folded usage + queue depths + health).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dump {
    /// Cumulative usage folded across every unit reporting to this aggregator.
    pub usage: UsageDelta,
    /// Number of management events folded so far.
    pub events: u64,
    /// Durable queue depths at sample time.
    pub depths: QueueDepths,
    /// Aggregate health bit.
    pub healthy: bool,
}

#[derive(Debug)]
struct Inner {
    usage: UsageDelta,
    events: u64,
    healthy: bool,
}

/// A cheaply-clonable handle to a shared usage/health aggregator.
#[derive(Clone, Debug)]
pub struct Metrics {
    inner: Arc<Mutex<Inner>>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// A fresh aggregator (zero usage, zero events, healthy).
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                usage: UsageDelta::default(),
                events: 0,
                healthy: true,
            })),
        }
    }

    /// Fold a usage increment into the running total (the tree aggregation, invariant #4).
    pub fn fold_usage(&self, delta: &UsageDelta) {
        let mut g = self.inner.lock().unwrap();
        g.usage.add(delta);
    }

    /// Count one management event observed.
    pub fn record_event(&self) {
        self.inner.lock().unwrap().events += 1;
    }

    /// Set the aggregate health bit.
    pub fn set_healthy(&self, healthy: bool) {
        self.inner.lock().unwrap().healthy = healthy;
    }

    /// The cumulative folded usage.
    pub fn usage(&self) -> UsageDelta {
        self.inner.lock().unwrap().usage
    }

    /// The number of events folded so far.
    pub fn events(&self) -> u64 {
        self.inner.lock().unwrap().events
    }

    /// Render a [`Dump`] joining the folded usage/health with the supplied durable queue depths.
    pub fn dump(&self, depths: QueueDepths) -> Dump {
        let g = self.inner.lock().unwrap();
        Dump {
            usage: g.usage,
            events: g.events,
            depths,
            healthy: g.healthy,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folds_usage_and_events() {
        let m = Metrics::new();
        m.fold_usage(&UsageDelta {
            input_tokens: 10,
            output_tokens: 5,
            api_calls: 1,
            ..Default::default()
        });
        m.fold_usage(&UsageDelta {
            input_tokens: 1,
            output_tokens: 2,
            api_calls: 1,
            ..Default::default()
        });
        m.record_event();
        m.record_event();

        let dump = m.dump(QueueDepths {
            pending_jobs: 3,
            pending_wakes: 0,
            sessions: 7,
            active: 2,
        });
        assert_eq!(dump.usage.input_tokens, 11);
        assert_eq!(dump.usage.output_tokens, 7);
        assert_eq!(dump.usage.api_calls, 2);
        assert_eq!(dump.events, 2);
        assert_eq!(dump.depths.sessions, 7);
        assert!(dump.healthy);
    }
}
