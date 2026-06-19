//! The [`EventSink`] — the single point that stamps the monotonic `seq` on every `AgentEvent`.
//!
//! Centralizing `seq` assignment is what makes the §17 stream lossless-primary and resyncable: a
//! lagging consumer reconciles against durable state by `seq` (§17.1 item 1). The sink is generic
//! over its delivery closure, so the same engine drive feeds both a broadcast fan-out (the actor)
//! and a discarding sink (the durable substrate adapter, which replays from the store instead).

use daemon_protocol::AgentEvent;
use std::sync::atomic::{AtomicU64, Ordering};

/// Stamps a monotonic `seq` on each event and forwards it to a delivery closure.
pub struct EventSink {
    seq: AtomicU64,
    out: Box<dyn Fn(AgentEvent) + Send + Sync>,
}

impl EventSink {
    /// A sink that forwards stamped events to `out`.
    pub fn new(out: impl Fn(AgentEvent) + Send + Sync + 'static) -> Self {
        Self {
            seq: AtomicU64::new(0),
            out: Box::new(out),
        }
    }

    /// A sink that drops every event (the substrate path replays from durable state instead).
    pub fn discarding() -> Self {
        Self::new(|_| {})
    }

    /// Stamp the next `seq`, build the event, and deliver it. Returns the assigned `seq`.
    pub fn emit(&self, make: impl FnOnce(u64) -> AgentEvent) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        (self.out)(make(seq));
        seq
    }

    /// The number of events emitted so far.
    pub fn emitted(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
}
