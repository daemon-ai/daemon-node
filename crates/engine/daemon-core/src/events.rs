// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The [`SessionLog`] — the single per-session sequencer that stamps the monotonic `seq` on every
//! item crossing the session boundary, in **both** directions.
//!
//! This generalises the old outbound-only event sink into the bidirectional session event log
//! (event-io spec §5.4). One `seq` counter orders the merged inbound/outbound stream, which is what
//! makes the §17 stream lossless-primary and resyncable: a lagging consumer reconciles against
//! durable state by `seq` (§17.1 item 1), and a second surface can observe a live conversation by
//! replaying the merged log from a cursor.
//!
//! Two delivery paths hang off the one sequencer:
//!
//! - `events_out` — the legacy outbound [`AgentEvent`] fan-out (the actor broadcast / a test
//!   collector / the discarding substrate path that replays from the store). Unchanged shape, so the
//!   coarse management projection and existing §17 consumers keep working untouched.
//! - `log_out` — the merged [`SessionLogEntry`] record that carries **both** directions with their
//!   `direction` / `origin` / `disposition` axes. The non-destructive cursored subscribe surface
//!   (the host's `LiveSessions`) is fed from here.
//!
//! `EventSink` is retained as a type alias so the many `EventSink::discarding()` test/tool sites and
//! the `daemon-core` re-export keep compiling.

use daemon_protocol::{
    AgentEvent, Disposition, Origin, OriginScope, SessionLogEntry, SessionPayload, TransportId,
};
use std::sync::atomic::{AtomicU64, Ordering};

/// The single per-session sequencer. Stamps a monotonic `seq` across both directions and forwards
/// each item to the outbound event fan-out and/or the merged-log record.
pub struct SessionLog {
    seq: AtomicU64,
    /// The session's own attribution for engine-emitted (outbound) entries.
    self_origin: Origin,
    events_out: Box<dyn Fn(AgentEvent) + Send + Sync>,
    log_out: Box<dyn Fn(SessionLogEntry) + Send + Sync>,
}

/// Back-compat alias: the sequencer *is* the bidirectional session log now.
pub type EventSink = SessionLog;

fn engine_origin() -> Origin {
    Origin {
        transport: TransportId::new("engine"),
        scope: OriginScope::Internal,
    }
}

impl SessionLog {
    /// A sequencer that forwards stamped outbound events to `out` and records nothing on the merged
    /// log (the common live/actor and test path; the merged log is opt-in via [`SessionLog::with_log`]).
    pub fn new(out: impl Fn(AgentEvent) + Send + Sync + 'static) -> Self {
        Self {
            seq: AtomicU64::new(0),
            self_origin: engine_origin(),
            events_out: Box::new(out),
            log_out: Box::new(|_| {}),
        }
    }

    /// A sequencer that feeds **both** the outbound event fan-out (`events_out`) and the merged
    /// bidirectional log (`log_out`), attributing engine-emitted entries to `self_origin`.
    pub fn with_log(
        self_origin: Origin,
        events_out: impl Fn(AgentEvent) + Send + Sync + 'static,
        log_out: impl Fn(SessionLogEntry) + Send + Sync + 'static,
    ) -> Self {
        Self {
            seq: AtomicU64::new(0),
            self_origin,
            events_out: Box::new(events_out),
            log_out: Box::new(log_out),
        }
    }

    /// A sequencer that drops every item (the substrate path replays from durable state instead).
    pub fn discarding() -> Self {
        Self::new(|_| {})
    }

    /// Stamp the next `seq`, build the outbound event, fan it out, record it on the merged log
    /// (`Outbound` / `Context`, attributed to the session origin), and return the assigned `seq`.
    pub fn emit(&self, make: impl FnOnce(u64) -> AgentEvent) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let event = make(seq);
        (self.events_out)(event.clone());
        (self.log_out)(SessionLogEntry {
            seq,
            direction: daemon_protocol::Direction::Outbound,
            origin: self.self_origin.clone(),
            disposition: Disposition::Context,
            payload: SessionPayload::Event(event),
        });
        seq
    }

    /// Record an **inbound** entry (world → session) on the merged log under the next `seq`. Used by
    /// the actor to make a surface message / steer / control command a first-class, observable,
    /// sequenced part of the same log. Returns the assigned `seq`.
    pub fn record_inbound(
        &self,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        (self.log_out)(SessionLogEntry {
            seq,
            direction: daemon_protocol::Direction::Inbound,
            origin,
            disposition,
            payload,
        });
        seq
    }

    /// Record a raw entry on the merged log under the next `seq` (escape hatch for callers that have
    /// already assembled a [`SessionPayload`] and chosen a direction/disposition). Returns the `seq`.
    pub fn record(
        &self,
        direction: daemon_protocol::Direction,
        origin: Origin,
        disposition: Disposition,
        payload: SessionPayload,
    ) -> u64 {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        (self.log_out)(SessionLogEntry {
            seq,
            direction,
            origin,
            disposition,
            payload,
        });
        seq
    }

    /// The number of `seq` values handed out so far (across both directions).
    pub fn emitted(&self) -> u64 {
        self.seq.load(Ordering::Relaxed)
    }
}
