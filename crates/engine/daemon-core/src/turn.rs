//! Turn context and effects (§4.2 / §4.3).
//!
//! A turn is modelled as a near-pure function over the conversation that produces a stream of
//! [`Effect`]s; the single-owner applier (in [`crate::engine`]) orders and applies them. The
//! [`TurnCx`] carries the ambient handles a phase/tool needs — cooperative cancellation, the event
//! sink, and the host request channel for blocking human-in-the-loop / delegation requests (§17).

use crate::conversation::Turn;
use crate::events::EventSink;
use daemon_common::{Budget, JobId, SessionId};
use daemon_protocol::HostRequestHandler;
use tokio_util::sync::CancellationToken;

/// The ambient context handed to phases and tools during a turn (§4.2).
pub struct TurnCx<'a> {
    /// Cooperative cancellation, checked at phase boundaries and in streams.
    pub cancel: CancellationToken,
    /// The event sink to stream progress without owning the host.
    pub events: &'a EventSink,
    /// The host request channel for blocking requests (§17 human-in-the-loop / delegation).
    pub host: &'a dyn HostRequestHandler,
    /// The session this turn belongs to.
    pub session_id: SessionId,
    /// The budget governing this turn's work.
    pub budget: Budget,
}

/// An effect a turn phase or tool produces; the single-owner applier orders and applies them
/// (§4.3). Phase 3 carries the subset needed to drive durable suspension; `Checkpoint`/`MemoryWrite`
/// and payload externalization arrive with the later engine slices.
pub enum Effect {
    /// Append a turn to the conversation (durable record).
    Persist(Turn),
    /// The engine delegated background work and now waits on `JobId` — drives suspension.
    Delegate(JobId),
}
