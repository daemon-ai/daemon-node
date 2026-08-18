// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! [`AttachmentHub`] — live observation and control as an *attachment* to a durable activation
//! (session-unification §7).
//!
//! The live actor rail gives an interactive session four things the activation path historically
//! lacked: a non-destructive merged event log (`Subscribe`/`log_after`), a destructive poll drain,
//! parked blocking host requests answered by `Respond`, and a way for `Steer`/`Interrupt` to reach
//! a turn already in flight. This module packages exactly those capabilities as a host-owned,
//! per-session hub the activation incarnation *attaches to* while it runs — so a durable session
//! serves the same client surface as a live one, and the live actor stops being a second
//! authority (retired at the stage-5 cutover).
//!
//! Stage 4 builds this dark: [`CoreIncarnation`](crate::engine_incarnation::CoreIncarnation)
//! publishes into an attached hub and parks interactive requests on it, and the conformance suite
//! proves activation-vs-live parity — but the wire (`Poll`/`Subscribe`/`Respond`) keeps routing to
//! the live rail until the cutover.

use super::internals::{api_origin, engine_origin, Drain, LogEntryParts, MergedLog, Pending};
use super::NodeEventFeed;
use daemon_api::{ApiError, LogPageView, LogStream, NodeEvent};
use daemon_common::{ReqId, SessionId};
use daemon_core::{SteerReq, TurnControl};
use daemon_protocol::{
    AgentCommand, AgentEvent, Direction, Disposition, HostRequest, HostRequestKind, HostResponse,
    HostResponseBody, Outbound, SessionPayload,
};
use dashmap::DashMap;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use tokio::sync::{oneshot, watch};

/// The per-session attachment surface for a durable activation (session-unification §7): one
/// merged log + drain + parked-request table + an occupied-turn control slot. Everything a client
/// could do to a live session, homed on the durable lifecycle.
pub struct AttachmentHub {
    session: SessionId,
    /// The non-destructive merged timeline (`log_after`/`subscribe`); its `append` emits the
    /// `SessionAdvanced` node-event when a feed is wired — durable sessions advance the same way
    /// live ones do.
    log: Arc<Mutex<MergedLog>>,
    /// The destructive single-consumer `poll` queue (live-drain parity).
    drain: Drain,
    /// Parked blocking host requests awaiting a client `respond` (live-parking parity).
    pending: Pending,
    /// The occupied-turn slot: the resident incarnation registers its [`TurnControl`] for the
    /// duration of a turn, so a durable mid-turn `Steer`/`Interrupt` is delivered INTO the
    /// in-flight turn instead of waiting for the next one (§7's timing contract).
    control: Mutex<Option<TurnControl>>,
    /// The occupied-slot notifier: `true` while a turn is resident. The stage-5 wake path uses
    /// this to decide "deliver into the resident turn" vs "wake an activation".
    occupied_tx: watch::Sender<bool>,
    /// The node-wide event feed (`ApprovalPending` when an approval parks).
    feed: Option<Arc<NodeEventFeed>>,
}

impl AttachmentHub {
    fn new(session: SessionId, epoch: u64, feed: Option<Arc<NodeEventFeed>>) -> Self {
        let (occupied_tx, _) = watch::channel(false);
        Self {
            log: Arc::new(Mutex::new(MergedLog::new(
                session.clone(),
                epoch,
                feed.clone(),
            ))),
            drain: Arc::new(Mutex::new(VecDeque::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            control: Mutex::new(None),
            occupied_tx,
            feed,
            session,
        }
    }

    // -- publication (the incarnation side) ------------------------------------------------------

    /// Publish one engine event to every attached consumer: append it to the merged log (which
    /// emits `SessionAdvanced`) and push it onto the poll drain — the same fan-out the live pump
    /// performs, minus the journal (the activation path journals at the turn boundary itself).
    pub fn publish_event(&self, ev: AgentEvent) {
        self.log.lock().unwrap().append(
            Direction::Outbound,
            LogEntryParts {
                origin: engine_origin(),
                disposition: Disposition::Context,
                payload: SessionPayload::Event(ev.clone()),
            },
        );
        self.drain.lock().unwrap().push_back(Outbound::Event(ev));
    }

    /// Park a blocking host request until a client answers via [`Self::respond`] — the live
    /// `ParkingHandler` semantics on the activation path: the request lands on the merged log and
    /// the poll drain (one ordered timeline), an approval badges `ApprovalPending` on the node
    /// feed, and the caller (the engine's turn) awaits the response. A hub dropped before an
    /// answer declines safely, exactly as a torn-down live session does.
    pub async fn park(&self, req: HostRequest) -> HostResponse {
        let request_id = req.request_id;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(request_id, tx);
        if matches!(req.kind, HostRequestKind::Approval { .. }) {
            if let Some(feed) = &self.feed {
                feed.emit(NodeEvent::ApprovalPending {
                    session: self.session.clone(),
                    request_id: request_id.0.to_string(),
                });
            }
        }
        self.log.lock().unwrap().append(
            Direction::Outbound,
            LogEntryParts {
                origin: engine_origin(),
                disposition: Disposition::Context,
                payload: SessionPayload::Request(req.clone()),
            },
        );
        self.drain.lock().unwrap().push_back(Outbound::Request(req));
        match rx.await {
            Ok(resp) => resp,
            // The hub was dropped before an answer arrived: decline safely.
            Err(_) => HostResponse {
                request_id,
                body: HostResponseBody::Approved {
                    approved: false,
                    allow_permanent: false,
                    reason: None,
                },
            },
        }
    }

    /// Register the in-flight turn's [`TurnControl`] (cheaply cloned; shared state) and flip the
    /// occupied notifier. Pair with [`Self::end_turn`] on every exit path.
    pub fn begin_turn(&self, control: &TurnControl) {
        *self.control.lock().unwrap() = Some(control.clone());
        // send_replace: a plain `send` drops the value when no receiver is subscribed yet.
        self.occupied_tx.send_replace(true);
    }

    /// Clear the occupied slot at the turn boundary (the incarnation passivates or lingers idle).
    pub fn end_turn(&self) {
        *self.control.lock().unwrap() = None;
        self.occupied_tx.send_replace(false);
    }

    // -- control + response (the client side) ----------------------------------------------------

    /// Deliver a steer INTO the resident turn, if one occupies the slot: the engine folds it at
    /// its next phase boundary (§7's timing contract — steering never waits for the next turn).
    /// Returns `false` when no turn is resident (the caller then routes it as a durable splice).
    /// A claimed steer is recorded on the merged log as the inbound command it was.
    pub fn deliver_steer(&self, request_id: ReqId, text: String) -> bool {
        let claimed = match self.control.lock().unwrap().as_ref() {
            Some(control) => {
                control.push_steer(SteerReq {
                    request_id,
                    text: text.clone(),
                });
                true
            }
            None => false,
        };
        if claimed {
            self.log.lock().unwrap().append(
                Direction::Inbound,
                LogEntryParts {
                    origin: api_origin(),
                    disposition: Disposition::Context,
                    payload: SessionPayload::Command(AgentCommand::Steer { request_id, text }),
                },
            );
        }
        claimed
    }

    /// Request cooperative cancellation of the resident turn, if one occupies the slot.
    pub fn interrupt(&self) -> bool {
        match self.control.lock().unwrap().as_ref() {
            Some(control) => {
                control.cancel();
                true
            }
            None => false,
        }
    }

    /// Answer a parked request (the live `respond` semantics): the response is recorded inbound on
    /// the merged log under the unified seq, then delivered to the awaiting turn.
    pub fn respond(&self, response: HostResponse) -> Result<(), ApiError> {
        let tx = self.pending.lock().unwrap().remove(&response.request_id);
        match tx {
            Some(tx) => {
                self.log.lock().unwrap().append(
                    Direction::Inbound,
                    LogEntryParts {
                        origin: api_origin(),
                        disposition: Disposition::Context,
                        payload: SessionPayload::Response(response.clone()),
                    },
                );
                let _ = tx.send(response);
                Ok(())
            }
            None => Err(ApiError::Other(format!(
                "no parked request {} on session {}",
                response.request_id.0, self.session
            ))),
        }
    }

    // -- observation (the client side) -----------------------------------------------------------

    /// Whether a turn currently occupies the control slot.
    pub fn occupied(&self) -> bool {
        *self.occupied_tx.borrow()
    }

    /// A watch on the occupied slot (the stage-5 wake path's "deliver vs activate" signal).
    pub fn occupancy(&self) -> watch::Receiver<bool> {
        self.occupied_tx.subscribe()
    }

    /// Destructively drain up to `max` outbound frames (live `poll` parity; `0` = all).
    pub fn poll(&self, max: u32) -> Vec<Outbound> {
        let mut q = self.drain.lock().unwrap();
        let take = if max == 0 {
            q.len()
        } else {
            q.len().min(max as usize)
        };
        q.drain(..take).collect()
    }

    /// A non-destructive page of merged-log entries with `seq > after_seq` (live `log_after`
    /// parity).
    pub fn log_after(&self, after_seq: u64, max: u32) -> LogPageView {
        self.log.lock().unwrap().page(after_seq, max)
    }

    /// A push stream that backfills `seq > after_seq` then continues live (live `subscribe`
    /// parity).
    pub fn subscribe(&self, after_seq: u64) -> LogStream {
        self.log.lock().unwrap().subscribe(after_seq)
    }
}

/// The host-owned hub registry: one [`AttachmentHub`] per attached session. The activation
/// incarnation looks its session up here at turn time ([`Self::get`] — absent = nothing attached,
/// zero overhead); the client surface attaches on demand ([`Self::attach`]).
pub struct AttachmentHubs {
    map: DashMap<SessionId, Arc<AttachmentHub>>,
    /// The node-wide event feed every hub's merged log advances (`SessionAdvanced`) and parks
    /// badge (`ApprovalPending`) into; `None` => no feed wired (e.g. substrate-only tests).
    feed: Option<Arc<NodeEventFeed>>,
}

impl AttachmentHubs {
    /// A registry whose hubs emit onto `feed` (when wired).
    pub fn new(feed: Option<Arc<NodeEventFeed>>) -> Self {
        Self {
            map: DashMap::new(),
            feed,
        }
    }

    /// Get-or-create the hub for `session`. `epoch` stamps the merged log's activation generation
    /// (L2 resync) and is only used at creation; an existing hub keeps its generation.
    pub fn attach(&self, session: &SessionId, epoch: u64) -> Arc<AttachmentHub> {
        self.map
            .entry(session.clone())
            .or_insert_with(|| {
                Arc::new(AttachmentHub::new(
                    session.clone(),
                    epoch,
                    self.feed.clone(),
                ))
            })
            .clone()
    }

    /// The hub for `session`, if one is attached.
    pub fn get(&self, session: &SessionId) -> Option<Arc<AttachmentHub>> {
        self.map.get(session).map(|h| h.clone())
    }

    /// Drop `session`'s hub (subscribers see their streams end; parked requests decline safely
    /// via the oneshot drop).
    pub fn detach(&self, session: &SessionId) {
        self.map.remove(session);
    }
}
