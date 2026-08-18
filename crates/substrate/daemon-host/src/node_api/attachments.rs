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
use daemon_api::{ApiError, DeliverySink, LogPageView, LogStream, SessionLogEntry};
use daemon_common::{ReqId, SessionId};
use daemon_core::{SteerReq, TurnControl};
use daemon_protocol::{
    AgentCommand, AgentEvent, DeliveryTarget, Direction, Disposition, HostRequest, HostRequestKind,
    HostResponse, HostResponseBody, Outbound, SessionPayload, SinkKind, TransportId,
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
    /// The session's delivery roster (§8 sidecar re-homing): the reply targets a live residency
    /// used to keep on its actor entry, now homed here so a durable session's `handover` /
    /// `delivery_targets` / transport enumeration serve identically.
    delivery: Mutex<Vec<DeliveryTarget>>,
    /// The in-process push feed (daemon-event-io-spec §5.9.3, the live pump's sink-push parity):
    /// every outbound entry this hub publishes is queued here; the per-hub pump task resolves the
    /// roster's CURRENT `Primary` targets and delivers to any registered [`DeliverySink`] owning
    /// one — so handover demotion stops/starts push delivery exactly like the live rail.
    push_tx: tokio::sync::mpsc::UnboundedSender<SessionLogEntry>,
}

impl AttachmentHub {
    fn new(
        session: SessionId,
        epoch: u64,
        feed: Option<Arc<NodeEventFeed>>,
        push_tx: tokio::sync::mpsc::UnboundedSender<SessionLogEntry>,
    ) -> Self {
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
            delivery: Mutex::new(Vec::new()),
            push_tx,
        }
    }

    // -- publication (the incarnation side) ------------------------------------------------------

    /// Publish one engine event to every attached consumer: append it to the merged log (which
    /// emits `SessionAdvanced`) and push it onto the poll drain — the same fan-out the live pump
    /// performs, minus the journal (the activation path journals at the turn boundary itself).
    pub fn publish_event(&self, ev: AgentEvent) {
        let entry = self.log.lock().unwrap().append(
            Direction::Outbound,
            LogEntryParts {
                origin: engine_origin(),
                disposition: Disposition::Context,
                payload: SessionPayload::Event(ev.clone()),
            },
        );
        // In-process sink push (§5.9.3): the pump task delivers to the roster's Primary sinks.
        let _ = self.push_tx.send(entry);
        self.drain.lock().unwrap().push_back(Outbound::Event(ev));
    }

    /// Park a blocking host request until a client answers via [`Self::respond`] — the live
    /// `ParkingHandler` semantics on the activation path: the request lands on the merged log and
    /// the poll drain (one ordered timeline), an approval badges an Approvals-domain
    /// `ProjectionChanged` on the node feed, and the caller (the engine's turn) awaits the
    /// response. A hub dropped before an answer declines safely, exactly as a torn-down live
    /// session does.
    pub async fn park(&self, req: HostRequest) -> HostResponse {
        let request_id = req.request_id;
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(request_id, tx);
        if matches!(req.kind, HostRequestKind::Approval { .. }) {
            if let Some(feed) = &self.feed {
                feed.note_domain_change(
                    daemon_api::ProjectionId::Approvals,
                    None,
                    daemon_api::ChangeScope::Key {
                        key: self.session.as_str().to_string(),
                    },
                    None,
                );
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

    /// Record an inbound client command on the merged log (the live `record_inbound` parity):
    /// the user's `StartTurn`/`Observe`/… shares the same seq timeline as the outbound events the
    /// turn it opens will stream, so a subscriber replays one ordered conversation.
    pub fn record_inbound(&self, command: AgentCommand) {
        self.log.lock().unwrap().append(
            Direction::Inbound,
            LogEntryParts {
                origin: api_origin(),
                disposition: Disposition::Context,
                payload: SessionPayload::Command(command),
            },
        );
    }

    /// Record a transport-scoped meta entry on the merged log (the live `record_meta` parity):
    /// observable on `log_after`/`subscribe`, never entering the prompt or the journal.
    pub fn record_meta(&self, origin: daemon_protocol::Origin, kind: String, body: Vec<u8>) {
        self.log.lock().unwrap().append(
            Direction::Inbound,
            LogEntryParts {
                origin,
                disposition: Disposition::Transport,
                payload: SessionPayload::Meta { kind, body },
            },
        );
    }

    /// Ask the RESIDENT turn for a read-only snapshot (served at its next phase boundary as an
    /// [`AgentEvent::Snapshot`] on this hub's stream). `false` when no turn occupies the slot —
    /// the caller serves the durable snapshot itself via [`Self::publish_snapshot`].
    pub fn request_snapshot(&self, request_id: ReqId) -> bool {
        match self.control.lock().unwrap().as_ref() {
            Some(control) => {
                control.push_snapshot(request_id);
                true
            }
            None => false,
        }
    }

    /// Serve an idle-path snapshot reply: the host projects the durable snapshot to a `ConvView`
    /// and this publishes it as the §17 [`AgentEvent::Snapshot`] on the drain + merged log —
    /// identical delivery shape to a resident turn's boundary reply.
    pub fn publish_snapshot(&self, request_id: ReqId, view: daemon_protocol::ConvView) {
        self.publish_event(AgentEvent::Snapshot {
            seq: 0,
            request_id,
            view,
        });
    }

    /// The merged log's activation generation (L2 resync; the live `log_epoch` parity).
    pub fn log_epoch(&self) -> u64 {
        self.log.lock().unwrap().epoch
    }

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

    // -- delivery roster (§8 sidecar re-homing) --------------------------------------------------

    /// Promote `target` to the session's `Primary` reply sink, demoting any current primary to
    /// `Spectator` (the live `handover` semantics, homed on the hub).
    pub fn handover(&self, target: DeliveryTarget) {
        let mut targets = self.delivery.lock().unwrap();
        for t in targets.iter_mut() {
            if t.kind == SinkKind::Primary {
                t.kind = SinkKind::Spectator;
            }
        }
        targets.retain(|t| !(t.transport == target.transport && t.route == target.route));
        targets.push(target);
    }

    /// Seed the `Primary` reply target if none exists yet (the routed-submit seeding semantics).
    pub fn seed_primary_target(&self, target: DeliveryTarget) {
        let mut targets = self.delivery.lock().unwrap();
        if !targets.iter().any(|t| t.kind == SinkKind::Primary) {
            targets.push(target);
        }
    }

    /// The session's current delivery targets.
    pub fn delivery_targets(&self) -> Vec<DeliveryTarget> {
        self.delivery.lock().unwrap().clone()
    }

    /// Whether this session's roster carries a `Primary` target on `transport`.
    pub fn primary_on(&self, transport: &TransportId) -> bool {
        self.delivery
            .lock()
            .unwrap()
            .iter()
            .any(|t| t.kind == SinkKind::Primary && &t.transport == transport)
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
    /// In-process outbound push sinks keyed by transport instance (daemon-event-io-spec §5.9.3),
    /// the durable half of the registry `LiveSessions` holds for residencies: each hub's pump
    /// resolves its roster's `Primary` targets per event and pushes to the sink owning each.
    sinks: Arc<DashMap<TransportId, Arc<dyn DeliverySink>>>,
}

impl AttachmentHubs {
    /// A registry whose hubs emit onto `feed` (when wired).
    pub fn new(feed: Option<Arc<NodeEventFeed>>) -> Self {
        Self {
            map: DashMap::new(),
            feed,
            sinks: Arc::new(DashMap::new()),
        }
    }

    /// Register (or replace) the in-process push [`DeliverySink`] for `transport` — the durable
    /// half of [`DeliveryHost`](super::delivery::DeliveryHost) registration.
    pub fn register_delivery_sink(&self, transport: TransportId, sink: Arc<dyn DeliverySink>) {
        self.sinks.insert(transport, sink);
    }

    /// Drop the in-process push sink for `transport` (durable delivery reverts to pull).
    pub fn unregister_delivery_sink(&self, transport: &TransportId) {
        self.sinks.remove(transport);
    }

    /// Get-or-create the hub for `session`. `epoch` stamps the merged log's activation generation
    /// (L2 resync) and is only used at creation; an existing hub keeps its generation. Creation
    /// spawns the hub's push pump (sink delivery); it ends when the hub is detached/dropped.
    pub fn attach(&self, session: &SessionId, epoch: u64) -> Arc<AttachmentHub> {
        self.map
            .entry(session.clone())
            .or_insert_with(|| {
                let (push_tx, push_rx) = tokio::sync::mpsc::unbounded_channel();
                let hub = Arc::new(AttachmentHub::new(
                    session.clone(),
                    epoch,
                    self.feed.clone(),
                    push_tx,
                ));
                spawn_push_pump(&hub, push_rx, self.sinks.clone());
                hub
            })
            .clone()
    }

    /// The hub for `session`, if one is attached.
    pub fn get(&self, session: &SessionId) -> Option<Arc<AttachmentHub>> {
        self.map.get(session).map(|h| h.clone())
    }

    /// Every currently-attached session id — the durable analogue of the live registry's
    /// residency enumeration (a hub means a client opened/observes the session).
    pub fn session_ids(&self) -> Vec<SessionId> {
        self.map.iter().map(|e| e.key().clone()).collect()
    }

    /// Drop `session`'s hub (subscribers see their streams end; parked requests decline safely
    /// via the oneshot drop).
    pub fn detach(&self, session: &SessionId) {
        self.map.remove(session);
    }

    /// Every attached session whose roster carries a `Primary` target on `transport` (§8 sidecar
    /// re-homing; the durable half of the transport's delivery enumeration).
    pub fn delivery_sessions(&self, transport: &TransportId) -> Vec<SessionId> {
        self.map
            .iter()
            .filter(|e| e.value().primary_on(transport))
            .map(|e| e.key().clone())
            .collect()
    }

    /// Push a synthesized outbound `entry` to the registered sink owning `target`'s transport
    /// (post-settle cron delivery on the durable rail). A no-op when no sink is registered.
    pub async fn push_to_target(&self, target: DeliveryTarget, entry: SessionLogEntry) {
        if let Some(sink) = self.sinks.get(&target.transport).map(|s| s.clone()) {
            sink.deliver(target, entry).await;
        }
    }

    /// Every distinct `Primary` target across attached sessions, deduplicated by
    /// `(transport, route)` — the durable half of a cron `deliver = "all"` broadcast.
    pub fn all_primary_targets(&self) -> Vec<DeliveryTarget> {
        let mut out: Vec<DeliveryTarget> = Vec::new();
        for e in self.map.iter() {
            for t in e.value().delivery.lock().unwrap().iter() {
                if t.kind == SinkKind::Primary
                    && !out
                        .iter()
                        .any(|o| o.transport == t.transport && o.route == t.route)
                {
                    out.push(t.clone());
                }
            }
        }
        out
    }
}

/// The per-hub in-process push pump (the live per-session pump's sink half, §5.9.3): for each
/// outbound entry the hub publishes, re-read the roster's CURRENT `Primary` targets and deliver
/// the entry to any registered sink owning one — so a `handover` demotion silently stops one sink
/// and starts the next. Holds the hub weakly (the pump must not keep a detached hub alive); it
/// ends when the hub is dropped (the sender closes) or the registry entry is gone.
fn spawn_push_pump(
    hub: &Arc<AttachmentHub>,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<SessionLogEntry>,
    sinks: Arc<DashMap<TransportId, Arc<dyn DeliverySink>>>,
) {
    let weak = Arc::downgrade(hub);
    tokio::spawn(async move {
        while let Some(entry) = rx.recv().await {
            let Some(hub) = weak.upgrade() else { break };
            let primaries: Vec<DeliveryTarget> = hub
                .delivery
                .lock()
                .unwrap()
                .iter()
                .filter(|t| t.kind == SinkKind::Primary)
                .cloned()
                .collect();
            drop(hub);
            for target in primaries {
                let sink = sinks.get(&target.transport).map(|s| s.clone());
                if let Some(sink) = sink {
                    sink.deliver(target, entry.clone()).await;
                }
            }
        }
    });
}
