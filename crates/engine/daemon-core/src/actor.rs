// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The agent session as an actor (§17 runtime handle).
//!
//! [`spawn_agent_session`] owns an [`Engine`] on a dedicated task and serves §17 commands over an
//! mpsc inbox while fanning [`AgentEvent`]s out over a broadcast. This is the live, in-process face
//! the host's `EngineUnit` wraps to present the engine as a `ManagedUnit`. The durable substrate
//! path does *not* use this actor — it drives the [`Engine`] directly through the activation seam.
//!
//! The loop is non-blocking: a running turn is driven as a future and `select!`-ed against the
//! inbox, so `Interrupt`/`Steer`/`Snapshot` are serviced *while a turn is in flight*. Those control
//! commands never touch `&mut engine` (which the turn holds) — they mutate a shared [`TurnControl`]
//! the turn observes at phase boundaries (the spec's at-boundary steering model). When idle:
//! `Snapshot` is answered immediately from the current snapshot, `Steer` opens a fresh
//! [`TurnTrigger::Steer`] turn, and a second `StartTurn` while busy is queued.

use crate::control::{SteerReq, TurnControl};
use crate::engine::{Engine, RewindError, RewindOutcome, TurnOutcome};
use crate::events::SessionLog;
use crate::provider::Provider;
use crate::Failure;
use daemon_common::ReqId;
use daemon_protocol::{
    AgentCommand, AgentEvent, Disposition, EndReason, HostRequestHandler, Origin, OriginScope,
    RewindAnchor, SessionLogEntry, SessionPayload, TransportId, TurnSummary, UserMsg,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};

/// The origin stamped on inbound commands submitted through the untagged convenience methods
/// ([`AgentHandle::start_turn`] etc.). Surface-aware callers use the `*_from` variants to attribute
/// the inbound item to the real transport.
fn local_origin() -> Origin {
    Origin {
        transport: TransportId::new("local"),
        scope: OriginScope::Internal,
        sender: None,
    }
}

/// Internal actor mailbox messages (the §17 commands plus their reply channels). Each carries the
/// [`Origin`] of the inbound submission so the actor can record it on the merged session log.
enum ActorMsg {
    StartTurn {
        input: UserMsg,
        request_id: ReqId,
        origin: Origin,
        reply: oneshot::Sender<Result<TurnSummary, Failure>>,
    },
    Steer {
        request_id: ReqId,
        text: String,
        origin: Origin,
    },
    /// Context-only append (no turn): folds into the conversation when idle, or queues on
    /// [`TurnControl`] to land in the following turn when busy (event-io §5.9 multi-party context).
    Observe {
        request_id: ReqId,
        input: UserMsg,
        origin: Origin,
    },
    Snapshot {
        request_id: ReqId,
        origin: Origin,
    },
    Interrupt {
        #[allow(dead_code)]
        reason: Option<String>,
        origin: Origin,
    },
    /// Rewind the conversation to `anchor` (conversation-rewind spec): interrupt any live turn, then
    /// truncate + reconstruct + bump epoch + emit `Rewound`. The reply carries the resolved outcome
    /// (or the rewind error) so the caller can drive the durable journal seal.
    RewindTo {
        anchor: RewindAnchor,
        request_id: ReqId,
        origin: Origin,
        reply: oneshot::Sender<Result<RewindOutcome, RewindError>>,
    },
    /// Live per-session model switch: swap the engine's provider. Applied at a turn boundary so an
    /// in-flight turn's prompt cache is never invalidated mid-conversation.
    SetProvider {
        provider: Arc<dyn Provider>,
        origin: Origin,
    },
    /// Live per-session edit-approval mode switch: set the engine's [`ApprovalPolicy`]. Applied at a
    /// turn boundary; consulted by the next gated tool action.
    SetApprovalPolicy {
        policy: crate::approval::ApprovalPolicy,
        origin: Origin,
    },
    Shutdown {
        origin: Origin,
    },
}

impl ActorMsg {
    fn origin(&self) -> &Origin {
        match self {
            ActorMsg::StartTurn { origin, .. }
            | ActorMsg::Steer { origin, .. }
            | ActorMsg::Observe { origin, .. }
            | ActorMsg::Snapshot { origin, .. }
            | ActorMsg::Interrupt { origin, .. }
            | ActorMsg::RewindTo { origin, .. }
            | ActorMsg::SetProvider { origin, .. }
            | ActorMsg::SetApprovalPolicy { origin, .. }
            | ActorMsg::Shutdown { origin } => origin,
        }
    }

    /// The merged-log payload + disposition for this inbound command. `StartTurn`/`Steer` enter the
    /// conversation (`Context`); the read-only/control commands are observability-only (`Transport`).
    fn as_inbound(&self) -> (SessionPayload, Disposition) {
        match self {
            ActorMsg::StartTurn {
                input, request_id, ..
            } => (
                SessionPayload::Command(AgentCommand::StartTurn {
                    input: input.clone(),
                    request_id: *request_id,
                }),
                Disposition::Context,
            ),
            ActorMsg::Steer {
                request_id, text, ..
            } => (
                SessionPayload::Command(AgentCommand::Steer {
                    text: text.clone(),
                    request_id: *request_id,
                }),
                Disposition::Context,
            ),
            ActorMsg::Observe {
                request_id, input, ..
            } => (
                SessionPayload::Command(AgentCommand::Observe {
                    input: input.clone(),
                    request_id: *request_id,
                }),
                // Context-bearing (enters the conversation), exactly like StartTurn/Steer.
                Disposition::Context,
            ),
            ActorMsg::Snapshot { request_id, .. } => (
                SessionPayload::Command(AgentCommand::Snapshot {
                    request_id: *request_id,
                }),
                Disposition::Transport,
            ),
            ActorMsg::Interrupt { reason, .. } => (
                SessionPayload::Command(AgentCommand::Interrupt {
                    reason: reason.clone(),
                }),
                Disposition::Transport,
            ),
            ActorMsg::RewindTo {
                anchor, request_id, ..
            } => (
                SessionPayload::Command(AgentCommand::RewindTo {
                    anchor: anchor.clone(),
                    request_id: *request_id,
                }),
                Disposition::Transport,
            ),
            // Observability-only: a model switch is not a wire command. Surface it as a meta marker.
            ActorMsg::SetProvider { .. } => (
                SessionPayload::Meta {
                    kind: "model.set".to_string(),
                    body: Vec::new(),
                },
                Disposition::Transport,
            ),
            // Observability-only: an edit-approval mode switch is not a wire command.
            ActorMsg::SetApprovalPolicy { policy, .. } => (
                SessionPayload::Meta {
                    kind: "mode.set".to_string(),
                    body: format!("{policy:?}").into_bytes(),
                },
                Disposition::Transport,
            ),
            ActorMsg::Shutdown { .. } => (
                SessionPayload::Command(AgentCommand::Shutdown),
                Disposition::Transport,
            ),
        }
    }
}

/// Record an inbound command on the merged session log under the next `seq`.
fn record_inbound(sink: &SessionLog, msg: &ActorMsg) {
    let (payload, disposition) = msg.as_inbound();
    sink.record_inbound(msg.origin().clone(), disposition, payload);
}

/// A handle to a running engine session: send §17 commands, subscribe to the §17 event stream or to
/// the merged bidirectional session log.
#[derive(Clone)]
pub struct AgentHandle {
    tx: mpsc::Sender<ActorMsg>,
    events: broadcast::Sender<AgentEvent>,
    log: broadcast::Sender<SessionLogEntry>,
    req_seq: Arc<AtomicU64>,
}

impl AgentHandle {
    fn next_req(&self) -> ReqId {
        ReqId(self.req_seq.fetch_add(1, Ordering::Relaxed))
    }

    /// Begin a turn from a user input attributed to `origin`, awaiting the terminal [`TurnSummary`].
    pub async fn start_turn_from(
        &self,
        origin: Origin,
        input: UserMsg,
    ) -> Result<TurnSummary, Failure> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(ActorMsg::StartTurn {
                input,
                request_id: self.next_req(),
                origin,
                reply,
            })
            .await
            .map_err(|_| Failure::Other("engine actor is gone".into()))?;
        rx.await
            .map_err(|_| Failure::Other("engine actor dropped the reply".into()))?
    }

    /// Begin a turn from a user input (untagged local origin), awaiting the terminal [`TurnSummary`].
    pub async fn start_turn(&self, input: UserMsg) -> Result<TurnSummary, Failure> {
        self.start_turn_from(local_origin(), input).await
    }

    /// Interrupt the in-flight turn (cooperative cancellation, honored at the next phase boundary).
    pub async fn interrupt(&self, reason: Option<String>) {
        let _ = self
            .tx
            .send(ActorMsg::Interrupt {
                reason,
                origin: local_origin(),
            })
            .await;
    }

    /// Rewind the conversation to `anchor` attributed to `origin` (conversation-rewind spec). Any
    /// live turn is interrupted first, then the conversation is truncated/reconstructed, the epoch is
    /// bumped, and `Rewound` is emitted on the event stream. Awaits the resolved [`RewindOutcome`].
    pub async fn rewind_to_from(
        &self,
        origin: Origin,
        request_id: ReqId,
        anchor: RewindAnchor,
    ) -> Result<RewindOutcome, Failure> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(ActorMsg::RewindTo {
                anchor,
                request_id,
                origin,
                reply,
            })
            .await
            .map_err(|_| Failure::Other("engine actor is gone".into()))?;
        rx.await
            .map_err(|_| Failure::Other("engine actor dropped the reply".into()))?
            .map_err(|e| Failure::Other(e.to_string()))
    }

    /// Rewind the conversation to `anchor` (untagged local origin).
    pub async fn rewind_to(
        &self,
        request_id: ReqId,
        anchor: RewindAnchor,
    ) -> Result<RewindOutcome, Failure> {
        self.rewind_to_from(local_origin(), request_id, anchor)
            .await
    }

    /// Inject steering text attributed to `origin`.
    pub async fn steer_from(&self, origin: Origin, request_id: ReqId, text: String) {
        let _ = self
            .tx
            .send(ActorMsg::Steer {
                request_id,
                text,
                origin,
            })
            .await;
    }

    /// Inject steering text. While a turn is running it is drained at the next phase boundary; when
    /// idle it opens a fresh steer turn. The ack rides the event stream as [`AgentEvent::Steered`].
    pub async fn steer(&self, request_id: ReqId, text: String) {
        self.steer_from(local_origin(), request_id, text).await;
    }

    /// Append context-only input attributed to `origin` **without** opening a turn (event-io §5.9):
    /// it folds into the conversation when idle and lands in the following turn when busy. No turn is
    /// started and no ack is emitted — the next mention-gated `start_turn` sees the accumulated text.
    pub async fn observe_from(&self, origin: Origin, request_id: ReqId, input: UserMsg) {
        let _ = self
            .tx
            .send(ActorMsg::Observe {
                request_id,
                input,
                origin,
            })
            .await;
    }

    /// Append context-only input (untagged local origin) without opening a turn.
    pub async fn observe(&self, request_id: ReqId, input: UserMsg) {
        self.observe_from(local_origin(), request_id, input).await;
    }

    /// Request a read-only snapshot. The reply rides the event stream as [`AgentEvent::Snapshot`]
    /// (served immediately when idle, or at the next phase boundary during a turn).
    pub async fn snapshot(&self, request_id: ReqId) {
        let _ = self
            .tx
            .send(ActorMsg::Snapshot {
                request_id,
                origin: local_origin(),
            })
            .await;
    }

    /// Swap the model provider for this session (a live model switch). Applied at the next turn
    /// boundary; an in-flight turn finishes on the old provider to preserve prompt caching.
    pub async fn set_provider(&self, provider: Arc<dyn Provider>) {
        let _ = self
            .tx
            .send(ActorMsg::SetProvider {
                provider,
                origin: local_origin(),
            })
            .await;
    }

    /// Set this session's edit-approval mode (a live §12 session-mode switch). Applied at the next
    /// turn boundary and consulted by the next gated tool action.
    pub async fn set_approval_policy(&self, policy: crate::approval::ApprovalPolicy) {
        let _ = self
            .tx
            .send(ActorMsg::SetApprovalPolicy {
                policy,
                origin: local_origin(),
            })
            .await;
    }

    /// Drain and shut the engine actor down.
    pub async fn shutdown(&self) {
        let _ = self
            .tx
            .send(ActorMsg::Shutdown {
                origin: local_origin(),
            })
            .await;
    }

    /// Subscribe to the lossless-primary §17 outbound event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Subscribe to the merged, bidirectional session event log (inbound + outbound, `seq`-stamped).
    pub fn subscribe_log(&self) -> broadcast::Receiver<SessionLogEntry> {
        self.log.subscribe()
    }
}

/// Map a turn outcome to the reply summary.
fn outcome_summary(outcome: Result<TurnOutcome, Failure>) -> Result<TurnSummary, Failure> {
    match outcome {
        Ok(TurnOutcome::Completed(s)) => Ok(s),
        Ok(TurnOutcome::Suspended(_)) => Ok(TurnSummary::ended(EndReason::Suspended)),
        Err(e) => Err(e),
    }
}

/// Spawn an engine session actor, returning its [`AgentHandle`]. The `host` services the engine's
/// blocking §17 requests (delegation / human-in-the-loop).
pub fn spawn_agent_session(mut engine: Engine, host: Arc<dyn HostRequestHandler>) -> AgentHandle {
    let (tx, mut rx) = mpsc::channel::<ActorMsg>(32);
    let (events_tx, _events_rx) = broadcast::channel::<AgentEvent>(256);
    let (log_tx, _log_rx) = broadcast::channel::<SessionLogEntry>(256);
    let actor_events = events_tx.clone();
    let actor_log = log_tx.clone();

    tokio::spawn(async move {
        let control = TurnControl::new();
        let self_origin = Origin {
            transport: TransportId::new("engine"),
            scope: OriginScope::Internal,
            sender: None,
        };
        let sink = SessionLog::with_log(
            self_origin,
            move |ev| {
                let _ = actor_events.send(ev);
            },
            move |entry| {
                let _ = actor_log.send(entry);
            },
        );
        let mut pending_starts: VecDeque<(UserMsg, oneshot::Sender<Result<TurnSummary, Failure>>)> =
            VecDeque::new();
        // A live model switch requested while a turn was running (or while idle): applied here at the
        // turn boundary so it never invalidates an in-flight turn's prompt cache.
        let mut pending_provider: Option<Arc<dyn Provider>> = None;
        // A live edit-approval mode switch requested while a turn was running (or while idle):
        // applied at the turn boundary alongside the provider switch.
        let mut pending_policy: Option<crate::approval::ApprovalPolicy> = None;
        // A rewind requested while a turn was running: the turn is cancelled (interrupt-first) and the
        // rewind is applied here at the boundary, once the abandoned turn has released `&mut engine`.
        type PendingRewind = (
            RewindAnchor,
            ReqId,
            oneshot::Sender<Result<RewindOutcome, RewindError>>,
        );
        let mut pending_rewind: Option<PendingRewind> = None;
        let mut shutting_down = false;

        loop {
            // Apply any pending live model switch before deciding/driving the next turn.
            if let Some(provider) = pending_provider.take() {
                engine.set_provider(provider);
            }
            // Apply any pending live edit-approval mode switch.
            if let Some(policy) = pending_policy.take() {
                engine.set_approval_policy(policy);
            }
            // ---- idle servicing: snapshots first (consistent read), then steer/queued starts ----
            for request_id in control.drain_snapshot() {
                let view = engine.conv_view();
                sink.emit(|seq| AgentEvent::Snapshot {
                    seq,
                    request_id,
                    view,
                });
            }

            // Decide the next turn to run (if any).
            let mut reply_slot: Option<oneshot::Sender<Result<TurnSummary, Failure>>> = None;
            let steers = control.drain_steer();
            if !steers.is_empty() {
                // Steer while idle: append the markers (acking each) and open a steer turn.
                for steer in &steers {
                    engine.push_steer_marker(steer);
                    let request_id = steer.request_id;
                    sink.emit(|seq| AgentEvent::Steered {
                        seq,
                        request_id,
                        accepted: true,
                    });
                }
            } else if let Some((input, reply)) = pending_starts.pop_front() {
                engine.push_user(input);
                reply_slot = Some(reply);
            } else if shutting_down {
                break;
            } else {
                // Nothing to run: wait for a command.
                let msg = match rx.recv().await {
                    Some(msg) => msg,
                    None => break,
                };
                // Record the inbound command on the merged log before acting on it.
                record_inbound(&sink, &msg);
                match msg {
                    ActorMsg::StartTurn { input, reply, .. } => {
                        pending_starts.push_back((input, reply));
                    }
                    ActorMsg::Steer {
                        request_id, text, ..
                    } => {
                        control.push_steer(SteerReq { request_id, text });
                    }
                    // Idle observe: fold the context straight into the conversation, opening no turn.
                    ActorMsg::Observe { input, .. } => {
                        engine.push_observe(input);
                    }
                    ActorMsg::Snapshot { request_id, .. } => control.push_snapshot(request_id),
                    // Idle: there is no in-flight turn to interrupt.
                    ActorMsg::Interrupt { .. } => {}
                    // Idle: no live turn to interrupt, so rewind immediately.
                    ActorMsg::RewindTo {
                        anchor,
                        request_id,
                        reply,
                        ..
                    } => {
                        let outcome = engine.rewind_to(&anchor, request_id, &sink);
                        let _ = reply.send(outcome);
                    }
                    ActorMsg::SetProvider { provider, .. } => {
                        pending_provider = Some(provider);
                    }
                    ActorMsg::SetApprovalPolicy { policy, .. } => {
                        pending_policy = Some(policy);
                    }
                    ActorMsg::Shutdown { .. } => break,
                }
                continue;
            }

            // ---- busy: drive the turn future, racing the inbox for control commands ----
            let summary = {
                let mut turn = Box::pin(engine.run_turn(&*host, &sink, &control));
                loop {
                    tokio::select! {
                        outcome = &mut turn => break outcome_summary(outcome),
                        msg = rx.recv() => match msg {
                            Some(msg) => {
                                // Record the inbound command on the merged log before acting on it.
                                record_inbound(&sink, &msg);
                                match msg {
                                    ActorMsg::Interrupt { .. } => control.cancel(),
                                    // Interrupt-first: cancel the live turn and stash the rewind; it is
                                    // applied at the boundary below once the turn releases `&mut engine`.
                                    ActorMsg::RewindTo {
                                        anchor,
                                        request_id,
                                        reply,
                                        ..
                                    } => {
                                        control.cancel();
                                        if let Some((.., stale)) = pending_rewind.replace((anchor, request_id, reply)) {
                                            // Supersede an earlier un-applied rewind; its caller errors out.
                                            drop(stale);
                                        }
                                    }
                                    ActorMsg::Steer { request_id, text, .. } => {
                                        control.push_steer(SteerReq { request_id, text });
                                    }
                                    // Busy observe: queue it; the boundary folds it into the
                                    // conversation as plain context for the following turn.
                                    ActorMsg::Observe { input, .. } => {
                                        control.push_observe(input);
                                    }
                                    ActorMsg::Snapshot { request_id, .. } => {
                                        control.push_snapshot(request_id);
                                    }
                                    ActorMsg::StartTurn { input, reply, .. } => {
                                        pending_starts.push_back((input, reply));
                                    }
                                    ActorMsg::SetProvider { provider, .. } => {
                                        pending_provider = Some(provider);
                                    }
                                    ActorMsg::SetApprovalPolicy { policy, .. } => {
                                        pending_policy = Some(policy);
                                    }
                                    ActorMsg::Shutdown { .. } => {
                                        control.cancel();
                                        shutting_down = true;
                                    }
                                }
                            }
                            None => {
                                control.cancel();
                                shutting_down = true;
                            }
                        },
                    }
                }
            };

            if let Some(reply) = reply_slot {
                let _ = reply.send(summary);
            }
            // Apply a rewind requested mid-turn: the abandoned turn has now released `&mut engine`, so
            // truncate/reconstruct/bump-epoch/emit-Rewound here at the boundary (interrupt-first done).
            if let Some((anchor, request_id, reply)) = pending_rewind.take() {
                let outcome = engine.rewind_to(&anchor, request_id, &sink);
                let _ = reply.send(outcome);
            }
            // Fresh cancellation token for the next turn.
            control.reset();
        }
        // Terminal teardown (§10/§11): the loop drained (Shutdown, or every handle dropped), so no
        // further turns can run on this incarnation — flush the context engine + memory providers
        // (a no-op if no turn ever started the session lifecycle).
        engine.end_session().await;
    });

    AgentHandle {
        tx,
        events: events_tx,
        log: log_tx,
        req_seq: Arc::new(AtomicU64::new(0)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{SystemPrompt, ToolCall};
    use crate::provider::MockProvider;
    use crate::tools::{Tool, ToolOutcome, ToolRegistry};
    use crate::turn::TurnCx;
    use daemon_common::{ReqId, SessionId};
    use daemon_protocol::{HostRequest, HostResponse, HostResponseBody, TurnTrigger};
    use std::time::Duration;

    struct NoopHost;

    #[async_trait::async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved {
                    approved: true,
                    allow_permanent: false,
                    reason: None,
                },
            }
        }
    }

    /// A tool that blocks until the turn is cancelled — lets a test hold a turn genuinely in flight.
    struct WaitForCancelTool;

    #[async_trait::async_trait]
    impl Tool for WaitForCancelTool {
        fn name(&self) -> &str {
            "wait"
        }
        fn schema(&self) -> &str {
            "{}"
        }
        async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
            cx.cancel.cancelled().await;
            ToolOutcome::text(call.call_id.clone(), true, "interrupted")
        }
    }

    async fn recv_until(
        rx: &mut broadcast::Receiver<AgentEvent>,
        pred: impl Fn(&AgentEvent) -> bool,
    ) -> AgentEvent {
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                .await
                .expect("timed out waiting for event")
                .expect("event stream closed");
            if pred(&ev) {
                return ev;
            }
        }
    }

    fn completing_engine(id: &str) -> Engine {
        Engine::fresh(
            SessionId::new(id),
            SystemPrompt::new("test"),
            Arc::new(MockProvider::completing("done")),
            Arc::new(ToolRegistry::new()),
        )
    }

    /// A §10 spy that records whether `on_session_end` fired (the actor teardown contract).
    struct EndSpyContext {
        ended: Arc<std::sync::atomic::AtomicBool>,
    }
    #[async_trait::async_trait]
    impl crate::context::ContextEngine for EndSpyContext {
        fn before_turn(
            &self,
            conv: &mut crate::conversation::Conversation,
            budget: Option<usize>,
        ) -> crate::context::Pressure {
            crate::context::Pressure {
                used_tokens: crate::context::estimate_tokens(conv),
                budget_tokens: budget,
            }
        }
        async fn compact(
            &self,
            conv: crate::conversation::Conversation,
            _budget: usize,
        ) -> crate::conversation::Conversation {
            conv
        }
        fn on_session_end(&self, _session: &SessionId, _conv: &crate::conversation::Conversation) {
            self.ended.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Draining the actor (Shutdown) is terminal teardown: the §10 context engine's
    /// `on_session_end` fires once the loop exits, without the host calling `end_session` itself.
    #[tokio::test]
    async fn shutdown_fires_session_end_teardown() {
        let ended = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let engine = completing_engine("teardown").with_context_engine(Arc::new(EndSpyContext {
            ended: ended.clone(),
        }));
        let handle = spawn_agent_session(engine, Arc::new(NoopHost));
        handle
            .start_turn(UserMsg::new("hi"))
            .await
            .expect("turn completes");
        handle.shutdown().await;
        // The teardown runs on the actor task after the loop breaks; poll briefly.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !ended.load(std::sync::atomic::Ordering::SeqCst) {
            assert!(
                std::time::Instant::now() < deadline,
                "session_end did not fire after shutdown"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// A snapshot request while idle is answered immediately on the event stream.
    #[tokio::test]
    async fn snapshot_when_idle_emits_snapshot_event() {
        let handle = spawn_agent_session(completing_engine("snap"), Arc::new(NoopHost));
        let mut rx = handle.subscribe();
        handle.snapshot(ReqId(5)).await;
        let ev = recv_until(
            &mut rx,
            |e| matches!(e, AgentEvent::Snapshot { request_id, .. } if *request_id == ReqId(5)),
        )
        .await;
        assert!(matches!(ev, AgentEvent::Snapshot { .. }));
        handle.shutdown().await;
    }

    /// Observe while **idle** folds context into the conversation but drives NO turn (W3 spike,
    /// event-io §5.9): the notification seam for host-originated input must therefore use
    /// `StartTurn` — an `Observe`d process-exit note would sit unseen until the user next speaks.
    #[tokio::test]
    async fn observe_when_idle_folds_context_but_opens_no_turn() {
        let handle = spawn_agent_session(completing_engine("observe-idle"), Arc::new(NoopHost));
        let mut rx = handle.subscribe();
        handle
            .observe(ReqId(7), UserMsg::new("[ambient] build finished"))
            .await;
        // The observed text is in the conversation (visible via a snapshot) and every event up to
        // that snapshot is turn-free — the observe drove nothing.
        handle.snapshot(ReqId(8)).await;
        let ev = recv_until(&mut rx, |e| {
            assert!(
                !matches!(e, AgentEvent::TurnStarted { .. }),
                "an idle Observe must not open a turn"
            );
            matches!(e, AgentEvent::Snapshot { .. })
        })
        .await;
        let AgentEvent::Snapshot { view, .. } = ev else {
            unreachable!()
        };
        assert!(view
            .turns
            .iter()
            .any(|t| format!("{t:?}").contains("[ambient] build finished")));
        // A real StartTurn afterwards runs a turn that carries the folded context.
        let driver = handle.clone();
        tokio::spawn(async move {
            let _ = driver.start_turn(UserMsg::new("go")).await;
        });
        recv_until(&mut rx, |e| {
            matches!(e, AgentEvent::TurnStarted { trigger, .. } if *trigger == TurnTrigger::User)
        })
        .await;
        handle.shutdown().await;
    }

    /// A steer while idle opens a fresh turn with `TurnTrigger::Steer`, acked via `Steered`.
    #[tokio::test]
    async fn steer_when_idle_opens_steer_turn() {
        let handle = spawn_agent_session(completing_engine("steer"), Arc::new(NoopHost));
        let mut rx = handle.subscribe();
        handle.steer(ReqId(3), "go".into()).await;
        recv_until(&mut rx, |e| {
            matches!(e, AgentEvent::Steered { request_id, accepted, .. } if *request_id == ReqId(3) && *accepted)
        })
        .await;
        recv_until(&mut rx, |e| {
            matches!(e, AgentEvent::TurnStarted { trigger, .. } if *trigger == TurnTrigger::Steer)
        })
        .await;
        handle.shutdown().await;
    }

    /// The merged session log records the inbound command (with its origin) and the outbound events
    /// under one monotonic `seq`, so a second surface can observe what was submitted — not just the
    /// engine's replies. This is the core asymmetry-closing guarantee of the event-io edge.
    #[tokio::test]
    async fn merged_log_records_inbound_and_outbound() {
        use daemon_protocol::{Direction, SessionPayload};

        let handle = spawn_agent_session(completing_engine("merged"), Arc::new(NoopHost));
        let mut log = handle.subscribe_log();

        let origin = Origin::new(
            "telegram",
            daemon_protocol::OriginScope::Dm { user: "u1".into() },
        );
        let driver = handle.clone();
        let origin_for_turn = origin.clone();
        tokio::spawn(async move {
            let _ = driver
                .start_turn_from(origin_for_turn, UserMsg::new("hi"))
                .await;
        });

        // The first inbound entry is the user's StartTurn, attributed to the telegram origin.
        let inbound = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let entry = log.recv().await.expect("log closed");
                if entry.direction == Direction::Inbound {
                    return entry;
                }
            }
        })
        .await
        .expect("timed out waiting for inbound entry");
        assert_eq!(inbound.origin, origin);
        assert!(matches!(
            inbound.payload,
            SessionPayload::Command(AgentCommand::StartTurn { .. })
        ));

        // An outbound entry follows on the same log, with a strictly greater seq.
        let outbound = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let entry = log.recv().await.expect("log closed");
                if entry.direction == Direction::Outbound {
                    return entry;
                }
            }
        })
        .await
        .expect("timed out waiting for outbound entry");
        assert!(outbound.seq > inbound.seq);
        assert!(matches!(outbound.payload, SessionPayload::Event(_)));

        handle.shutdown().await;
    }

    /// A turn held in a tool can be interrupted mid-flight, finalizing as `Interrupted`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn interrupt_during_tool_yields_interrupted() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(WaitForCancelTool));
        let engine = Engine::fresh(
            SessionId::new("interrupt"),
            SystemPrompt::new("test"),
            Arc::new(MockProvider::delegating("wait", "done")),
            Arc::new(registry),
        );
        let handle = spawn_agent_session(engine, Arc::new(NoopHost));
        let mut rx = handle.subscribe();

        let driver = handle.clone();
        tokio::spawn(async move {
            let _ = driver.start_turn(UserMsg::new("hi")).await;
        });

        // Wait until the tool is in flight, then interrupt.
        recv_until(&mut rx, |e| matches!(e, AgentEvent::ToolStarted { .. })).await;
        handle.interrupt(Some("stop".into())).await;

        let finished = recv_until(&mut rx, |e| matches!(e, AgentEvent::TurnFinished { .. })).await;
        match finished {
            AgentEvent::TurnFinished { summary, .. } => {
                assert_eq!(summary.end_reason, EndReason::Interrupted)
            }
            _ => unreachable!(),
        }
        handle.shutdown().await;
    }

    /// A live model switch (`SetSessionModel` → `set_provider`) swaps the provider on the running
    /// actor at the next turn boundary: the first turn uses the original provider, and the turn after
    /// the swap streams text from the new provider — with no session rebuild.
    #[tokio::test]
    async fn set_provider_swaps_model_at_next_turn() {
        async fn turn_text(
            handle: &AgentHandle,
            rx: &mut broadcast::Receiver<AgentEvent>,
        ) -> String {
            let driver = handle.clone();
            tokio::spawn(async move {
                let _ = driver.start_turn(UserMsg::new("hi")).await;
            });
            let mut text = String::new();
            loop {
                let ev = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("timed out")
                    .expect("stream closed");
                match ev {
                    AgentEvent::TextDelta { text: t, .. } => text.push_str(&t),
                    AgentEvent::TurnFinished { .. } => return text,
                    _ => {}
                }
            }
        }

        let engine = Engine::fresh(
            SessionId::new("swap"),
            SystemPrompt::new("test"),
            Arc::new(MockProvider::completing("first")),
            Arc::new(ToolRegistry::new()),
        );
        let handle = spawn_agent_session(engine, Arc::new(NoopHost));
        let mut rx = handle.subscribe();

        assert_eq!(turn_text(&handle, &mut rx).await, "first");

        handle
            .set_provider(Arc::new(MockProvider::completing("second")))
            .await;

        assert_eq!(turn_text(&handle, &mut rx).await, "second");
        handle.shutdown().await;
    }

    /// `RewindTo` while a turn is live interrupts it first, then truncates: the abandoned turn
    /// finalizes as `Interrupted` and `Rewound` is emitted only after, so no abandoned-turn event
    /// races the truncation (conversation-rewind spec §4 interrupt-first).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rewind_while_busy_interrupts_first() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(WaitForCancelTool));
        let engine = Engine::fresh(
            SessionId::new("rewind-busy"),
            SystemPrompt::new("test"),
            Arc::new(MockProvider::delegating("wait", "done")),
            Arc::new(registry),
        );
        let handle = spawn_agent_session(engine, Arc::new(NoopHost));
        let mut rx = handle.subscribe();

        let driver = handle.clone();
        tokio::spawn(async move {
            let _ = driver.start_turn(UserMsg::new("hi")).await;
        });

        // Hold the turn genuinely in flight (the tool blocks until cancelled), then rewind.
        recv_until(&mut rx, |e| matches!(e, AgentEvent::ToolStarted { .. })).await;
        let rewinder = handle.clone();
        let rewound = tokio::spawn(async move {
            rewinder
                .rewind_to(ReqId(9), RewindAnchor::UserTurn { ordinal: 0 })
                .await
        });

        // The live turn finalizes as Interrupted before the conversation is truncated.
        let finished = recv_until(&mut rx, |e| matches!(e, AgentEvent::TurnFinished { .. })).await;
        assert!(matches!(
            finished,
            AgentEvent::TurnFinished { summary, .. } if summary.end_reason == EndReason::Interrupted
        ));

        // Rewound is emitted after the interrupt, retaining zero turns (rewound to the first user).
        let ev = recv_until(&mut rx, |e| matches!(e, AgentEvent::Rewound { .. })).await;
        assert!(matches!(
            ev,
            AgentEvent::Rewound { to_cursor, request_id, .. } if to_cursor == 0 && request_id == ReqId(9)
        ));

        let outcome = rewound.await.unwrap().expect("rewind resolves");
        assert_eq!(outcome.retained_turns, 0);
        handle.shutdown().await;
    }
}
