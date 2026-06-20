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
use crate::engine::{Engine, TurnOutcome};
use crate::events::EventSink;
use crate::Failure;
use daemon_common::ReqId;
use daemon_protocol::{AgentEvent, EndReason, HostRequestHandler, TurnSummary, UserMsg};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};

/// Internal actor mailbox messages (the §17 commands plus their reply channels).
enum ActorMsg {
    StartTurn {
        input: UserMsg,
        reply: oneshot::Sender<Result<TurnSummary, Failure>>,
    },
    Steer {
        request_id: ReqId,
        text: String,
    },
    Snapshot {
        request_id: ReqId,
    },
    Interrupt {
        #[allow(dead_code)]
        reason: Option<String>,
    },
    Shutdown,
}

/// A handle to a running engine session: send §17 commands, subscribe to the §17 event stream.
#[derive(Clone)]
pub struct AgentHandle {
    tx: mpsc::Sender<ActorMsg>,
    events: broadcast::Sender<AgentEvent>,
}

impl AgentHandle {
    /// Begin a turn from a user input, awaiting the terminal [`TurnSummary`].
    pub async fn start_turn(&self, input: UserMsg) -> Result<TurnSummary, Failure> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(ActorMsg::StartTurn { input, reply })
            .await
            .map_err(|_| Failure::Other("engine actor is gone".into()))?;
        rx.await
            .map_err(|_| Failure::Other("engine actor dropped the reply".into()))?
    }

    /// Interrupt the in-flight turn (cooperative cancellation, honored at the next phase boundary).
    pub async fn interrupt(&self, reason: Option<String>) {
        let _ = self.tx.send(ActorMsg::Interrupt { reason }).await;
    }

    /// Inject steering text. While a turn is running it is drained at the next phase boundary; when
    /// idle it opens a fresh steer turn. The ack rides the event stream as [`AgentEvent::Steered`].
    pub async fn steer(&self, request_id: ReqId, text: String) {
        let _ = self.tx.send(ActorMsg::Steer { request_id, text }).await;
    }

    /// Request a read-only snapshot. The reply rides the event stream as [`AgentEvent::Snapshot`]
    /// (served immediately when idle, or at the next phase boundary during a turn).
    pub async fn snapshot(&self, request_id: ReqId) {
        let _ = self.tx.send(ActorMsg::Snapshot { request_id }).await;
    }

    /// Drain and shut the engine actor down.
    pub async fn shutdown(&self) {
        let _ = self.tx.send(ActorMsg::Shutdown).await;
    }

    /// Subscribe to the lossless-primary §17 event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
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
    let actor_events = events_tx.clone();

    tokio::spawn(async move {
        let control = TurnControl::new();
        let sink = EventSink::new(move |ev| {
            let _ = actor_events.send(ev);
        });
        let mut pending_starts: VecDeque<(UserMsg, oneshot::Sender<Result<TurnSummary, Failure>>)> =
            VecDeque::new();
        let mut shutting_down = false;

        loop {
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
                match msg {
                    ActorMsg::StartTurn { input, reply } => {
                        pending_starts.push_back((input, reply));
                    }
                    ActorMsg::Steer { request_id, text } => {
                        control.push_steer(SteerReq { request_id, text });
                    }
                    ActorMsg::Snapshot { request_id } => control.push_snapshot(request_id),
                    // Idle: there is no in-flight turn to interrupt.
                    ActorMsg::Interrupt { .. } => {}
                    ActorMsg::Shutdown => break,
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
                            Some(ActorMsg::Interrupt { .. }) => control.cancel(),
                            Some(ActorMsg::Steer { request_id, text }) => {
                                control.push_steer(SteerReq { request_id, text });
                            }
                            Some(ActorMsg::Snapshot { request_id }) => {
                                control.push_snapshot(request_id);
                            }
                            Some(ActorMsg::StartTurn { input, reply }) => {
                                pending_starts.push_back((input, reply));
                            }
                            Some(ActorMsg::Shutdown) | None => {
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
            // Fresh cancellation token for the next turn.
            control.reset();
        }
    });

    AgentHandle {
        tx,
        events: events_tx,
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
                body: HostResponseBody::Approved(true),
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

    /// A snapshot request while idle is answered immediately on the event stream.
    #[tokio::test]
    async fn snapshot_when_idle_emits_snapshot_event() {
        let handle = spawn_agent_session(completing_engine("snap"), Arc::new(NoopHost));
        let mut rx = handle.subscribe();
        handle.snapshot(ReqId(5)).await;
        let ev = recv_until(&mut rx, |e| {
            matches!(e, AgentEvent::Snapshot { request_id, .. } if *request_id == ReqId(5))
        })
        .await;
        assert!(matches!(ev, AgentEvent::Snapshot { .. }));
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
}
