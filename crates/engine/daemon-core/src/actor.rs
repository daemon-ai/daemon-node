//! The agent session as an actor (§17 runtime handle).
//!
//! [`spawn_agent_session`] owns an [`Engine`] on a dedicated task and serves §17 commands over an
//! mpsc inbox while fanning [`AgentEvent`]s out over a broadcast. This is the live, in-process face
//! the host's `EngineUnit` wraps to present the engine as a `ManagedUnit`. The durable substrate
//! path does *not* use this actor — it drives the [`Engine`] directly through the activation seam.

use crate::engine::{Engine, TurnOutcome};
use crate::events::EventSink;
use crate::Failure;
use daemon_protocol::{AgentEvent, EndReason, HostRequestHandler, TurnSummary, UserMsg};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

/// Internal actor mailbox messages (the §17 commands plus their reply channels).
enum ActorMsg {
    StartTurn {
        input: UserMsg,
        reply: oneshot::Sender<Result<TurnSummary, Failure>>,
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

    /// Interrupt the in-flight turn (cooperative cancellation).
    pub async fn interrupt(&self, reason: Option<String>) {
        let _ = self.tx.send(ActorMsg::Interrupt { reason }).await;
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

/// Spawn an engine session actor, returning its [`AgentHandle`]. The `host` services the engine's
/// blocking §17 requests (delegation / human-in-the-loop).
pub fn spawn_agent_session(mut engine: Engine, host: Arc<dyn HostRequestHandler>) -> AgentHandle {
    let (tx, mut rx) = mpsc::channel::<ActorMsg>(32);
    let (events_tx, _events_rx) = broadcast::channel::<AgentEvent>(256);
    let actor_events = events_tx.clone();

    tokio::spawn(async move {
        let cancel = CancellationToken::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                ActorMsg::StartTurn { input, reply } => {
                    engine.push_user(input);
                    let sink_tx = actor_events.clone();
                    let sink = EventSink::new(move |ev| {
                        let _ = sink_tx.send(ev);
                    });
                    let outcome = engine.run_turn(&*host, &sink, cancel.clone()).await;
                    let summary = match outcome {
                        Ok(TurnOutcome::Completed(s)) => Ok(s),
                        Ok(TurnOutcome::Suspended(_)) => {
                            Ok(TurnSummary::ended(EndReason::Suspended))
                        }
                        Err(e) => Err(e),
                    };
                    let _ = reply.send(summary);
                }
                ActorMsg::Interrupt { .. } => cancel.cancel(),
                ActorMsg::Shutdown => break,
            }
        }
    });

    AgentHandle {
        tx,
        events: events_tx,
    }
}
