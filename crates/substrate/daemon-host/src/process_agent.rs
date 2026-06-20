//! [`ProcessAgentUnit`] — a **foreign** agent process presented as an `Engine`-leaf managed unit.
//!
//! Where [`crate::unit::EngineUnit`] backs a unit with an in-process `daemon-core` engine, this backs
//! it with a child process that speaks §17 over a [`daemon_provision`] cut: `AgentCommand`/
//! `HostResponse` framed (CBOR) down its stdin, `AgentEvent`/`HostRequest` framed up its stdout. Both
//! flow through the same [`crate::section17`] adapter, so a foreign brain is indistinguishable from
//! `daemon-core` to its supervisor (`UnitKind::Engine`) — the whole point of the §17 leaf being a
//! universal agent-runner contract.
//!
//! Unlike the durable placement cut ([`crate::cut`]), there is **no** store/credential brokering: a
//! foreign brain owns its own state, so its lifecycle is adapter-owned (the child is killed on drop,
//! relaunched from its launch profile) rather than hydrated/dehydrated from a `daemon-core` snapshot.

use crate::section17::{AgentUnit, Section17Session};
use async_trait::async_trait;
use daemon_common::UnitId;
use daemon_protocol::{
    AgentCommand, AgentEvent, HostRequestHandler, Inbound, Outbound,
};
use daemon_provision::{ChildGuard, CutChannel, CutWriter, Placement};
use std::sync::Arc;
use tokio::sync::broadcast;

/// A [`Section17Session`] over a foreign agent process driven across a cut.
struct ProcessSection17 {
    writer: CutWriter,
    events: broadcast::Sender<AgentEvent>,
    /// Owns the child process (when placed over a real cut); killed on drop so a unit never leaks an
    /// OS process. `None` when driven over an in-memory channel (tests).
    _child: Option<ChildGuard>,
}

impl ProcessSection17 {
    /// Start pumping a foreign agent over `channel`: spawn the reader task (events up, blocking
    /// requests answered via `host` and framed back down) and retain the writer for `submit`.
    fn from_channel(
        channel: CutChannel,
        child: Option<ChildGuard>,
        host: Arc<dyn HostRequestHandler>,
    ) -> Self {
        let (writer, mut reader) = channel.split();
        let (events, _) = broadcast::channel::<AgentEvent>(256);

        let events_relay = events.clone();
        let reply_writer = writer.clone();
        tokio::spawn(async move {
            while let Some(bytes) = reader.recv().await {
                match decode_up(&bytes) {
                    Some(Outbound::Event(ev)) => {
                        let _ = events_relay.send(ev);
                    }
                    Some(Outbound::Request(req)) => {
                        let resp = host.request(req).await;
                        let _ = reply_writer
                            .send(&encode_down(&Inbound::Response(resp)))
                            .await;
                    }
                    Some(_) | None => continue,
                }
            }
        });

        Self {
            writer,
            events,
            _child: child,
        }
    }
}

#[async_trait]
impl Section17Session for ProcessSection17 {
    async fn submit(&self, cmd: AgentCommand) {
        let _ = self
            .writer
            .send(&encode_down(&Inbound::Command(cmd)))
            .await;
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }
}

/// A foreign agent process presented to its supervisor as a `UnitKind::Engine` managed unit.
pub struct ProcessAgentUnit;

impl ProcessAgentUnit {
    /// Wrap a live [`Placement`] (a spawned foreign agent + its cut) as a managed unit `id`.
    pub fn start(id: UnitId, placement: Placement) -> AgentUnit {
        let Placement { channel, child } = placement;
        AgentUnit::start(id, move |host: Arc<dyn HostRequestHandler>| {
            Arc::new(ProcessSection17::from_channel(channel, Some(child), host))
                as Arc<dyn Section17Session>
        })
    }

    /// Wrap a foreign agent reachable over an in-memory `channel` (no OS child) as a managed unit.
    /// Used by tests to exercise the cut framing without spawning a process.
    #[cfg(test)]
    pub fn from_channel(id: UnitId, channel: CutChannel) -> AgentUnit {
        AgentUnit::start(id, move |host: Arc<dyn HostRequestHandler>| {
            Arc::new(ProcessSection17::from_channel(channel, None, host))
                as Arc<dyn Section17Session>
        })
    }
}

/// Encode a down-frame (CBOR). Frame types are always serializable; a failure is a programming error.
fn encode_down(frame: &Inbound) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(frame, &mut buf).expect("encode Inbound");
    buf
}

/// Decode an up-frame; `None` on a malformed frame.
fn decode_up(bytes: &[u8]) -> Option<Outbound> {
    ciborium::from_reader(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_common::{Budget, ReqId};
    use daemon_protocol::{
        EndReason, HostRequest, HostRequestKind, HostResponseBody, TurnSummary, TurnTrigger,
    };
    use daemon_supervision::{
        Ack, ManageCommand, ManageEvent, ManageRequest, ManageRequestHandler, ManageResponse,
        ManageResponseBody, ManagedUnit, StreamLagged, UnitKind, WorkRef,
    };
    use std::time::Duration;

    /// A supervisor handler that approves everything (the answer-authority for the foreign unit).
    struct Approver;

    #[async_trait]
    impl ManageRequestHandler for Approver {
        async fn request(&self, req: ManageRequest) -> ManageResponse {
            ManageResponse {
                request_id: req.request_id,
                body: ManageResponseBody::Approved(true),
            }
        }
    }

    fn encode_up(frame: &Outbound) -> Vec<u8> {
        let mut buf = Vec::new();
        ciborium::into_writer(frame, &mut buf).expect("encode Outbound");
        buf
    }

    fn decode_down(bytes: &[u8]) -> Option<Inbound> {
        ciborium::from_reader(bytes).ok()
    }

    /// Drive a foreign unit over an in-memory cut: the "agent" raises an approval request, then on
    /// the approval emits a `TurnStarted` -> `TurnFinished{Completed}` pair. Proves the cut framing
    /// round-trips both directions and maps up to the management protocol identically to an engine.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn foreign_unit_round_trips_events_and_a_request() {
        // Two duplex pipes form the bidirectional cut (write to `a` is read from `b`).
        let (p2c_a, p2c_b) = tokio::io::duplex(64 * 1024);
        let (c2p_a, c2p_b) = tokio::io::duplex(64 * 1024);
        let parent = CutChannel::from_parts(Box::new(c2p_b), Box::new(p2c_a));
        let child = CutChannel::from_parts(Box::new(p2c_b), Box::new(c2p_a));

        // The "foreign agent": a non-engine task speaking the §17 cut dialect.
        let (cw, mut cr) = child.split();
        tokio::spawn(async move {
            while let Some(bytes) = cr.recv().await {
                match decode_down(&bytes) {
                    Some(Inbound::Command(AgentCommand::StartTurn { .. })) => {
                        let req = Outbound::Request(HostRequest {
                            request_id: ReqId(1),
                            kind: HostRequestKind::Approval {
                                prompt: "may I?".into(),
                            },
                        });
                        let _ = cw.send(&encode_up(&req)).await;
                    }
                    Some(Inbound::Response(resp)) => {
                        assert!(matches!(resp.body, HostResponseBody::Approved(true)));
                        let started = Outbound::Event(AgentEvent::TurnStarted {
                            seq: 0,
                            trigger: TurnTrigger::User,
                        });
                        let finished = Outbound::Event(AgentEvent::TurnFinished {
                            seq: 1,
                            summary: TurnSummary::ended(EndReason::Completed),
                        });
                        let _ = cw.send(&encode_up(&started)).await;
                        let _ = cw.send(&encode_up(&finished)).await;
                    }
                    _ => {}
                }
            }
        });

        let unit = ProcessAgentUnit::from_channel(UnitId::new("foreign"), parent);
        assert_eq!(unit.kind(), UnitKind::Engine);
        unit.install_request_handler(Arc::new(Approver));
        let mut events = unit.events();

        assert_eq!(
            unit.command(ManageCommand::Assign {
                request_id: ReqId(0),
                work: WorkRef::inline("w", "do the thing"),
                budget: Budget::unlimited(),
            })
            .await,
            Ack::Accepted
        );

        // Expect Started then Finished{Completed} mapped up from the foreign agent's §17 events.
        let mut saw_started = false;
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(5), events.recv())
                .await
                .expect("timed out awaiting management events");
            match ev {
                Ok(ManageEvent::Started { .. }) => saw_started = true,
                Ok(ManageEvent::Finished { outcome, .. }) => {
                    assert!(saw_started, "Finished arrived before Started");
                    assert_eq!(outcome.end_reason, daemon_supervision::EndReason::Completed);
                    break;
                }
                Ok(_) => {}
                Err(StreamLagged::Lagged { .. }) => {}
                Err(StreamLagged::Closed) => panic!("event stream closed before Finished"),
            }
        }
    }
}
