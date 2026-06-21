//! [`ProcessAgentUnit`] — a **foreign** agent process presented as an `Engine`-leaf managed unit.
//!
//! Where [`crate::unit::EngineUnit`] backs a unit with an in-process `daemon-core` engine, this backs
//! it with a child process that speaks §17 over a [`daemon_provision`] cut. It is a thin factory over
//! the generic [`CodecSession`](crate::foreign::CodecSession) driver wired with the
//! [`NativeCutCodec`](crate::foreign::NativeCutCodec): `AgentCommand`/`HostResponse` framed (CBOR)
//! down its stdin, `AgentEvent`/`HostRequest` framed up its stdout. The session flows through the
//! same [`crate::agent_session`] adapter, so a foreign brain is indistinguishable from `daemon-core` to
//! its supervisor (`UnitKind::Engine`) — the whole point of the §17 leaf being a universal
//! agent-runner contract. Other foreign protocols are just other codecs on the same driver.
//!
//! Unlike the durable placement cut ([`crate::cut`]), there is **no** store/credential brokering: a
//! foreign brain owns its own state, so its lifecycle is adapter-owned (the child is killed on drop,
//! relaunched from its launch profile) rather than hydrated/dehydrated from a `daemon-core` snapshot.

use crate::agent_session::{AgentSession, AgentUnit};
use crate::foreign::{CodecSession, NativeCutCodec};
use daemon_common::UnitId;
use daemon_protocol::HostRequestHandler;
use daemon_provision::Placement;
use std::sync::Arc;

/// A foreign agent process presented to its supervisor as a `UnitKind::Engine` managed unit.
pub struct ProcessAgentUnit;

impl ProcessAgentUnit {
    /// Wrap a live [`Placement`] (a spawned foreign agent + its cut) as a managed unit `id`.
    pub fn start(id: UnitId, placement: Placement) -> AgentUnit {
        Self::start_journaled(id, placement, None)
    }

    /// As [`Self::start`], but durably journals the foreign agent's transcript (finished blocks +
    /// lifecycle, sealed per turn) into `journal` when provided.
    pub fn start_journaled(
        id: UnitId,
        placement: Placement,
        journal: Option<Arc<crate::journal::JournalFeeder>>,
    ) -> AgentUnit {
        let Placement { channel, child } = placement;
        AgentUnit::start_journaled(id, journal, move |host: Arc<dyn HostRequestHandler>| {
            Arc::new(CodecSession::from_channel(
                channel,
                Some(child),
                host,
                NativeCutCodec,
            )) as Arc<dyn AgentSession>
        })
    }

    /// Wrap a foreign agent reachable over an in-memory `channel` (no OS child) as a managed unit.
    /// Used by tests to exercise the cut framing without spawning a process.
    #[cfg(test)]
    pub fn from_channel(id: UnitId, channel: daemon_provision::CutChannel) -> AgentUnit {
        AgentUnit::start_journaled(id, None, move |host: Arc<dyn HostRequestHandler>| {
            Arc::new(CodecSession::from_channel(
                channel,
                None,
                host,
                NativeCutCodec,
            )) as Arc<dyn AgentSession>
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use daemon_common::{Budget, ReqId};
    use daemon_protocol::{
        AgentCommand, AgentEvent, EndReason, HostRequest, HostRequestKind, HostResponseBody,
        Inbound, Outbound, TurnSummary, TurnTrigger,
    };
    use daemon_provision::CutChannel;
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
