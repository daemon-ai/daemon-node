//! The transport-agnostic §17 session seam and the §17 ⇄ management translation adapter.
//!
//! §17 (`AgentCommand` in, `AgentEvent`/`HostRequest` out) is the *leaf dialect every brain speaks*,
//! whether it is an in-process `daemon-core` engine ([`crate::unit::EngineUnit`]) or a foreign agent
//! process driven over a cut ([`crate::process_agent::ProcessAgentUnit`]). This module factors the
//! part that is identical for both:
//!
//! - [`Section17Session`] — a running §17 session as a transport: submit a command, subscribe to the
//!   event stream. Blocking host requests are answered by the [`HostRequestHandler`] the session was
//!   constructed with (the [`ManageToHost`] adapter), so the request direction never appears in this
//!   trait.
//! - [`AgentUnit`] — the shared `UnitKind::Engine` [`ManagedUnit`] over any `Section17Session`. It
//!   realizes the supervision §4 mapping table (total upward, partial downward) so a host presents an
//!   engine — ours or foreign — identically to its supervisor.

use crate::journal::JournalFeeder;
use async_trait::async_trait;
use daemon_api::Outbound;
use daemon_common::{JobId, UnitId};
use daemon_protocol::{
    AgentCommand, AgentEvent, CompletionSource as P17CompletionSource, EndReason as P17EndReason,
    HostRequest, HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, TurnTrigger,
    UserMsg,
};
use daemon_supervision::{
    Ack, ApprovalReq, ChoiceReq, CompletionSource, DelegationSpec, EscalationReq, EventStream,
    FailureClass, FailureView, InputReq, ManageCommand, ManageEvent, ManageRequest,
    ManageRequestHandler, ManageRequestKind, ManageResponseBody, ManagedUnit, Outcome, ProcId,
    ProgressDelta, StartTrigger, ToolRef, ToolResultRef, UnitKind, WorkId, WorkRef,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// The slot the parent's upward request handler is installed into at attach time.
pub(crate) type HandlerSlot = Arc<Mutex<Option<Arc<dyn ManageRequestHandler>>>>;

/// A bounded ring buffer of §17 [`Outbound`] items (streamed events + raised host requests) retained
/// per unit for the rich, transcript-fidelity per-`UnitId` drain ([`AgentUnit::drain_outbound`], the
/// host side of `ControlApi::unit_outbound`). Live-only and best-effort: when full the oldest item
/// is dropped. Durable/queryable history (reconnect, scroll-back) is out of scope for this layer.
pub(crate) type OutboundDrain = Arc<Mutex<VecDeque<Outbound>>>;

/// How many recent §17 `Outbound` items a unit retains for its rich drain.
const OUTBOUND_DRAIN_CAP: usize = 1024;

/// Push one item onto a unit's bounded outbound drain, dropping the oldest when at capacity.
fn push_outbound(drain: &OutboundDrain, item: Outbound) {
    let mut q = drain.lock().unwrap();
    if q.len() >= OUTBOUND_DRAIN_CAP {
        q.pop_front();
    }
    q.push_back(item);
}

/// A running §17 session, transport-agnostic. Commands go in; `AgentEvent`s come out on the
/// broadcast. Blocking host requests are answered by the [`HostRequestHandler`] the session was
/// built with, so they never surface on this trait.
///
/// Public so an out-of-tree adapter crate (e.g. `daemon-acp`) can implement its own session over a
/// foreign protocol and wrap it with [`AgentUnit::start_journaled`].
#[async_trait]
pub trait Section17Session: Send + Sync {
    /// Submit a §17 command. Must return promptly: a `StartTurn` runs the turn in the background so
    /// progress streams out as events.
    async fn submit(&self, cmd: AgentCommand);

    /// Subscribe to the lossless-primary §17 event stream.
    fn subscribe(&self) -> broadcast::Receiver<AgentEvent>;
}

/// An engine (in-process or foreign) presented to its supervisor as a `UnitKind::Engine`
/// [`ManagedUnit`] over a [`Section17Session`] (host-spec §9).
pub struct AgentUnit {
    id: UnitId,
    session: Arc<dyn Section17Session>,
    handler: HandlerSlot,
    events: broadcast::Sender<ManageEvent>,
    last_work: Arc<Mutex<Option<WorkId>>>,
    outbound: OutboundDrain,
}

impl AgentUnit {
    /// Start a unit identified by `id` over a session built by `build`. `build` is handed the
    /// [`HostRequestHandler`] the session must route its blocking §17 requests through (the
    /// [`ManageToHost`] adapter that escalates to the installed [`ManageRequestHandler`]).
    ///
    /// When `journal` is `Some`, the full §17 `Outbound` stream (events + raised requests) is fed
    /// into it so the unit's finished transcript blocks + lifecycle are durably sealed per turn (the
    /// fleet/foreign production journaling path); `None` disables journaling.
    ///
    /// Public so an adapter crate can present its own [`Section17Session`] as a managed engine unit
    /// with the same drain + verifiable-journal wiring as the in-tree backends.
    pub fn start_journaled(
        id: UnitId,
        journal: Option<Arc<JournalFeeder>>,
        build: impl FnOnce(Arc<dyn HostRequestHandler>) -> Arc<dyn Section17Session>,
    ) -> Self {
        let handler: HandlerSlot = Arc::new(Mutex::new(None));
        let outbound: OutboundDrain = Arc::new(Mutex::new(VecDeque::new()));
        let host = Arc::new(ManageToHost {
            handler: handler.clone(),
            outbound: outbound.clone(),
            journal: journal.clone(),
        });
        let session = build(host);

        let (events, _) = broadcast::channel::<ManageEvent>(256);
        let last_work = Arc::new(Mutex::new(None));

        // Relay: subscribe to the §17 stream and (a) retain each event verbatim on the rich outbound
        // drain (transcript fidelity — structured `detail` / `ContentDelta` survive untouched), then
        // (b) feed the verifiable journal (coalesced into finished blocks, sealed per turn), then
        // (c) map it up to a ManageEvent for the coarse management broadcast. Subscribing here
        // (before any turn) keeps both translations lossless for live consumers (§4 / §2.2).
        let mut agent_rx = session.subscribe();
        let out = events.clone();
        let last_work_relay = last_work.clone();
        let drain_relay = outbound.clone();
        let journal_relay = journal.clone();
        tokio::spawn(async move {
            loop {
                match agent_rx.recv().await {
                    Ok(ev) => {
                        let frame = Outbound::Event(ev.clone());
                        push_outbound(&drain_relay, frame.clone());
                        if let Some(feeder) = &journal_relay {
                            feeder.feed(&frame).await;
                        }
                        if let Some(mapped) = map_event(ev, &last_work_relay) {
                            let _ = out.send(mapped);
                        }
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        });

        Self {
            id,
            session,
            handler,
            events,
            last_work,
            outbound,
        }
    }
}

/// Resolve a [`WorkRef`] to the engine's `StartTurn` input (§4: `work` resolves to a `UserMsg`).
fn resolve_work(work: &WorkRef) -> UserMsg {
    if let Some(payload) = &work.payload {
        UserMsg::new(payload.text.clone())
    } else if let Some(content) = &work.content {
        UserMsg::new(format!("content:{}", content.as_str()))
    } else {
        UserMsg::new(work.id.as_str().to_owned())
    }
}

/// Map a §17 [`AgentEvent`] up to a generic [`ManageEvent`] (supervision §4 mapping table).
///
/// Returns `None` for `TurnFinished { Suspended }`: suspension is an engine-internal phase boundary,
/// not a management-terminal outcome — the unit reappears with `Started { BackgroundCompletion }` on
/// resume, so emitting `Finished` here would be wrong.
fn map_event(ev: AgentEvent, last_work: &Arc<Mutex<Option<WorkId>>>) -> Option<ManageEvent> {
    let mapped = match ev {
        AgentEvent::TurnStarted { seq, trigger } => ManageEvent::Started {
            seq,
            trigger: map_trigger(trigger, last_work),
        },
        AgentEvent::TextDelta { seq, text } => ManageEvent::Progress {
            seq,
            delta: ProgressDelta::Text(text),
        },
        AgentEvent::ReasoningDelta { seq, text } => ManageEvent::Progress {
            seq,
            delta: ProgressDelta::Reasoning(text),
        },
        AgentEvent::ToolStarted { seq, call } => ManageEvent::Progress {
            seq,
            delta: ProgressDelta::ToolStarted(ToolRef {
                call_id: call.call_id,
                name: call.name,
            }),
        },
        AgentEvent::ToolFinished { seq, result } => ManageEvent::Progress {
            seq,
            delta: ProgressDelta::ToolFinished(ToolResultRef {
                call_id: result.call_id,
                ok: result.ok,
            }),
        },
        AgentEvent::Usage { seq, delta } => ManageEvent::Usage { seq, delta },
        AgentEvent::RateLimit { seq, snapshot } => ManageEvent::RateLimit { seq, snapshot },
        AgentEvent::TurnFinished { seq, summary } => match summary.end_reason {
            P17EndReason::Suspended => return None,
            end_reason => ManageEvent::Finished {
                seq,
                outcome: Outcome {
                    end_reason: map_end_reason(end_reason),
                    summary: summary.final_text,
                    artifacts: Vec::new(),
                },
            },
        },
        AgentEvent::Error { seq, failure } => ManageEvent::Error {
            seq,
            failure: FailureView::new(FailureClass::Internal, failure),
        },
        // Opaque structured stream content has no coarse management projection: the supervisor /
        // fleet dashboard stays payload-agnostic by design (it never interprets `kind`/`body`). A
        // rich consumer reads it verbatim off the §17 `Outbound` drain (`unit_outbound`), not here.
        // The opaque `detail` on `ToolStarted`/`ToolFinished` is likewise dropped above: the coarse
        // `ProgressDelta::ToolStarted`/`ToolFinished` keep only `call_id`/`name`/`ok`.
        AgentEvent::ContentDelta { .. } => return None,
        // Steered / Snapshot are control-correlated replies, not management progress.
        _ => return None,
    };
    Some(mapped)
}

fn map_trigger(trigger: TurnTrigger, last_work: &Arc<Mutex<Option<WorkId>>>) -> StartTrigger {
    match trigger {
        TurnTrigger::User | TurnTrigger::Steer => {
            let work = last_work
                .lock()
                .unwrap()
                .clone()
                .unwrap_or_else(|| WorkId::new("assigned"));
            StartTrigger::Assigned(work)
        }
        TurnTrigger::BackgroundCompletion { source } => StartTrigger::BackgroundCompletion {
            source: match source {
                P17CompletionSource::Delegation(job) => {
                    CompletionSource::Delegation(UnitId::new(job.0))
                }
                P17CompletionSource::Process(job) => CompletionSource::Process(ProcId::new(job.0)),
            },
        },
    }
}

fn map_end_reason(end_reason: P17EndReason) -> daemon_supervision::EndReason {
    use daemon_supervision::EndReason;
    match end_reason {
        P17EndReason::Completed => EndReason::Completed,
        P17EndReason::Interrupted => EndReason::Interrupted,
        P17EndReason::BudgetExhausted => EndReason::BudgetExhausted,
        P17EndReason::Failed => EndReason::Failed(FailureClass::Internal),
        // Suspended is filtered before this point.
        P17EndReason::Suspended => EndReason::Interrupted,
        _ => EndReason::Failed(FailureClass::Internal),
    }
}

#[async_trait]
impl ManagedUnit for AgentUnit {
    fn id(&self) -> UnitId {
        self.id.clone()
    }

    fn kind(&self) -> UnitKind {
        UnitKind::Engine
    }

    async fn command(&self, cmd: ManageCommand) -> Ack {
        match cmd {
            ManageCommand::Assign { work, .. } => {
                *self.last_work.lock().unwrap() = Some(work.id.clone());
                let input = resolve_work(&work);
                // The turn runs asynchronously (the session backgrounds it); progress streams up.
                self.session
                    .submit(AgentCommand::StartTurn {
                        input,
                        request_id: daemon_common::ReqId(0),
                    })
                    .await;
                Ack::Accepted
            }
            ManageCommand::Cancel { reason } => {
                self.session.submit(AgentCommand::Interrupt { reason }).await;
                Ack::Accepted
            }
            ManageCommand::Snapshot { .. } => Ack::Accepted,
            ManageCommand::Shutdown { .. } => {
                self.session.submit(AgentCommand::Shutdown).await;
                Ack::Accepted
            }
            // No-ops at a single conversation (supervision §4 mapping table).
            ManageCommand::Pause | ManageCommand::Resume | ManageCommand::Scale { .. } => {
                Ack::Unsupported
            }
            _ => Ack::Unsupported,
        }
    }

    fn events(&self) -> EventStream<ManageEvent> {
        EventStream::new(self.events.subscribe())
    }

    fn install_request_handler(&self, handler: Arc<dyn ManageRequestHandler>) {
        *self.handler.lock().unwrap() = Some(handler);
    }

    /// The rich, transcript-fidelity per-unit drill-down: the full §17 `Outbound` stream this engine
    /// retained, in order (the drill-down counterpart to the coarse [`Self::events`] management
    /// stream). Preserves structured tool `detail` / `ContentDelta` and blocking host requests
    /// untouched, so a transcript consumer can render any unit in the tree.
    fn drain_outbound(&self, max: u32) -> Vec<Outbound> {
        let mut q = self.outbound.lock().unwrap();
        let take = if max == 0 {
            q.len()
        } else {
            (max as usize).min(q.len())
        };
        q.drain(..take).collect()
    }
}

/// The §17 `HostRequestHandler` the session sees: forwards each blocking §17 request up to the
/// installed management [`ManageRequestHandler`] (escalating up the chain), then maps the reply back
/// down (host-spec §9: §17 `HostRequest`s → `ManageRequest`s).
pub(crate) struct ManageToHost {
    handler: HandlerSlot,
    /// The unit's rich outbound drain: a raised request is retained here (in causal order with the
    /// event stream) so the transcript consumer sees the pending interactive prompt.
    outbound: OutboundDrain,
    /// The verifiable-journal feeder, so a raised request graduates into a durable request block.
    journal: Option<Arc<JournalFeeder>>,
}

#[async_trait]
impl HostRequestHandler for ManageToHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        // Retain the blocking request on the rich drain before escalating, so a transcript consumer
        // can render the pending prompt (approval / input / choice / delegate).
        push_outbound(&self.outbound, Outbound::Request(req.clone()));
        if let Some(feeder) = &self.journal {
            feeder.feed(&Outbound::Request(req.clone())).await;
        }

        let installed = self.handler.lock().unwrap().clone();
        let request_id = req.request_id;
        let is_delegate = matches!(req.kind, HostRequestKind::Delegate { .. });

        let Some(handler) = installed else {
            // No supervisor attached: answer conservatively so the engine can make progress.
            let body = if is_delegate {
                HostResponseBody::Delegated(JobId::new("undelegated"))
            } else {
                HostResponseBody::Approved(false)
            };
            return HostResponse { request_id, body };
        };

        let kind = map_request_kind(req.kind);
        let response = handler.request(ManageRequest { request_id, kind }).await;
        HostResponse {
            request_id,
            body: map_response_body(response.body, is_delegate),
        }
    }
}

fn map_request_kind(kind: HostRequestKind) -> ManageRequestKind {
    match kind {
        HostRequestKind::Approval { prompt } => ManageRequestKind::Approval(ApprovalReq { prompt }),
        HostRequestKind::Input { prompt } => ManageRequestKind::Input(InputReq { prompt }),
        HostRequestKind::Choice { prompt, options } => {
            ManageRequestKind::Choice(ChoiceReq { prompt, options })
        }
        HostRequestKind::Delegate { label, budget } => {
            ManageRequestKind::Delegate(vec![DelegationSpec {
                work: WorkRef::inline(label.clone(), label),
                budget,
                toolset: Vec::new(),
            }])
        }
        _ => ManageRequestKind::Escalate(EscalationReq {
            reason: "unmapped §17 request".into(),
        }),
    }
}

fn map_response_body(body: ManageResponseBody, is_delegate: bool) -> HostResponseBody {
    match body {
        ManageResponseBody::Approved(ok) => HostResponseBody::Approved(ok),
        ManageResponseBody::Input(text) => HostResponseBody::Input(text),
        ManageResponseBody::Chosen(index) => HostResponseBody::Chosen(index),
        ManageResponseBody::Delegated(units) => HostResponseBody::Delegated(
            units
                .first()
                .map(|u| JobId::new(u.as_str().to_owned()))
                .unwrap_or_else(|| JobId::new("delegated")),
        ),
        _ if is_delegate => HostResponseBody::Delegated(JobId::new("delegated")),
        _ => HostResponseBody::Approved(false),
    }
}
