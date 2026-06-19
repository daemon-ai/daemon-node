//! The §17 ⇄ management protocol translation adapter (host-spec §9; supervision §4 mapping table).
//!
//! The host is the only node that translates: it presents each engine it drives as a
//! `UnitKind::Engine` [`ManagedUnit`] to the supervisor above it. [`EngineUnit`] wraps a running
//! engine's [`AgentHandle`] and realizes the §4 mapping table — total upward (every §17 event maps
//! to a `ManageEvent`/`ManageRequest`) and partial downward (`Pause`/`Resume`/`Scale` an engine
//! cannot honor return [`Ack::Unsupported`]). §17 is adapted, never re-exported as the generic
//! types, so `daemon-core` stays free of `daemon-supervision`.

use async_trait::async_trait;
use daemon_common::{JobId, UnitId};
use daemon_core::{spawn_agent_session, AgentHandle, Engine};
use daemon_protocol::{
    AgentEvent, CompletionSource as P17CompletionSource, EndReason as P17EndReason, HostRequest,
    HostRequestHandler, HostRequestKind, HostResponse, HostResponseBody, TurnTrigger, UserMsg,
};
use daemon_supervision::{
    Ack, ApprovalReq, ChoiceReq, CompletionSource, DelegationSpec, EndReason, EscalationReq,
    EventStream, FailureClass, FailureView, InputReq, ManageCommand, ManageEvent, ManageRequest,
    ManageRequestHandler, ManageRequestKind, ManageResponseBody, ManagedUnit,
    Outcome, ProcId, ProgressDelta, StartTrigger, ToolRef, ToolResultRef, UnitKind, WorkId, WorkRef,
};
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// The slot the parent's upward request handler is installed into at attach time.
type HandlerSlot = Arc<Mutex<Option<Arc<dyn ManageRequestHandler>>>>;

/// An engine presented to its supervisor as a `UnitKind::Engine` [`ManagedUnit`] (host-spec §9).
pub struct EngineUnit {
    id: UnitId,
    handle: AgentHandle,
    handler: HandlerSlot,
    events: broadcast::Sender<ManageEvent>,
    last_work: Arc<Mutex<Option<WorkId>>>,
}

impl EngineUnit {
    /// Spawn an engine session and present it as a managed unit identified by `id`.
    pub fn spawn(id: UnitId, engine: Engine) -> Self {
        let handler: HandlerSlot = Arc::new(Mutex::new(None));
        let host = Arc::new(ManageToHost {
            handler: handler.clone(),
        });
        let handle = spawn_agent_session(engine, host);

        let (events, _) = broadcast::channel::<ManageEvent>(256);
        let last_work = Arc::new(Mutex::new(None));

        // Relay: subscribe to the §17 stream and map each event up to a ManageEvent. Subscribing
        // here (before any turn) keeps the translation lossless for live consumers (§4 / §2.2).
        let mut agent_rx = handle.subscribe();
        let out = events.clone();
        let last_work_relay = last_work.clone();
        tokio::spawn(async move {
            loop {
                match agent_rx.recv().await {
                    Ok(ev) => {
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
            handle,
            handler,
            events,
            last_work,
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

fn map_end_reason(end_reason: P17EndReason) -> EndReason {
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
impl ManagedUnit for EngineUnit {
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
                let handle = self.handle.clone();
                // The turn runs asynchronously; its progress streams up as ManageEvents.
                tokio::spawn(async move {
                    let _ = handle.start_turn(input).await;
                });
                Ack::Accepted
            }
            ManageCommand::Cancel { reason } => {
                let handle = self.handle.clone();
                tokio::spawn(async move {
                    handle.interrupt(reason).await;
                });
                Ack::Accepted
            }
            ManageCommand::Snapshot { .. } => Ack::Accepted,
            ManageCommand::Shutdown { .. } => {
                let handle = self.handle.clone();
                tokio::spawn(async move {
                    handle.shutdown().await;
                });
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
}

/// The §17 `HostRequestHandler` the engine sees: forwards each blocking §17 request up to the
/// installed management [`ManageRequestHandler`] (escalating up the chain), then maps the reply
/// back down (host-spec §9: §17 `HostRequest`s → `ManageRequest`s).
struct ManageToHost {
    handler: HandlerSlot,
}

#[async_trait]
impl HostRequestHandler for ManageToHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
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
        let response = handler
            .request(ManageRequest {
                request_id,
                kind,
            })
            .await;
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
