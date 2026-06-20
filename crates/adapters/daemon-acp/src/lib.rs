//! `daemon-acp` — the Agent Client Protocol (ACP) foreign-agent adapter.
//!
//! ACP is JSON-RPC 2.0 over newline-delimited stdio, and unlike Claude-Code `stream-json` it is
//! **symmetric**: the agent calls *back* into the client for services (permission prompts, and —
//! when advertised — filesystem / terminal access). This crate bridges an ACP agent to the daemon's
//! §17 session seam ([`daemon_host::Section17Session`]) so it presents to the orchestrator as an
//! ordinary `UnitKind::Engine` managed unit, with its finished transcript blocks flowing into the
//! verifiable journal exactly like any other engine.
//!
//! The heavy [`agent_client_protocol`] dependency (a scoped builder/connection runtime) is isolated
//! here: [`AcpSession`] owns the connection on a dedicated task fed by an mpsc command queue, so the
//! session outlives a single prompt, and the crate's runtime model never leaks into `daemon-host`.
//!
//! **Scope (permission-first):** this adapter advertises no `fs`/`terminal` client capabilities, so
//! the only symmetric callback it bridges is `session/request_permission` → a §17
//! [`HostRequest`](daemon_protocol::HostRequest) `Approval`. Filesystem / terminal callbacks are a
//! follow-up on the same seam.

#![forbid(unsafe_code)]

use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, EnvVariable, InitializeRequest, McpServer, McpServerStdio,
    NewSessionRequest, PermissionOptionKind, PromptRequest, PromptResponse,
    CancelNotification, RequestPermissionOutcome, RequestPermissionRequest,
    RequestPermissionResponse, SelectedPermissionOutcome, SessionNotification, SessionUpdate,
    StopReason, TextContent, ToolCall, ToolCallStatus, ToolCallUpdate,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, Agent, Client, ConnectionTo, Responder};
use async_trait::async_trait;
use daemon_common::{ReqId, UnitId};
use daemon_host::{AgentUnit, JournalFeeder, Section17Session};
use daemon_protocol::{
    AgentCommand, AgentEvent, EndReason, HostRequest, HostRequestHandler, HostRequestKind,
    HostResponseBody, ToolCallView, ToolDetail, ToolResultView, TurnSummary, TurnTrigger,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

/// How to spawn the foreign ACP agent subprocess (program + args + environment + working dir).
#[derive(Clone, Debug)]
pub struct AcpLaunch {
    /// The agent program to exec.
    pub program: PathBuf,
    /// Arguments passed to the agent.
    pub args: Vec<String>,
    /// Environment variables set for the agent (added to the inherited environment).
    pub env: Vec<(String, String)>,
    /// The working directory advertised to the agent in `session/new`.
    pub cwd: PathBuf,
}

impl AcpLaunch {
    /// Construct a launch spec for `program`, defaulting the cwd to the current directory.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            }
    }

    /// Set the agent arguments.
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set the agent environment.
    pub fn env(mut self, env: Vec<(String, String)>) -> Self {
        self.env = env;
        self
    }

    /// Set the working directory advertised in `session/new`.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = cwd.into();
        self
    }

    fn into_agent(self) -> (AcpAgent, PathBuf) {
        let name = self
            .program
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "acp-agent".to_string());
        let stdio = McpServerStdio::new(name, self.program)
            .args(self.args)
            .env(
                self.env
                    .into_iter()
                    .map(|(k, v)| EnvVariable::new(k, v))
                    .collect(),
            );
        (AcpAgent::new(McpServer::Stdio(stdio)), self.cwd)
    }
}

/// A §17 session over a foreign ACP agent. Construct via [`acp_unit`] to present it as a managed
/// engine unit; the connection runs on a dedicated task and is fed commands through an mpsc queue.
pub struct AcpSession {
    commands: mpsc::UnboundedSender<AgentCommand>,
    events: broadcast::Sender<AgentEvent>,
}

/// Present a foreign ACP agent as a `UnitKind::Engine` managed unit identified by `id`, journaling
/// its finished transcript blocks + lifecycle (sealed per turn) into `journal` when provided.
pub fn acp_unit(id: UnitId, launch: AcpLaunch, journal: Option<Arc<JournalFeeder>>) -> AgentUnit {
    AgentUnit::start_journaled(id, journal, move |host: Arc<dyn HostRequestHandler>| {
        AcpSession::connect(launch, host)
    })
}

impl AcpSession {
    /// Spawn the ACP connection driver and return the live session. `host` answers the symmetric
    /// permission callbacks the agent raises.
    pub fn connect(launch: AcpLaunch, host: Arc<dyn HostRequestHandler>) -> Arc<dyn Section17Session> {
        let (commands, command_rx) = mpsc::unbounded_channel::<AgentCommand>();
        let (events, _) = broadcast::channel::<AgentEvent>(256);
        let seq = Arc::new(AtomicU64::new(0));

        tokio::spawn(drive(launch, host, events.clone(), seq, command_rx));

        Arc::new(AcpSession { commands, events })
    }
}

#[async_trait]
impl Section17Session for AcpSession {
    async fn submit(&self, cmd: AgentCommand) {
        let _ = self.commands.send(cmd);
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }
}

/// Drive the ACP connection for the lifetime of the session: initialize, open a session, then relay
/// each queued §17 command as a prompt / cancel until the queue closes or `Shutdown` arrives.
async fn drive(
    launch: AcpLaunch,
    host: Arc<dyn HostRequestHandler>,
    events: broadcast::Sender<AgentEvent>,
    seq: Arc<AtomicU64>,
    mut command_rx: mpsc::UnboundedReceiver<AgentCommand>,
) {
    let (agent, cwd) = launch.into_agent();

    // Notification handler: stream `session/update`s up as §17 events.
    let notif_events = events.clone();
    let notif_seq = seq.clone();
    // Permission handler: bridge `session/request_permission` to a §17 blocking host request.
    let perm_host = host.clone();
    let perm_req_ids = Arc::new(AtomicU64::new(1));

    // Loop body captures: drive prompts and emit turn lifecycle events.
    let loop_events = events.clone();
    let loop_seq = seq.clone();

    let result = Client
        .builder()
        .name("daemon-acp")
        .on_receive_notification(
            move |notif: SessionNotification, _cx| {
                let events = notif_events.clone();
                let seq = notif_seq.clone();
                async move {
                    for ev in map_update(notif.update, &seq) {
                        let _ = events.send(ev);
                    }
                    Ok(())
                }
            },
            agent_client_protocol::on_receive_notification!(),
        )
        .on_receive_request(
            move |req: RequestPermissionRequest,
                  responder: Responder<RequestPermissionResponse>,
                  _cx| {
                let host = perm_host.clone();
                let req_ids = perm_req_ids.clone();
                async move {
                    let outcome = resolve_permission(req, &host, &req_ids).await;
                    responder.respond(RequestPermissionResponse::new(outcome))
                }
            },
            agent_client_protocol::on_receive_request!(),
        )
        .connect_with(agent, |cx: ConnectionTo<Agent>| async move {
            cx.send_request(InitializeRequest::new(ProtocolVersion::V1))
                .block_task()
                .await?;
            let session = cx
                .send_request(NewSessionRequest::new(cwd))
                .block_task()
                .await?;
            let session_id = session.session_id;

            while let Some(cmd) = command_rx.recv().await {
                match cmd {
                    AgentCommand::StartTurn { input, .. } => {
                        let _ = loop_events.send(AgentEvent::TurnStarted {
                            seq: loop_seq.fetch_add(1, Ordering::Relaxed),
                            trigger: TurnTrigger::User,
                        });
                        let summary = match cx
                            .send_request(PromptRequest::new(
                                session_id.clone(),
                                vec![ContentBlock::Text(TextContent::new(input.text))],
                            ))
                            .block_task()
                            .await
                        {
                            Ok(PromptResponse { stop_reason, .. }) => map_stop(stop_reason),
                            Err(_) => TurnSummary::ended(EndReason::Failed),
                        };
                        let _ = loop_events.send(AgentEvent::TurnFinished {
                            seq: loop_seq.fetch_add(1, Ordering::Relaxed),
                            summary,
                        });
                    }
                    AgentCommand::Steer { text, .. } => {
                        let _ = loop_events.send(AgentEvent::TurnStarted {
                            seq: loop_seq.fetch_add(1, Ordering::Relaxed),
                            trigger: TurnTrigger::Steer,
                        });
                        let summary = match cx
                            .send_request(PromptRequest::new(
                                session_id.clone(),
                                vec![ContentBlock::Text(TextContent::new(text))],
                            ))
                            .block_task()
                            .await
                        {
                            Ok(PromptResponse { stop_reason, .. }) => map_stop(stop_reason),
                            Err(_) => TurnSummary::ended(EndReason::Failed),
                        };
                        let _ = loop_events.send(AgentEvent::TurnFinished {
                            seq: loop_seq.fetch_add(1, Ordering::Relaxed),
                            summary,
                        });
                    }
                    AgentCommand::Interrupt { .. } => {
                        let _ = cx.send_notification(CancelNotification::new(session_id.clone()));
                    }
                    // No ACP analogue for a read-only snapshot; the live event stream is the view.
                    AgentCommand::Snapshot { .. } => {}
                    AgentCommand::Shutdown => break,
                    _ => {}
                }
            }
            Ok(())
        })
        .await;

    if let Err(err) = result {
        tracing::warn!(error = %err, "acp connection ended with error");
    }
}

/// Map an ACP `session/update` to zero or more §17 [`AgentEvent`]s (assigning the monotonic `seq`).
fn map_update(update: SessionUpdate, seq: &AtomicU64) -> Vec<AgentEvent> {
    let next = || seq.fetch_add(1, Ordering::Relaxed);
    match update {
        SessionUpdate::AgentMessageChunk(chunk) => match text_of(&chunk) {
            Some(text) => vec![AgentEvent::TextDelta { seq: next(), text }],
            None => Vec::new(),
        },
        SessionUpdate::AgentThoughtChunk(chunk) => match text_of(&chunk) {
            Some(text) => vec![AgentEvent::ReasoningDelta { seq: next(), text }],
            None => Vec::new(),
        },
        SessionUpdate::ToolCall(call) => vec![tool_started(call, next())],
        SessionUpdate::ToolCallUpdate(update) => tool_finished(update, seq).into_iter().collect(),
        // User echoes, plans, modes, usage, etc. have no §17 leaf projection here.
        _ => Vec::new(),
    }
}

fn text_of(chunk: &ContentChunk) -> Option<String> {
    match &chunk.content {
        ContentBlock::Text(t) => Some(t.text.clone()),
        _ => None,
    }
}

fn tool_started(call: ToolCall, seq: u64) -> AgentEvent {
    let call_id = call.tool_call_id.0.to_string();
    let detail = call
        .raw_input
        .as_ref()
        .map(|v| ToolDetail::new(call.title.clone(), cbor_bytes(v)));
    let args_summary = call.raw_input.as_ref().map(summarize).unwrap_or_default();
    AgentEvent::ToolStarted {
        seq,
        call: ToolCallView {
            call_id,
            name: call.title,
            args_summary,
            detail,
        },
    }
}

fn tool_finished(update: ToolCallUpdate, seq: &AtomicU64) -> Option<AgentEvent> {
    // Only a terminal status graduates to a §17 tool result; intermediate updates are streaming
    // noise the coarse surface ignores (the rich view tracks them off the live stream).
    let status = update.fields.status?;
    let ok = match status {
        ToolCallStatus::Completed => true,
        ToolCallStatus::Failed => false,
        ToolCallStatus::Pending | ToolCallStatus::InProgress => return None,
        _ => return None,
    };
    let detail = update
        .fields
        .raw_output
        .as_ref()
        .map(|v| ToolDetail::new("tool_result", cbor_bytes(v)));
    let summary = update
        .fields
        .raw_output
        .as_ref()
        .map(summarize)
        .or(update.fields.title)
        .unwrap_or_default();
    Some(AgentEvent::ToolFinished {
        seq: seq.fetch_add(1, Ordering::Relaxed),
        result: ToolResultView {
            call_id: update.tool_call_id.0.to_string(),
            ok,
            summary,
            detail,
        },
    })
}

/// Map an ACP [`StopReason`] to a §17 [`TurnSummary`].
fn map_stop(reason: StopReason) -> TurnSummary {
    let end_reason = match reason {
        StopReason::EndTurn => EndReason::Completed,
        StopReason::MaxTokens | StopReason::MaxTurnRequests => EndReason::BudgetExhausted,
        StopReason::Cancelled => EndReason::Interrupted,
        StopReason::Refusal => EndReason::Failed,
        _ => EndReason::Completed,
    };
    TurnSummary::ended(end_reason)
}

/// Bridge an ACP permission request to a §17 blocking host request and map the host's decision back
/// to a permission option (approve → first allow option; deny → first reject option).
async fn resolve_permission(
    req: RequestPermissionRequest,
    host: &Arc<dyn HostRequestHandler>,
    req_ids: &AtomicU64,
) -> RequestPermissionOutcome {
    let prompt = req
        .tool_call
        .fields
        .title
        .clone()
        .unwrap_or_else(|| format!("permission for tool {}", req.tool_call.tool_call_id.0));
    let request_id = ReqId(req_ids.fetch_add(1, Ordering::Relaxed));
    let response = host
        .request(HostRequest {
            request_id,
            kind: HostRequestKind::Approval { prompt },
        })
        .await;
    let approved = matches!(response.body, HostResponseBody::Approved(true));

    let wanted = |kind: &PermissionOptionKind| {
        if approved {
            matches!(
                kind,
                PermissionOptionKind::AllowOnce | PermissionOptionKind::AllowAlways
            )
        } else {
            matches!(
                kind,
                PermissionOptionKind::RejectOnce | PermissionOptionKind::RejectAlways
            )
        }
    };

    match req.options.iter().find(|opt| wanted(&opt.kind)) {
        Some(opt) => RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            opt.option_id.clone(),
        )),
        None => RequestPermissionOutcome::Cancelled,
    }
}

/// CBOR-encode an opaque JSON payload for a [`ToolDetail`] body (CBOR by convention).
fn cbor_bytes(value: &serde_json::Value) -> Vec<u8> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).expect("cbor-encode tool detail");
    buf
}

/// A short, non-secret human summary of a JSON payload for the coarse management view.
fn summarize(value: &serde_json::Value) -> String {
    let mut s = value.to_string();
    const MAX: usize = 200;
    if s.len() > MAX {
        s.truncate(MAX);
        s.push('…');
    }
    s
}
