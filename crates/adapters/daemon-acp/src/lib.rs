// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-acp` — the Agent Client Protocol (ACP) foreign-agent adapter.
//!
//! ACP is JSON-RPC 2.0 over newline-delimited stdio, and unlike Claude-Code `stream-json` it is
//! **symmetric**: the agent calls *back* into the client for services (permission prompts, and —
//! when advertised — filesystem / terminal access). This crate bridges an ACP agent to the daemon's
//! §17 session seam ([`daemon_host::AgentSession`]) so it presents to the orchestrator as an
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
    CancelNotification, ContentBlock, ContentChunk, EnvVariable, InitializeRequest, McpServer,
    McpServerStdio, NewSessionRequest, PermissionOptionKind, PromptRequest, PromptResponse,
    RequestPermissionOutcome, RequestPermissionRequest, RequestPermissionResponse,
    SelectedPermissionOutcome, SessionConfigKind, SessionConfigOption, SessionConfigOptionCategory,
    SessionConfigSelectOption, SessionConfigSelectOptions, SessionId, SessionNotification,
    SessionUpdate, SetSessionConfigOptionRequest, StopReason, TextContent, ToolCall,
    ToolCallStatus, ToolCallUpdate,
};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::{AcpAgent, Agent, Client, ConnectionTo, Responder};
use async_trait::async_trait;
use daemon_common::env_policy::EnvPolicy;
use daemon_common::{ReqId, UnitId};
use daemon_host::{AgentSession, AgentUnit, JournalFeeder};
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
    /// The node-validated profile model to steer the agent to (Layer 1), applied best-effort after
    /// `session/new` via `session/set_config_option` when the agent advertises a `Model` selector.
    /// `None` leaves the agent on its own default model. This is *not* an arbitrary spawn input:
    /// only the model string flows here — the launch recipe still comes from the catalog by name.
    pub desired_model: Option<String>,
    /// The declared env policy for the agent subprocess (Cluster E): always
    /// [`EnvPolicy::InheritFull`] — an ACP agent is a trusted foreign-engine node component that
    /// inherits the full daemon env by design (provider keys etc.), with `env` extras added on
    /// top. `Clean` is **not currently representable** here: the `agent_client_protocol`
    /// transport owns the actual spawn and exposes no env-clearing hook, so this field is private
    /// with no `Clean` constructor (declaration-only enforcement until upstream grows one).
    policy: EnvPolicy,
}

impl AcpLaunch {
    /// Construct a launch spec for `program`, defaulting the cwd to the current directory.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env: Vec::new(),
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/")),
            desired_model: None,
            policy: EnvPolicy::InheritFull,
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

    /// Set the desired model to steer the agent to (best-effort; see [`AcpLaunch::desired_model`]).
    pub fn model(mut self, model: Option<String>) -> Self {
        self.desired_model = model;
        self
    }

    fn into_agent(self) -> (AcpAgent, PathBuf) {
        // Declared env policy (Cluster E), stated so the spawn is auditable even though the
        // `agent_client_protocol` transport owns the actual process spawn: `InheritFull` — the
        // library-spawned agent inherits the full daemon env, and `self.env` extras are passed
        // through below exactly as before.
        match &self.policy {
            EnvPolicy::InheritFull => { /* the transport's spawn inherits; extras follow */ }
            EnvPolicy::Clean { .. } => {
                // No constructor produces `Clean` for ACP (see the `policy` field docs): the
                // transport exposes no env-clearing hook, so a Clean ACP launch is
                // unrepresentable today.
                unreachable!("AcpLaunch env policy is always InheritFull (no Clean constructor)")
            }
        }
        let name = self
            .program
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "acp-agent".to_string());
        let stdio = McpServerStdio::new(name, self.program).args(self.args).env(
            self.env
                .into_iter()
                .map(|(k, v)| EnvVariable::new(k, v))
                .collect(),
        );
        (AcpAgent::new(McpServer::Stdio(stdio)), self.cwd)
    }
}

/// The verified outcome of an ACP `initialize` handshake against a candidate binary (I7 discovery):
/// the agent answered `initialize`, so it *is* an ACP agent, and reported this protocol version +
/// capabilities. Captured by [`probe`] from the `InitializeResponse` the live [`drive`] path
/// otherwise discards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcpProbe {
    /// The protocol version the agent reported (rendered).
    pub protocol_version: String,
    /// The agent capabilities advertised at `initialize`, flattened to opaque key/value pairs.
    pub capabilities: Vec<(String, String)>,
}

/// The discovery/probe timeout: a candidate that does not complete the ACP `initialize` handshake
/// within this window is treated as "not an ACP agent" (no hang on a mis-curated binary).
const PROBE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Probe a candidate by spawning it and attempting the ACP `initialize` handshake, capturing the
/// `InitializeResponse` (the "is this an ACP agent?" confirmation + its metadata). Returns `None` if
/// the binary fails to spawn, does not speak ACP, or exceeds [`PROBE_TIMEOUT`]. This is the half of
/// the connection [`drive`] discards — surfaced standalone for catalog discovery.
pub async fn probe(launch: AcpLaunch) -> Option<AcpProbe> {
    let (agent, _cwd) = launch.into_agent();
    let captured = Arc::new(std::sync::Mutex::new(None));
    let sink = captured.clone();
    let run = Client.builder().name("daemon-acp-probe").connect_with(
        agent,
        move |cx: ConnectionTo<Agent>| {
            let sink = sink.clone();
            async move {
                let resp = cx
                    .send_request(InitializeRequest::new(ProtocolVersion::V1))
                    .block_task()
                    .await?;
                // Serialize the whole response generically so we need not track the exact schema:
                // pull the protocol version + flatten capabilities into opaque key/value pairs.
                if let Ok(value) = serde_json::to_value(&resp) {
                    *sink.lock().unwrap() = Some(value);
                }
                Ok(())
            }
        },
    );
    match tokio::time::timeout(PROBE_TIMEOUT, run).await {
        Ok(Ok(())) => {}
        // A connection error (failed spawn / not ACP) or a timeout: not a confirmed ACP agent.
        _ => return None,
    }
    let value = captured.lock().unwrap().take()?;
    Some(AcpProbe {
        protocol_version: extract_protocol_version(&value),
        capabilities: flatten_capabilities(&value),
    })
}

/// Best-effort protocol-version extraction from a serialized `InitializeResponse` (camelCase or
/// snake_case key), rendered to a string.
fn extract_protocol_version(value: &serde_json::Value) -> String {
    for key in ["protocolVersion", "protocol_version"] {
        if let Some(v) = value.get(key) {
            return match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
        }
    }
    String::new()
}

/// Flatten the `agentCapabilities` object of a serialized `InitializeResponse` into opaque key/value
/// pairs (one level deep). Schema-agnostic so a capabilities-shape change does not break discovery.
fn flatten_capabilities(value: &serde_json::Value) -> Vec<(String, String)> {
    let caps = value
        .get("agentCapabilities")
        .or_else(|| value.get("agent_capabilities"));
    let Some(serde_json::Value::Object(map)) = caps else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (k, v) in map {
        match v {
            serde_json::Value::Object(inner) => {
                for (ik, iv) in inner {
                    out.push((format!("{k}.{ik}"), scalar_string(iv)));
                }
            }
            other => out.push((k.clone(), scalar_string(other))),
        }
    }
    out
}

fn scalar_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Resolve a program name on `$PATH` (or as a direct path), returning the resolved path when found.
/// The "is it installed?" half of discovery, independent of whether it actually speaks ACP.
fn which(program: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(program);
    if p.is_absolute() || program.contains('/') {
        return p.exists().then(|| p.to_path_buf());
    }
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(program))
        .find(|candidate| candidate.is_file())
}

/// One row of the curated direct-binary recipe table (I7): a display name, the program to exec, the
/// protocol-mode args, and the wire protocol the daemon drives it with.
struct Curated {
    /// Catalog display name / key (e.g. `"gemini"`, `"claude"`).
    name: &'static str,
    /// The program to exec (resolved on `$PATH`).
    program: &'static str,
    /// Protocol-mode invocation args (best-effort; see the table doc-comment).
    args: &'static [&'static str],
    /// The wire protocol this agent speaks (selects the adapter + whether discovery handshakes).
    protocol: daemon_api::AgentProtocol,
}

/// The curated **direct-binary** foreign-agent recipe table (I7): agents whose own CLI speaks a wire
/// protocol the daemon drives on stdio, so they auto-detect on `$PATH`. Two protocols are covered:
///
/// * [`AgentProtocol::Acp`](daemon_api::AgentProtocol::Acp) — a symmetric JSON-RPC handshake, so
///   discovery **confirms** the agent by completing an `initialize` exchange (fills `version` /
///   `capabilities` → *verified*). The acp-mode flags are best-effort and self-correcting: a
///   mis-curated flag (or an agent that doesn't actually speak ACP) simply means the probe does not
///   confirm, and the entry shows installed-on-PATH but *unverified* rather than failing.
/// * [`AgentProtocol::StreamJson`](daemon_api::AgentProtocol::StreamJson) — the Claude-Code NDJSON
///   dialect (also Amp). There is **no handshake**, so discovery is PATH-only and the entry is always
///   surfaced *unverified* (`version` stays `None`); the `args` are the streaming-input invocation
///   used when the agent is actually spawned.
///
/// Adapter-wrapped agents that only speak ACP through a separate shim (`npx pi-acp`,
/// `@agentclientprotocol/claude-agent-acp`, `codex-acp`, ...) and IDE-embedded agents are **not**
/// listed here — they stay manual-register entries (`source = Manual`).
const CURATED: &[Curated] = &[
    // --- ACP direct-binary agents (best-effort acp-mode flags, confirmed via `initialize`) -------
    acp("gemini", "gemini", &["--experimental-acp"]),
    acp("qwen", "qwen", &["--acp"]),
    acp("goose", "goose", &["acp"]),
    acp("opencode", "opencode", &["acp"]),
    acp("codex", "codex", &["acp"]),
    acp("kimi", "kimi", &["acp"]),
    acp("crow-cli", "crow-cli", &[]),
    acp("cursor-agent", "cursor-agent", &["acp"]),
    acp("copilot", "copilot", &["--acp", "--stdio"]),
    acp("droid", "droid", &["exec", "--output-format", "acp"]),
    acp("iflow", "iflow", &["--experimental-acp"]),
    acp("qoder", "qodercli", &["--acp"]),
    acp("kilocode", "kilocode", &["acp"]),
    acp("mistral-vibe", "mistral-vibe", &[]),
    acp("junie", "junie", &[]),
    acp("eca", "eca", &[]),
    // --- stream-json direct-binary agents (no handshake → always surfaced unverified) -------------
    stream_json(
        "claude",
        "claude",
        &[
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ],
    ),
    stream_json("amp", "amp", &["--stream-json", "--stream-json-input"]),
];

/// Build an [`AgentProtocol::Acp`](daemon_api::AgentProtocol::Acp) curated row (const-fn so the
/// table stays a compile-time constant).
const fn acp(name: &'static str, program: &'static str, args: &'static [&'static str]) -> Curated {
    Curated {
        name,
        program,
        args,
        protocol: daemon_api::AgentProtocol::Acp,
    }
}

/// Build an [`AgentProtocol::StreamJson`](daemon_api::AgentProtocol::StreamJson) curated row.
const fn stream_json(
    name: &'static str,
    program: &'static str,
    args: &'static [&'static str],
) -> Curated {
    Curated {
        name,
        program,
        args,
        protocol: daemon_api::AgentProtocol::StreamJson,
    }
}

/// Build an unprobed [`daemon_api::AgentEntry`] from a curated row: the recipe, its protocol, and a
/// cheap PATH `installed` check. Shared by `discover` (which then handshakes ACP entries) and
/// `builtin` (which never handshakes) so both stay in lockstep on recipe + protocol shape.
fn curated_entry(row: &Curated) -> daemon_api::AgentEntry {
    let recipe = daemon_api::AgentRecipe {
        program: Some(row.program.to_string()),
        args: row.args.iter().map(|s| (*s).to_string()).collect(),
        env: Vec::new(),
        endpoint: None,
    };
    daemon_api::AgentEntry {
        name: row.name.to_string(),
        installed: recipe_installed(&recipe),
        recipe,
        source: daemon_api::AgentSource::Builtin,
        protocol: row.protocol,
        version: None,
        capabilities: Vec::new(),
    }
}

/// The server-side ACP discoverer (I7): probes the curated direct-binary recipe table on `$PATH` via
/// the ACP `initialize` handshake. Implements [`daemon_host::AgentDiscovery`] so the host's
/// `agent_discover` / `agent_register` ops can confirm + enrich entries without `daemon-host` linking
/// the ACP runtime (which would be a dependency cycle — `daemon-acp` depends on `daemon-host`).
/// Stream-json entries are PATH-probed only (no `initialize` handshake).
#[derive(Clone, Debug, Default)]
pub struct AcpDiscoverer;

impl AcpDiscoverer {
    /// A fresh discoverer over the curated recipe table.
    pub fn new() -> Self {
        Self
    }
}

/// Build an [`AcpLaunch`] from a wire [`daemon_api::AgentRecipe`]'s program + args + env (stdio
/// agents only; an endpoint recipe has no local binary to spawn). Public so the node's
/// foreign-engine resolution (profile `engine = Foreign{agent}` -> catalog recipe -> spawn) reuses
/// the exact mapping discovery uses.
pub fn launch_from_recipe(recipe: &daemon_api::AgentRecipe) -> Option<AcpLaunch> {
    let program = recipe.program.as_ref()?;
    Some(
        AcpLaunch::new(program.clone())
            .args(recipe.args.clone())
            .env(recipe.env.clone()),
    )
}

/// Whether a recipe's stdio program currently resolves on `$PATH` (or as a direct path) — the
/// cheap "is it installed *right now*?" re-check the foreign-engine spawn path runs, since
/// installed-ness can change between profile validation and spawn. `false` for endpoint-only
/// recipes (no local binary).
pub fn recipe_installed(recipe: &daemon_api::AgentRecipe) -> bool {
    recipe
        .program
        .as_deref()
        .is_some_and(|program| which(program).is_some())
}

#[async_trait]
impl daemon_host::AgentDiscovery for AcpDiscoverer {
    async fn discover(&self) -> Vec<daemon_api::AgentEntry> {
        let mut out = Vec::with_capacity(CURATED.len());
        for row in CURATED {
            let mut entry = curated_entry(row);
            // Confirm ACP agents (and capture metadata) via the `initialize` handshake only when the
            // binary is present. Stream-json agents have no handshake, so their `version`/`caps`
            // stay empty — installed-on-PATH is the whole probe, and they surface as unverified.
            if entry.installed && row.protocol == daemon_api::AgentProtocol::Acp {
                if let Some(launch) = launch_from_recipe(&entry.recipe) {
                    if let Some(p) = probe(launch).await {
                        entry.version = Some(p.protocol_version);
                        entry.capabilities = p.capabilities;
                    }
                }
            }
            out.push(entry);
        }
        out
    }

    async fn probe(&self, mut entry: daemon_api::AgentEntry) -> daemon_api::AgentEntry {
        // The installed check is protocol-independent (a cheap PATH probe on stdio recipes); the
        // `initialize` handshake is ACP-only — a stream-json agent has no handshake, so its
        // version/caps stay empty (installed-on-PATH is the whole probe).
        if let Some(program) = entry.recipe.program.as_deref() {
            entry.installed = which(program).is_some();
        }
        if entry.protocol != daemon_api::AgentProtocol::Acp {
            return entry;
        }
        if let Some(launch) = launch_from_recipe(&entry.recipe) {
            if let Some(p) = probe(launch).await {
                entry.version = Some(p.protocol_version);
                entry.capabilities = p.capabilities;
            }
        }
        entry
    }

    fn builtin(&self, name: &str) -> Option<daemon_api::AgentEntry> {
        // Recipe + PATH check only — deliberately NO initialize probe, so the validation /
        // spawn-resolution fast path never spawns candidate processes. The protocol comes from the
        // curated row (so a stream-json builtin like `claude`/`amp` is spawnable-by-name via the
        // fleet's StreamJson branch, not just ACP entries).
        let row = CURATED.iter().find(|row| row.name == name)?;
        Some(curated_entry(row))
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
    pub fn connect(launch: AcpLaunch, host: Arc<dyn HostRequestHandler>) -> Arc<dyn AgentSession> {
        let (commands, command_rx) = mpsc::unbounded_channel::<AgentCommand>();
        let (events, _) = broadcast::channel::<AgentEvent>(256);
        let seq = Arc::new(AtomicU64::new(0));

        tokio::spawn(drive(launch, host, events.clone(), seq, command_rx));

        Arc::new(AcpSession { commands, events })
    }
}

#[async_trait]
impl AgentSession for AcpSession {
    async fn submit(&self, cmd: AgentCommand) {
        let _ = self.commands.send(cmd);
    }

    fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.events.subscribe()
    }

    /// Foreign ACP agents are not rewindable: ACP has no truncate-at-anchor primitive and the agent
    /// owns its own conversation state, so the daemon cannot make it forget post-anchor turns.
    fn rewindable(&self) -> bool {
        false
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
    let desired_model = launch.desired_model.clone();
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

            // Layer 1: best-effort model selection over the agent's advertised config options.
            // Any failure or missing selector is logged and the session proceeds on the agent's
            // default model — model selection never aborts the session.
            if let Some(wanted) = desired_model.as_deref() {
                apply_model(&cx, &session_id, session.config_options.as_deref(), wanted).await;
            }

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
                    // Conversation rewind is unsupported for foreign ACP agents: ACP has no
                    // truncate-at-anchor primitive (session/load replays the full history,
                    // session/fork is unstable and forks the whole context, session/resume does not
                    // truncate), and the agent — not the daemon — owns the conversation state, so the
                    // daemon cannot make it forget post-anchor turns. Sessions advertise this up front
                    // via `SessionInfo::rewindable = false`, so a client never offers rewind here; if
                    // one is submitted anyway it is dropped (no fake/partial rewind).
                    AgentCommand::RewindTo { request_id, .. } => {
                        tracing::warn!(
                            request_id = request_id.0,
                            "RewindTo is unsupported for a foreign ACP agent (not rewindable); dropping"
                        );
                    }
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

/// Layer 1 best-effort model selection: after `session/new`, if the agent advertised a
/// `Model`-category `Select` config option, try to switch it to the node-validated profile model
/// `wanted` via `session/set_config_option`. Matching prefers the option value's id, then its
/// display name case-insensitively. Every outcome is logged (`applied` / `already-current` /
/// `not-offered` / `no-selector`) and the function never returns an error that aborts the session —
/// on any miss or failure the agent keeps its own default model.
async fn apply_model(
    cx: &ConnectionTo<Agent>,
    session_id: &SessionId,
    config_options: Option<&[SessionConfigOption]>,
    wanted: &str,
) {
    // Locate the Model-category Select selector, if the agent advertised one.
    let selector = config_options.into_iter().flatten().find_map(|opt| {
        if opt.category != Some(SessionConfigOptionCategory::Model) {
            return None;
        }
        match &opt.kind {
            SessionConfigKind::Select(select) => Some((opt, select)),
            _ => None,
        }
    });
    let Some((option, select)) = selector else {
        tracing::info!(
            requested_model = wanted,
            outcome = "no-selector",
            "foreign ACP agent advertised no Model selector; keeping its default model"
        );
        return;
    };

    // id-exact first, then a case-insensitive display-name fallback.
    let matched = select_options(&select.options)
        .find(|v| v.value.0.as_ref() == wanted)
        .or_else(|| select_options(&select.options).find(|v| v.name.eq_ignore_ascii_case(wanted)));
    let Some(value) = matched else {
        tracing::info!(
            requested_model = wanted,
            outcome = "not-offered",
            "foreign ACP agent's Model selector does not offer the requested model; keeping default"
        );
        return;
    };

    if value.value == select.current_value {
        tracing::info!(
            requested_model = wanted,
            model = %value.value.0,
            outcome = "already-current",
            "foreign ACP agent's Model selector is already on the requested model"
        );
        return;
    }

    match cx
        .send_request(SetSessionConfigOptionRequest::new(
            session_id.clone(),
            option.id.clone(),
            value.value.clone(),
        ))
        .block_task()
        .await
    {
        Ok(resp) => {
            // Read back the resulting current_value for the same option, falling back to the value
            // we requested when the agent does not echo the option in its response.
            let current = resp
                .config_options
                .iter()
                .find(|o| o.id == option.id)
                .and_then(|o| match &o.kind {
                    SessionConfigKind::Select(s) => Some(s.current_value.0.to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| value.value.0.to_string());
            tracing::info!(
                requested_model = wanted,
                model = %value.value.0,
                current_value = %current,
                outcome = "applied",
                "selected model on foreign ACP agent"
            );
        }
        Err(err) => {
            tracing::warn!(
                requested_model = wanted,
                model = %value.value.0,
                error = %err,
                outcome = "error",
                "failed to set model on foreign ACP agent; keeping its default"
            );
        }
    }
}

/// Flatten a select option's (possibly grouped) choices into one iterator of value entries, so the
/// id/name matching in [`apply_model`] treats ungrouped and grouped selectors uniformly.
fn select_options(
    options: &SessionConfigSelectOptions,
) -> Box<dyn Iterator<Item = &SessionConfigSelectOption> + '_> {
    match options {
        SessionConfigSelectOptions::Ungrouped(list) => Box::new(list.iter()),
        SessionConfigSelectOptions::Grouped(groups) => {
            Box::new(groups.iter().flat_map(|g| g.options.iter()))
        }
        // `SessionConfigSelectOptions` is `#[non_exhaustive]`: an unknown future grouping simply
        // yields no matchable values, so the model stays `not-offered` (default model kept).
        _ => Box::new(std::iter::empty()),
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
            kind: HostRequestKind::Approval {
                prompt,
                allow_permanent_offered: false,
            },
        })
        .await;
    let approved = matches!(
        response.body,
        HostResponseBody::Approved { approved: true, .. }
    );

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

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_protocol::{HostRequest, HostResponse, HostResponseBody};

    struct NoopHost;

    #[async_trait]
    impl HostRequestHandler for NoopHost {
        async fn request(&self, req: HostRequest) -> HostResponse {
            HostResponse {
                request_id: req.request_id,
                body: HostResponseBody::Approved {
                    approved: false,
                    allow_permanent: false,
                    reason: None,
                },
            }
        }
    }

    /// A foreign ACP session is not rewindable: ACP has no truncate-at-anchor primitive, so the
    /// capability is `false` (the GUI/TUI reads this — via `SessionInfo::rewindable` / the unit — to
    /// hide rewind for ACP agents). Pure capability check; no agent process needs to speak ACP.
    #[tokio::test]
    async fn acp_session_is_not_rewindable() {
        // `/nonexistent-acp-agent` never connects, but `rewindable()` is a static capability that
        // does not depend on the connection, so the session object reports it immediately.
        let launch = AcpLaunch::new("/nonexistent-acp-agent");
        let session = AcpSession::connect(launch, Arc::new(NoopHost));
        assert!(
            !session.rewindable(),
            "foreign ACP sessions must advertise rewindable=false"
        );
    }
}
