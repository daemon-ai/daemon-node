// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-pytool-client` — the supervised client for the out-of-process **Python tool worker**.
//!
//! [`PyToolHost`] is to the Python tool worker what `MettaCoprocessor` is to the MeTTa worker and
//! `LocalProvider` is to the inference worker: it lazily spawns the worker child over a
//! length-framed [`CutChannel`](daemon_provision::CutChannel), speaks the
//! [`daemon_pytool::protocol`], respawns the worker after a crash / transport fault, and trips a
//! crash-loop "meltdown" to [`PyToolError::Fatal`] when restarts exceed a budget within a sliding
//! window.
//!
//! Two things differ from the metta client:
//! - **Concurrent in-flight calls.** A background reader task routes each reply to the matching
//!   request's [`oneshot`](tokio::sync::oneshot), so a model-emitted *parallel* tool batch runs
//!   concurrently in the worker (the metta client serializes every op behind one lock).
//! - **A proxy per tool.** [`discover`] spawns the worker once, lists its tools, and returns a
//!   [`PyToolProxy`] (`impl `[`daemon_core::Tool`]) for each — so Python tools register into the
//!   ordinary [`ToolRegistry`](daemon_core::ToolRegistry) and the engine never knows they are
//!   out-of-process. A proxy's `run()` issues a `CallTool` round-trip and maps the reply onto a
//!   [`ToolOutcome`](daemon_core::ToolOutcome) (content + `untrusted` fence + structured detail).

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use daemon_common::SessionId;
use daemon_core::{
    Tool, ToolCall, ToolConcurrency, ToolOutcome, ToolProvider, ToolProviderError, TurnCx,
};
use daemon_protocol::ToolDetail;
use daemon_provision::{
    ChildGuard, CutReader, CutWriter, Placement, PlacementSpec, ProcessProvisioner, Provisioner,
};
use daemon_pytool::protocol::{
    self, Command, Concurrency, ErrorClass, Event, ResultDetail, ToolManifest, PROTOCOL_VERSION,
};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;

/// Construction + tuning for a [`PyToolHost`]'s worker.
#[derive(Clone, Debug)]
pub struct PyToolConfig {
    /// The program to exec for the worker (e.g. a `python3` interpreter, or a standalone worker bin).
    pub program: PathBuf,
    /// Arguments passed to the worker (e.g. `["-m", "daemon_pytool", "--tools-dir", <dir>]`).
    pub args: Vec<String>,
    /// Extra environment variables set on the worker child (e.g. `PYTHONPATH`).
    pub env: Vec<(String, String)>,
    /// How long to wait for [`Event::Ready`] after spawning.
    pub spawn_timeout: Duration,
    /// How long to wait for a reply (discovery or a tool call) before declaring a transport fault.
    pub op_timeout: Duration,
    /// Crash-loop meltdown: max restarts allowed within [`PyToolConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which [`PyToolConfig::max_restarts`] is counted.
    pub restart_window: Duration,
}

impl PyToolConfig {
    /// A config running `program` with `args`, with sensible default timeouts + meltdown budget.
    pub fn new(program: impl Into<PathBuf>, args: Vec<String>) -> Self {
        Self {
            program: program.into(),
            args,
            env: Vec::new(),
            spawn_timeout: Duration::from_secs(30),
            op_timeout: Duration::from_secs(60),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// A classified Python-tool failure (mirrors [`ErrorClass`] + transport faults).
#[derive(Debug, thiserror::Error)]
pub enum PyToolError {
    /// A bad request (undecodable args, unknown tool) — the caller must fix it.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// The op / tool is not supported by this worker.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A transient transport/worker fault — a retry on a fresh worker may succeed.
    #[error("transient: {0}")]
    Transient(String),
    /// Unrecoverable (crash-loop meltdown, internal bug).
    #[error("fatal: {0}")]
    Fatal(String),
    /// The call was cancelled / timed out cooperatively.
    #[error("cancelled")]
    Cancelled,
}

impl PyToolError {
    fn from_class(class: ErrorClass, message: String) -> Self {
        match class {
            ErrorClass::BadRequest => PyToolError::BadRequest(message),
            ErrorClass::Unsupported => PyToolError::Unsupported(message),
            ErrorClass::Transient => PyToolError::Transient(message),
            ErrorClass::Fatal => PyToolError::Fatal(message),
            ErrorClass::Cancelled => PyToolError::Cancelled,
        }
    }

    /// Whether the failure warrants tearing down the worker so the next call respawns it.
    fn should_replace_worker(&self) -> bool {
        matches!(self, PyToolError::Transient(_) | PyToolError::Fatal(_))
    }
}

/// A tool's reply (the success payload of a `CallTool` round-trip).
#[derive(Clone, Debug)]
pub struct ToolReply {
    /// Whether the tool succeeded.
    pub ok: bool,
    /// The textual result content.
    pub content: String,
    /// An optional structured detail envelope.
    pub detail: Option<ResultDetail>,
    /// Whether the content is external/untrusted (the §12 pipeline fences it).
    pub untrusted: bool,
}

/// A supervised client over a single Python tool worker process.
pub struct PyToolHost {
    inner: Arc<Inner>,
}

struct Inner {
    cfg: PyToolConfig,
    worker: AsyncMutex<Option<Arc<Worker>>>,
    restarts: AsyncMutex<Vec<Instant>>,
    /// The tools discovered by the most recent spawn (offered to the model via the proxies).
    tools: Mutex<Vec<ToolManifest>>,
    next_id: AtomicU64,
}

impl PyToolHost {
    /// Build a host for `cfg`. The worker is spawned lazily (on [`discover`](PyToolHost::discover)
    /// or the first tool call).
    pub fn new(cfg: PyToolConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                cfg,
                worker: AsyncMutex::new(None),
                restarts: AsyncMutex::new(Vec::new()),
                tools: Mutex::new(Vec::new()),
                next_id: AtomicU64::new(1),
            }),
        }
    }

    /// Spawn the worker (if needed) and return the tools it most recently advertised.
    pub async fn discover(&self) -> Result<Vec<ToolManifest>, PyToolError> {
        self.ensure_worker().await?;
        Ok(self
            .inner
            .tools
            .lock()
            .expect("pytool tools poisoned")
            .clone())
    }

    /// Invoke a tool on the (lazily spawned) worker. Tears the worker down on a fault that warrants
    /// a fresh process so the next call respawns it.
    pub async fn call_tool(
        &self,
        call_id: &str,
        name: &str,
        args: &str,
        session_id: &str,
        deadline_ms: u64,
    ) -> Result<ToolReply, PyToolError> {
        // Per-call deadline: a caller-supplied `deadline_ms` wins; otherwise the worker is given the
        // transport watchdog (`op_timeout`) as an advisory budget, so a cooperative tool can self-
        // limit before the client tears the worker down. `0` here means "no explicit deadline".
        let deadline_ms = if deadline_ms == 0 {
            self.inner.cfg.op_timeout.as_millis() as u64
        } else {
            deadline_ms
        };
        let cmd = Command::CallTool {
            request_id: 0,
            call_id: call_id.to_string(),
            name: name.to_string(),
            args: args.to_string(),
            session_id: session_id.to_string(),
            deadline_ms,
        };
        match self.round_trip(cmd).await? {
            Event::Result {
                ok,
                content,
                detail,
                untrusted,
                ..
            } => Ok(ToolReply {
                ok,
                content,
                detail,
                untrusted,
            }),
            Event::Error { class, message, .. } => Err(PyToolError::from_class(class, message)),
            other => Err(PyToolError::Transient(format!(
                "unexpected reply to CallTool: {other:?}"
            ))),
        }
    }

    /// Best-effort: ask the worker to cancel an in-flight call. The worker still sends the call's
    /// terminal `Result`/`Error`; this just lets a cooperative tool stop early.
    pub async fn cancel(&self, call_id: &str) {
        let worker = { self.inner.worker.lock().await.clone() };
        if let Some(worker) = worker {
            let _ = worker
                .send(&Command::Cancel {
                    call_id: call_id.to_string(),
                })
                .await;
        }
    }

    /// Probe the worker (spawning it if needed); `Ok(())` if it answers `Pong`.
    pub async fn ping(&self) -> Result<(), PyToolError> {
        match self.round_trip(Command::Ping { request_id: 0 }).await? {
            Event::Pong { .. } => Ok(()),
            other => Err(PyToolError::Transient(format!(
                "unexpected reply to Ping: {other:?}"
            ))),
        }
    }

    /// Gracefully stop the worker (if any). Idempotent.
    pub async fn shutdown(&self) {
        let mut guard = self.inner.worker.lock().await;
        if let Some(worker) = guard.take() {
            worker.shutdown().await;
        }
    }

    /// Issue one request-bearing command, assigning its id centrally and routing the matching reply
    /// back through the worker's reader task. Replaces the worker on a fault that warrants it.
    async fn round_trip(&self, mut cmd: Command) -> Result<Event, PyToolError> {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        cmd.set_request_id(id);

        let worker = self.ensure_worker().await?;
        let result = worker.round_trip(id, &cmd, self.inner.cfg.op_timeout).await;

        if let Err(ref failure) = result {
            if failure.should_replace_worker() {
                self.replace_worker(&worker).await;
            }
        }
        result
    }

    /// Return the live worker, spawning a fresh one (under the meltdown budget) if there is none or
    /// the current one has died.
    async fn ensure_worker(&self) -> Result<Arc<Worker>, PyToolError> {
        let mut guard = self.inner.worker.lock().await;
        if let Some(worker) = guard.as_ref() {
            if worker.is_alive() {
                return Ok(worker.clone());
            }
            // The reader saw EOF: drop the dead handle before respawning.
            if let Some(dead) = guard.take() {
                dead.shutdown().await;
            }
        }
        self.enforce_restart_budget().await?;
        let worker = Worker::spawn(&self.inner.cfg).await?;
        *self.inner.tools.lock().expect("pytool tools poisoned") = worker.tools.clone();
        *guard = Some(worker.clone());
        Ok(worker)
    }

    /// Tear down `dead` if it is still the current worker (so the next call respawns it).
    async fn replace_worker(&self, dead: &Arc<Worker>) {
        let mut guard = self.inner.worker.lock().await;
        if let Some(cur) = guard.as_ref() {
            if Arc::ptr_eq(cur, dead) {
                let dead = guard.take().expect("worker present");
                dead.shutdown().await;
            }
        }
    }

    /// Enforce the crash-loop meltdown budget before a (re)spawn.
    async fn enforce_restart_budget(&self) -> Result<(), PyToolError> {
        let mut restarts = self.inner.restarts.lock().await;
        let now = Instant::now();
        restarts.retain(|t| now.duration_since(*t) < self.inner.cfg.restart_window);
        if restarts.len() as u32 >= self.inner.cfg.max_restarts {
            return Err(PyToolError::Fatal(format!(
                "python tool worker crash-loop: {} restarts within {:?}",
                restarts.len(),
                self.inner.cfg.restart_window
            )));
        }
        restarts.push(now);
        Ok(())
    }
}

/// A live worker process: the framed writer, the per-request reply routing table fed by a reader
/// task, the child guard, and the tools it advertised at spawn.
struct Worker {
    writer: CutWriter,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Event>>>>,
    alive: Arc<AtomicBool>,
    child: AsyncMutex<ChildGuard>,
    reader: Mutex<Option<JoinHandle<()>>>,
    tools: Vec<ToolManifest>,
}

impl Worker {
    /// Spawn the worker, complete the `Ready` + `ListTools` handshake, and start the reply router.
    async fn spawn(cfg: &PyToolConfig) -> Result<Arc<Worker>, PyToolError> {
        let spec = PlacementSpec {
            program: cfg.program.clone(),
            args: cfg.args.clone(),
            env: cfg.env.clone(),
        };
        let session = SessionId::new("daemon-pytool-worker");
        let Placement { channel, child } = ProcessProvisioner::new()
            .place(&session, spec)
            .await
            .map_err(|e| PyToolError::Transient(format!("spawn python tool worker: {e}")))?;
        let (writer, mut reader) = channel.split();

        // Handshake step 1: the worker's unsolicited `Ready`.
        match tokio::time::timeout(cfg.spawn_timeout, reader.recv()).await {
            Err(_) => return Err(PyToolError::Transient("worker spawn timed out".into())),
            Ok(None) => return Err(PyToolError::Transient("worker exited before ready".into())),
            Ok(Some(bytes)) => match protocol::decode::<Event>(&bytes) {
                Ok(Event::Ready {
                    worker,
                    sdk_version,
                    protocol_version,
                }) => {
                    if protocol_version != PROTOCOL_VERSION {
                        tracing::warn!(
                            worker_version = protocol_version,
                            client_version = PROTOCOL_VERSION,
                            "python tool worker protocol version mismatch; proceeding"
                        );
                    }
                    tracing::info!(worker = %worker, sdk = %sdk_version, "python tool worker ready");
                }
                Ok(Event::Error { class, message, .. }) => {
                    return Err(PyToolError::from_class(class, message))
                }
                Ok(other) => {
                    return Err(PyToolError::Fatal(format!("expected Ready, got {other:?}")))
                }
                Err(e) => return Err(PyToolError::Fatal(format!("undecodable ready frame: {e}"))),
            },
        }

        // Notify the worker of the negotiated protocol version (fire-and-forget).
        let init = protocol::encode(&Command::Initialize {
            protocol_version: PROTOCOL_VERSION,
        })
        .map_err(|e| PyToolError::Fatal(format!("encode init: {e}")))?;
        writer
            .send(&init)
            .await
            .map_err(|e| PyToolError::Transient(format!("send init: {e}")))?;

        // Handshake step 2: discover tools.
        let tools = Self::list_tools(&writer, &mut reader, cfg.op_timeout).await?;

        // Start the reply router over the remaining frames.
        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Event>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let reader_pending = pending.clone();
        let reader_alive = alive.clone();
        let reader_task = tokio::spawn(async move {
            while let Some(bytes) = reader.recv().await {
                match protocol::decode::<Event>(&bytes) {
                    Ok(event) => route_reply(event, &reader_pending),
                    Err(e) => tracing::warn!(error = %e, "python tool worker: undecodable frame"),
                }
            }
            // EOF: the worker died. Mark it dead and fail every in-flight call (dropping the senders
            // makes their awaits return a recv error, which the round-trip maps to a transient fault).
            reader_alive.store(false, Ordering::Relaxed);
            reader_pending
                .lock()
                .expect("pytool pending poisoned")
                .clear();
        });

        Ok(Arc::new(Worker {
            writer,
            pending,
            alive,
            child: AsyncMutex::new(child),
            reader: Mutex::new(Some(reader_task)),
            tools,
        }))
    }

    /// Send `ListTools` and read frames until the matching `Tools` reply (handshake, pre-router).
    async fn list_tools(
        writer: &CutWriter,
        reader: &mut CutReader,
        timeout: Duration,
    ) -> Result<Vec<ToolManifest>, PyToolError> {
        let request_id = 0;
        let bytes = protocol::encode(&Command::ListTools { request_id })
            .map_err(|e| PyToolError::Fatal(format!("encode list_tools: {e}")))?;
        writer
            .send(&bytes)
            .await
            .map_err(|e| PyToolError::Transient(format!("send list_tools: {e}")))?;
        loop {
            match tokio::time::timeout(timeout, reader.recv()).await {
                Err(_) => return Err(PyToolError::Transient("list_tools timed out".into())),
                Ok(None) => {
                    return Err(PyToolError::Transient(
                        "worker exited during discovery".into(),
                    ))
                }
                Ok(Some(bytes)) => match protocol::decode::<Event>(&bytes) {
                    Ok(Event::Tools { tools, .. }) => return Ok(tools),
                    Ok(Event::Error { class, message, .. }) => {
                        return Err(PyToolError::from_class(class, message))
                    }
                    // Ignore any unrelated frame during the handshake.
                    Ok(_) => {}
                    Err(e) => {
                        return Err(PyToolError::Fatal(format!(
                            "undecodable discovery frame: {e}"
                        )))
                    }
                },
            }
        }
    }

    fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    /// Send one command, await its matching reply (by `request_id`), bounded by `timeout`.
    async fn round_trip(
        &self,
        request_id: u64,
        cmd: &Command,
        timeout: Duration,
    ) -> Result<Event, PyToolError> {
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .expect("pytool pending poisoned")
            .insert(request_id, tx);
        if let Err(e) = self.send(cmd).await {
            self.pending
                .lock()
                .expect("pytool pending poisoned")
                .remove(&request_id);
            return Err(PyToolError::Transient(format!("send command: {e}")));
        }
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(event)) => Ok(event),
            Ok(Err(_)) => {
                // The router dropped the sender (worker EOF) before replying.
                Err(PyToolError::Transient(
                    "worker exited during request".into(),
                ))
            }
            Err(_) => {
                self.pending
                    .lock()
                    .expect("pytool pending poisoned")
                    .remove(&request_id);
                Err(PyToolError::Transient(format!(
                    "worker watchdog: no reply within {timeout:?}"
                )))
            }
        }
    }

    /// Encode and send one command frame.
    async fn send(&self, cmd: &Command) -> std::io::Result<()> {
        let bytes = protocol::encode(cmd)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        self.writer.send(&bytes).await
    }

    /// Best-effort graceful stop: mark dead, ask the worker to exit, kill + reap the child, stop the
    /// reader, and fail any stragglers.
    async fn shutdown(&self) {
        self.alive.store(false, Ordering::Relaxed);
        let _ = self.send(&Command::Shutdown).await;
        self.child.lock().await.shutdown().await;
        if let Some(reader) = self.reader.lock().expect("pytool reader poisoned").take() {
            reader.abort();
        }
        self.pending
            .lock()
            .expect("pytool pending poisoned")
            .clear();
    }
}

/// Route one reply to the request awaiting it. A `request_id`-bearing reply (`Tools`/`Result`/`Pong`,
/// or an `Error` for a specific request) completes that request; a worker-level `Error` (no
/// `request_id`) fails every in-flight call.
fn route_reply(event: Event, pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Event>>>>) {
    let id = match &event {
        Event::Tools { request_id, .. }
        | Event::Result { request_id, .. }
        | Event::Pong { request_id } => Some(*request_id),
        Event::Error { request_id, .. } => *request_id,
        Event::Ready { .. } => None,
    };
    let mut map = pending.lock().expect("pytool pending poisoned");
    match id {
        Some(id) => {
            if let Some(tx) = map.remove(&id) {
                let _ = tx.send(event);
            }
        }
        None => {
            // A worker-level fault (or a stray Ready): fail every in-flight call by dropping the
            // senders so their round-trips return a transient fault and the worker is replaced.
            map.clear();
        }
    }
}

/// A [`daemon_core::Tool`] proxy that forwards a call to the Python worker. Holds a shared
/// [`PyToolHost`] (so all proxies share one worker process) and the tool's discovered manifest.
pub struct PyToolProxy {
    host: Arc<PyToolHost>,
    manifest: ToolManifest,
}

impl PyToolProxy {
    /// Build a proxy for `manifest`, backed by `host`.
    pub fn new(host: Arc<PyToolHost>, manifest: ToolManifest) -> Self {
        Self { host, manifest }
    }
}

#[async_trait::async_trait]
impl Tool for PyToolProxy {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn schema(&self) -> &str {
        &self.manifest.schema
    }

    fn deferrable(&self) -> bool {
        // Python tools are dynamic breadth too: part of the searchable long tail, not the core set.
        true
    }

    fn concurrency(&self) -> ToolConcurrency {
        // Map the worker's declared concurrency class onto the engine's §12 batch policy: a Python
        // tool authored as `concurrency="parallel"` joins a concurrent batch (the host + worker
        // already support concurrent in-flight calls via the reply router), while the default
        // `exclusive` serializes — matching the engine's all-or-nothing parallel-batch rule.
        match self.manifest.concurrency {
            Concurrency::Parallel => ToolConcurrency::Parallel,
            Concurrency::Exclusive => ToolConcurrency::Exclusive,
        }
    }

    async fn run(&self, call: &ToolCall, cx: &TurnCx<'_>) -> ToolOutcome {
        let call_id = call.call_id.clone();
        let session_id = cx.session_id.to_string();
        let call_fut =
            self.host
                .call_tool(&call_id, &self.manifest.name, &call.args, &session_id, 0);
        tokio::select! {
            biased;
            // Cooperative cancellation: tell the worker to stop, return a failed (cancelled) result.
            _ = cx.cancel.cancelled() => {
                self.host.cancel(&call_id).await;
                ToolOutcome::text(call_id, false, "python tool call cancelled")
            }
            reply = call_fut => match reply {
                Ok(reply) => reply_to_outcome(call_id, reply, self.manifest.untrusted),
                Err(e) => ToolOutcome::text(call_id, false, format!("python tool error: {e}")),
            }
        }
    }
}

/// Map a [`ToolReply`] onto a [`ToolOutcome`], honouring the untrusted fence (per-call override or
/// the manifest default) and converting a structured detail into the §17 [`ToolDetail`] envelope
/// (the JSON body is serialized to bytes; the GUI decodes it per `kind`).
fn reply_to_outcome(call_id: String, reply: ToolReply, manifest_untrusted: bool) -> ToolOutcome {
    let untrusted = reply.untrusted || manifest_untrusted;
    let mut outcome = if untrusted {
        ToolOutcome::untrusted_text(call_id, reply.ok, reply.content)
    } else {
        ToolOutcome::text(call_id, reply.ok, reply.content)
    };
    if let Some(detail) = reply.detail {
        let body = serde_json::to_vec(&detail.body).unwrap_or_default();
        outcome = outcome.with_detail(ToolDetail::new(detail.kind, body));
    }
    outcome
}

/// The Python tool surface as a [`daemon_core::ToolProvider`] — the shared discovery seam the host
/// uses for every dynamic tool source (Python now, MCP later). Wraps one [`PyToolHost`] (one worker
/// process, respawned lazily); [`discover`](ToolProvider::discover) returns a [`PyToolProxy`] per
/// tool the worker currently advertises.
pub struct PyToolProvider {
    host: Arc<PyToolHost>,
    label: String,
}

impl PyToolProvider {
    /// Build a provider over a worker described by `cfg`. The worker is spawned lazily on the first
    /// `discover`/call.
    pub fn new(cfg: PyToolConfig) -> Self {
        Self {
            host: Arc::new(PyToolHost::new(cfg)),
            label: "python".to_string(),
        }
    }

    /// Override the diagnostic label (default `"python"`).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// The shared host backing this provider's proxies.
    pub fn host(&self) -> Arc<PyToolHost> {
        self.host.clone()
    }

    fn proxies(&self, manifests: Vec<ToolManifest>) -> Vec<Arc<dyn Tool>> {
        manifests
            .into_iter()
            .map(|m| Arc::new(PyToolProxy::new(self.host.clone(), m)) as Arc<dyn Tool>)
            .collect()
    }
}

#[async_trait::async_trait]
impl ToolProvider for PyToolProvider {
    fn label(&self) -> &str {
        &self.label
    }

    async fn discover(&self) -> Result<Vec<Arc<dyn Tool>>, ToolProviderError> {
        let manifests = self.host.discover().await?;
        Ok(self.proxies(manifests))
    }
}

/// Spawn the worker described by `cfg`, discover its tools, and return a [`PyToolProxy`] (as
/// `Arc<dyn Tool>`) for each — ready to register into a [`ToolRegistry`](daemon_core::ToolRegistry).
/// All proxies share one [`PyToolHost`] (one worker process), which respawns lazily on a fault. A
/// convenience wrapper over [`PyToolProvider`] for callers wiring a single Python surface.
pub async fn discover(cfg: PyToolConfig) -> Result<Vec<Arc<dyn Tool>>, PyToolError> {
    let provider = PyToolProvider::new(cfg);
    let manifests = provider.host.discover().await?;
    Ok(provider.proxies(manifests))
}
