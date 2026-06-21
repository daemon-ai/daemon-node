//! [`LocalProvider`] — a [`Provider`] over a supervised `daemon-infer` worker process.
//!
//! Local inference engines (`llama.cpp`, `mistral.rs`) run in a separate `daemon-infer` child so a
//! GPU driver fault or allocator OOM crashes the worker, not the daemon. This one engine-agnostic
//! provider drives either engine: it spawns the worker via [`ProcessProvisioner`] over a
//! length-framed [`CutChannel`], speaks [`daemon_infer::protocol`], and maps the worker's classified
//! [`ErrorClass`](daemon_infer::protocol::ErrorClass) onto the §8 [`Failure`] taxonomy so the
//! existing `daemon-core` recovery loop (retry / compact / abort) drives recovery unchanged.
//!
//! On top of §8 recovery this provider owns the *worker lifecycle*: it lazily spawns the worker,
//! respawns it after a crash / watchdog kill / OOM (so the retry hits a fresh process), and trips a
//! local crash-loop "meltdown" to [`Failure::Fatal`] when restarts exceed a budget. An inner
//! time-to-first-token / inter-token watchdog kills a hung worker and surfaces
//! [`Failure::TransientTransport`].
//!
//! Generations are serialized on one worker (a single local model is single-stream), so the worker
//! is held behind a mutex for the duration of each call.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use daemon_common::{SessionId, UsageDelta};
use daemon_core::{
    Capabilities, Failure, ModelOutput, Provider, Request, RequestMsg, StreamEvent, ToolCallFormat,
};
use daemon_infer::protocol::{self, Command, Engine, Event, ModelParams, Sampling};
use daemon_provision::{
    ChildGuard, CutWriter, Placement, PlacementSpec, ProcessProvisioner, Provisioner,
};
use futures::stream::BoxStream;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use crate::{finalize_output, RawToolCall};

/// Construction + tuning for a [`LocalProvider`]'s worker.
#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// Path to the `daemon-infer` worker binary.
    pub worker_bin: PathBuf,
    /// Which engine the worker is spawned for.
    pub engine: Engine,
    /// The model: a local GGUF path (llama) or a directory / Hugging Face id (mistral.rs).
    pub model: String,
    /// Model load knobs.
    pub params: ModelParams,
    /// Extra environment variables set on the worker child (e.g. `CUDA_VISIBLE_DEVICES`).
    pub env: Vec<(String, String)>,
    /// Sampling parameters applied to every generation.
    pub sampling: Sampling,
    /// The output-token cap (`0` = the worker default).
    pub max_tokens: u32,
    /// How long to wait for `Event::Ready` after `Command::Load`.
    pub load_timeout: Duration,
    /// Watchdog: max wait for the first event of a generation (time to first token).
    pub ttft_timeout: Duration,
    /// Watchdog: max wait between events once streaming has started.
    pub inter_token_timeout: Duration,
    /// Crash-loop meltdown: max restarts allowed within [`WorkerConfig::restart_window`].
    pub max_restarts: u32,
    /// The sliding window over which [`WorkerConfig::max_restarts`] is counted.
    pub restart_window: Duration,
}

impl WorkerConfig {
    /// A config with sensible watchdog/meltdown defaults for `engine`/`model`.
    pub fn new(worker_bin: impl Into<PathBuf>, engine: Engine, model: impl Into<String>) -> Self {
        Self {
            worker_bin: worker_bin.into(),
            engine,
            model: model.into(),
            params: ModelParams::default(),
            env: Vec::new(),
            sampling: Sampling::default(),
            max_tokens: 0,
            load_timeout: Duration::from_secs(120),
            ttft_timeout: Duration::from_secs(60),
            inter_token_timeout: Duration::from_secs(30),
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
        }
    }
}

/// A [`Provider`] backed by a supervised local-inference worker.
pub struct LocalProvider {
    inner: Arc<LocalInner>,
}

struct LocalInner {
    cfg: WorkerConfig,
    capabilities: Capabilities,
    worker: Mutex<Option<Worker>>,
    restarts: Mutex<Vec<Instant>>,
}

impl LocalProvider {
    /// Build a provider for `cfg`. The worker is spawned lazily on the first generation.
    pub fn new(cfg: WorkerConfig) -> Self {
        let capabilities = default_capabilities(&cfg);
        Self {
            inner: Arc::new(LocalInner {
                cfg,
                capabilities,
                worker: Mutex::new(None),
                restarts: Mutex::new(Vec::new()),
            }),
        }
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        self.inner.run_generation(req, |_| {}).await
    }

    fn stream(&self, req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
        let inner = self.inner.clone();
        let (tx, rx) = futures::channel::mpsc::unbounded();
        let deltas = tx.clone();
        tokio::spawn(async move {
            let on_delta = move |ev: StreamEvent| {
                let _ = deltas.unbounded_send(Ok(ev));
            };
            match inner.run_generation(req, on_delta).await {
                Ok(output) => {
                    let _ = tx.unbounded_send(Ok(StreamEvent::Done(output)));
                }
                Err(failure) => {
                    let _ = tx.unbounded_send(Err(failure));
                }
            }
        });
        Box::pin(rx)
    }
}

impl LocalInner {
    /// Drive one generation on the (lazily spawned) worker, forwarding deltas via `on_delta` and
    /// returning the finalized [`ModelOutput`]. Tears the worker down on a failure that warrants a
    /// fresh process so the next call (a §8 retry) respawns it.
    async fn run_generation(
        &self,
        req: Request,
        on_delta: impl FnMut(StreamEvent) + Send,
    ) -> Result<ModelOutput, Failure> {
        let mut guard = self.worker.lock().await;
        if guard.is_none() {
            *guard = Some(self.spawn_worker().await?);
        }
        let worker = guard.as_mut().expect("worker present after spawn");

        let result = drive_generation(worker, &self.cfg, &req, on_delta).await;

        if let Err(ref failure) = result {
            if should_replace_worker(failure) {
                if let Some(mut dead) = guard.take() {
                    dead.shutdown().await;
                }
            }
        }
        result
    }

    /// Spawn + load a fresh worker, enforcing the crash-loop meltdown budget.
    async fn spawn_worker(&self) -> Result<Worker, Failure> {
        {
            let mut restarts = self.restarts.lock().await;
            let now = Instant::now();
            restarts.retain(|t| now.duration_since(*t) < self.cfg.restart_window);
            if restarts.len() as u32 >= self.cfg.max_restarts {
                return Err(Failure::Fatal(format!(
                    "daemon-infer worker crash-loop: {} restarts within {:?}",
                    restarts.len(),
                    self.cfg.restart_window
                )));
            }
            restarts.push(now);
        }
        Worker::spawn(&self.cfg).await
    }
}

/// A live worker process: the framed writer, an event inbox fed by a reader task, and the child guard.
struct Worker {
    writer: CutWriter,
    events: tokio::sync::mpsc::UnboundedReceiver<Event>,
    child: ChildGuard,
    next_request_id: u64,
    reader: JoinHandle<()>,
}

impl Worker {
    /// Spawn the worker, send `Load`, and block until it reports `Ready` (or fails / times out).
    async fn spawn(cfg: &WorkerConfig) -> Result<Worker, Failure> {
        let spec = PlacementSpec {
            program: cfg.worker_bin.clone(),
            args: vec!["--engine".to_string(), cfg.engine.as_str().to_string()],
            env: cfg.env.clone(),
        };
        let session = SessionId::new("daemon-infer-worker");
        let Placement { channel, child } = ProcessProvisioner::new()
            .place(&session, spec)
            .await
            .map_err(|e| Failure::TransientTransport(format!("spawn daemon-infer worker: {e}")))?;

        let (writer, mut framed_reader) = channel.split();
        let (ev_tx, ev_rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
        let reader = tokio::spawn(async move {
            while let Some(bytes) = framed_reader.recv().await {
                match protocol::decode::<Event>(&bytes) {
                    Ok(event) => {
                        if ev_tx.send(event).is_err() {
                            break;
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "daemon-infer: undecodable event frame"),
                }
            }
            // EOF / broken pipe: dropping `ev_tx` closes the inbox, which a generation reads as a
            // mid-stream worker exit.
        });

        let mut worker = Worker {
            writer,
            events: ev_rx,
            child,
            next_request_id: 1,
            reader,
        };

        let load = Command::Load {
            engine: cfg.engine,
            model: cfg.model.clone(),
            params: cfg.params.clone(),
        };
        if let Err(e) = worker.send(&load).await {
            worker.shutdown().await;
            return Err(Failure::TransientTransport(format!("send load: {e}")));
        }

        match tokio::time::timeout(cfg.load_timeout, worker.events.recv()).await {
            Err(_) => {
                worker.shutdown().await;
                Err(Failure::TransientTransport("worker load timed out".into()))
            }
            Ok(None) => {
                worker.shutdown().await;
                Err(Failure::TransientTransport(
                    "worker exited during load".into(),
                ))
            }
            Ok(Some(Event::Ready { .. })) => Ok(worker),
            Ok(Some(Event::Error { class, message, .. })) => {
                worker.shutdown().await;
                Err(class_to_failure(class, message))
            }
            Ok(Some(other)) => {
                worker.shutdown().await;
                Err(Failure::Fatal(format!(
                    "unexpected event during load: {other:?}"
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

    /// Best-effort graceful stop: ask the worker to exit, kill + reap the child, stop the reader.
    async fn shutdown(&mut self) {
        let _ = self.send(&Command::Shutdown).await;
        self.child.shutdown().await;
        self.reader.abort();
    }
}

/// Run one generation against `worker`, applying the TTFT / inter-token watchdog and assembling the
/// finalized output through the §9 repair pipeline.
async fn drive_generation(
    worker: &mut Worker,
    cfg: &WorkerConfig,
    req: &Request,
    mut on_delta: impl FnMut(StreamEvent) + Send,
) -> Result<ModelOutput, Failure> {
    let request_id = worker.next_request_id;
    worker.next_request_id += 1;

    let cmd = Command::Generate {
        request_id,
        system: req.system.clone(),
        messages: req.messages.iter().map(to_proto_msg).collect(),
        tools: req
            .tools
            .iter()
            .map(|t| protocol::ToolDef {
                name: t.name.clone(),
                schema: t.schema.clone(),
            })
            .collect(),
        sampling: cfg.sampling,
        max_tokens: cfg.max_tokens,
    };
    worker
        .send(&cmd)
        .await
        .map_err(|e| Failure::TransientTransport(format!("send generate: {e}")))?;

    let mut text = String::new();
    let mut reasoning = String::new();
    let mut raw_calls: Vec<RawToolCall> = Vec::new();
    let usage;
    let mut first = true;

    loop {
        let budget = if first {
            cfg.ttft_timeout
        } else {
            cfg.inter_token_timeout
        };
        let event = match tokio::time::timeout(budget, worker.events.recv()).await {
            Err(_) => {
                return Err(Failure::TransientTransport(format!(
                    "worker watchdog: no event within {budget:?}"
                )));
            }
            Ok(None) => {
                return Err(Failure::TransientTransport(
                    "worker exited during generation".into(),
                ));
            }
            Ok(Some(event)) => event,
        };
        first = false;

        match event {
            Event::TextDelta {
                request_id: rid,
                text: delta,
            } if rid == request_id => {
                on_delta(StreamEvent::TextDelta(delta.clone()));
                text.push_str(&delta);
            }
            Event::ReasoningDelta {
                request_id: rid,
                text: delta,
            } if rid == request_id => {
                on_delta(StreamEvent::ReasoningDelta(delta.clone()));
                reasoning.push_str(&delta);
            }
            Event::ToolCall {
                request_id: rid,
                call,
            } if rid == request_id => {
                raw_calls.push(RawToolCall {
                    id: call.call_id,
                    name: call.name,
                    args: call.args,
                });
            }
            Event::Done {
                request_id: rid,
                usage: u,
            } if rid == request_id => {
                usage = UsageDelta {
                    input_tokens: u.input_tokens,
                    output_tokens: u.output_tokens,
                    api_calls: 1,
                };
                break;
            }
            Event::Error {
                request_id: rid,
                class,
                message,
            } if rid.is_none() || rid == Some(request_id) => {
                return Err(class_to_failure(class, message));
            }
            // Stale frame from a prior request (serialized generations make this rare) — ignore.
            _ => {}
        }
    }

    let reasoning = (!reasoning.is_empty()).then_some(reasoning);
    Ok(finalize_output(
        text,
        reasoning,
        raw_calls,
        usage,
        &req.tool_names(),
    ))
}

/// Map one engine [`RequestMsg`] onto the worker protocol's [`protocol::Msg`].
fn to_proto_msg(msg: &RequestMsg) -> protocol::Msg {
    protocol::Msg {
        role: msg.role.clone(),
        content: msg.content.clone(),
        tool_calls: msg
            .tool_calls
            .iter()
            .map(|c| protocol::ToolCall {
                call_id: c.call_id.clone(),
                name: c.name.clone(),
                args: c.args.clone(),
            })
            .collect(),
        tool_call_id: msg.tool_call_id.clone(),
    }
}

/// Map a worker [`ErrorClass`](protocol::ErrorClass) onto the §8 [`Failure`] taxonomy.
///
/// A VRAM/host OOM maps to [`Failure::ProviderOverloaded`] (retry with backoff on a fresh worker —
/// the respawn reclaims the allocation); a context overflow maps to [`Failure::ContextOverflow`]
/// (compact + retry), keeping the same worker.
fn class_to_failure(class: protocol::ErrorClass, message: String) -> Failure {
    use protocol::ErrorClass as C;
    match class {
        C::ContextOverflow => Failure::ContextOverflow(message),
        C::OutOfMemory => Failure::ProviderOverloaded(message),
        C::Transient => Failure::TransientTransport(message),
        C::Fatal => Failure::Fatal(message),
        C::Cancelled => Failure::Cancelled,
    }
}

/// Whether a failure warrants tearing down the worker so the next call respawns a fresh process.
/// Transport faults (crash / watchdog kill), OOM (reclaim VRAM), and fatals replace the worker; a
/// context overflow is a prompt issue (the worker is healthy) and cancellation leaves it reusable.
fn should_replace_worker(failure: &Failure) -> bool {
    matches!(
        failure,
        Failure::TransientTransport(_) | Failure::ProviderOverloaded(_) | Failure::Fatal(_)
    )
}

/// The pre-load capabilities advertised for `cfg`, derived from the engine + configured context.
fn default_capabilities(cfg: &WorkerConfig) -> Capabilities {
    let max_context = (cfg.params.n_ctx > 0).then_some(cfg.params.n_ctx);
    match cfg.engine {
        Engine::Llama => Capabilities {
            supports_native_tools: false,
            supports_streaming: true,
            tool_call_format: ToolCallFormat::HermesXml,
            max_context,
        },
        Engine::MistralRs => Capabilities {
            supports_native_tools: true,
            supports_streaming: true,
            tool_call_format: ToolCallFormat::Native,
            max_context,
        },
    }
}
