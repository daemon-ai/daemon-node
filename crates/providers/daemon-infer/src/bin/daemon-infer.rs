//! The `daemon-infer` worker binary: a length-framed stdio loop over [`daemon_infer::protocol`].
//!
//! Spawned by the daemon's `LocalProvider` with `--engine {llama|mistralrs}`. It loads a model
//! (`Command::Load`), then streams generations (`Command::Generate` -> `TextDelta`/`ToolCall`/`Done`),
//! honoring `Cancel`/`Ping`/`Shutdown`. With no engine feature compiled it runs as an inert stub:
//! `Load` returns a `Fatal` "no backend" error and the worker stays responsive (then exits on
//! `Shutdown` / EOF). stdout is the cut; all diagnostics go to stderr.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use daemon_infer::backend::{BackendChunk, GenerateRequest, InferenceBackend};
use daemon_infer::backends;
use daemon_infer::protocol::{self, Command, Engine, Event};
use daemon_provision::{CutChannel, CutWriter};
use tokio::sync::mpsc::{self, UnboundedSender};
use tokio_util::sync::CancellationToken;

#[tokio::main]
async fn main() {
    let selected_engine = parse_engine_arg();
    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    let mut backend: Option<Arc<dyn InferenceBackend>> = None;
    let mut inflight: HashMap<u64, CancellationToken> = HashMap::new();
    // Generation tasks report their request_id here on completion so the inflight map is pruned.
    let (done_tx, mut done_rx) = mpsc::unbounded_channel::<u64>();

    loop {
        tokio::select! {
            // A completed generation: drop its cancel token.
            Some(rid) = done_rx.recv() => {
                inflight.remove(&rid);
            }
            // An inbound command frame (or EOF / broken pipe -> exit).
            maybe = reader.recv() => {
                let Some(bytes) = maybe else { break };
                let cmd: Command = match protocol::decode(&bytes) {
                    Ok(cmd) => cmd,
                    Err(e) => {
                        eprintln!("daemon-infer: undecodable command frame: {e}");
                        continue;
                    }
                };
                match cmd {
                    Command::Load { engine, model, params } => {
                        if let Some(sel) = selected_engine {
                            if sel != engine {
                                eprintln!(
                                    "daemon-infer: --engine {} but Load requested {}; honoring Load",
                                    sel.as_str(),
                                    engine.as_str()
                                );
                            }
                        }
                        match backends::load(engine, &model, &params).await {
                            Ok(loaded) => {
                                let capabilities = loaded.capabilities();
                                backend = Some(Arc::from(loaded));
                                send_event(&writer, &Event::Ready { capabilities }).await;
                            }
                            Err(e) => {
                                send_event(
                                    &writer,
                                    &Event::Error {
                                        request_id: None,
                                        class: e.class,
                                        message: e.message,
                                    },
                                )
                                .await;
                            }
                        }
                    }
                    Command::Generate {
                        request_id,
                        system,
                        messages,
                        tools,
                        sampling,
                        max_tokens,
                    } => {
                        let Some(backend) = backend.clone() else {
                            send_event(
                                &writer,
                                &Event::Error {
                                    request_id: Some(request_id),
                                    class: protocol::ErrorClass::Fatal,
                                    message: "no model loaded".into(),
                                },
                            )
                            .await;
                            continue;
                        };
                        let cancel = CancellationToken::new();
                        inflight.insert(request_id, cancel.clone());
                        let req = GenerateRequest {
                            request_id,
                            system,
                            messages,
                            tools,
                            sampling,
                            max_tokens,
                        };
                        let writer = writer.clone();
                        let done_tx = done_tx.clone();
                        tokio::spawn(run_generation(backend, writer, req, cancel, done_tx));
                    }
                    Command::Cancel { request_id } => {
                        if let Some(token) = inflight.get(&request_id) {
                            token.cancel();
                        }
                    }
                    Command::Ping => send_event(&writer, &Event::Pong).await,
                    Command::Shutdown => break,
                }
            }
        }
    }

    // Cancel anything still running so generation tasks unwind before exit.
    for (_id, token) in inflight.drain() {
        token.cancel();
    }
}

/// Drive one generation, forwarding chunks as events then a terminal `Done`/`Error`.
async fn run_generation(
    backend: Arc<dyn InferenceBackend>,
    writer: CutWriter,
    req: GenerateRequest,
    cancel: CancellationToken,
    done_tx: UnboundedSender<u64>,
) {
    let request_id = req.request_id;
    let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<BackendChunk>();
    let generate = backend.generate(req, chunk_tx, cancel);
    tokio::pin!(generate);

    let terminal = loop {
        tokio::select! {
            biased;
            maybe = chunk_rx.recv() => {
                if let Some(chunk) = maybe {
                    forward_chunk(&writer, request_id, chunk).await;
                }
                // `None` means the backend dropped its sender; the `generate` arm finalizes.
            }
            result = &mut generate => break result,
        }
    };

    // Drain any chunks buffered before the backend returned.
    while let Ok(chunk) = chunk_rx.try_recv() {
        forward_chunk(&writer, request_id, chunk).await;
    }

    let event = match terminal {
        Ok(usage) => Event::Done { request_id, usage },
        Err(e) => Event::Error {
            request_id: Some(request_id),
            class: e.class,
            message: e.message,
        },
    };
    send_event(&writer, &event).await;
    let _ = done_tx.send(request_id);
}

/// Map a [`BackendChunk`] onto its wire [`Event`] and send it.
async fn forward_chunk(writer: &CutWriter, request_id: u64, chunk: BackendChunk) {
    let event = match chunk {
        BackendChunk::Text(text) => Event::TextDelta { request_id, text },
        BackendChunk::Reasoning(text) => Event::ReasoningDelta { request_id, text },
        BackendChunk::Tool(call) => Event::ToolCall { request_id, call },
    };
    send_event(writer, &event).await;
}

/// Encode and send one event. A send failure means the parent went away — diagnosed to stderr.
async fn send_event(writer: &CutWriter, event: &Event) {
    let bytes = match protocol::encode(event) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("daemon-infer: failed to encode event: {e}");
            return;
        }
    };
    if let Err(e) = writer.send(&bytes).await {
        eprintln!("daemon-infer: failed to send event (parent gone?): {e}");
    }
}

/// Parse an optional `--engine <name>` flag (the spawn-time selector; `Load` is authoritative).
fn parse_engine_arg() -> Option<Engine> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--engine=") {
            return Engine::parse(value);
        }
        if arg == "--engine" {
            return args.next().and_then(|v| Engine::parse(&v));
        }
    }
    None
}
