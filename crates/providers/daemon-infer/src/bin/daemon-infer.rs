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
    // A one-shot `quantize` subcommand (not part of the stdio protocol): offline-quantize a GGUF
    // via llama.cpp's native quantizer, then exit. The daemon's model manager shells out to this.
    let raw_args: Vec<String> = std::env::args().collect();
    if raw_args.get(1).map(String::as_str) == Some("quantize") {
        std::process::exit(run_quantize_cli(&raw_args[2..]));
    }

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
                        constraint,
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
                            constraint,
                        };
                        let writer = writer.clone();
                        let done_tx = done_tx.clone();
                        tokio::spawn(run_generation(backend, writer, req, cancel, done_tx));
                    }
                    Command::Embed { request_id, texts } => {
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
                        let writer = writer.clone();
                        // Embeddings are a bounded request/response (no streaming, no cancel token):
                        // run on a task so a slow embed does not block the command loop.
                        tokio::spawn(async move {
                            let event = match backend.embed(texts).await {
                                Ok(vectors) => {
                                    let dims = vectors.first().map(|v| v.len() as u32).unwrap_or(0);
                                    Event::Embeddings {
                                        request_id,
                                        vectors,
                                        dims,
                                    }
                                }
                                Err(e) => Event::Error {
                                    request_id: Some(request_id),
                                    class: e.class,
                                    message: e.message,
                                },
                            };
                            send_event(&writer, &event).await;
                        });
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

/// Run the one-shot `quantize` subcommand, returning a process exit code.
///
/// Usage: `daemon-infer quantize --in <f16.gguf> --out <q4km.gguf> --ftype Q4_K_M [--nthread N]`.
/// Without the `llama` feature this is a clear non-zero error (no engine linked).
fn run_quantize_cli(args: &[String]) -> i32 {
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut ftype: Option<String> = None;
    let mut nthread: i32 = 0;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--in" | "--input" => input = it.next().cloned(),
            "--out" | "--output" => output = it.next().cloned(),
            "--ftype" | "--type" => ftype = it.next().cloned(),
            "--nthread" | "--threads" => {
                nthread = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
            }
            other => eprintln!("daemon-infer quantize: ignoring unknown arg '{other}'"),
        }
    }
    let (Some(input), Some(output), Some(ftype)) = (input, output, ftype) else {
        eprintln!(
            "daemon-infer quantize: required --in <gguf> --out <gguf> --ftype <Q4_K_M> [--nthread N]"
        );
        return 2;
    };

    #[cfg(feature = "llama")]
    {
        match backends::quantize::run_quantize(&input, &output, &ftype, nthread) {
            Ok(()) => {
                println!("daemon-infer quantize: wrote {output}");
                0
            }
            Err(e) => {
                eprintln!("daemon-infer quantize: {e}");
                1
            }
        }
    }
    #[cfg(not(feature = "llama"))]
    {
        let _ = (input, output, ftype, nthread);
        eprintln!(
            "daemon-infer quantize: built without the `llama` feature; rebuild the worker with --features llama"
        );
        3
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
