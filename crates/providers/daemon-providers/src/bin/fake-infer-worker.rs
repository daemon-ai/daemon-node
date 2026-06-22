//! A scripted fake `daemon-infer` worker for [`LocalProvider`] integration tests.
//!
//! It speaks the real [`daemon_infer::protocol`] over the same length-framed stdio cut as the
//! production worker, but instead of running an engine it plays a scenario selected by
//! `DAEMON_FAKE_SCENARIO`, optionally varying behavior by spawn index (a counter persisted in
//! `DAEMON_FAKE_STATE`) so a test can assert "crash once, then succeed on the respawn".
//!
//! Scenarios: `stream` (default) | `tool` | `exit-midgen` | `hang` | `load-error` | `exit-on-start`.

use std::time::Duration;

use daemon_infer::protocol::{
    self, Capabilities, Command, ErrorClass, Event, ToolCall, ToolCallFormat, Usage,
};
use daemon_provision::{CutChannel, CutWriter};

#[tokio::main]
async fn main() {
    let scenario = std::env::var("DAEMON_FAKE_SCENARIO").unwrap_or_else(|_| "stream".to_string());
    let spawn_index = bump_spawn_counter();

    if scenario == "exit-on-start" {
        std::process::exit(1);
    }

    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    while let Some(bytes) = reader.recv().await {
        let cmd: Command = match protocol::decode(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("fake-infer-worker: undecodable command: {e}");
                continue;
            }
        };
        match cmd {
            Command::Load { .. } => {
                if scenario == "load-error" {
                    send(
                        &writer,
                        &Event::Error {
                            request_id: None,
                            class: ErrorClass::Fatal,
                            message: "fake: model unloadable".into(),
                        },
                    )
                    .await;
                } else {
                    send(
                        &writer,
                        &Event::Ready {
                            capabilities: Capabilities {
                                supports_native_tools: scenario == "tool",
                                supports_streaming: true,
                                tool_call_format: ToolCallFormat::Native,
                                max_context: Some(4096),
                            },
                        },
                    )
                    .await;
                }
            }
            Command::Generate { request_id, .. } => {
                run_generate(&writer, &scenario, spawn_index, request_id).await;
            }
            Command::Embed { request_id, texts } => {
                run_embed(&writer, &scenario, request_id, &texts).await;
            }
            Command::Cancel { .. } => {}
            Command::Ping => send(&writer, &Event::Pong).await,
            Command::Shutdown => break,
        }
    }
}

/// Play the scenario's generation behavior for `request_id`.
async fn run_generate(writer: &CutWriter, scenario: &str, spawn_index: u64, request_id: u64) {
    // `exit-midgen` / `hang` misbehave on the *first* spawn (index 0) and stream cleanly on the
    // respawn (index >= 1), so a test can assert restart-then-retry succeeds.
    let misbehave = spawn_index == 0;
    match scenario {
        "tool" => {
            send(
                writer,
                &Event::ToolCall {
                    request_id,
                    call: ToolCall {
                        call_id: "call-1".into(),
                        name: "read_file".into(),
                        args: r#"{"path":"x"}"#.into(),
                    },
                },
            )
            .await;
            send_done(writer, request_id).await;
        }
        "exit-midgen" if misbehave => {
            send_text(writer, request_id, "par").await;
            // Crash mid-generation: the parent sees the cut close before `Done`.
            std::process::exit(1);
        }
        "hang" if misbehave => {
            // Never answer: the parent's TTFT/inter-token watchdog must kill us.
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
        // "stream" and the post-restart paths of exit-midgen/hang.
        _ => {
            send_text(writer, request_id, "Hello").await;
            send_text(writer, request_id, " world").await;
            send_done(writer, request_id).await;
        }
    }
}

/// The fake embedding dimensionality.
const FAKE_DIM: usize = 8;

/// Answer an embed request with deterministic per-text vectors (or an error under `embed-error`).
async fn run_embed(writer: &CutWriter, scenario: &str, request_id: u64, texts: &[String]) {
    if scenario == "embed-error" {
        send(
            writer,
            &Event::Error {
                request_id: Some(request_id),
                class: ErrorClass::Fatal,
                message: "fake: embeddings unavailable".into(),
            },
        )
        .await;
        return;
    }
    let vectors: Vec<Vec<f32>> = texts.iter().map(|t| fake_embed(t)).collect();
    send(
        writer,
        &Event::Embeddings {
            request_id,
            vectors,
            dims: FAKE_DIM as u32,
        },
    )
    .await;
}

/// A stable bag-of-words hash embedding (so shared tokens produce similar vectors), L2-normalized.
fn fake_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; FAKE_DIM];
    for token in text.split_whitespace() {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in token.to_ascii_lowercase().as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        v[(h % FAKE_DIM as u64) as usize] += 1.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    v
}

async fn send_text(writer: &CutWriter, request_id: u64, text: &str) {
    send(
        writer,
        &Event::TextDelta {
            request_id,
            text: text.into(),
        },
    )
    .await;
}

async fn send_done(writer: &CutWriter, request_id: u64) {
    send(
        writer,
        &Event::Done {
            request_id,
            usage: Usage {
                input_tokens: 5,
                output_tokens: 2,
                cache_read_tokens: 0,
            },
        },
    )
    .await;
}

async fn send(writer: &CutWriter, event: &Event) {
    let bytes = protocol::encode(event).expect("encode event");
    let _ = writer.send(&bytes).await;
}

/// Read-increment the spawn counter in `DAEMON_FAKE_STATE` (if set), returning this spawn's index.
fn bump_spawn_counter() -> u64 {
    let Ok(path) = std::env::var("DAEMON_FAKE_STATE") else {
        return 0;
    };
    let current = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    let _ = std::fs::write(&path, (current + 1).to_string());
    current
}
