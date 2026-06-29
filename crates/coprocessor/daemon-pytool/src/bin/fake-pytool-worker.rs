// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `fake-pytool-worker` — a hermetic, Python-free worker that speaks the [`daemon_pytool::protocol`].
//!
//! It exists so the supervised client (`daemon-pytool-client`) and `tests/daemon-conformance` can
//! exercise the spawn / discover / call / respawn path in CI without a system Python. It mirrors
//! what the real `daemon_pytool` Python package does on the wire: emit `Ready`, answer `ListTools`
//! with a single `py_echo` tool, echo `CallTool` args back as the result, reply to `Ping`, and exit
//! on `Shutdown`.
//!
//! Behaviour knobs (args), used by the client's tests:
//! - `--crash-on-call`: exit the process immediately on the first `CallTool` (to test respawn).
//! - `--hang-on-call`: never reply to a `CallTool` (to test the client's op-timeout + teardown).

#![forbid(unsafe_code)]

use daemon_provision::CutChannel;
use daemon_pytool::protocol::{self, Command, Concurrency, Event, ToolManifest, PROTOCOL_VERSION};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let crash_on_call = args.iter().any(|a| a == "--crash-on-call");
    let hang_on_call = args.iter().any(|a| a == "--hang-on-call");

    let channel = CutChannel::from_stdio();
    let (writer, mut reader) = channel.split();

    // Announce readiness (the unsolicited first frame).
    let ready = Event::Ready {
        worker: "fake-pytool-worker".into(),
        sdk_version: env!("CARGO_PKG_VERSION").into(),
        protocol_version: PROTOCOL_VERSION,
    };
    if let Ok(bytes) = protocol::encode(&ready) {
        let _ = writer.send(&bytes).await;
    }

    while let Some(bytes) = reader.recv().await {
        let cmd = match protocol::decode::<Command>(&bytes) {
            Ok(cmd) => cmd,
            Err(e) => {
                eprintln!("fake-pytool-worker: undecodable command: {e}");
                continue;
            }
        };
        let reply = match cmd {
            Command::Initialize { .. } => continue,
            Command::ListTools { request_id } => Event::Tools {
                request_id,
                tools: vec![ToolManifest {
                    name: "py_echo".into(),
                    description: "Echo the provided text back to the caller.".into(),
                    schema: r#"{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}"#.into(),
                    concurrency: Concurrency::Parallel,
                    untrusted: false,
                }],
            },
            Command::CallTool {
                request_id,
                call_id,
                args,
                ..
            } => {
                if crash_on_call {
                    std::process::exit(1);
                }
                if hang_on_call {
                    continue;
                }
                let text = serde_json::from_str::<serde_json::Value>(&args)
                    .ok()
                    .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(str::to_owned))
                    .unwrap_or_default();
                Event::Result {
                    request_id,
                    call_id,
                    ok: true,
                    content: text.clone(),
                    detail: Some(protocol::ResultDetail {
                        kind: "py_echo".into(),
                        body: serde_json::json!({ "echoed": text }),
                    }),
                    untrusted: false,
                }
            }
            Command::Cancel { .. } => continue,
            Command::Ping { request_id } => Event::Pong { request_id },
            Command::Shutdown => break,
        };
        match protocol::encode(&reply) {
            Ok(frame) => {
                if writer.send(&frame).await.is_err() {
                    break;
                }
            }
            Err(e) => eprintln!("fake-pytool-worker: encode failed: {e}"),
        }
    }
}
