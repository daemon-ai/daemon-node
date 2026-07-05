// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Integration tests for [`LocalProvider`] against the scripted `fake-infer-worker` binary.
//!
//! These exercise the full supervised-worker path (spawn -> load -> generate over the real
//! length-framed protocol cut) without any engine, covering streaming, tool-call decode, and the
//! recovery surface the §8 loop relies on: worker crash, watchdog timeout, crash-loop meltdown, and
//! a fatal load.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use daemon_core::{EmbeddingProvider, Failure, Provider, Request, RequestMsg, StreamEvent};
use daemon_infer::protocol::Engine;
use daemon_providers::{LocalEmbedder, LocalProvider, WorkerConfig};
use futures::StreamExt;

/// The path to the scripted fake worker (built as a bin target of this crate).
const FAKE_WORKER: &str = env!("CARGO_BIN_EXE_fake-infer-worker");

/// A worker config pointing at the fake worker, with short watchdog/meltdown windows for fast tests.
fn worker_config(scenario: &str, state: &Path) -> WorkerConfig {
    let mut wc = WorkerConfig::new(PathBuf::from(FAKE_WORKER), Engine::Llama, "fake-model");
    wc.env = vec![
        ("DAEMON_FAKE_SCENARIO".to_string(), scenario.to_string()),
        ("DAEMON_FAKE_STATE".to_string(), state.display().to_string()),
    ];
    wc.load_timeout = Duration::from_secs(5);
    wc.ttft_timeout = Duration::from_millis(300);
    wc.inter_token_timeout = Duration::from_millis(300);
    wc.max_restarts = 3;
    wc.restart_window = Duration::from_secs(30);
    wc
}

/// A unique per-test path for the fake worker's spawn-counter file.
fn temp_state(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("daemon-fake-{tag}-{}-{nanos}", std::process::id()))
}

fn user_request(text: &str) -> Request {
    Request {
        system: "you are a test".into(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: text.into(),
            ..Default::default()
        }],
        tools: Vec::new(),
        auth: None,
        constraint: None,
        cache_system: false,
        params: Default::default(),
        task: None,
    }
}

#[tokio::test]
async fn chat_streams_and_completes() {
    let state = temp_state("stream");
    let provider = LocalProvider::new(worker_config("stream", &state));
    let out = provider.chat(user_request("hi")).await.expect("chat ok");
    assert_eq!(out.text, "Hello world");
    assert_eq!(out.usage.output_tokens, 2);
    assert_eq!(out.usage.input_tokens, 5);
}

#[tokio::test]
async fn stream_emits_deltas_then_done() {
    let state = temp_state("stream2");
    let provider = LocalProvider::new(worker_config("stream", &state));
    let mut events = provider.stream(user_request("hi"));

    let mut deltas = Vec::new();
    let mut done_text = None;
    while let Some(event) = events.next().await {
        match event.expect("event ok") {
            StreamEvent::TextDelta(t) => deltas.push(t),
            StreamEvent::Done(out) => {
                done_text = Some(out.text);
                break;
            }
            _ => {}
        }
    }
    assert_eq!(deltas, vec!["Hello".to_string(), " world".to_string()]);
    assert_eq!(done_text.as_deref(), Some("Hello world"));
}

#[tokio::test]
async fn tool_call_is_decoded() {
    let state = temp_state("tool");
    let provider = LocalProvider::new(worker_config("tool", &state));
    let out = provider.chat(user_request("read")).await.expect("chat ok");
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].name, "read_file");
}

#[tokio::test]
async fn worker_crash_midgen_then_retry_succeeds() {
    let state = temp_state("midgen");
    let provider = LocalProvider::new(worker_config("exit-midgen", &state));

    let first = provider.chat(user_request("hi")).await;
    assert!(
        matches!(first, Err(Failure::TransientTransport(_))),
        "expected transient on mid-gen crash, got {first:?}"
    );

    // The next call respawns a fresh worker (spawn index 1) which streams cleanly — proving the
    // provider replaces a crashed worker so the §8 retry lands on a healthy process.
    let second = provider.chat(user_request("hi")).await.expect("retry ok");
    assert_eq!(second.text, "Hello world");
}

#[tokio::test]
async fn worker_hang_trips_watchdog_then_retry() {
    let state = temp_state("hang");
    let provider = LocalProvider::new(worker_config("hang", &state));

    let first = provider.chat(user_request("hi")).await;
    assert!(
        matches!(first, Err(Failure::TransientTransport(_))),
        "expected transient on watchdog timeout, got {first:?}"
    );

    let second = provider.chat(user_request("hi")).await.expect("retry ok");
    assert_eq!(second.text, "Hello world");
}

#[tokio::test]
async fn crash_loop_trips_meltdown_fatal() {
    let state = temp_state("loop");
    let mut cfg = worker_config("exit-on-start", &state);
    cfg.max_restarts = 2;
    let provider = LocalProvider::new(cfg);

    // Each attempt respawns until restarts exceed the budget, then the provider declares a meltdown
    // Fatal instead of fork-bombing the box.
    let mut last = provider.chat(user_request("hi")).await;
    for _ in 0..5 {
        if matches!(last, Err(Failure::Fatal(_))) {
            break;
        }
        last = provider.chat(user_request("hi")).await;
    }
    assert!(
        matches!(last, Err(Failure::Fatal(_))),
        "expected meltdown Fatal, got {last:?}"
    );
}

#[tokio::test]
async fn local_embedder_returns_deterministic_vectors() {
    let state = temp_state("embed");
    let embedder = LocalEmbedder::new(worker_config("stream", &state), 8, "fake-embed");
    let vectors = embedder
        .embed(&["hello world".to_string(), "foo".to_string()])
        .await
        .expect("embed ok");
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), 8);
    assert_eq!(embedder.dimensions(), 8);
    assert_eq!(embedder.model(), "fake-embed");

    // The same text re-embeds to the same vector (deterministic worker).
    let again = embedder.embed(&["hello world".to_string()]).await.unwrap();
    assert_eq!(again[0], vectors[0]);
}

#[tokio::test]
async fn local_embed_error_is_classified() {
    let state = temp_state("embederr");
    let embedder = LocalEmbedder::new(worker_config("embed-error", &state), 8, "fake-embed");
    let result = embedder.embed(&["x".to_string()]).await;
    assert!(
        matches!(result, Err(Failure::Fatal(_))),
        "expected fatal embed error, got {result:?}"
    );
}

#[tokio::test]
async fn fatal_load_error_is_fatal() {
    let state = temp_state("loaderr");
    let provider = LocalProvider::new(worker_config("load-error", &state));
    let result = provider.chat(user_request("hi")).await;
    assert!(
        matches!(result, Err(Failure::Fatal(_))),
        "expected fatal load error, got {result:?}"
    );
}
