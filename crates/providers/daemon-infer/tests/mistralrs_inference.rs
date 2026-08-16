// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Real-model mistral.rs generation smoke for the engine lane (ignored by default).
//!
//! Gated on the `mistralrs` feature and `#[ignore]` so the default `cargo test --workspace`
//! (stub worker, no engine) never builds or runs it. It loads a small instruct model from the
//! shared HF cache (downloading on first run) and drives one sampled generation, asserting
//! non-empty text — and, with tools offered, that no tool-call markup leaks into the text
//! (tools ride the native API, so the text stream must stay clean).
//!
//! ```text
//! nix develop --command bash -c '
//!   cargo test -p daemon-infer --features mistralrs -j 12 \
//!     --test mistralrs_inference -- --ignored --nocapture --test-threads 1'
//! ```
//!
//! `--test-threads 1` matters on a cold cache: concurrent first-run downloads race on the
//! hf-hub blob lock and one loader fails spuriously.
//!
//! Override the model with `DAEMON_INFER_TEST_MISTRALRS_MODEL` (an HF repo id or local path).
#![cfg(feature = "mistralrs")]

use daemon_infer::backend::{BackendChunk, GenerateRequest};
use daemon_infer::backends;
use daemon_infer::protocol::{Engine, ModelParams, Msg, Sampling, ToolDef};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// The model under test: a small instruct model whose safetensors mistral.rs loads directly.
fn model_id() -> String {
    std::env::var("DAEMON_INFER_TEST_MISTRALRS_MODEL")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "HuggingFaceTB/SmolLM2-135M-Instruct".to_string())
}

fn params() -> ModelParams {
    ModelParams {
        n_gpu_layers: 0,
        n_ctx: 1024,
        n_threads: None,
        flash_attn: false,
        isq: None,
        embeddings: false,
        mmproj: None,
    }
}

fn request(tools: Vec<ToolDef>) -> GenerateRequest {
    GenerateRequest {
        request_id: 1,
        system: "You are a helpful assistant.".to_string(),
        messages: vec![Msg {
            role: "user".to_string(),
            content: "Write one short sentence about the sky.".to_string(),
            ..Default::default()
        }],
        tools,
        // Light sampling with a fixed seed (mirrors the llama smoke): tiny models collapse under
        // pure greedy decoding.
        sampling: Sampling {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            seed: 42,
            ..Default::default()
        },
        max_tokens: 64,
        constraint: None,
        stop: Vec::new(),
    }
}

async fn run(req: GenerateRequest) -> (daemon_infer::protocol::Usage, String, usize) {
    let backend = backends::load(Engine::MistralRs, &model_id(), &params())
        .await
        .expect("load mistral.rs model");
    assert!(backend.capabilities().supports_native_tools);
    assert!(backend.capabilities().supports_streaming);

    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let generate = backend.generate(req, tx, cancel);
    let collect = async {
        let mut text = String::new();
        let mut tool_chunks = 0usize;
        while let Some(chunk) = rx.recv().await {
            match chunk {
                BackendChunk::Text(t) | BackendChunk::Reasoning(t) => text.push_str(&t),
                BackendChunk::Tool(_) => tool_chunks += 1,
            }
        }
        (text, tool_chunks)
    };
    let (usage, (text, tool_chunks)) = tokio::join!(generate, collect);
    (usage.expect("generation completed"), text, tool_chunks)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "downloads/loads a real HF model and runs mistral.rs generation"]
async fn mistralrs_generates_text() {
    let (usage, text, _) = run(request(Vec::new())).await;
    eprintln!(
        "generated (in={} out={}): {text:?}",
        usage.input_tokens, usage.output_tokens
    );
    assert!(usage.output_tokens > 0, "expected output tokens");
    assert!(!text.trim().is_empty(), "expected non-empty generated text");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "downloads/loads a real HF model and runs mistral.rs generation"]
async fn mistralrs_offered_tools_keep_text_clean() {
    let tools = vec![ToolDef {
        name: "mnemosyne_recall".to_string(),
        schema: r#"{"type":"object","description":"Recall a stored memory.","properties":{"query":{"type":"string"}}}"#.to_string(),
    }];
    let (usage, text, tool_chunks) = run(request(tools)).await;
    eprintln!(
        "generated with tools (in={} out={} tool_chunks={tool_chunks}): {text:?}",
        usage.input_tokens, usage.output_tokens
    );
    // A 135M model may or may not decide to call the tool; what must hold is that the text
    // stream carries no tool-call markup (native decode owns that channel).
    assert!(
        !text.contains("<tool_call>") && !text.contains("</tool_call>"),
        "tool-call markup leaked into the text stream: {text:?}"
    );
    assert!(
        usage.output_tokens > 0 || tool_chunks > 0,
        "expected some output"
    );
}
