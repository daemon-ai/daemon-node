// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Real-GGUF llama.cpp generation tests for the engine lane (ignored by default).
//!
//! These are gated on the `llama` feature and `#[ignore]` so the default `cargo test --workspace`
//! (stub worker, no engine) never builds or runs them. They load an actual GGUF off disk and drive
//! one greedy generation through [`daemon_infer::backends`], asserting non-empty output.
//!
//! Intended to run in the Vulkan dev shell against the Vulkan-capable llama.cpp build:
//!
//! ```text
//! nix develop .#vulkan --command bash -c '
//!   cargo build -p daemon-infer --features llama,dynamic-link
//!   DAEMON_INFER_TEST_GGUF=/path/to/SmolLM2-135M-Instruct-Q2_K_L.gguf \
//!   DAEMON_INFER_TEST_NGL=99 \
//!   cargo test -p daemon-infer --features llama --test llama_inference -- --ignored --nocapture'
//! ```
//!
//! With `--nocapture`, llama.cpp's own load logs (e.g. `ggml_vulkan: Found N Vulkan devices`) print
//! to stderr, confirming the GPU device was actually selected (not a silent CPU fallback). The
//! deterministic GPU-build assertion is [`daemon_infer::backends::gpu_offload_supported`].
#![cfg(feature = "llama")]

use daemon_infer::backend::{BackendChunk, GenerateRequest};
use daemon_infer::backends;
use daemon_infer::protocol::{Engine, ModelParams, Msg, Sampling};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

/// The GGUF under test (a small instruct model, e.g. SmolLM2-135M Q2_K_L). Skips when unset.
fn model_path() -> Option<String> {
    std::env::var("DAEMON_INFER_TEST_GGUF")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// GPU layers to offload (`99` = "all"); `0` forces CPU. Defaults high so the test exercises Vulkan.
fn n_gpu_layers() -> u32 {
    std::env::var("DAEMON_INFER_TEST_NGL")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(99)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "loads a real GGUF and runs llama.cpp generation (set DAEMON_INFER_TEST_GGUF)"]
async fn llama_generates_text_on_gpu() {
    let Some(path) = model_path() else {
        eprintln!("skipping: set DAEMON_INFER_TEST_GGUF to a local .gguf to run this test");
        return;
    };
    let n_gpu_layers = n_gpu_layers();

    eprintln!("llama.cpp system info: {}", backends::system_info());
    assert!(
        backends::gpu_offload_supported(),
        "linked llama.cpp has no GPU backend — build the worker/prebuilt with --features vulkan"
    );

    let params = ModelParams {
        n_gpu_layers,
        n_ctx: 512,
        n_threads: None,
        flash_attn: false,
        isq: None,
        embeddings: false,
    };
    let backend = backends::load(Engine::Llama, &path, &params)
        .await
        .expect("load GGUF for generation");
    assert!(
        backend.capabilities().supports_streaming,
        "llama backend should stream"
    );

    let req = GenerateRequest {
        request_id: 1,
        system: "You are a helpful assistant.".to_string(),
        messages: vec![Msg {
            role: "user".to_string(),
            content: "Write one short sentence about the sky.".to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }],
        tools: Vec::new(),
        // Light sampling with a fixed seed: greedy decoding on a 135M Q2_K model tends to collapse
        // into a repeated control token, so a little temperature keeps the output textual.
        sampling: Sampling {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            seed: 42,
        },
        max_tokens: 64,
        constraint: None,
    };

    let (tx, mut rx) = mpsc::unbounded_channel();
    let cancel = CancellationToken::new();
    let generate = backend.generate(req, tx, cancel);
    let collect = async {
        let mut out = String::new();
        let mut chunks = 0usize;
        while let Some(chunk) = rx.recv().await {
            chunks += 1;
            match chunk {
                BackendChunk::Text(t) | BackendChunk::Reasoning(t) => out.push_str(&t),
                BackendChunk::Tool(_) => {}
            }
        }
        (out, chunks)
    };
    let (usage, (text, chunks)) = tokio::join!(generate, collect);
    let usage = usage.expect("generation completed");

    eprintln!(
        "generated (in={} out={} cache={} chunks={}): {text:?}",
        usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, chunks
    );
    assert!(
        usage.output_tokens > 0,
        "expected at least one output token"
    );
    assert!(
        !text.trim().is_empty(),
        "expected non-empty generated text from llama.cpp (out_tokens={}, chunks={})",
        usage.output_tokens,
        chunks
    );
}
