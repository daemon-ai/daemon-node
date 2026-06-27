//! Real-GGUF llama.cpp embedding tests for the engine lane (ignored by default).
//!
//! Gated on the `llama` feature and `#[ignore]`, like `llama_inference.rs`. Loads an embedding GGUF
//! (e.g. all-MiniLM-L12-v2 Q2_K) in embedding mode and asserts the pooled, L2-normalized vectors are
//! the expected dimensionality and semantically ordered (similar texts score higher than unrelated
//! ones). Runs in the Vulkan dev shell against the Vulkan-capable llama.cpp build:
//!
//! ```text
//! nix develop .#vulkan --command bash -c '
//!   cargo build -p daemon-infer --features llama,dynamic-link
//!   DAEMON_INFER_TEST_EMBED_GGUF=/path/to/all-MiniLM-L12-v2.Q2_K.gguf \
//!   DAEMON_INFER_TEST_NGL=99 \
//!   cargo test -p daemon-infer --features llama --test llama_embed -- --ignored --nocapture'
//! ```
#![cfg(feature = "llama")]

use daemon_infer::backends;
use daemon_infer::protocol::{Engine, ModelParams};

/// The embedding GGUF under test (e.g. all-MiniLM-L12-v2 Q2_K). Skips when unset.
fn embed_model_path() -> Option<String> {
    std::env::var("DAEMON_INFER_TEST_EMBED_GGUF")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn n_gpu_layers() -> u32 {
    std::env::var("DAEMON_INFER_TEST_NGL")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(99)
}

/// Dot product; for L2-normalized vectors this is cosine similarity.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "loads a real embedding GGUF and runs llama.cpp embeddings (set DAEMON_INFER_TEST_EMBED_GGUF)"]
async fn llama_embeds_text_on_gpu() {
    let Some(path) = embed_model_path() else {
        eprintln!("skipping: set DAEMON_INFER_TEST_EMBED_GGUF to a local embedding .gguf");
        return;
    };
    let n_gpu_layers = n_gpu_layers();

    eprintln!("llama.cpp system info: {}", backends::system_info());
    assert!(
        backends::gpu_offload_supported(),
        "linked llama.cpp has no GPU backend — build the worker/prebuilt with --features vulkan"
    );

    // Embedding mode: a pooled-embedding context (mean pool + L2 norm) rather than generation.
    let params = ModelParams {
        n_gpu_layers,
        n_ctx: 0, // use the model's trained context
        n_threads: None,
        flash_attn: false,
        isq: None,
        embeddings: true,
    };
    let backend = backends::load(Engine::Llama, &path, &params)
        .await
        .expect("load embedding GGUF");

    let texts = vec![
        "The cat sat on the warm windowsill.".to_string(),
        "A kitten rested by the sunny window.".to_string(),
        "Quarterly revenue exceeded analyst expectations.".to_string(),
    ];
    let vectors = backend.embed(texts.clone()).await.expect("embed batch");

    assert_eq!(vectors.len(), texts.len(), "one vector per input");
    // all-MiniLM-L12-v2 is a 384-dim BERT embedder.
    for (i, v) in vectors.iter().enumerate() {
        eprintln!("vec[{i}] dims={} norm={:.4}", v.len(), l2_norm(v));
        assert_eq!(
            v.len(),
            384,
            "all-MiniLM-L12-v2 produces 384-dim embeddings"
        );
        assert!(
            (l2_norm(v) - 1.0).abs() < 1e-3,
            "embeddings should be L2-normalized (got norm {})",
            l2_norm(v)
        );
    }

    // Semantic ordering: the two cat/window sentences should be closer to each other than either is
    // to the unrelated finance sentence.
    let sim_related = dot(&vectors[0], &vectors[1]);
    let sim_unrelated_a = dot(&vectors[0], &vectors[2]);
    let sim_unrelated_b = dot(&vectors[1], &vectors[2]);
    eprintln!(
        "cosine related={sim_related:.4} unrelated_a={sim_unrelated_a:.4} unrelated_b={sim_unrelated_b:.4}"
    );
    assert!(
        sim_related > sim_unrelated_a && sim_related > sim_unrelated_b,
        "related sentences ({sim_related:.4}) should out-score unrelated ones \
         ({sim_unrelated_a:.4}, {sim_unrelated_b:.4})"
    );
}
