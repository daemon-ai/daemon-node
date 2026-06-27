//! Capstone: drive the REAL Vulkan-enabled `daemon-infer` worker through the daemon's own
//! [`LocalProvider`] / [`LocalEmbedder`] wiring — the exact supervised-worker path the node composes
//! (spawn -> `Command::Load` -> `Generate`/`Embed` over the length-framed protocol cut). This proves
//! the full local-inference stack end-to-end on the GPU, not just the in-process backend.
//!
//! Ignored + env-gated; run in the Vulkan dev shell after building a llama worker:
//!
//! ```text
//! nix develop .#vulkan --command bash -c '
//!   cargo build -p daemon-infer --features llama,dynamic-link
//!   export DAEMON_INFER_WORKER_BIN="$PWD/target/debug/daemon-infer"
//!   export DAEMON_INFER_TEST_GGUF=/path/to/SmolLM2-135M-Instruct-Q2_K_L.gguf
//!   export DAEMON_INFER_TEST_EMBED_GGUF=/path/to/all-MiniLM-L12-v2.Q2_K.gguf
//!   cargo test -p daemon-providers --test local_provider_vulkan -- --ignored --nocapture'
//! ```

use std::path::PathBuf;
use std::time::Duration;

use daemon_core::{EmbeddingProvider, Provider, Request, RequestMsg};
use daemon_infer::protocol::{Engine, ModelParams, Sampling};
use daemon_providers::{LocalEmbedder, LocalProvider, WorkerConfig};

fn worker_bin() -> Option<PathBuf> {
    std::env::var_os("DAEMON_INFER_WORKER_BIN")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
}

fn n_gpu_layers() -> u32 {
    std::env::var("DAEMON_INFER_TEST_NGL")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(99)
}

fn llama_worker(bin: PathBuf, model: &str, embeddings: bool) -> WorkerConfig {
    let mut wc = WorkerConfig::new(bin, Engine::Llama, model);
    wc.params = ModelParams {
        n_gpu_layers: n_gpu_layers(),
        n_ctx: 512,
        n_threads: None,
        flash_attn: false,
        isq: None,
        embeddings,
    };
    wc.sampling = Sampling {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 40,
        seed: 42,
    };
    wc.max_tokens = 64;
    wc.load_timeout = Duration::from_secs(120);
    wc.ttft_timeout = Duration::from_secs(60);
    wc.inter_token_timeout = Duration::from_secs(30);
    wc
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "drives the real Vulkan worker (set DAEMON_INFER_WORKER_BIN + DAEMON_INFER_TEST_GGUF)"]
async fn local_provider_chats_on_vulkan() {
    let (Some(bin), Ok(model)) = (worker_bin(), std::env::var("DAEMON_INFER_TEST_GGUF")) else {
        eprintln!("skipping: set DAEMON_INFER_WORKER_BIN and DAEMON_INFER_TEST_GGUF");
        return;
    };
    let provider = LocalProvider::new(llama_worker(bin, &model, false));
    let req = Request {
        system: "You are a helpful assistant.".to_string(),
        messages: vec![RequestMsg {
            role: "user".to_string(),
            content: "Write one short sentence about the sky.".to_string(),
            ..Default::default()
        }],
        tools: Vec::new(),
        auth: None,
        constraint: None,
        cache_system: false,
    };
    let out = provider.chat(req).await.expect("chat ok");
    eprintln!(
        "LocalProvider(Vulkan) chat -> out_tokens={} text={:?}",
        out.usage.output_tokens, out.text
    );
    assert!(
        !out.text.trim().is_empty(),
        "expected non-empty text from the supervised Vulkan worker"
    );
    assert!(out.usage.output_tokens > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "drives the real Vulkan worker (set DAEMON_INFER_WORKER_BIN + DAEMON_INFER_TEST_EMBED_GGUF)"]
async fn local_embedder_embeds_on_vulkan() {
    let (Some(bin), Ok(model)) = (worker_bin(), std::env::var("DAEMON_INFER_TEST_EMBED_GGUF"))
    else {
        eprintln!("skipping: set DAEMON_INFER_WORKER_BIN and DAEMON_INFER_TEST_EMBED_GGUF");
        return;
    };
    // `LocalEmbedder::new` forces `params.embeddings = true`; this is the same path `build_embedder`
    // takes in the daemon after `ModelManager.resolve` hands it a cached GGUF.
    let embedder = LocalEmbedder::new(llama_worker(bin, &model, true), 384, "all-MiniLM-L12-v2");
    let vectors = embedder
        .embed(&[
            "The cat sat on the warm windowsill.".to_string(),
            "A kitten rested by the sunny window.".to_string(),
            "Quarterly revenue exceeded analyst expectations.".to_string(),
        ])
        .await
        .expect("embed ok");

    assert_eq!(vectors.len(), 3);
    assert_eq!(vectors[0].len(), 384);
    assert_eq!(embedder.dimensions(), 384);

    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let related = dot(&vectors[0], &vectors[1]);
    let unrelated = dot(&vectors[0], &vectors[2]);
    eprintln!("LocalEmbedder(Vulkan) cosine related={related:.4} unrelated={unrelated:.4}");
    assert!(
        related > unrelated,
        "related sentences ({related:.4}) should out-score the unrelated one ({unrelated:.4})"
    );
}
