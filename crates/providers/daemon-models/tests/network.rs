// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Live-network integration tests (ignored by default: run with `--ignored` and network access).
//!
//! These hit the real Hugging Face Hub. They are excluded from the normal suite (which uses the
//! `wiremock`-backed `hf_mock.rs`) so CI stays hermetic.

use daemon_common::{ModelEngine, SearchQuery};
use daemon_models::{HfClient, ManagerConfig, ModelManager};

/// A stable, well-known GGUF repo for read-path checks.
const STABLE_REPO: &str = "TheBloke/Llama-2-7B-GGUF";

#[tokio::test]
#[ignore = "hits the live Hugging Face Hub"]
async fn live_search_returns_gguf_repos() {
    let client = HfClient::new(None);
    let query = SearchQuery::new("llama gguf", ModelEngine::Llama);
    let page = daemon_models::hf::search::search(&client, &query)
        .await
        .expect("live search");
    assert!(!page.results.is_empty(), "expected some GGUF repos");
}

#[tokio::test]
#[ignore = "hits the live Hugging Face Hub"]
async fn live_files_lists_quantized_ggufs() {
    let client = HfClient::new(None);
    let files =
        daemon_models::hf::files::list_files(&client, STABLE_REPO, "main", ModelEngine::Llama)
            .await
            .expect("live files");
    assert!(files.iter().any(|f| f.path.ends_with(".gguf")));
    assert!(files.iter().any(|f| f.quant.is_some()));
}

/// End-to-end acquire: list the smallest GGUF in a tiny test repo, download it through the manager,
/// and confirm it is cataloged with a valid on-disk artifact.
#[tokio::test]
#[ignore = "downloads a (small) GGUF from the live Hub"]
async fn live_pull_small_gguf_catalogs() {
    let cache_dir = std::env::temp_dir().join(format!("daemon-models-it-{}", std::process::id()));
    let manager = ModelManager::new(ManagerConfig {
        cache_dir: Some(cache_dir.clone()),
        registry_path: Some(cache_dir.join("catalog.json")),
        endpoint: None,
        ..Default::default()
    })
    .await
    .expect("manager");

    // A tiny GGUF used by llama.cpp CI keeps the transfer small.
    let repo = "ggml-org/models";
    let files = manager
        .model_files(repo, Some("main"), ModelEngine::Llama)
        .await
        .expect("files");
    let smallest = files
        .iter()
        .filter(|f| f.path.ends_with(".gguf") && f.size_bytes > 0)
        .min_by_key(|f| f.size_bytes)
        .expect("at least one gguf");

    let model = daemon_common::ModelRef::new(
        ModelEngine::Llama,
        daemon_common::ModelSource::hf_file(repo, &smallest.path),
    );
    let artifact = manager.resolve(&model).await.expect("resolve/download");
    assert!(artifact.local_path.exists(), "artifact on disk");
    assert!(daemon_models::gguf::verify_gguf_magic(&artifact.local_path).unwrap());

    let catalog = manager.catalog().await;
    assert!(catalog.iter().any(|m| m.model == model));

    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// The recommender picks a real GGUF file from a live repo that fits a generous budget.
#[tokio::test]
#[ignore = "hits the live Hugging Face Hub"]
async fn live_recommend_llama_picks_a_gguf() {
    let cache_dir = std::env::temp_dir().join(format!("daemon-models-rec-{}", std::process::id()));
    let manager = ModelManager::new(ManagerConfig {
        cache_dir: Some(cache_dir.clone()),
        registry_path: Some(cache_dir.join("catalog.json")),
        endpoint: None,
        ..Default::default()
    })
    .await
    .expect("manager");

    // A generous 64 GiB budget so a 7B repo certainly has a fitting quant.
    let rec = manager
        .recommend(
            STABLE_REPO,
            Some("main"),
            ModelEngine::Llama,
            Some(64 << 30),
        )
        .await
        .expect("recommend");
    assert!(rec.file.is_some(), "a GGUF file should be recommended");
    assert!(rec.fits, "a quant should fit a 64 GiB budget");
    assert!(!rec.candidates.is_empty());

    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// End-to-end quantize: download a tiny F16/high-precision GGUF and quantize it to Q4_K_M via the
/// llama-enabled worker. Requires `DAEMON_INFER_WORKER_BIN` to point at a worker built with
/// `--features llama` (the engine lane is cmake-based and not built in the default suite).
#[tokio::test]
#[ignore = "needs a llama-enabled worker (DAEMON_INFER_WORKER_BIN) + network"]
async fn live_quantize_small_gguf() {
    use daemon_common::QuantizeState;

    let Some(worker_bin) = std::env::var_os("DAEMON_INFER_WORKER_BIN") else {
        eprintln!("skipping: set DAEMON_INFER_WORKER_BIN to a llama-enabled worker");
        return;
    };
    let cache_dir = std::env::temp_dir().join(format!("daemon-models-qz-{}", std::process::id()));
    let manager = ModelManager::new(ManagerConfig {
        cache_dir: Some(cache_dir.clone()),
        registry_path: Some(cache_dir.join("catalog.json")),
        endpoint: None,
        quantize_worker_bin: Some(worker_bin.into()),
        ..Default::default()
    })
    .await
    .expect("manager");

    // A tiny single-model GGUF (full precision) keeps the download + quantize fast. We pass the source
    // explicitly because the `ggml-org/models` repo is a grab-bag of unrelated models.
    let id = manager
        .quantize(
            "ggml-org/models",
            Some("main"),
            "Q4_K_M",
            Some("tinyllamas/stories15M.gguf".to_string()),
        )
        .await
        .expect("start quantize");

    // Poll to completion (quantizing a tiny model is quick, but allow generous time).
    let mut final_state = QuantizeState::Queued;
    for _ in 0..600 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if let Some(s) = manager.quantizes().await.into_iter().find(|s| s.id == id) {
            final_state = s.state.clone();
            if matches!(s.state, QuantizeState::Completed | QuantizeState::Failed) {
                if let Some(err) = s.error {
                    panic!("quantize failed: {err}");
                }
                assert!(s.output_path.is_some_and(|p| p.exists()));
                break;
            }
        }
    }
    assert_eq!(final_state, QuantizeState::Completed);

    let _ = std::fs::remove_dir_all(&cache_dir);
}

/// A `ModelManager` that downloads into the *real* shared HF cache (so the artifact persists for the
/// `daemon-infer` Vulkan inference/embedding tests), but keeps its catalog manifest in a throwaway
/// temp file so we do not clutter the cache directory.
async fn shared_cache_manager(tag: &str) -> ModelManager {
    let registry =
        std::env::temp_dir().join(format!("daemon-models-{tag}-{}.json", std::process::id()));
    ModelManager::new(ManagerConfig {
        cache_dir: None, // follow HF_*/XDG precedence so downloads land in the shared hub cache
        registry_path: Some(registry),
        endpoint: None,
        ..Default::default()
    })
    .await
    .expect("manager")
}

/// Full chat-model flow against the real Hub: search -> list files (the `_L` quant labels we added
/// must now resolve `Q2_K_L`) -> recommend -> download the chosen `SmolLM2-135M-Instruct-Q2_K_L.gguf`
/// into the shared cache -> verify the GGUF magic. Prints the on-disk path as
/// `DAEMON_INFER_TEST_GGUF=<path>` so the Vulkan inference test can consume it.
#[tokio::test]
#[ignore = "downloads SmolLM2-135M Q2_K_L (~90 MB) from the live Hub"]
async fn live_smollm2_search_files_recommend_download() {
    const REPO: &str = "bartowski/SmolLM2-135M-Instruct-GGUF";
    const FILE: &str = "SmolLM2-135M-Instruct-Q2_K_L.gguf";

    let manager = shared_cache_manager("smollm2").await;

    // Step 1: search surfaces the repo.
    let page = manager
        .search(SearchQuery::new(
            "SmolLM2 Instruct GGUF",
            ModelEngine::Llama,
        ))
        .await
        .expect("search");
    assert!(!page.results.is_empty(), "expected SmolLM2 search hits");
    assert!(
        page.results.iter().any(|h| h.repo.contains("SmolLM2")),
        "expected a SmolLM2 repo among hits: {:?}",
        page.results.iter().map(|h| &h.repo).collect::<Vec<_>>()
    );

    // Step 2: list files — the Q2_K_L embed/output variant must be present AND labeled.
    let files = manager
        .model_files(REPO, Some("main"), ModelEngine::Llama)
        .await
        .expect("files");
    let q2kl = files.iter().find(|f| f.path == FILE).unwrap_or_else(|| {
        panic!(
            "repo should expose {FILE}; saw {:?}",
            files.iter().map(|f| &f.path).collect::<Vec<_>>()
        )
    });
    assert_eq!(
        q2kl.quant.as_deref(),
        Some("Q2_K_L"),
        "Q2_K_L must be recognized after the quant-label fix"
    );

    // The recommender should pick a real, labeled GGUF for a generous budget.
    let rec = manager
        .recommend(REPO, Some("main"), ModelEngine::Llama, Some(64 << 30))
        .await
        .expect("recommend");
    assert!(rec.file.is_some(), "a GGUF file should be recommended");
    assert!(!rec.candidates.is_empty());

    // Step 3: download the specific Q2_K_L file and verify it on disk.
    let model = daemon_common::ModelRef::new(
        ModelEngine::Llama,
        daemon_common::ModelSource::hf_file(REPO, FILE),
    );
    let artifact = manager.resolve(&model).await.expect("resolve/download");
    assert!(artifact.local_path.exists(), "artifact on disk");
    assert!(
        daemon_models::gguf::verify_gguf_magic(&artifact.local_path).unwrap(),
        "downloaded file must start with the GGUF magic"
    );
    println!("DAEMON_INFER_TEST_GGUF={}", artifact.local_path.display());
}

/// Full embedding-model flow: list files for the all-MiniLM-L12-v2 GGUF repo, download the small
/// `Q2_K` quant into the shared cache, and verify the GGUF magic. Prints the on-disk path as
/// `DAEMON_INFER_TEST_EMBED_GGUF=<path>` for the embedding test.
#[tokio::test]
#[ignore = "downloads all-MiniLM-L12-v2 Q2_K (~25 MB) from the live Hub"]
async fn live_minilm_embed_files_download() {
    const REPO: &str = "leliuga/all-MiniLM-L12-v2-GGUF";
    const FILE: &str = "all-MiniLM-L12-v2.Q2_K.gguf";

    let manager = shared_cache_manager("minilm").await;

    let files = manager
        .model_files(REPO, Some("main"), ModelEngine::Llama)
        .await
        .expect("files");
    assert!(
        files.iter().any(|f| f.path == FILE),
        "repo should expose {FILE}; saw {:?}",
        files.iter().map(|f| &f.path).collect::<Vec<_>>()
    );

    let model = daemon_common::ModelRef::new(
        ModelEngine::Llama,
        daemon_common::ModelSource::hf_file(REPO, FILE),
    );
    let artifact = manager.resolve(&model).await.expect("resolve/download");
    assert!(artifact.local_path.exists(), "artifact on disk");
    assert!(
        daemon_models::gguf::verify_gguf_magic(&artifact.local_path).unwrap(),
        "downloaded file must start with the GGUF magic"
    );
    println!(
        "DAEMON_INFER_TEST_EMBED_GGUF={}",
        artifact.local_path.display()
    );
}
