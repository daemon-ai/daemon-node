// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Phase 4: integration test crate; raw fs/reqwest/Command are expected in tests.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

//! The wire v48 quant filter/enrichment at the [`ModelManager`] level, against an in-process
//! `wiremock` Hub (no live network): canonical-family normalization over file listings, the
//! cache-first (never blocking) no-filter enrichment, the filtered fetch with indeterminate
//! hits kept, repo-name families on the repository strategy, and the SWR files cache absorbing
//! repeat listings.

use daemon_common::{ModelEngine, SearchQuery};
use daemon_models::{ManagerConfig, ModelManager};
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir =
        std::env::temp_dir().join(format!("daemon-models-quant-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn manager_over(dir: &std::path::Path, endpoint: String) -> ModelManager {
    ModelManager::new(ManagerConfig {
        cache_dir: Some(dir.join("hub")),
        fallback_cache_dir: None,
        registry_path: Some(dir.join("catalog.json")),
        endpoint: Some(endpoint),
        quantize_worker_bin: None,
    })
    .await
    .expect("manager")
}

fn hit_row(id: &str) -> serde_json::Value {
    json!({
        "id": id,
        "downloads": 10,
        "likes": 1,
        "pipeline_tag": "text-generation",
        "gated": false,
        "private": false
    })
}

fn tree_row(path: &str) -> serde_json::Value {
    json!({"type": "file", "path": path, "size": 1_000_000u64})
}

async fn mount_search(server: &MockServer, ids: &[&str]) {
    let models: Vec<serde_json::Value> = ids.iter().map(|id| hit_row(id)).collect();
    Mock::given(method("GET"))
        .and(path("/models-json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "numTotalItems": models.len(),
            "models": models
        })))
        .mount(server)
        .await;
}

async fn mount_tree(server: &MockServer, repo: &str, files: &[&str]) {
    let rows: Vec<serde_json::Value> = files.iter().map(|f| tree_row(f)).collect();
    Mock::given(method("GET"))
        .and(path(format!("/api/models/{repo}/tree/main")))
        .respond_with(ResponseTemplate::new(200).set_body_json(rows))
        .mount(server)
        .await;
}

/// With a quant selection active the node fetches the page's file listings, folds raw labels
/// into canonical families, serves only matching hits (indeterminate ones KEPT), and marks the
/// page `quant_filter_applied`.
#[tokio::test]
async fn quant_filter_serves_only_matching_hits_and_keeps_indeterminate() {
    let server = MockServer::start().await;
    mount_search(&server, &["org/q4-repo", "org/f16-repo", "org/broken"]).await;
    mount_tree(
        &server,
        "org/q4-repo",
        &[
            "Model-Q4_K_M.gguf",
            "Model-Q4_K_S.gguf",
            "Model-IQ2_XS.gguf",
        ],
    )
    .await;
    mount_tree(&server, "org/f16-repo", &["Model-F16.gguf"]).await;
    Mock::given(method("GET"))
        .and(path("/api/models/org/broken/tree/main"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;

    let dir = temp_dir("filter");
    let manager = manager_over(&dir, server.uri()).await;
    let mut query = SearchQuery::new("model", "llama_cpp");
    query.quants = Some(vec!["Q4".to_string()]);
    let page = manager
        .search(ModelEngine::Llama, query)
        .await
        .expect("filtered search");

    assert!(page.quant_filter_applied);
    let repos: Vec<&str> = page.results.iter().map(|h| h.repo.as_str()).collect();
    assert!(repos.contains(&"org/q4-repo"), "matching hit served");
    assert!(
        !repos.contains(&"org/f16-repo"),
        "non-matching hit filtered out"
    );
    assert!(
        repos.contains(&"org/broken"),
        "indeterminate hit (listing failed) is kept, never hidden"
    );
    let q4 = page
        .results
        .iter()
        .find(|h| h.repo == "org/q4-repo")
        .unwrap();
    assert_eq!(
        q4.quants.as_deref(),
        Some(["Q4".to_string(), "IQ".to_string()].as_slice()),
        "distinct families in canonical order"
    );
    let broken = page
        .results
        .iter()
        .find(|h| h.repo == "org/broken")
        .unwrap();
    assert_eq!(broken.quants, None, "indeterminate hit stays unenriched");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Without a quant selection the page is served immediately — a cold cache serves
/// `quants: None`, never blocking on the Hub — while a detached warm task fetches the page's
/// listings in the background. A follow-up serve of the same query then carries the families
/// from the cache (the progressive-enrichment contract the client's page-0 re-read consumes),
/// and the claim guard + SWR cache keep it at exactly ONE tree request total.
#[tokio::test]
async fn no_filter_serves_immediately_and_warms_in_background() {
    let server = MockServer::start().await;
    mount_search(&server, &["org/repo"]).await;
    mount_tree(&server, "org/repo", &["Model-Q8_0.gguf", "Model-BF16.gguf"]).await;

    let dir = temp_dir("swr");
    let manager = manager_over(&dir, server.uri()).await;

    let page = manager
        .search(ModelEngine::Llama, SearchQuery::new("model", "llama_cpp"))
        .await
        .expect("search");
    assert!(!page.quant_filter_applied);
    assert_eq!(
        page.results[0].quants, None,
        "cold cache: served unenriched, not blocked on the warm"
    );

    // The detached warm lands shortly after; a re-serve of the query picks the families up.
    let mut served = None;
    for _ in 0..100 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let page = manager
            .search(ModelEngine::Llama, SearchQuery::new("model", "llama_cpp"))
            .await
            .expect("re-serve");
        if let Some(q) = page.results[0].quants.clone() {
            served = Some(q);
            break;
        }
    }
    assert_eq!(
        served.as_deref(),
        Some(["Q8".to_string(), "BF16".to_string()].as_slice()),
        "background warm enriched the follow-up serve"
    );

    // Claim guard + SWR cache: the repeat serves above spawned no duplicate fetches.
    let tree_hits = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .filter(|r| r.url.path().contains("/tree/"))
        .count();
    assert_eq!(
        tree_hits, 1,
        "one warm fetch total — repeats absorbed by the claim guard and SWR cache"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The repository strategy (mistral.rs) reads families off the repo NAME (method formats like
/// GPTQ / bnb-4bit) — free to compute, always served, filterable without any tree fetch.
#[tokio::test]
async fn repository_strategy_families_ride_the_repo_name() {
    let server = MockServer::start().await;
    mount_search(
        &server,
        &[
            "TheBloke/Llama-2-7B-GPTQ",
            "unsloth/llama-3-8b-bnb-4bit",
            "mistralai/Mistral-7B-v0.1",
        ],
    )
    .await;

    let dir = temp_dir("repo-name");
    let manager = manager_over(&dir, server.uri()).await;

    // Unfiltered: every hit carries its name-derived families.
    let page = manager
        .search(
            ModelEngine::MistralRs,
            SearchQuery::new("llama", "mistral_rs"),
        )
        .await
        .expect("search");
    let by_repo = |r: &str| {
        page.results
            .iter()
            .find(|h| h.repo == r)
            .unwrap()
            .quants
            .clone()
    };
    assert_eq!(
        by_repo("TheBloke/Llama-2-7B-GPTQ"),
        Some(vec!["GPTQ".into()])
    );
    assert_eq!(
        by_repo("unsloth/llama-3-8b-bnb-4bit"),
        Some(vec!["BNB4".into()])
    );
    assert_eq!(
        by_repo("mistralai/Mistral-7B-v0.1"),
        Some(Vec::new()),
        "a plain repo name yields no families (known, empty)"
    );

    // Filtered: only the GPTQ repo survives; no tree endpoint is ever consulted.
    let mut query = SearchQuery::new("llama", "mistral_rs");
    query.quants = Some(vec!["GPTQ".to_string()]);
    let page = manager
        .search(ModelEngine::MistralRs, query)
        .await
        .expect("filtered search");
    assert!(page.quant_filter_applied);
    let repos: Vec<&str> = page.results.iter().map(|h| h.repo.as_str()).collect();
    assert_eq!(repos, vec!["TheBloke/Llama-2-7B-GPTQ"]);
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|r| !r.url.path().contains("/tree/")),
        "repo-name enrichment never fetches listings"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
