// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! HF read-surface tests against an in-process `wiremock` server (no live network).

use daemon_common::{ModelEngine, SearchQuery};
use daemon_models::hf::{files, search, HfClient};
use serde_json::json;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The primary `models-json` search: rich filters (library + pipeline + params buckets), a total
/// count, numeric-parameter coercion, and the node-supplied canonical `web_url`.
#[tokio::test]
async fn search_uses_models_json_with_filters_and_total() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models-json"))
        .and(query_param("library", "gguf"))
        .and(query_param("pipeline_tag", "text-generation"))
        .and(query_param("search", "llama"))
        .and(query_param("withCount", "true"))
        .and(query_param("num_parameters", "min:7B,max:32B"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "numTotalItems": 2,
            "models": [
                {
                    "id": "TheBloke/Llama-2-7B-GGUF",
                    "downloads": 12345,
                    "likes": 67,
                    "numParameters": 6_738_415_616u64,
                    "pipeline_tag": "text-generation",
                    "lastModified": "2024-01-02T03:04:05.000Z",
                    "gated": false,
                    "private": false
                },
                {
                    "id": "org/Other-GGUF",
                    "downloads": 9,
                    "likes": 1,
                    "gated": "auto",
                    "private": false
                }
            ]
        })))
        .mount(&server)
        .await;

    let client = HfClient::with_endpoint(server.uri(), None);
    let mut query = SearchQuery::new("llama", "llama_cpp");
    query.params_min = Some(7_000_000_000);
    query.params_max = Some(32_000_000_000);
    let page = search::search(&client, ModelEngine::Llama, &query)
        .await
        .expect("search");
    assert_eq!(page.results.len(), 2);
    assert_eq!(page.total, Some(2));
    assert!(page.params_filter_applied);
    assert!(!page.degraded);
    assert!(!page.has_more, "2 of 2 served on page 0");
    let first = &page.results[0];
    assert_eq!(first.repo, "TheBloke/Llama-2-7B-GGUF");
    assert_eq!(first.author.as_deref(), Some("TheBloke"));
    assert_eq!(first.downloads, 12345);
    assert_eq!(first.num_parameters, Some(6_738_415_616));
    assert_eq!(
        first.web_url.as_deref(),
        Some(format!("{}/TheBloke/Llama-2-7B-GGUF", server.uri()).as_str())
    );
    assert!(!first.gated);
    assert!(page.results[1].gated, "string 'auto' means gated");
}

/// A primary-endpoint 5xx falls back (bounded) to `/api/models`, marked `degraded`: no total, no
/// parameter filter.
#[tokio::test]
async fn search_falls_back_to_api_models_on_5xx_as_degraded() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models-json"))
        .respond_with(ResponseTemplate::new(500))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("filter", "gguf"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {
                "id": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 12345,
                "likes": 67,
                "pipeline_tag": "text-generation",
                "gated": false,
                "private": false
            }
        ])))
        .mount(&server)
        .await;

    let client = HfClient::with_endpoint(server.uri(), None);
    let mut query = SearchQuery::new("llama", "llama_cpp");
    query.params_min = Some(7_000_000_000);
    let page = search::search(&client, ModelEngine::Llama, &query)
        .await
        .expect("fallback search");
    assert_eq!(page.results.len(), 1);
    assert!(page.degraded, "fallback pages are marked degraded");
    assert_eq!(page.total, None, "the fallback endpoint has no total");
    assert!(
        !page.params_filter_applied,
        "the fallback endpoint cannot filter by parameters"
    );
}

/// Response drift on the primary (a payload without `models`) also takes the bounded fallback.
#[tokio::test]
async fn search_falls_back_on_primary_response_drift() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models-json"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"unexpected": true})))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
        .mount(&server)
        .await;

    let client = HfClient::with_endpoint(server.uri(), None);
    let query = SearchQuery::new("llama", "llama_cpp");
    let page = search::search(&client, ModelEngine::Llama, &query)
        .await
        .expect("drift falls back");
    assert!(page.degraded);
    assert!(page.results.is_empty());
}

/// A 4xx from the primary is a real error — never silently-unfiltered fallback results.
#[tokio::test]
async fn search_4xx_is_an_error_not_a_fallback() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models-json"))
        .respond_with(ResponseTemplate::new(400))
        .mount(&server)
        .await;
    // No /api/models mock: a fallback attempt would 404 and fail the test differently.

    let client = HfClient::with_endpoint(server.uri(), None);
    let query = SearchQuery::new("llama", "llama_cpp");
    let err = search::search(&client, ModelEngine::Llama, &query)
        .await
        .expect_err("a 4xx surfaces as an error");
    let requests = server.received_requests().await.expect("captured");
    assert!(
        requests.iter().all(|r| r.url.path() != "/api/models"),
        "the fallback endpoint must not be consulted on a 4xx"
    );
    let _ = err;
}

#[tokio::test]
async fn files_lists_gguf_with_quant_and_shards() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models/org/repo/tree/main"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"type": "file", "path": "README.md", "size": 100},
            {"type": "file", "path": "Model-Q4_K_M.gguf", "size": 4_000_000_000u64},
            {"type": "file", "path": "Model-Q8_0-00001-of-00002.gguf", "size": 5_000_000_000u64},
            {"type": "file", "path": "Model-Q8_0-00002-of-00002.gguf", "size": 5_000_000_000u64},
            {"type": "directory", "path": "subdir"}
        ])))
        .mount(&server)
        .await;

    let client = HfClient::with_endpoint(server.uri(), None);
    let files = files::list_files(&client, "org/repo", "main", ModelEngine::Llama)
        .await
        .expect("files");
    // README + directory dropped; three GGUF files kept.
    assert_eq!(files.len(), 3);
    let q4 = files
        .iter()
        .find(|f| f.path == "Model-Q4_K_M.gguf")
        .unwrap();
    assert_eq!(q4.quant.as_deref(), Some("Q4_K_M"));
    assert!(!q4.is_split);
    let shard1 = files
        .iter()
        .find(|f| f.path == "Model-Q8_0-00001-of-00002.gguf")
        .unwrap();
    assert!(shard1.is_split);
    assert!(shard1.is_first_shard);
}

#[tokio::test]
async fn files_for_mistralrs_keeps_repo_siblings() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models/org/mistral/tree/main"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {"type": "file", "path": "config.json", "size": 700},
            {"type": "file", "path": "tokenizer.json", "size": 2_000_000},
            {"type": "file", "path": "model.safetensors", "size": 9_000_000_000u64},
            {"type": "file", "path": "notes.txt", "size": 10}
        ])))
        .mount(&server)
        .await;

    let client = HfClient::with_endpoint(server.uri(), None);
    let files = files::list_files(&client, "org/mistral", "main", ModelEngine::MistralRs)
        .await
        .expect("files");
    let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
    assert!(paths.contains(&"config.json"));
    assert!(paths.contains(&"tokenizer.json"));
    assert!(paths.contains(&"model.safetensors"));
    assert!(!paths.contains(&"notes.txt"));

    // `list_all` returns every file (the mistral.rs prewarm set).
    let all = files::list_all(&client, "org/mistral", "main")
        .await
        .unwrap();
    assert_eq!(all.len(), 4);
}

#[tokio::test]
async fn not_found_maps_to_error() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models/missing/repo/tree/main"))
        .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
        .mount(&server)
        .await;
    let client = HfClient::with_endpoint(server.uri(), None);
    let err = files::list_files(&client, "missing/repo", "main", ModelEngine::Llama)
        .await
        .unwrap_err();
    assert!(matches!(err, daemon_models::ModelError::NotFound(_)));
}
