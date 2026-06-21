//! HF read-surface tests against an in-process `wiremock` server (no live network).

use daemon_common::{ModelEngine, SearchQuery};
use daemon_models::hf::{files, search, HfClient};
use serde_json::json;
use wiremock::matchers::{method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

#[tokio::test]
async fn search_parses_and_filters_gguf() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/models"))
        .and(query_param("filter", "gguf"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!([
            {
                "id": "TheBloke/Llama-2-7B-GGUF",
                "downloads": 12345,
                "likes": 67,
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
        ])))
        .mount(&server)
        .await;

    let client = HfClient::with_endpoint(server.uri(), None);
    let query = SearchQuery::new("llama", ModelEngine::Llama);
    let page = search::search(&client, &query).await.expect("search");
    assert_eq!(page.results.len(), 2);
    let first = &page.results[0];
    assert_eq!(first.repo, "TheBloke/Llama-2-7B-GGUF");
    assert_eq!(first.author.as_deref(), Some("TheBloke"));
    assert_eq!(first.downloads, 12345);
    assert!(!first.gated);
    assert!(page.results[1].gated, "string 'auto' means gated");
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
    let q4 = files.iter().find(|f| f.path == "Model-Q4_K_M.gguf").unwrap();
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
    let all = files::list_all(&client, "org/mistral", "main").await.unwrap();
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
