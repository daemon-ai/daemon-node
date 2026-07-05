// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Web-tool coverage: hosted-backend auth+parse against an in-process `wiremock` server, the local
//! readability fallback against a served HTML page, and tool-level dispatch (`web_search` /
//! `web_extract`) through a hand-built `TurnCx` with fake backends — including the egress reject and
//! the keyed -> local fall-through.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use daemon_common::{Budget, SessionId};
use daemon_core::events::EventSink;
use daemon_core::exec::LocalEnvironment;
use daemon_core::{Tool, ToolCall, TurnCx};
use daemon_protocol::{HostRequest, HostRequestHandler, HostResponse, HostResponseBody};
use daemon_tool_web::{
    FetchOpts, FetchedDoc, FirecrawlFetch, LocalFetch, SearchOpts, SearchResults, SearchTopic,
    SecretSource, TavilySearch, WebError, WebExtractTool, WebFetchBackend, WebSearchBackend,
    WebSearchTool,
};
use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// An in-memory [`SecretSource`] for tests.
struct MapSecrets(HashMap<String, String>);

impl MapSecrets {
    fn with(key: &str, val: &str) -> Arc<dyn SecretSource> {
        let mut m = HashMap::new();
        m.insert(key.to_string(), val.to_string());
        Arc::new(MapSecrets(m))
    }
    fn empty() -> Arc<dyn SecretSource> {
        Arc::new(MapSecrets(HashMap::new()))
    }
}

impl SecretSource for MapSecrets {
    fn secret(&self, key: &str) -> Option<String> {
        self.0.get(key).cloned()
    }
}

#[tokio::test]
async fn tavily_sends_bearer_and_parses_hits() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/search"))
        .and(header("authorization", "Bearer tav-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "query": "rust async",
            "answer": "Async Rust uses futures.",
            "results": [
                {"title": "Tokio", "url": "https://tokio.rs", "content": "runtime", "score": 0.9},
                {"title": "Futures", "url": "https://docs.rs/futures", "content": "combinators"}
            ]
        })))
        .mount(&server)
        .await;

    let backend = TavilySearch::new(MapSecrets::with("tavily", "tav-key"))
        .with_endpoint(format!("{}/search", server.uri()));
    let opts = SearchOpts {
        max_results: 5,
        topic: SearchTopic::General,
    };
    let results = backend
        .search("rust async", &opts)
        .await
        .expect("search ok");
    assert_eq!(results.provider, "tavily");
    assert_eq!(results.answer.as_deref(), Some("Async Rust uses futures."));
    assert_eq!(results.hits.len(), 2);
    assert_eq!(results.hits[0].title, "Tokio");
    assert_eq!(results.hits[0].score, Some(0.9));
}

#[tokio::test]
async fn tavily_missing_key_is_reported() {
    let backend = TavilySearch::new(MapSecrets::empty());
    let opts = SearchOpts {
        max_results: 5,
        topic: SearchTopic::General,
    };
    let err = backend.search("q", &opts).await.expect_err("missing key");
    assert!(matches!(err, WebError::MissingKey(k) if k == "tavily"));
}

#[tokio::test]
async fn firecrawl_sends_bearer_and_parses_markdown() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/scrape"))
        .and(header("authorization", "Bearer fc-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": {
                "markdown": "# Title\n\nBody text.",
                "metadata": {"title": "Title"}
            }
        })))
        .mount(&server)
        .await;

    let backend = FirecrawlFetch::new(MapSecrets::with("firecrawl", "fc-key"))
        .with_endpoint(format!("{}/v1/scrape", server.uri()));
    assert!(backend.available(), "keyed backend is available");
    let doc = backend
        .fetch("https://example.com", &FetchOpts::default())
        .await
        .expect("fetch ok");
    assert_eq!(doc.provider, "firecrawl");
    assert_eq!(doc.title.as_deref(), Some("Title"));
    assert!(doc.content.contains("Body text."));
}

#[tokio::test]
async fn firecrawl_unavailable_without_key() {
    let backend = FirecrawlFetch::new(MapSecrets::empty());
    assert!(!backend.available(), "no key => skip in the chain");
}

#[tokio::test]
async fn local_fetch_extracts_html_to_markdown() {
    let server = MockServer::start().await;
    let html = r#"<!doctype html><html><head><title>Hello Daemon</title></head>
        <body><article><h1>Hello Daemon</h1>
        <p>This is the main article body with enough words to satisfy readability heuristics so the
        extractor keeps it as the primary content rather than discarding it as boilerplate.</p>
        <p>A second paragraph adds more substantial text to the document body.</p>
        </article></body></html>"#;
    Mock::given(method("GET"))
        .and(path("/article"))
        .respond_with(ResponseTemplate::new(200).set_body_string(html))
        .mount(&server)
        .await;

    let backend = LocalFetch::new();
    // Calling the backend directly bypasses the tool-level egress check, so the 127.0.0.1 mock is
    // reachable (the SSRF guard lives in WebExtractTool::run, tested separately).
    let doc = backend
        .fetch(&format!("{}/article", server.uri()), &FetchOpts::default())
        .await
        .expect("local fetch ok");
    assert_eq!(doc.provider, "local");
    assert!(doc.content.contains("main article body"));
}

/// Bug repro: a page that `302`s to link-local metadata space must be rejected mid-chain, not
/// followed. The initial (loopback-mock) URL is intentionally reachable — only the redirect hop is
/// re-validated by the shared egress client. Pre-fix, reqwest auto-followed the hop and returned the
/// metadata endpoint's body to the model.
#[tokio::test]
async fn local_fetch_rejects_redirect_to_link_local() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/redirect"))
        .respond_with(
            ResponseTemplate::new(302)
                .insert_header("location", "http://169.254.169.254/latest/meta-data/"),
        )
        .mount(&server)
        .await;

    let backend = LocalFetch::new();
    let err = backend
        .fetch(&format!("{}/redirect", server.uri()), &FetchOpts::default())
        .await
        .expect_err("redirect into link-local space must be rejected");
    assert!(
        matches!(err, WebError::Rejected(_)),
        "expected WebError::Rejected, got {err:?}"
    );
}

/// Builds a throwaway [`TurnCx`] so a tool's `run` can be exercised in isolation.
struct NoopHost;
#[async_trait]
impl HostRequestHandler for NoopHost {
    async fn request(&self, req: HostRequest) -> HostResponse {
        HostResponse {
            request_id: req.request_id,
            body: HostResponseBody::Approved(true),
        }
    }
}

async fn run_tool(tool: &dyn Tool, args: &str) -> daemon_core::ToolOutcome {
    let events = EventSink::discarding();
    let exec = LocalEnvironment::sandbox("web-tool-test");
    let host = NoopHost;
    let cx = TurnCx {
        cancel: tokio_util::sync::CancellationToken::new(),
        events: &events,
        host: &host,
        session_id: SessionId::new("s"),
        profile: None,
        budget: Budget::unlimited(),
        exec: &exec,
        tool_result_budget: 0,
        approval_policy: daemon_core::ApprovalPolicy::AutoAllow,
        pre_approved: false,
        checkpoints: None,
        tool_timeout: None,
    };
    let call = ToolCall {
        call_id: "c1".into(),
        name: tool.name().to_string(),
        args: args.into(),
    };
    tool.run(&call, &cx).await
}

struct FakeSearch;
#[async_trait]
impl WebSearchBackend for FakeSearch {
    fn name(&self) -> &str {
        "fake"
    }
    async fn search(&self, query: &str, _opts: &SearchOpts) -> Result<SearchResults, WebError> {
        Ok(SearchResults {
            query: query.to_string(),
            answer: None,
            hits: Vec::new(),
            provider: "fake".to_string(),
        })
    }
}

#[tokio::test]
async fn web_search_tool_dispatch_marks_untrusted() {
    let tool = WebSearchTool::new(Arc::new(FakeSearch));
    let outcome = run_tool(&tool, r#"{"query":"hello"}"#).await;
    assert!(outcome.result.ok);
    assert!(outcome.untrusted, "search results are external/untrusted");
    assert!(outcome.result.content.contains("via fake"));
}

#[tokio::test]
async fn web_extract_tool_rejects_unsafe_url() {
    let tool = WebExtractTool::new(vec![Arc::new(LocalFetch::new())]);
    let outcome = run_tool(&tool, r#"{"url":"http://localhost:9000/secret"}"#).await;
    assert!(!outcome.result.ok, "loopback host must be rejected");
}

/// A keyed backend that is never available (no key) — it should be skipped.
struct UnavailableFetch;
#[async_trait]
impl WebFetchBackend for UnavailableFetch {
    fn name(&self) -> &str {
        "unavailable"
    }
    fn available(&self) -> bool {
        false
    }
    async fn fetch(&self, _url: &str, _opts: &FetchOpts) -> Result<FetchedDoc, WebError> {
        Err(WebError::MissingKey("unavailable".into()))
    }
}

struct FakeLocal;
#[async_trait]
impl WebFetchBackend for FakeLocal {
    fn name(&self) -> &str {
        "fakelocal"
    }
    async fn fetch(&self, url: &str, _opts: &FetchOpts) -> Result<FetchedDoc, WebError> {
        Ok(FetchedDoc {
            url: url.to_string(),
            title: Some("Doc".into()),
            content: "extracted body".into(),
            provider: "fakelocal".to_string(),
        })
    }
}

#[tokio::test]
async fn web_extract_tool_falls_through_to_available_backend() {
    let tool = WebExtractTool::new(vec![Arc::new(UnavailableFetch), Arc::new(FakeLocal)]);
    let outcome = run_tool(&tool, r#"{"url":"https://example.com/page"}"#).await;
    assert!(outcome.result.ok);
    assert!(outcome.untrusted, "extracted page is external/untrusted");
    assert!(outcome.result.content.contains("via fakelocal"));
    assert!(outcome.result.content.contains("extracted body"));
}
