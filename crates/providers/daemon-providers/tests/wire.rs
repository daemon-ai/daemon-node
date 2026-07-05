// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Integration tests for the genai-backed provider against an in-process mock HTTP server
//! (wiremock): non-streaming decode, SSE streaming decode, and §8 error classification — driving the
//! real `genai` request/response path with the endpoint pointed at the mock.
//!
//! The mock matches `POST` only (genai appends the adapter's path to our `with_endpoint` base, so we
//! don't pin the exact path) and returns provider-shaped bodies that `genai` decodes.

use daemon_core::{
    EmbeddingProvider, Failure, Provider, Request, RequestMsg, StreamEvent, ToolDef,
};
use daemon_providers::{GenAiEmbedder, GenAiProvider};
use futures::StreamExt;
use std::time::Duration;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

fn req() -> Request {
    Request {
        system: "you are a test".into(),
        messages: vec![RequestMsg {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }],
        tools: vec![ToolDef {
            name: "read_file".into(),
            schema: r#"{"type":"object","properties":{"path":{"type":"string"}}}"#.into(),
        }],
        auth: Some("sk-test".into()),
        constraint: None,
        cache_system: false,
        params: Default::default(),
        task: None,
    }
}

/// Point the provider at the mock server (genai appends the adapter path to this base).
fn endpoint(server: &MockServer) -> String {
    format!("{}/", server.uri())
}

#[tokio::test]
async fn openai_chat_decodes_text_and_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "the answer"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
        })))
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    let out = provider.chat(req()).await.expect("chat succeeds");
    assert_eq!(out.text, "the answer");
    assert_eq!(out.usage.input_tokens, 11);
    assert_eq!(out.usage.output_tokens, 4);
}

/// OpenAI caches prompt prefixes automatically and exposes the hit count under
/// `usage.prompt_tokens_details.cached_tokens`; genai decodes it, which we map onto
/// `UsageDelta::cache_read_tokens` — so cloud cache savings are observed without any explicit
/// breakpoint markers (unlike Anthropic).
#[tokio::test]
async fn openai_cached_tokens_decode_to_cache_read() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-3", "object": "chat.completion", "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "prompt_tokens_details": {"cached_tokens": 80}
            }
        })))
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    let out = provider.chat(req()).await.expect("chat succeeds");
    assert_eq!(out.usage.input_tokens, 100);
    assert_eq!(out.usage.cache_read_tokens, 80);
}

/// Every request carries a conversation-stable `prompt_cache_key` (derived from the system + tools
/// prefix) so OpenAI keeps a conversation routed to the same cache-warm backend. The same request
/// prefix yields the same key across turns.
#[tokio::test]
async fn openai_request_carries_stable_prompt_cache_key() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-4", "object": "chat.completion", "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        })))
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    provider.chat(req()).await.expect("first call succeeds");
    provider.chat(req()).await.expect("second call succeeds");

    let received = server.received_requests().await.expect("captured requests");
    let keys: Vec<String> = received
        .iter()
        .map(|r| {
            let body: serde_json::Value = serde_json::from_slice(&r.body).expect("json body");
            body["prompt_cache_key"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert!(
        keys[0].starts_with("daemon-"),
        "prompt_cache_key should be sent on the OpenAI wire: {keys:?}"
    );
    assert_eq!(
        keys[0], keys[1],
        "the same request prefix must yield a stable key across turns"
    );
}

#[tokio::test]
async fn openai_chat_decodes_tool_call() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-2", "object": "chat.completion", "model": "gpt-test",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"}}]
            }}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        })))
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    let out = provider.chat(req()).await.expect("chat succeeds");
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].name, "read_file");
    assert_eq!(out.tool_calls[0].args, r#"{"path":"a"}"#);
}

#[tokio::test]
async fn openai_stream_emits_deltas_and_done() {
    let server = MockServer::start().await;
    let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n\
               data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" there\"}}]}\n\n\
               data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\n\
               data: [DONE]\n\n";
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    let mut stream = provider.stream(req());
    let mut text_deltas = Vec::new();
    let mut done: Option<daemon_core::ModelOutput> = None;
    while let Some(ev) = stream.next().await {
        match ev.expect("no stream error") {
            StreamEvent::TextDelta(t) => text_deltas.push(t),
            StreamEvent::Done(out) => done = Some(out),
            _ => {}
        }
    }
    assert_eq!(text_deltas.join(""), "Hi there");
    let done = done.expect("a terminal Done event");
    assert_eq!(done.text, "Hi there");
}

#[tokio::test]
async fn openai_429_classifies_as_rate_limit() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "30")
                .set_body_string("rate limited"),
        )
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    match provider.chat(req()).await {
        Err(Failure::RateLimit { retry_after, .. }) => {
            assert_eq!(retry_after, Some(Duration::from_secs(30)));
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_400_context_overflow_classified() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_string("This model's maximum context length is 8192 tokens"),
        )
        .mount(&server)
        .await;

    let provider = GenAiProvider::openai("gpt-test").with_endpoint(endpoint(&server));
    assert!(matches!(
        provider.chat(req()).await,
        Err(Failure::ContextOverflow(_))
    ));
}

#[tokio::test]
async fn openai_embed_batch_decodes_vectors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 4, "total_tokens": 4}
        })))
        .mount(&server)
        .await;

    let embedder = GenAiEmbedder::openai("text-embedding-3-small")
        .with_endpoint(endpoint(&server))
        .with_auth("sk-test")
        .with_dimensions(3);
    let vectors = embedder
        .embed(&["hello".to_string(), "world".to_string()])
        .await
        .expect("embed succeeds");
    assert_eq!(vectors.len(), 2);
    assert_eq!(embedder.dimensions(), 3);
    let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
    assert!(close(vectors[0][0], 0.1) && close(vectors[0][2], 0.3));
    assert!(close(vectors[1][0], 0.4) && close(vectors[1][2], 0.6));
}

#[tokio::test]
async fn openai_embed_429_classifies_as_rate_limit() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after", "12")
                .set_body_string("slow down"),
        )
        .mount(&server)
        .await;

    let embedder = GenAiEmbedder::openai("text-embedding-3-small")
        .with_endpoint(endpoint(&server))
        .with_auth("sk-test");
    match embedder.embed(&["hi".to_string()]).await {
        Err(Failure::RateLimit { retry_after, .. }) => {
            assert_eq!(retry_after, Some(Duration::from_secs(12)));
        }
        other => panic!("expected RateLimit, got {other:?}"),
    }
}

#[tokio::test]
async fn anthropic_chat_decodes_text_thinking_and_tool_use() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {"path": "x"}}
            ],
            "usage": {"input_tokens": 7, "output_tokens": 9}
        })))
        .mount(&server)
        .await;

    let provider = GenAiProvider::anthropic("claude-test").with_endpoint(endpoint(&server));
    let out = provider.chat(req()).await.expect("chat succeeds");
    assert_eq!(out.text, "answer");
    assert_eq!(out.tool_calls.len(), 1);
    assert_eq!(out.tool_calls[0].name, "read_file");
    assert_eq!(out.usage.output_tokens, 9);
}

/// With a price sheet attached, decoded usage carries an estimated `cost_micros` derived from the
/// token breakdown (fresh input + cache read/write + output), so cost flows through the usage stream.
#[tokio::test]
async fn anthropic_chat_computes_cost_from_pricing() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_c", "type": "message", "role": "assistant", "model": "claude-test",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            // input_tokens excludes cache; genai's Anthropic adapter normalizes prompt_tokens to
            // include cache_creation + cache_read, so prompt_tokens = 1000 + 2000 + 5000 = 8000.
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 2000,
                "cache_read_input_tokens": 5000
            }
        })))
        .mount(&server)
        .await;

    // $3 / $15 per Mtok (cache read 0.1x, write 1.25x).
    let provider = GenAiProvider::anthropic("claude-test")
        .with_endpoint(endpoint(&server))
        .with_pricing(daemon_common::Pricing::from_io(3_000_000, 15_000_000));
    let out = provider.chat(req()).await.expect("chat succeeds");

    // prompt_tokens=8000, cache_read=5000, cache_write=2000 => fresh input = 1000.
    // fresh 1000 * 3.0 = 3_000; read 5000 * 0.3 = 1_500; write 2000 * 3.75 = 7_500; out 500 * 15 = 7_500.
    assert_eq!(out.usage.input_tokens, 8000);
    assert_eq!(out.usage.cache_read_tokens, 5000);
    assert_eq!(out.usage.cache_write_tokens, 2000);
    assert_eq!(out.usage.cost_micros, 3_000 + 1_500 + 7_500 + 7_500);
}

/// Without a price sheet, `cost_micros` stays unset (0) — cost is simply not computed.
#[tokio::test]
async fn anthropic_chat_without_pricing_leaves_cost_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_d", "type": "message", "role": "assistant", "model": "claude-test",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1000, "output_tokens": 500}
        })))
        .mount(&server)
        .await;

    let provider = GenAiProvider::anthropic("claude-test").with_endpoint(endpoint(&server));
    let out = provider.chat(req()).await.expect("chat succeeds");
    assert_eq!(out.usage.cost_micros, 0);
}

/// The engine's cache-policy markers (`cache_system` + a last-message breakpoint) serialize onto the
/// Anthropic wire request as `cache_control` blocks on the system prefix and the marked message.
#[tokio::test]
async fn anthropic_cache_breakpoints_serialize_to_wire() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-test",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 3, "output_tokens": 1}
        })))
        .mount(&server)
        .await;

    let mut request = req();
    request.cache_system = true;
    request.messages.last_mut().unwrap().cache_breakpoint = true;

    let provider = GenAiProvider::anthropic("claude-test").with_endpoint(endpoint(&server));
    provider.chat(request).await.expect("chat succeeds");

    let received = server.received_requests().await.expect("captured requests");
    let body: serde_json::Value = serde_json::from_slice(&received[0].body).expect("json body");

    // System is sent as a structured array carrying a cache_control breakpoint (not a plain string).
    let system = &body["system"];
    assert!(
        system.is_array(),
        "cached system should be a structured block array, got {system}"
    );
    assert!(
        system
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b.get("cache_control").is_some()),
        "system prefix should carry a cache_control breakpoint: {system}"
    );

    // The last user message carries a content-part cache_control breakpoint.
    let last = body["messages"].as_array().unwrap().last().unwrap();
    let has_cc = last["content"]
        .as_array()
        .map(|parts| parts.iter().any(|p| p.get("cache_control").is_some()))
        .unwrap_or(false);
    assert!(
        has_cc,
        "last message should carry a cache_control block: {last}"
    );
}

/// The `DaemonApi` provider path: genai's **OpenAI** adapter pointed at the daemon-api gateway base
/// (`.../api/v1/`, trailing slash) must hit exactly `/api/v1/chat/completions` — proving the `v1`
/// segment is neither dropped (missing-slash `Url::join`) nor doubled (`/v1/v1/...`), and that the
/// OpenAI Chat-Completions wire is used (never the Anthropic-native `/v1/messages`) even for a
/// `claude-*`/`author/slug` model id. This mirrors how `bins/daemon` builds the DaemonApi provider:
/// `GenAiProvider::openai(model).with_endpoint(<daemon base with trailing slash>)`.
#[tokio::test]
async fn daemon_api_openai_adapter_hits_api_v1_chat_completions() {
    use wiremock::matchers::path;

    let server = MockServer::start().await;
    // Match ONLY the exact OpenAI-compatible chat path under the gateway's `/api/v1` mount.
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "cmpl-daemon",
            "object": "chat.completion",
            "model": "anthropic/claude-sonnet-4-5",
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "routed through the gateway"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
        })))
        .mount(&server)
        .await;

    // The daemon-api base carries a trailing slash (as `NodeConfig::daemon_api_base` guarantees), so
    // genai's relative `Url::join("chat/completions")` resolves under `/api/v1/`.
    let base = format!("{}/api/v1/", server.uri());
    let provider = GenAiProvider::openai("anthropic/claude-sonnet-4-5").with_endpoint(base);
    let out = provider
        .chat(req())
        .await
        .expect("chat succeeds against the gateway path");
    assert_eq!(out.text, "routed through the gateway");

    // Assert the exact path the adapter posted to (no dropped/doubled `v1`, OpenAI wire not
    // Anthropic-native). The mock only matches `/api/v1/chat/completions`, so a hit already proves
    // it; re-assert on the captured request for an explicit, readable guarantee.
    let received = server.received_requests().await.expect("captured requests");
    assert_eq!(received.len(), 1, "exactly one upstream call");
    assert_eq!(
        received[0].url.path(),
        "/api/v1/chat/completions",
        "DaemonApi must post to the OpenAI-compatible /api/v1/chat/completions path"
    );
}
