//! Integration tests for the genai-backed provider against an in-process mock HTTP server
//! (wiremock): non-streaming decode, SSE streaming decode, and §8 error classification — driving the
//! real `genai` request/response path with the endpoint pointed at the mock.
//!
//! The mock matches `POST` only (genai appends the adapter's path to our `with_endpoint` base, so we
//! don't pin the exact path) and returns provider-shaped bodies that `genai` decodes.

use daemon_core::{EmbeddingProvider, Failure, Provider, Request, RequestMsg, StreamEvent, ToolDef};
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
