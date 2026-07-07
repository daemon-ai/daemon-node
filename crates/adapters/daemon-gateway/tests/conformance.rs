// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
// Integration test crate: a raw `reqwest::Client` is the OpenAI-wire client under test (the egress
// SSRF gate is a production-code concern, not a test-harness one), mirroring the daemon-http tests.
#![allow(clippy::disallowed_types)]

//! End-to-end conformance for the gateway surface over a real bound socket, driven by a mock
//! [`GatewayBackend`]: non-streaming + streaming completion, `GET /v1/models`, and the bearer `401`.

use std::sync::Arc;

use async_trait::async_trait;
use daemon_api::{ModelDescriptor, ProviderSelector};
use daemon_common::UsageDelta;
use daemon_core::{ModelOutput, Provider, Request, StreamEvent, ToolCall};
use daemon_gateway::{Completion, GatewayBackend, GatewayError, GatewayPrincipal};
use futures::{stream, StreamExt};

const TOKEN: &str = "test-gateway-token";

/// The shared bearer resolution both test backends use: the fixed test token authenticates as the
/// admin caller; everything else is rejected (`None` → 401). Per-session tokens are exercised in the
/// binary's own registry tests.
async fn authorize_test_token(token: &str) -> Option<GatewayPrincipal> {
    (token == TOKEN).then_some(GatewayPrincipal::Admin)
}

/// A deterministic backend: `catalog()` returns a fixed list; `complete()` echoes the last user
/// message. When the model is `"tooly"` it emits a tool call; when streaming it splits the reply
/// into two text deltas + a terminal `Done`.
struct MockBackend;

#[async_trait]
impl GatewayBackend for MockBackend {
    async fn catalog(&self) -> Vec<ModelDescriptor> {
        vec![
            ModelDescriptor::cloud("gpt-4o", ProviderSelector::GenAi, Some(128_000)),
            ModelDescriptor {
                id: "local-gguf".into(),
                provider: ProviderSelector::LlamaCpp,
                display_name: None,
                context_length: Some(8192),
                input_price_micros_per_mtok: None,
                output_price_micros_per_mtok: None,
                local: true,
            },
        ]
    }

    async fn authorize(&self, token: &str) -> Option<GatewayPrincipal> {
        authorize_test_token(token).await
    }

    async fn complete(
        &self,
        _principal: &GatewayPrincipal,
        model: &str,
        req: Request,
        stream_flag: bool,
    ) -> Result<Completion, GatewayError> {
        if model == "nope" {
            return Err(GatewayError::UnknownModel(model.into()));
        }
        let last_user = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_default();
        let usage = UsageDelta {
            input_tokens: 7,
            output_tokens: 3,
            ..Default::default()
        };
        let mut out = ModelOutput {
            text: format!("echo: {last_user}"),
            usage,
            ..Default::default()
        };
        if model == "tooly" {
            out.text = String::new();
            out.tool_calls = vec![ToolCall {
                call_id: "call_1".into(),
                name: "get_weather".into(),
                args: "{\"city\":\"NYC\"}".into(),
            }];
        }
        if stream_flag {
            let full = out.text.clone();
            let (a, b) = full.split_at(full.len() / 2);
            let events = vec![
                Ok(StreamEvent::TextDelta(a.to_string())),
                Ok(StreamEvent::TextDelta(b.to_string())),
                Ok(StreamEvent::Done(out)),
            ];
            Ok(Completion::Stream(Box::pin(stream::iter(events))))
        } else {
            Ok(Completion::Once(out))
        }
    }
}

/// Bind the gateway on an ephemeral loopback port and return its base URL (`http://127.0.0.1:<p>`).
async fn spawn_gateway() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let backend: Arc<dyn GatewayBackend> = Arc::new(MockBackend);
    tokio::spawn(async move {
        let _ = daemon_gateway::serve(listener, backend).await;
    });
    format!("http://{addr}")
}

#[tokio::test]
async fn non_streaming_completion() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "ping"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "echo: ping");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["total_tokens"], 10);
}

#[tokio::test]
async fn non_streaming_tool_call() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({
            "model": "tooly",
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let call = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(call["id"], "call_1");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    assert_eq!(call["function"]["arguments"], "{\"city\":\"NYC\"}");
}

#[tokio::test]
async fn streaming_completion_emits_chunks_and_done() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "stream": true,
            "messages": [{"role": "user", "content": "stream me"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "content-type: {ct}");
    let body = resp.text().await.unwrap();
    // The stream terminates with the OpenAI sentinel.
    assert!(body.contains("data: [DONE]"), "missing [DONE]: {body}");
    // The first chunk carries the assistant role; the reassembled content is the echoed reply.
    assert!(body.contains("\"role\":\"assistant\""), "no role: {body}");
    let reassembled: String = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<serde_json::Value>(d).ok())
        .filter_map(|v| {
            v["choices"][0]["delta"]["content"]
                .as_str()
                .map(str::to_string)
        })
        .collect();
    assert_eq!(reassembled, "echo: stream me");
    // The terminal chunk carries the finish reason.
    assert!(
        body.contains("\"finish_reason\":\"stop\""),
        "no finish: {body}"
    );
}

#[tokio::test]
async fn models_reflects_catalog() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base}/v1/models"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let ids: Vec<String> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&"gpt-4o".to_string()));
    assert!(ids.contains(&"local-gguf".to_string()));
}

#[tokio::test]
async fn unknown_model_is_404() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({
            "model": "nope",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
}

/// A backend that drives a real `daemon_core::Provider` (the deterministic `MockProvider`) through
/// the gateway — the actual provider seam the binary's backend uses (minus credential brokering).
/// Used by the env-gated e2e to prove an OpenAI-wire client can complete a turn against a
/// node-served provider end to end.
struct ProviderBackend;

#[async_trait]
impl GatewayBackend for ProviderBackend {
    async fn catalog(&self) -> Vec<ModelDescriptor> {
        vec![ModelDescriptor {
            id: "mock-local".into(),
            provider: ProviderSelector::Mock,
            display_name: None,
            context_length: Some(8192),
            input_price_micros_per_mtok: None,
            output_price_micros_per_mtok: None,
            local: true,
        }]
    }

    async fn authorize(&self, token: &str) -> Option<GatewayPrincipal> {
        authorize_test_token(token).await
    }

    async fn complete(
        &self,
        _principal: &GatewayPrincipal,
        _model: &str,
        req: Request,
        stream_flag: bool,
    ) -> Result<Completion, GatewayError> {
        let provider = daemon_core::MockProvider::completing("served by the node provider");
        if stream_flag {
            // Collect the provider's stream into an owned buffer (the mock yields one terminal
            // Done) so the returned stream is `'static` without capturing the provider.
            let events: Vec<_> = provider
                .stream(req)
                .map(|ev| ev.map_err(|f| GatewayError::Provider(f.to_string())))
                .collect()
                .await;
            Ok(Completion::Stream(Box::pin(stream::iter(events))))
        } else {
            let out = provider
                .chat(req)
                .await
                .map_err(|f| GatewayError::Provider(f.to_string()))?;
            Ok(Completion::Once(out))
        }
    }
}

/// The env-gated real-seam e2e (per the plan, the "real agent/model" e2e is gated behind
/// `DAEMON_E2E_AGENTS=1`; the default suite stays offline/mock). This variant is offline-safe — it
/// drives a real `daemon_core::Provider` rather than a real external agent binary — so it exercises
/// the actual provider call path an OpenAI-wire client hits, without a live model or key.
#[tokio::test]
async fn real_provider_seam_e2e() {
    if std::env::var("DAEMON_E2E_AGENTS").as_deref() != Ok("1") {
        eprintln!("skipping real_provider_seam_e2e (set DAEMON_E2E_AGENTS=1 to run)");
        return;
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let backend: Arc<dyn GatewayBackend> = Arc::new(ProviderBackend);
    tokio::spawn(async move {
        let _ = daemon_gateway::serve(listener, backend).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // Non-stream: the OpenAI-wire client gets the provider's reply.
    let body: serde_json::Value = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({
            "model": "mock-local",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "served by the node provider"
    );

    // Stream: same reply, delivered as SSE terminating in [DONE].
    let text = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(TOKEN)
        .json(&serde_json::json!({
            "model": "mock-local",
            "stream": true,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(text.contains("served by the node provider"));
    assert!(text.contains("data: [DONE]"));
}

#[tokio::test]
async fn missing_token_is_401() {
    let base = spawn_gateway().await;
    let client = reqwest::Client::new();
    // No Authorization header.
    let resp = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);

    // A wrong token is also rejected.
    let resp = client
        .get(format!("{base}/v1/models"))
        .bearer_auth("wrong")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 401);
}
