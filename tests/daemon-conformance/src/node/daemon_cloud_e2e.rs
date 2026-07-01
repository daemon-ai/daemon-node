// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! TRACK 5 DAEMON CLOUD DISCOVERY→CONFIGURE→CHAT E2E: the whole zero-env "pick provider → pick a
//! model it offers → set the key → chat" chain proven end to end against one node, where the mock
//! upstream doubles as the Daemon Cloud gateway (it serves BOTH the keyless `GET /api/v1/models`
//! listing and the bearer-authed `POST /api/v1/chat/completions` inference).
//!
//! It complements (does not duplicate):
//! - `provider_discovery.rs` — the node *wiring* of the `CloudCatalog` hook (credential threading,
//!   the static fallback). This file drives the ops over the real socket as part of a full flow.
//! - `live_agent_e2e.rs` — the authenticated login→turn chain. This file reuses that turn idiom but
//!   fronts it with discovery and asserts the CORRECTED `requires_key` semantics: Daemon Cloud lists
//!   KEYLESS yet `requires_key == true` (a key is needed to RUN TURNS), and a Daemon Cloud profile
//!   with NO credential fails the turn clearly (never a silent success).

use super::harness::*;

use daemon_api::{
    ModelDescriptor, Outbound, ProfileSpec, ProviderDescriptor, ProviderKindWire, ProviderSelector,
};
use daemon_host::CloudCatalog;
use daemon_protocol::{AgentCommand, AgentEvent, EndReason, UserMsg};
use std::sync::Mutex;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// A discovery hook whose Daemon Cloud model listing is a **real, keyless** `GET {base}/models`
/// against the mock gateway (mirroring the binary's `daemon_cloud_gateway_models`), recording the
/// LIST key it was handed so the test can prove Daemon Cloud lists with none.
struct DiscoveryChatCatalog {
    /// The Daemon Cloud gateway base (ends in `/api/v1/`); `{base}models` is the listing endpoint.
    models_base: String,
    /// The LIST key the node passed for the last `daemon_cloud` `provider_models` call.
    last_daemon_cloud_key: Arc<Mutex<Option<Option<String>>>>,
}

#[async_trait::async_trait]
impl CloudCatalog for DiscoveryChatCatalog {
    async fn list(&self) -> Vec<ModelDescriptor> {
        Vec::new()
    }

    async fn providers(&self) -> Vec<ProviderDescriptor> {
        vec![
            ProviderDescriptor {
                id: "llama_cpp".into(),
                display_name: "llama.cpp (local)".into(),
                kind: ProviderKindWire::Local,
                wire_selector: ProviderSelector::LlamaCpp,
                requires_key: false,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "mistral_rs".into(),
                display_name: "mistral.rs (local)".into(),
                kind: ProviderKindWire::Local,
                wire_selector: ProviderSelector::MistralRs,
                requires_key: false,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "anthropic".into(),
                display_name: "Anthropic".into(),
                kind: ProviderKindWire::Cloud,
                wire_selector: ProviderSelector::GenAi,
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: None,
            },
            ProviderDescriptor {
                id: "daemon_cloud".into(),
                display_name: "Daemon Cloud".into(),
                kind: ProviderKindWire::DaemonCloud,
                wire_selector: ProviderSelector::DaemonApi,
                // Needs a key to RUN TURNS; model LISTING stays keyless (proven below).
                requires_key: true,
                supports_model_discovery: true,
                default_base_url: Some(self.models_base.clone()),
            },
        ]
    }

    async fn provider_models(
        &self,
        provider_id: &str,
        key: Option<String>,
    ) -> Vec<ModelDescriptor> {
        match provider_id {
            "daemon_cloud" => {
                *self.last_daemon_cloud_key.lock().unwrap() = Some(key.clone());
                // Keyless `GET {base}/models` against the mock gateway (no Authorization header).
                let url = format!("{}models", self.models_base);
                let body: serde_json::Value = reqwest::Client::new()
                    .get(&url)
                    .send()
                    .await
                    .expect("gateway /models reachable")
                    .json()
                    .await
                    .expect("gateway /models is JSON");
                let rows = body.get("data").cloned().unwrap_or(body);
                let rows: Vec<serde_json::Value> = serde_json::from_value(rows).unwrap_or_default();
                rows.into_iter()
                    .filter_map(|m| {
                        Some(ModelDescriptor {
                            id: m.get("id")?.as_str()?.to_string(),
                            provider: ProviderSelector::DaemonApi,
                            display_name: m
                                .get("name")
                                .and_then(|v| v.as_str())
                                .map(str::to_string),
                            context_length: m
                                .get("context_length")
                                .and_then(|v| v.as_u64())
                                .map(|v| v as u32),
                            input_price_micros_per_mtok: None,
                            output_price_micros_per_mtok: None,
                            local: false,
                        })
                    })
                    .collect()
            }
            // A genai vendor: a synthesized model tagged with whether a key authenticated the LIST.
            "anthropic" => vec![ModelDescriptor {
                id: "claude-sonnet-4-5".into(),
                provider: ProviderSelector::GenAi,
                display_name: None,
                context_length: None,
                input_price_micros_per_mtok: None,
                output_price_micros_per_mtok: None,
                local: key.is_none(),
            }],
            // Local engines are served by the host from its ModelManager catalog, not here.
            _ => Vec::new(),
        }
    }
}

/// Mount the Daemon Cloud gateway's keyless model listing (`GET /api/v1/models`, OpenAI `{ "data" }`
/// envelope, `author/slug` ids). No auth matcher: a hit proves the picker browses without a key.
async fn mount_models(server: &MockServer) {
    let body = r#"{"data":[{"id":"anthropic/claude-sonnet-4-5","name":"Claude Sonnet 4.5","context_length":200000}]}"#;
    Mock::given(method("GET"))
        .and(path("/api/v1/models"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(body.as_bytes().to_vec(), "application/json"),
        )
        .mount(server)
        .await;
}

/// Mount the gateway's OpenAI-compatible chat path returning `content` over SSE (the engine drives
/// interactive turns through the provider's streaming path). Matches ONLY
/// `POST /api/v1/chat/completions`, so a hit proves the `v1` segment is neither dropped nor doubled
/// and the OpenAI wire (not Anthropic-native `/v1/messages`) was used.
async fn mount_chat(server: &MockServer, content: &str) {
    let content_json = serde_json::Value::String(content.to_string());
    let sse = format!(
        "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":{content_json}}}}}]}}\n\n\
         data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}],\
         \"usage\":{{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}}}\n\n\
         data: [DONE]\n\n"
    );
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(server)
        .await;
}

/// The `DaemonApi` (Daemon Cloud) profile a GUI would create from a discovered model, pinned at the
/// mock gateway base.
fn daemon_cloud_profile(base: &str, model: &str) -> ProfileSpec {
    ProfileSpec {
        base_url: Some(base.to_string()),
        system_prompt: "You are the Daemon Cloud agent.".into(),
        ..ProfileSpec::new("gateway", ProviderSelector::DaemonApi, model)
    }
}

/// Poll the local client, accumulating drained items until the turn reaches a terminal event
/// (`TurnFinished` or `Error`) or the deadline elapses. Returns everything drained so far. The
/// deadline is generous: an auth failure (a keyless Daemon Cloud turn) is `Failure::Auth`, which the
/// §8 recovery loop bounds by rotate-then-retry (default 3 retries with 2/4/8s backoff) before it
/// aborts — a clear terminal error, but ~15-18s out.
async fn drain_until_terminal(client: &ApiClient, session: &SessionId) -> Vec<Outbound> {
    let deadline = Instant::now() + Duration::from_secs(45);
    let mut acc: Vec<Outbound> = Vec::new();
    while Instant::now() < deadline {
        match client
            .call(ApiRequest::Poll {
                session: session.clone(),
                max: 0,
            })
            .await
            .expect("poll")
        {
            ApiResponse::Drained(items) => acc.extend(items),
            other => panic!("expected Drained, got {other:?}"),
        }
        let terminal = acc.iter().any(|o| {
            matches!(
                o,
                Outbound::Event(AgentEvent::TurnFinished { .. } | AgentEvent::Error { .. })
            )
        });
        if terminal {
            return acc;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the turn never reached a terminal event (TurnFinished/Error)");
}

/// THE POSITIVE CHAIN: `ProviderCatalog` lists Daemon Cloud (`requires_key == true`) + genai vendors
/// + local engines; `ProviderModels(daemon_cloud)` lists KEYLESS via the mock gateway; a profile
/// created from a discovered Daemon Cloud model + `CredentialSet` + a turn through the Daemon Cloud
/// (OpenAI-adapter + endpoint) path to the mock upstream SUCCEEDS carrying the Bearer over SSE.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_cloud_discovery_configure_and_chat() {
    let server = MockServer::start().await;
    mount_models(&server).await;
    mount_chat(&server, "routed through the Daemon Cloud gateway").await;
    let base = format!("{}/api/v1/", server.uri());

    let last_key = Arc::new(Mutex::new(None));
    let catalog: Arc<dyn CloudCatalog> = Arc::new(DiscoveryChatCatalog {
        models_base: base.clone(),
        last_daemon_cloud_key: last_key.clone(),
    });
    let (node, _cred_store, handle) =
        assemble_daemon_api_gateway_with_catalog(base.clone(), catalog);

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server_task = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // 1) Discover providers: Daemon Cloud requires a key (to run turns), lists keyless; genai vendors
    // require a key; local engines do not.
    let providers = match client
        .call(ApiRequest::ProviderCatalog)
        .await
        .expect("provider catalog")
    {
        ApiResponse::ProviderCatalog(p) => p,
        other => panic!("expected ProviderCatalog, got {other:?}"),
    };
    let daemon_cloud = providers
        .iter()
        .find(|p| p.id == "daemon_cloud")
        .expect("daemon_cloud present in catalog");
    assert_eq!(daemon_cloud.kind, ProviderKindWire::DaemonCloud);
    assert_eq!(daemon_cloud.wire_selector, ProviderSelector::DaemonApi);
    assert!(
        daemon_cloud.requires_key,
        "Daemon Cloud needs a key to run turns (corrected semantics)"
    );
    assert_eq!(
        daemon_cloud.default_base_url.as_deref(),
        Some(base.as_str())
    );
    assert!(
        providers
            .iter()
            .any(|p| p.id == "anthropic" && p.requires_key),
        "a genai vendor requires a key"
    );
    assert!(
        providers
            .iter()
            .any(|p| p.id == "llama_cpp" && !p.requires_key),
        "local engines do not require a key"
    );

    // 2) List Daemon Cloud's models KEYLESS via the gateway.
    let models = match client
        .call(ApiRequest::ProviderModels {
            provider: "daemon_cloud".into(),
            credential_ref: None,
            transient_key: None,
        })
        .await
        .expect("provider models")
    {
        ApiResponse::ProviderModels(m) => m,
        other => panic!("expected ProviderModels, got {other:?}"),
    };
    let model_id = "anthropic/claude-sonnet-4-5";
    assert!(
        models.iter().any(|m| m.id == model_id),
        "the gateway model must be listed, got {:?}",
        models.iter().map(|m| &m.id).collect::<Vec<_>>()
    );
    assert_eq!(
        *last_key.lock().unwrap(),
        Some(None),
        "Daemon Cloud lists with NO LIST credential"
    );
    // The listing hit the gateway `/models` without an Authorization header (keyless proof).
    let listing = server.received_requests().await.expect("captured requests");
    let models_reqs: Vec<_> = listing
        .iter()
        .filter(|r| r.url.path() == "/api/v1/models")
        .collect();
    assert_eq!(models_reqs.len(), 1, "exactly one keyless /models listing");
    assert!(
        models_reqs[0].headers.get("authorization").is_none(),
        "the /models listing must carry no Authorization header"
    );

    // 3) Provision the inference key (mandatory for a requires_key provider), then create + select a
    // profile from the discovered model.
    assert!(matches!(
        client
            .call(ApiRequest::CredentialSet {
                profile: "gateway".into(),
                secret: "sk-daemon-cloud-key".into(),
            })
            .await
            .expect("credential set"),
        ApiResponse::Ok
    ));
    assert!(matches!(
        client
            .call(ApiRequest::ProfileCreate {
                spec: daemon_cloud_profile(&base, model_id),
            })
            .await
            .expect("profile create"),
        ApiResponse::Ok
    ));
    assert!(matches!(
        client
            .call(ApiRequest::ProfileSelect {
                id: "gateway".into(),
            })
            .await
            .expect("profile select"),
        ApiResponse::Ok
    ));

    // 4) Drive a real turn through the Daemon Cloud path to the mock upstream.
    let session = SessionId::new("daemon-cloud-1");
    assert!(matches!(
        client
            .call(ApiRequest::Submit {
                session: session.clone(),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hello daemon cloud"),
                    request_id: daemon_common::ReqId(1),
                },
                origin: None,
                profile: None,
            })
            .await
            .expect("submit"),
        ApiResponse::Ok
    ));
    let drained = drain_until_terminal(&client, &session).await;
    let final_text = drained.iter().find_map(|o| match o {
        Outbound::Event(AgentEvent::TurnFinished { summary, .. }) => {
            Some(summary.final_text.clone().unwrap_or_default())
        }
        _ => None,
    });
    assert!(
        final_text
            .as_deref()
            .is_some_and(|t| t.contains("routed through the Daemon Cloud gateway")),
        "the assistant reply must come from the mock upstream, got {final_text:?}"
    );

    // 5) The turn hit the OpenAI-compatible chat path carrying the provisioned Bearer.
    let received = server.received_requests().await.expect("captured requests");
    let chat: Vec<_> = received
        .iter()
        .filter(|r| r.url.path() == "/api/v1/chat/completions")
        .collect();
    assert_eq!(chat.len(), 1, "exactly one upstream chat call");
    let auth = chat[0]
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .expect("authorization header present on the chat call");
    assert_eq!(
        auth, "Bearer sk-daemon-cloud-key",
        "the CredentialSet key must flow to the upstream as the provider bearer"
    );

    server_task.abort();
    handle.shutdown().await;
}

/// THE NEGATIVE: a Daemon Cloud profile with NO credential must fail the turn CLEARLY — an
/// `AgentEvent::Error` and/or a `TurnFinished{ end_reason: Failed }` — and NEVER produce assistant
/// text (never a silent success). The gateway rejects the unauthenticated inference request.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn daemon_cloud_turn_without_credential_errors() {
    let server = MockServer::start().await;
    // The gateway rejects an unauthenticated inference request (no credential was provisioned).
    Mock::given(method("POST"))
        .and(path("/api/v1/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_raw(
            br#"{"error":{"message":"missing bearer"}}"#.to_vec(),
            "application/json",
        ))
        .mount(&server)
        .await;
    let base = format!("{}/api/v1/", server.uri());

    let last_key = Arc::new(Mutex::new(None));
    let catalog: Arc<dyn CloudCatalog> = Arc::new(DiscoveryChatCatalog {
        models_base: base.clone(),
        last_daemon_cloud_key: last_key,
    });
    let (node, _cred_store, handle) =
        assemble_daemon_api_gateway_with_catalog(base.clone(), catalog);

    let path = temp_socket();
    let _ = std::fs::remove_file(&path);
    let listener = UnixListener::bind(&path).expect("bind socket");
    let server_task = tokio::spawn(serve_api_unix(listener, node.clone()));
    let client = ApiClient::new(path.clone());

    // Create + select a Daemon Cloud profile with NO CredentialSet.
    assert!(matches!(
        client
            .call(ApiRequest::ProfileCreate {
                spec: daemon_cloud_profile(&base, "anthropic/claude-sonnet-4-5"),
            })
            .await
            .expect("profile create"),
        ApiResponse::Ok
    ));
    assert!(matches!(
        client
            .call(ApiRequest::ProfileSelect {
                id: "gateway".into(),
            })
            .await
            .expect("profile select"),
        ApiResponse::Ok
    ));

    let session = SessionId::new("daemon-cloud-nocred");
    assert!(matches!(
        client
            .call(ApiRequest::Submit {
                session: session.clone(),
                command: AgentCommand::StartTurn {
                    input: UserMsg::new("hello without a key"),
                    request_id: daemon_common::ReqId(1),
                },
                origin: None,
                profile: None,
            })
            .await
            .expect("submit"),
        ApiResponse::Ok
    ));

    let drained = drain_until_terminal(&client, &session).await;
    let errored = drained.iter().any(|o| match o {
        Outbound::Event(AgentEvent::Error { .. }) => true,
        Outbound::Event(AgentEvent::TurnFinished { summary, .. }) => {
            summary.end_reason == EndReason::Failed
        }
        _ => false,
    });
    assert!(
        errored,
        "a Daemon Cloud turn with no credential must error clearly, got {drained:?}"
    );
    // It must NEVER silently succeed with assistant text.
    let any_text = drained.iter().any(|o| match o {
        Outbound::Event(AgentEvent::TextDelta { text, .. }) => !text.is_empty(),
        Outbound::Event(AgentEvent::TurnFinished { summary, .. }) => {
            summary.final_text.as_deref().is_some_and(|t| !t.is_empty())
        }
        _ => false,
    });
    assert!(
        !any_text,
        "an unauthenticated Daemon Cloud turn must produce no assistant text"
    );

    server_task.abort();
    handle.shutdown().await;
}
