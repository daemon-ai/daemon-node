// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The [`Provider`] adapter over the [`genai`] multi-provider client.

use crate::{classify_genai_error, finalize_output, RawToolCall};
use async_trait::async_trait;
use daemon_common::{Pricing, UsageDelta};
use daemon_core::provider::CacheTtl;
use daemon_core::{
    Capabilities, EmbeddingProvider, Failure, ModelOutput, Provider, Request, RequestImage,
    RequestMsg, RequestParams, ResponseMeta, StreamEvent, ToolCallFormat,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{
    CacheControl, ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatStreamEvent,
    ContentPart, MessageContent, StopReason, StreamEnd, Tool, ToolCall as GenToolCall,
    ToolResponse,
};
use genai::embed::{EmbedOptions, EmbedRequest};
use genai::resolver::{AuthData, Endpoint};
use genai::{Client, ModelIden, ServiceTarget};

/// A networked model provider backed by [`genai`].
///
/// One client serves any `genai`-supported provider: the adapter (OpenAI, Anthropic, Gemini, Groq,
/// DeepSeek, xAI, OpenRouter, Cohere, Ollama, …) is *inferred from the model name* via
/// [`Client::default_model`] / [`AdapterKind::from_model`] (namespaced ids like `groq::…` force the
/// adapter), so there is no daemon-side provider registry. The credential lease secret
/// (`Request.auth`) is threaded per-call as the API key; when absent, genai falls back to the
/// adapter's `default_key_env_name()` from the environment. An optional `endpoint` override points
/// at a custom base URL (the in-process wire tests), and an optional explicit `adapter` forces the
/// protocol for a model name that does not self-identify (used by those tests).
pub struct GenAiProvider {
    client: Client,
    /// Explicit adapter override. `None` => infer from the model name (the default path). `Some`
    /// forces the protocol regardless of the model id (wire tests, custom gateways).
    adapter: Option<AdapterKind>,
    model: String,
    endpoint: Option<String>,
    max_tokens: u32,
    /// The model's price sheet, when known: used to fill `UsageDelta::cost_micros` at decode time
    /// so cost flows through the usage stream like any other usage field. `None` => `cost_micros`
    /// stays `0` (cost not computed).
    pricing: Option<Pricing>,
}

impl GenAiProvider {
    /// A provider for `model`, with the genai adapter inferred from the (optionally namespaced)
    /// model name — the primary construction path. Uses genai's default endpoint and the model's
    /// published output-token cap ([`known_max_output`]) so a large-output model (e.g. Claude 4,
    /// `o3`) is not silently clamped; unknown models fall back to [`DEFAULT_MAX_OUTPUT_TOKENS`].
    pub fn for_model(model: impl Into<String>) -> Self {
        let model = model.into();
        let max_tokens = known_max_output(&model).unwrap_or(crate::DEFAULT_MAX_OUTPUT_TOKENS);
        Self {
            client: Client::default(),
            adapter: None,
            model,
            endpoint: None,
            max_tokens,
            pricing: None,
        }
    }

    /// A provider for an *explicit* `adapter`/`model` — forces the protocol regardless of the model
    /// id. Prefer [`GenAiProvider::for_model`]; this exists for wire tests and custom gateways.
    pub fn new(adapter: AdapterKind, model: impl Into<String>) -> Self {
        Self {
            adapter: Some(adapter),
            ..Self::for_model(model)
        }
    }

    /// Override the API base URL (the wire tests point this at a mock server).
    pub fn with_endpoint(mut self, base_url: impl Into<String>) -> Self {
        self.endpoint = Some(base_url.into());
        self
    }

    /// Override the output-token cap.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Attach the model's price sheet so decoded usage carries an estimated `cost_micros`.
    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// The OpenAI Chat Completions provider for `model` (explicit adapter; for wire tests).
    pub fn openai(model: impl Into<String>) -> Self {
        Self::new(AdapterKind::OpenAI, model)
    }

    /// The **Daemon Cloud** provider for `model`: a clone of genai's OpenRouter adapter behaviour
    /// achieved through genai's PUBLIC API only — an OpenAI-compatible [`AdapterKind::OpenAI`] pinned
    /// at [`DAEMON_CLOUD_BASE`] via [`with_endpoint`](Self::with_endpoint). genai's `OpenRouterAdapter`
    /// is itself a pass-through delegating to `OpenAIAdapter` with a fixed endpoint + its own
    /// `key_env`, so this is byte-identical on the wire while avoiding an OpenRouter env-key fallback;
    /// OpenRouter remains available as its own genai vendor via [`for_model`](Self::for_model). Model
    /// ids are OpenRouter-style `author/slug`; the bearer flows per-call via the credential broker.
    /// Override the base (self-hosted gateway) with [`with_endpoint`](Self::with_endpoint).
    pub fn daemon_cloud(model: impl Into<String>) -> Self {
        Self::openai(model).with_endpoint(DAEMON_CLOUD_BASE)
    }

    /// The Anthropic Messages provider for `model` (explicit adapter; for wire tests).
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self::new(AdapterKind::Anthropic, model)
    }

    /// The output-token cap actually applied to `req`: the per-call [`RequestParams::max_tokens`]
    /// override when set, else the provider's model cap. Also surfaced on [`ResponseMeta::params`].
    fn effective_max_tokens(&self, req: &Request) -> u32 {
        req.params.max_tokens.unwrap_or(self.max_tokens)
    }

    fn options(&self, req: &Request) -> ChatOptions {
        let mut opts = ChatOptions::default()
            .with_capture_usage(true)
            .with_capture_content(true)
            .with_capture_reasoning_content(true)
            .with_capture_tool_calls(true)
            .with_normalize_reasoning_content(true)
            .with_max_tokens(self.effective_max_tokens(req))
            // A conversation-stable cache key derived from the stable prefix (system + tools). Used
            // by OpenAI to keep a conversation's requests routed to the same backend so its automatic
            // prefix cache stays warm; ignored by adapters that do not honor it.
            .with_prompt_cache_key(prompt_cache_key(req));
        // Per-call sampling overrides ([`Request::params`]); applied only when set so the default
        // request leaves genai's own defaults untouched. genai 0.6 has no `top_k` setter, so
        // `params.top_k` is ignored here (the local worker honors it).
        if let Some(temperature) = req.params.temperature {
            opts = opts.with_temperature(temperature);
        }
        if let Some(top_p) = req.params.top_p {
            opts = opts.with_top_p(top_p);
        }
        if let Some(seed) = req.params.seed {
            opts = opts.with_seed(seed);
        }
        opts
    }

    fn chat_request(&self, req: &Request) -> ChatRequest {
        let marker = cache_marker(req.cache_ttl);
        let mut messages: Vec<ChatMessage> = req
            .messages
            .iter()
            .map(|msg| to_chat_message(msg, marker.clone()))
            .collect();
        // A cache breakpoint on the tools+system prefix (the largest stable block) when the engine
        // requests it. genai ignores `cache_control` on the request-level `system` string for
        // Anthropic, so the system must be passed as a leading System *message* to carry the
        // breakpoint; otherwise it stays the plain request-level system.
        let cache_system = req.cache_system && !req.system.is_empty();
        if cache_system {
            messages.insert(
                0,
                ChatMessage::system(req.system.clone()).with_options(marker),
            );
        }
        let mut chat = ChatRequest::new(messages);
        if !req.system.is_empty() && !cache_system {
            chat = chat.with_system(req.system.clone());
        }
        if !req.tools.is_empty() {
            let tools: Vec<Tool> = req
                .tools
                .iter()
                .map(|def| {
                    let schema: serde_json::Value = serde_json::from_str(&def.schema)
                        .unwrap_or_else(|_| serde_json::json!({"type": "object"}));
                    Tool::new(def.name.clone()).with_schema(schema)
                })
                .collect();
            chat = chat.with_tools(tools);
        }
        chat
    }
}

/// A conversation-stable OpenAI `prompt_cache_key` derived from the request's stable prefix (the
/// system prompt + the offered tools' names/schemas). The engine keeps that prefix byte-stable
/// across a conversation's turns, so this key is identical turn to turn — exactly what OpenAI wants
/// to route repeat requests to the same cache-warm backend. Other adapters ignore the key.
///
/// `DefaultHasher` (fixed-key SipHash) is process-independent and deterministic, so the key is the
/// same across daemon restarts for the same prefix; this is a routing hint, not a security boundary.
fn prompt_cache_key(req: &Request) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    req.system.hash(&mut hasher);
    for tool in &req.tools {
        tool.name.hash(&mut hasher);
        tool.schema.hash(&mut hasher);
    }
    format!("daemon-{:016x}", hasher.finish())
}

/// The Daemon Cloud gateway base URL (OpenRouter-clone; OpenAI-compatible). Pinned here (a public
/// endpoint, not host config) so [`GenAiProvider::daemon_cloud`] stays config-crate-free; the host
/// profile's `base_url` overrides it for a self-hosted gateway.
pub const DAEMON_CLOUD_BASE: &str = "https://api.daemon.ai/api/v1/";

/// The genai cloud vendors the node enumerates for provider + model discovery. genai 0.6.5 exposes
/// no public enumeration of [`AdapterKind`], so this is the curated set the picker offers. Each is
/// queried for models only when a credential resolves (a transient/stored key, or its env var), so a
/// no-key node makes no network calls for it.
pub const DISCOVERY_ADAPTERS: &[AdapterKind] = &[
    AdapterKind::OpenAI,
    AdapterKind::Anthropic,
    AdapterKind::Gemini,
    AdapterKind::Groq,
    AdapterKind::DeepSeek,
    AdapterKind::Xai,
    AdapterKind::Cohere,
    AdapterKind::OpenRouter,
];

/// Live model ids for a single genai `adapter`, credential-aware: when `key` is `Some`, the LIST
/// call authenticates with it (a first-run transient key or a stored credential); when `None`, genai
/// resolves the adapter's `default_key_env_name()` from the environment. Ids are namespaced so
/// [`AdapterKind::from_model`] round-trips the adapter at chat time. A listing error yields an empty
/// list (one unreachable/keyless vendor never blanks the picker). This is the genai-native model
/// discovery the `ProviderModels` op drives per vendor.
pub async fn genai_models_for(adapter: AdapterKind, key: Option<&str>) -> Vec<String> {
    let client = Client::default();
    // `AuthData` converts into a genai `ProviderConfig { endpoint: None, auth: Some(..) }`; `()`
    // resolves the adapter's default endpoint + env key.
    let listed = match key {
        Some(k) => {
            client
                .all_model_names(adapter, AuthData::from_single(k.to_string()))
                .await
        }
        None => client.all_model_names(adapter, ()).await,
    };
    match listed {
        Ok(names) => names
            .iter()
            .map(|name| namespace_model(adapter, name))
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Live model ids from `genai` for every [`DISCOVERY_ADAPTERS`] adapter whose API key is present in
/// the environment ([`AdapterKind::default_key_env_name`]), each namespaced so
/// [`AdapterKind::from_model`] round-trips the adapter. Adapters without a key are skipped (their
/// models come from the static fallback catalog); a per-adapter listing error is swallowed so one
/// unreachable provider does not blank the picker. This is the genai-native replacement for a
/// daemon-side cloud-model registry.
pub async fn genai_listed_models() -> Vec<String> {
    let mut out = Vec::new();
    for &adapter in DISCOVERY_ADAPTERS {
        let has_key = adapter
            .default_key_env_name()
            .map(|env| std::env::var(env).is_ok())
            .unwrap_or(false);
        if !has_key {
            continue;
        }
        // `None` => genai resolves the adapter's default endpoint + env key for the listing call.
        out.extend(genai_models_for(adapter, None).await);
    }
    out
}

/// A human label for a genai cloud vendor adapter (the picker's provider row title). Falls back to
/// genai's lowercase adapter name for any vendor without a curated label.
fn vendor_display_name(adapter: AdapterKind) -> String {
    match adapter {
        AdapterKind::OpenAI => "OpenAI",
        AdapterKind::Anthropic => "Anthropic",
        AdapterKind::Gemini => "Google Gemini",
        AdapterKind::Groq => "Groq",
        AdapterKind::DeepSeek => "DeepSeek",
        AdapterKind::Xai => "xAI",
        AdapterKind::Cohere => "Cohere",
        AdapterKind::OpenRouter => "OpenRouter",
        other => return other.as_lower_str().to_string(),
    }
    .to_string()
}

/// The genai cloud vendors the picker offers as `(stable id, display name)` pairs. The id is the
/// adapter's genai lowercase name (e.g. `"anthropic"`, `"open_router"`) and is the discriminator the
/// `ProviderModels.provider` field carries (every genai vendor shares `ProviderSelector::GenAi`).
pub fn discovery_vendor_ids() -> Vec<(String, String)> {
    DISCOVERY_ADAPTERS
        .iter()
        .map(|&a| (a.as_lower_str().to_string(), vendor_display_name(a)))
        .collect()
}

/// Live models for a genai vendor identified by its discovery id (see [`discovery_vendor_ids`]),
/// credential-aware (see [`genai_models_for`]). Empty for an unknown id.
pub async fn genai_models_for_id(vendor_id: &str, key: Option<&str>) -> Vec<String> {
    let Some(&adapter) = DISCOVERY_ADAPTERS
        .iter()
        .find(|a| a.as_lower_str() == vendor_id)
    else {
        return Vec::new();
    };
    genai_models_for(adapter, key).await
}

/// Prefix a model id with its `genai` adapter namespace (`groq::…`) unless the id already
/// self-identifies — i.e. [`AdapterKind::from_model`] resolves the same adapter — so a GUI that
/// stores the returned id round-trips through inference at chat time.
fn namespace_model(adapter: AdapterKind, model: &str) -> String {
    let self_identifies = AdapterKind::from_model(model)
        .map(|a| a == adapter)
        .unwrap_or(false);
    if self_identifies {
        model.to_string()
    } else {
        format!("{}::{}", adapter.as_lower_str(), model)
    }
}

/// An [`EmbeddingProvider`] backed by [`genai`]'s embeddings API.
///
/// Mirrors [`GenAiProvider`]: one adapter serves any `genai`-supported embedding model; the lease
/// secret (`with_auth`) is applied per call via the resolved [`ServiceTarget`], and an optional
/// `endpoint` override points at a custom base URL (the wire tests use a mock). The embedding model
/// is a *separate* model from any chat model (e.g. `text-embedding-3-small`).
pub struct GenAiEmbedder {
    client: Client,
    adapter: AdapterKind,
    model: String,
    endpoint: Option<String>,
    auth: Option<String>,
    dims: usize,
}

impl GenAiEmbedder {
    /// An embedder for `adapter`/`model` using `genai`'s default endpoint.
    pub fn new(adapter: AdapterKind, model: impl Into<String>) -> Self {
        Self {
            client: Client::default(),
            adapter,
            model: model.into(),
            endpoint: None,
            auth: None,
            dims: 0,
        }
    }

    /// The OpenAI embeddings provider for `model` (e.g. `text-embedding-3-small`).
    pub fn openai(model: impl Into<String>) -> Self {
        Self::new(AdapterKind::OpenAI, model)
    }

    /// Override the API base URL (the wire tests point this at a mock server).
    pub fn with_endpoint(mut self, base_url: impl Into<String>) -> Self {
        self.endpoint = Some(base_url.into());
        self
    }

    /// Set the bearer credential (the §7 lease secret) applied to each call.
    pub fn with_auth(mut self, auth: impl Into<String>) -> Self {
        self.auth = Some(auth.into());
        self
    }

    /// Declare the embedding dimensionality (for store/index validation; `0` = unknown).
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dims = dims;
        self
    }
}

#[async_trait]
impl EmbeddingProvider for GenAiEmbedder {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Failure> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let target = resolve_target(
            &self.client,
            Some(self.adapter),
            &self.model,
            self.endpoint.as_deref(),
            self.auth.as_deref(),
        )
        .await?;
        let req = EmbedRequest::from_texts(texts.to_vec());
        let opts = EmbedOptions::default().with_capture_usage(true);
        let resp = self
            .client
            .exec_embed(target, req, Some(&opts))
            .await
            .map_err(classify_genai_error)?;
        Ok(resp.into_vectors())
    }
}

/// Build the per-call [`ServiceTarget`]: `genai`'s resolved default for the model, with the auth key
/// (the lease secret) and any endpoint override applied. When `adapter` is `None` the genai adapter
/// is *inferred* from the model name (namespace or prefix; see [`AdapterKind::from_model`]); a
/// `Some` forces it. Free (not a method) so the streaming task can resolve without borrowing the
/// provider. When `auth` is `None`, genai resolves the adapter's `default_key_env_name()` from the
/// environment as the fallback credential.
async fn resolve_target(
    client: &Client,
    adapter: Option<AdapterKind>,
    model: &str,
    endpoint: Option<&str>,
    auth: Option<&str>,
) -> Result<ServiceTarget, Failure> {
    let model_iden = match adapter {
        Some(adapter) => ModelIden::new(adapter, model.to_string()),
        None => client
            .default_model(model)
            .map_err(|e| Failure::Provider(format!("infer adapter for {model:?}: {e}")))?,
    };
    let mut target = client
        .resolve_service_target(model_iden)
        .await
        .map_err(|e| Failure::Provider(format!("resolve target: {e}")))?;
    if let Some(key) = auth {
        target.auth = AuthData::from_single(key.to_string());
    }
    if let Some(base) = endpoint {
        target.endpoint = Endpoint::from_owned(base.to_string());
    }
    Ok(target)
}

/// Map an engine [`CacheTtl`] to the genai `cache_control` marker: the default 5-minute tier is
/// the bare `{"type":"ephemeral"}` (no `ttl` key, hermes' 5m marker), the extended tier serializes
/// `{"type":"ephemeral","ttl":"1h"}` on the Anthropic wire. Every breakpoint of one request shares
/// the marker, so genai's "1h before 5m" ordering constraint can never trip.
fn cache_marker(ttl: CacheTtl) -> CacheControl {
    match ttl {
        CacheTtl::FiveMin => CacheControl::Ephemeral,
        CacheTtl::OneHour => CacheControl::Ephemeral1h,
    }
}

/// Map one flattened [`RequestMsg`] into a `genai` [`ChatMessage`], preserving native tool linkage.
/// A [`RequestMsg::cache_breakpoint`] becomes the request's `cache_control` `marker` so a
/// prefix-caching provider (Anthropic) caches the conversation up to that message; other providers
/// ignore it.
fn to_chat_message(msg: &RequestMsg, marker: CacheControl) -> ChatMessage {
    let chat_msg = match msg.role.as_str() {
        "tool" => {
            let call_id = msg.tool_call_id.clone().unwrap_or_default();
            ChatMessage::from(ToolResponse::new(call_id, msg.content.clone()))
        }
        "assistant" if !msg.tool_calls.is_empty() => {
            let mut parts: Vec<ContentPart> = Vec::new();
            if !msg.content.is_empty() {
                parts.push(ContentPart::from_text(msg.content.clone()));
            }
            for tc in &msg.tool_calls {
                let args: serde_json::Value =
                    serde_json::from_str(&tc.args).unwrap_or(serde_json::Value::Null);
                parts.push(ContentPart::ToolCall(GenToolCall {
                    call_id: tc.call_id.clone(),
                    fn_name: tc.name.clone(),
                    fn_arguments: args,
                    thought_signatures: None,
                }));
            }
            ChatMessage::assistant(MessageContent::from_parts(parts))
        }
        "assistant" => ChatMessage::assistant(msg.content.clone()),
        _ if !msg.images.is_empty() => {
            // A multimodal user message: the text (when present) followed by one image part each.
            let mut parts: Vec<ContentPart> = Vec::new();
            if !msg.content.is_empty() {
                parts.push(ContentPart::from_text(msg.content.clone()));
            }
            for img in &msg.images {
                parts.push(to_image_part(img));
            }
            ChatMessage::user(MessageContent::from_parts(parts))
        }
        _ => ChatMessage::user(msg.content.clone()),
    };
    if msg.cache_breakpoint {
        chat_msg.with_options(marker)
    } else {
        chat_msg
    }
}

/// Map a resolved [`RequestImage`] onto a genai binary [`ContentPart`]. The tool layer has already
/// produced base64 bytes + MIME (path/URL/`data:` resolution and SSRF checks live there), so this is
/// a pure re-wrap; providers that do not accept images ignore the part.
fn to_image_part(img: &RequestImage) -> ContentPart {
    ContentPart::from_binary_base64(img.mime.clone(), img.data_base64.clone(), None)
}

/// The published **output**-token cap for a well-known cloud chat model (the max a single generation
/// may emit — distinct from the context window in [`known_context_window`]), matched by id prefix so
/// dated / `-latest` aliases resolve. Prefixes are ordered most-specific first because the first
/// match wins. `None` for unknown models, in which case the caller applies
/// [`DEFAULT_MAX_OUTPUT_TOKENS`](crate::DEFAULT_MAX_OUTPUT_TOKENS). Kept local to the provider so this
/// crate stays free of the `daemon-api` catalog type (mirrors [`known_context_window`]).
pub(crate) fn known_max_output(model: &str) -> Option<u32> {
    const TABLE: &[(&str, u32)] = &[
        // Anthropic Claude: 4.x families publish large output windows; 3.5 doubles 3.x.
        ("claude-opus-4", 32_000),
        ("claude-sonnet-4", 64_000),
        ("claude-3-5-sonnet", 8_192),
        ("claude-3-5-haiku", 8_192),
        ("claude-3-opus", 4_096),
        ("claude-3-haiku", 4_096),
        ("claude-3-sonnet", 4_096),
        // OpenAI: reasoning models (o*) allow very large completions; 4o/4.1 are mid-range.
        ("gpt-4o-mini", 16_384),
        ("gpt-4o", 16_384),
        ("gpt-4.1", 32_768),
        ("o4-mini", 100_000),
        ("o3", 100_000),
        ("o1", 100_000),
        // Google Gemini.
        ("gemini-2.5", 65_536),
        ("gemini-2.0", 8_192),
        ("gemini-1.5", 8_192),
    ];
    TABLE
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|&(_, cap)| cap)
}

/// The published context window for a well-known cloud chat model, matched by id prefix so dated /
/// `-latest` aliases resolve. `None` for unknown models (the engine then has no denominator). Kept
/// local to the provider so this crate stays free of the `daemon-api` catalog type.
fn known_context_window(model: &str) -> Option<u32> {
    const TABLE: &[(&str, u32)] = &[
        ("claude-opus-4", 200_000),
        ("claude-sonnet-4", 200_000),
        ("claude-3-5-sonnet", 200_000),
        ("claude-3-5-haiku", 200_000),
        ("claude-3-opus", 200_000),
        ("gpt-4o", 128_000),
        ("gpt-4.1", 1_000_000),
        ("o3", 200_000),
        ("o4-mini", 200_000),
    ];
    TABLE
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map(|&(_, ctx)| ctx)
}

/// Map `genai`'s [`Usage`](genai::chat::Usage) into the canonical [`UsageDelta`], including the
/// Anthropic/OpenAI prompt-cache + reasoning-token breakdowns the provider surfaces in the
/// `*_details` sub-objects.
fn usage_from(usage: &genai::chat::Usage, pricing: Option<&Pricing>) -> UsageDelta {
    let prompt = usage.prompt_tokens_details.as_ref();
    let completion = usage.completion_tokens_details.as_ref();
    let cache_read = prompt.and_then(|d| d.cached_tokens).unwrap_or(0).max(0) as u64;
    let cache_write = prompt
        .and_then(|d| d.cache_creation_tokens)
        .unwrap_or(0)
        .max(0) as u64;
    let reasoning = completion
        .and_then(|d| d.reasoning_tokens)
        .unwrap_or(0)
        .max(0) as u64;
    let mut delta = UsageDelta {
        input_tokens: usage.prompt_tokens.unwrap_or(0).max(0) as u64,
        output_tokens: usage.completion_tokens.unwrap_or(0).max(0) as u64,
        api_calls: 1,
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        reasoning_tokens: reasoning,
        cost_micros: 0,
    };
    if let Some(pricing) = pricing {
        delta.cost_micros = delta.estimate_cost_micros(pricing);
    }
    delta
}

/// Map a genai [`AdapterKind`] onto the OpenTelemetry `gen_ai.provider.name` well-known value where
/// one exists, else fall back to genai's lowercase adapter name (a valid custom value per the spec).
fn provider_name_otel(kind: AdapterKind) -> String {
    match kind {
        AdapterKind::OpenAI | AdapterKind::OpenAIResp => "openai",
        AdapterKind::Anthropic => "anthropic",
        AdapterKind::Gemini => "gcp.gemini",
        AdapterKind::Groq => "groq",
        AdapterKind::DeepSeek => "deepseek",
        AdapterKind::Xai => "x_ai",
        AdapterKind::Cohere => "cohere",
        other => return other.as_lower_str().to_string(),
    }
    .to_string()
}

/// Normalize a genai [`StopReason`] onto the OpenTelemetry `gen_ai.response.finish_reasons` values
/// (`stop`/`length`/`tool_calls`/`content_filter`), passing through anything provider-specific.
fn finish_reason_otel(reason: &StopReason) -> String {
    match reason {
        StopReason::Completed(_) => "stop".to_string(),
        StopReason::MaxTokens(_) => "length".to_string(),
        StopReason::ToolCall(_) => "tool_calls".to_string(),
        StopReason::ContentFilter(_) => "content_filter".to_string(),
        StopReason::StopSequence(_) => "stop".to_string(),
        StopReason::Other(raw) => raw.clone(),
    }
}

/// Map `genai` tool calls into the pre-repair [`RawToolCall`]s.
fn raw_calls(calls: Vec<GenToolCall>) -> Vec<RawToolCall> {
    calls
        .into_iter()
        .map(|c| RawToolCall {
            id: c.call_id,
            name: c.fn_name,
            args: if c.fn_arguments.is_null() {
                "{}".to_string()
            } else {
                c.fn_arguments.to_string()
            },
        })
        .collect()
}

/// Decode a non-streaming [`ChatResponse`] into the canonical [`ModelOutput`] through §9 repair,
/// carrying the provider response metadata (finish reason, response id/model, vendor) and the
/// applied `max_tokens` cap onto the output for telemetry.
fn decode_response(
    resp: ChatResponse,
    valid_tools: &[String],
    pricing: Option<&Pricing>,
    max_tokens: u32,
) -> ModelOutput {
    let usage = usage_from(&resp.usage, pricing);
    let reasoning = resp.reasoning_content.clone();
    let provider_name = Some(provider_name_otel(resp.model_iden.adapter_kind));
    let response_model = Some(resp.provider_model_iden.model_name.to_string());
    let response_id = resp.response_id.clone();
    let finish_reason = resp.stop_reason.as_ref().map(finish_reason_otel);
    let text = resp.content.joined_texts().unwrap_or_default();
    let calls = raw_calls(resp.content.into_tool_calls());
    let mut out = finalize_output(text, reasoning, calls, usage, valid_tools);
    out.meta = Some(Box::new(ResponseMeta {
        finish_reason,
        response_id,
        response_model,
        provider_name,
        params: Some(RequestParams {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }),
    }));
    out
}

/// Assemble a [`ModelOutput`] from a captured [`StreamEnd`] through §9 repair, carrying the provider
/// response metadata (finish reason, response id, vendor + resolved model from the stream's
/// [`ModelIden`]) and the applied `max_tokens` cap onto the output for telemetry.
fn decode_stream_end(
    end: StreamEnd,
    valid_tools: &[String],
    pricing: Option<&Pricing>,
    model_iden: &ModelIden,
    max_tokens: u32,
) -> ModelOutput {
    let usage = end
        .captured_usage
        .as_ref()
        .map(|u| usage_from(u, pricing))
        .unwrap_or_default();
    let reasoning = end.captured_reasoning_content.clone();
    let response_id = end.captured_response_id.clone();
    let finish_reason = end.captured_stop_reason.as_ref().map(finish_reason_otel);
    let provider_name = Some(provider_name_otel(model_iden.adapter_kind));
    let response_model = Some(model_iden.model_name.to_string());
    let text = end
        .captured_texts()
        .map(|texts| texts.join(""))
        .unwrap_or_default();
    let calls = raw_calls(end.captured_into_tool_calls().unwrap_or_default());
    let mut out = finalize_output(text, reasoning, calls, usage, valid_tools);
    out.meta = Some(Box::new(ResponseMeta {
        finish_reason,
        response_id,
        response_model,
        provider_name,
        params: Some(RequestParams {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }),
    }));
    out
}

#[async_trait]
impl Provider for GenAiProvider {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: true,
            supports_streaming: true,
            tool_call_format: ToolCallFormat::Native,
            // The published context window for a well-known cloud model (the context-fill HUD's
            // denominator). `None` when the model is unknown to the static table.
            max_context: known_context_window(&self.model),
        }
    }

    async fn chat(&self, req: Request) -> Result<ModelOutput, Failure> {
        let target = resolve_target(
            &self.client,
            self.adapter,
            &self.model,
            self.endpoint.as_deref(),
            req.auth.as_deref(),
        )
        .await?;
        let chat = self.chat_request(&req);
        let opts = self.options(&req);
        let valid = req.tool_names();
        let resp = self
            .client
            .exec_chat(target, chat, Some(&opts))
            .await
            .map_err(classify_genai_error)?;
        Ok(decode_response(
            resp,
            &valid,
            self.pricing.as_ref(),
            self.effective_max_tokens(&req),
        ))
    }

    fn stream(&self, req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
        let client = self.client.clone();
        let adapter = self.adapter;
        let model = self.model.clone();
        let endpoint = self.endpoint.clone();
        let auth = req.auth.clone();
        let chat = self.chat_request(&req);
        let opts = self.options(&req);
        let valid = req.tool_names();
        let pricing = self.pricing;
        let max_tokens = self.effective_max_tokens(&req);
        let (tx, rx) = futures::channel::mpsc::unbounded();

        tokio::spawn(async move {
            let target = match resolve_target(
                &client,
                adapter,
                &model,
                endpoint.as_deref(),
                auth.as_deref(),
            )
            .await
            {
                Ok(t) => t,
                Err(f) => {
                    let _ = tx.unbounded_send(Err(f));
                    return;
                }
            };
            let resp = match client.exec_chat_stream(target, chat, Some(&opts)).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.unbounded_send(Err(classify_genai_error(e)));
                    return;
                }
            };
            let model_iden = resp.model_iden.clone();
            let mut stream = resp.stream;
            while let Some(event) = stream.next().await {
                match event {
                    Ok(ChatStreamEvent::Chunk(chunk)) => {
                        if tx
                            .unbounded_send(Ok(StreamEvent::TextDelta(chunk.content)))
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(ChatStreamEvent::ReasoningChunk(chunk)) => {
                        if tx
                            .unbounded_send(Ok(StreamEvent::ReasoningDelta(chunk.content)))
                            .is_err()
                        {
                            return;
                        }
                    }
                    Ok(ChatStreamEvent::End(end)) => {
                        let out = decode_stream_end(
                            end,
                            &valid,
                            pricing.as_ref(),
                            &model_iden,
                            max_tokens,
                        );
                        let _ = tx.unbounded_send(Ok(StreamEvent::Done(out)));
                        return;
                    }
                    // Start / ToolCallChunk / ThoughtSignatureChunk: the authoritative tool calls and
                    // text are captured in `End` (capture_* options), so per-chunk tool deltas are not
                    // re-assembled here.
                    Ok(_) => {}
                    Err(e) => {
                        let _ = tx.unbounded_send(Err(classify_genai_error(e)));
                        return;
                    }
                }
            }
        });

        Box::pin(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_max_output_resolves_by_prefix_most_specific_first() {
        // Dated / -latest aliases resolve via the prefix.
        assert_eq!(known_max_output("claude-sonnet-4-20250514"), Some(64_000));
        assert_eq!(known_max_output("claude-opus-4-8"), Some(32_000));
        assert_eq!(known_max_output("claude-3-5-sonnet-latest"), Some(8_192));
        assert_eq!(known_max_output("claude-3-opus-20240229"), Some(4_096));
        // gpt-4o-mini must win over the gpt-4o prefix (more specific first).
        assert_eq!(known_max_output("gpt-4o-mini"), Some(16_384));
        assert_eq!(known_max_output("gpt-4o-2024-08-06"), Some(16_384));
        assert_eq!(known_max_output("o3-mini"), Some(100_000));
        assert_eq!(known_max_output("gemini-2.5-pro"), Some(65_536));
    }

    #[test]
    fn unknown_model_has_no_published_cap() {
        assert_eq!(known_max_output("some-future-model"), None);
    }

    #[test]
    fn for_model_sources_the_models_output_cap_not_the_flat_fallback() {
        // A large-output model gets its real cap, not the conservative fallback.
        assert_eq!(
            GenAiProvider::for_model("claude-sonnet-4").max_tokens,
            64_000
        );
        assert_eq!(GenAiProvider::for_model("o3").max_tokens, 100_000);
        // An unknown model falls back to the shared conservative default.
        assert_eq!(
            GenAiProvider::for_model("some-future-model").max_tokens,
            crate::DEFAULT_MAX_OUTPUT_TOKENS
        );
    }

    #[test]
    fn with_max_tokens_overrides_the_sourced_cap() {
        let p = GenAiProvider::for_model("claude-sonnet-4").with_max_tokens(123);
        assert_eq!(p.max_tokens, 123);
    }

    #[test]
    fn options_apply_per_call_params_when_set() {
        let p = GenAiProvider::for_model("gpt-4o").with_max_tokens(1000);
        let req = Request::default().with_params(RequestParams {
            temperature: Some(0.2),
            top_p: Some(0.9),
            max_tokens: Some(500),
            seed: Some(7),
            // genai 0.6 has no top_k setter; it is silently ignored.
            top_k: Some(40),
        });
        let opts = p.options(&req);
        assert_eq!(opts.temperature, Some(0.2));
        assert_eq!(opts.top_p, Some(0.9));
        assert_eq!(opts.seed, Some(7));
        // The per-call cap overrides the model cap.
        assert_eq!(opts.max_tokens, Some(500));
    }

    #[test]
    fn options_default_params_leave_sampling_unset_and_use_model_cap() {
        let p = GenAiProvider::for_model("gpt-4o").with_max_tokens(1000);
        let opts = p.options(&Request::default());
        assert_eq!(opts.temperature, None);
        assert_eq!(opts.top_p, None);
        assert_eq!(opts.seed, None);
        assert_eq!(opts.max_tokens, Some(1000));
    }

    #[test]
    fn effective_max_tokens_prefers_per_call_override() {
        let p = GenAiProvider::for_model("gpt-4o").with_max_tokens(1000);
        assert_eq!(p.effective_max_tokens(&Request::default()), 1000);
        let req = Request::default().with_params(RequestParams {
            max_tokens: Some(42),
            ..Default::default()
        });
        assert_eq!(p.effective_max_tokens(&req), 42);
    }

    #[test]
    fn to_chat_message_maps_images_into_binary_parts() {
        let msg = RequestMsg {
            role: "user".into(),
            content: "describe this".into(),
            images: vec![RequestImage {
                mime: "image/png".into(),
                data_base64: "AAAA".into(),
            }],
            ..Default::default()
        };
        let chat = to_chat_message(&msg, CacheControl::Ephemeral);
        let parts = chat.content.parts();
        assert!(parts.iter().any(|p| p.is_text()), "keeps the text part");
        assert!(
            parts.iter().any(|p| p.is_image()),
            "emits an image binary part"
        );
    }

    #[test]
    fn to_chat_message_text_only_user_has_no_image_part() {
        let msg = RequestMsg {
            role: "user".into(),
            content: "no image".into(),
            ..Default::default()
        };
        let chat = to_chat_message(&msg, CacheControl::Ephemeral);
        assert!(!chat.content.parts().iter().any(|p| p.is_image()));
    }
}
