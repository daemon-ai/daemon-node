//! The [`Provider`] adapter over the [`genai`] multi-provider client.

use crate::{classify_genai_error, finalize_output, RawToolCall};
use async_trait::async_trait;
use daemon_common::UsageDelta;
use daemon_core::{
    Capabilities, EmbeddingProvider, Failure, ModelOutput, Provider, Request, RequestMsg,
    StreamEvent, ToolCallFormat,
};
use futures::stream::BoxStream;
use futures::StreamExt;
use genai::adapter::AdapterKind;
use genai::chat::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponse, ChatStreamEvent, ContentPart,
    MessageContent, StreamEnd, Tool, ToolCall as GenToolCall, ToolResponse,
};
use genai::embed::{EmbedOptions, EmbedRequest};
use genai::resolver::{AuthData, Endpoint};
use genai::{Client, ModelIden, ServiceTarget};

/// A networked model provider backed by [`genai`].
///
/// One adapter serves any `genai`-supported provider; the daemon picks the [`AdapterKind`] + model
/// from config. The credential lease secret (`Request.auth`) is threaded per-call as the provider's
/// API key, so the engine's §7 credential broker stays the source of truth. An optional `endpoint`
/// override points the client at a custom base URL (used by the in-process wire tests).
pub struct GenAiProvider {
    client: Client,
    adapter: AdapterKind,
    model: String,
    endpoint: Option<String>,
    max_tokens: u32,
}

/// The default output cap sent to providers that require one (e.g. Anthropic Messages).
const DEFAULT_MAX_TOKENS: u32 = 4096;

impl GenAiProvider {
    /// A provider for `adapter`/`model` using `genai`'s default endpoint and a default output cap.
    pub fn new(adapter: AdapterKind, model: impl Into<String>) -> Self {
        Self {
            client: Client::default(),
            adapter,
            model: model.into(),
            endpoint: None,
            max_tokens: DEFAULT_MAX_TOKENS,
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

    /// The OpenAI Chat Completions provider for `model`.
    pub fn openai(model: impl Into<String>) -> Self {
        Self::new(AdapterKind::OpenAI, model)
    }

    /// The Anthropic Messages provider for `model`.
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self::new(AdapterKind::Anthropic, model)
    }

    fn options(&self) -> ChatOptions {
        ChatOptions::default()
            .with_capture_usage(true)
            .with_capture_content(true)
            .with_capture_reasoning_content(true)
            .with_capture_tool_calls(true)
            .with_normalize_reasoning_content(true)
            .with_max_tokens(self.max_tokens)
    }

    fn chat_request(&self, req: &Request) -> ChatRequest {
        let messages: Vec<ChatMessage> = req.messages.iter().map(to_chat_message).collect();
        let mut chat = ChatRequest::new(messages);
        if !req.system.is_empty() {
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
            self.adapter,
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

/// Build the per-call [`ServiceTarget`]: `genai`'s resolved default for the adapter/model, with the
/// auth key (the lease secret) and any endpoint override applied. Free (not a method) so the
/// streaming task can resolve it without borrowing the provider.
async fn resolve_target(
    client: &Client,
    adapter: AdapterKind,
    model: &str,
    endpoint: Option<&str>,
    auth: Option<&str>,
) -> Result<ServiceTarget, Failure> {
    let model_iden = ModelIden::new(adapter, model.to_string());
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

/// Map one flattened [`RequestMsg`] into a `genai` [`ChatMessage`], preserving native tool linkage.
fn to_chat_message(msg: &RequestMsg) -> ChatMessage {
    match msg.role.as_str() {
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
        _ => ChatMessage::user(msg.content.clone()),
    }
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
fn usage_from(usage: &genai::chat::Usage) -> UsageDelta {
    let prompt = usage.prompt_tokens_details.as_ref();
    let completion = usage.completion_tokens_details.as_ref();
    let cache_read = prompt.and_then(|d| d.cached_tokens).unwrap_or(0).max(0) as u64;
    let cache_write = prompt
        .and_then(|d| d.cache_creation_tokens)
        .unwrap_or(0)
        .max(0) as u64;
    let reasoning = completion.and_then(|d| d.reasoning_tokens).unwrap_or(0).max(0) as u64;
    UsageDelta {
        input_tokens: usage.prompt_tokens.unwrap_or(0).max(0) as u64,
        output_tokens: usage.completion_tokens.unwrap_or(0).max(0) as u64,
        api_calls: 1,
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        reasoning_tokens: reasoning,
        cost_micros: 0,
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

/// Decode a non-streaming [`ChatResponse`] into the canonical [`ModelOutput`] through §9 repair.
fn decode_response(resp: ChatResponse, valid_tools: &[String]) -> ModelOutput {
    let usage = usage_from(&resp.usage);
    let reasoning = resp.reasoning_content.clone();
    let text = resp.content.joined_texts().unwrap_or_default();
    let calls = raw_calls(resp.content.into_tool_calls());
    finalize_output(text, reasoning, calls, usage, valid_tools)
}

/// Assemble a [`ModelOutput`] from a captured [`StreamEnd`] through §9 repair.
fn decode_stream_end(end: StreamEnd, valid_tools: &[String]) -> ModelOutput {
    let usage = end
        .captured_usage
        .as_ref()
        .map(usage_from)
        .unwrap_or_default();
    let reasoning = end.captured_reasoning_content.clone();
    let text = end
        .captured_texts()
        .map(|texts| texts.join(""))
        .unwrap_or_default();
    let calls = raw_calls(end.captured_into_tool_calls().unwrap_or_default());
    finalize_output(text, reasoning, calls, usage, valid_tools)
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
        let opts = self.options();
        let valid = req.tool_names();
        let resp = self
            .client
            .exec_chat(target, chat, Some(&opts))
            .await
            .map_err(classify_genai_error)?;
        Ok(decode_response(resp, &valid))
    }

    fn stream(&self, req: Request) -> BoxStream<'_, Result<StreamEvent, Failure>> {
        let client = self.client.clone();
        let adapter = self.adapter;
        let model = self.model.clone();
        let endpoint = self.endpoint.clone();
        let auth = req.auth.clone();
        let chat = self.chat_request(&req);
        let opts = self.options();
        let valid = req.tool_names();
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
                        let out = decode_stream_end(end, &valid);
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
