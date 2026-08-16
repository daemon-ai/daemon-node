// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The mistral.rs backend.
//!
//! Unlike `llama-cpp-4`, mistral.rs is async and tokio-native: [`Model`] is `Send + Sync` and
//! [`Model::stream_chat_request`] yields a stream of [`Response`] chunks, so [`generate`] forwards
//! chunks directly from the worker's tokio runtime and cancels by dropping the stream on the cancel
//! token.
//!
//! Tools ride the engine's native API: offered tools go through `RequestBuilder::set_tools` /
//! `set_tool_choice` (the engine renders them through the model's own chat template), and calls
//! come back structured on `Delta::tool_calls` — no text preamble, no markup parsing. This is what
//! makes the `supports_native_tools` capability claim factual. Paged-attention and CUDA/Metal perf
//! tuning remain the later `mistralrs-depth` items; the [`InferenceBackend`] contract is identical
//! to the llama backend so the daemon-side `LocalProvider` does not change.

use std::collections::HashMap;

use mistralrs::{
    Function, IsqType, RequestBuilder, Response, TextMessageRole, TextModelBuilder, Tool,
    ToolCallResponse, ToolCallType, ToolChoice, ToolType,
};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendChunk, BackendError, GenerateRequest, InferenceBackend};
use crate::protocol::{Capabilities, ModelParams, ToolCall, ToolCallFormat, Usage};

/// A loaded mistral.rs model.
pub struct MistralRsBackend {
    model: mistralrs::Model,
    capabilities: Capabilities,
}

impl MistralRsBackend {
    /// Build a text model from `model` (an HF repo id or local path). Phase-1 seam: ISQ in-situ
    /// quantization is applied when requested; deeper builder options arrive in Phase 2.
    ///
    /// Prompt caching is engine-managed here: mistral.rs enables prefix caching by default
    /// (`prefix_cache_n = 16`) — block-level when PagedAttention is on, sequence-level otherwise —
    /// and this one [`MistralRsBackend`] holds a persistent [`mistralrs::Model`] for the worker's
    /// lifetime, so a shared prefix (system prompt + prior turns) is reused across generations for
    /// lower TTFT. We deliberately pass no cache-disabling builder flag; do not add
    /// `with_prefix_cache_n(None)`/`no_kv_cache` unless prefix reuse must be turned off.
    pub async fn load(model: &str, params: &ModelParams) -> Result<Self, BackendError> {
        let mut builder = TextModelBuilder::new(model.to_string());
        if let Some(isq) = params.isq.as_deref().and_then(parse_isq) {
            builder = builder.with_isq(isq);
        }
        let model = builder
            .build()
            .await
            .map_err(|e| classify_build(&e.to_string()))?;
        let capabilities = Capabilities {
            supports_native_tools: true,
            supports_streaming: true,
            tool_call_format: ToolCallFormat::Native,
            // Advertise the configured context window when set (the model default otherwise).
            max_context: (params.n_ctx > 0).then_some(params.n_ctx),
            // mistral.rs advertises tools through its native tool API, not our template renderer.
            template_tools: false,
        };
        Ok(Self {
            model,
            capabilities,
        })
    }
}

#[async_trait::async_trait]
impl InferenceBackend for MistralRsBackend {
    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    async fn generate(
        &self,
        req: GenerateRequest,
        tx: UnboundedSender<BackendChunk>,
        cancel: CancellationToken,
    ) -> Result<Usage, BackendError> {
        let request = build_request(&req);
        let mut stream = self
            .model
            .stream_chat_request(request)
            .await
            .map_err(|e| BackendError::transient(format!("mistralrs request: {e}")))?;

        // Text streams live; tool calls arrive structured on the delta (native Tool/ToolChoice),
        // so there is nothing to buffer or parse out of the text.
        let mut produced: u64 = 0;
        // The final streaming chunk carries authoritative prompt/completion token counts.
        let mut token_usage: Option<(u64, u64)> = None;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Err(BackendError::cancelled()),
                next = stream.next() => {
                    let Some(resp) = next else { break };
                    match resp {
                        Response::Chunk(chunk) => {
                            if let Some(usage) = chunk.usage.as_ref() {
                                token_usage = Some((
                                    usage.prompt_tokens as u64,
                                    usage.completion_tokens as u64,
                                ));
                            }
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(text) = choice.delta.content.as_ref() {
                                    if !text.is_empty() {
                                        produced += 1;
                                        if tx.send(BackendChunk::Text(text.clone())).is_err() {
                                            // Consumer dropped the receiver (cancel/abort upstream).
                                            break;
                                        }
                                    }
                                }
                                for call in choice.delta.tool_calls.iter().flatten() {
                                    produced += 1;
                                    let sent = tx.send(BackendChunk::Tool(ToolCall {
                                        call_id: call.id.clone(),
                                        name: call.function.name.clone(),
                                        args: call.function.arguments.clone(),
                                    }));
                                    if sent.is_err() {
                                        break;
                                    }
                                }
                                if choice.finish_reason.is_some() {
                                    break;
                                }
                            }
                        }
                        Response::InternalError(e) | Response::ValidationError(e) => {
                            return Err(classify_build(&e.to_string()));
                        }
                        Response::ModelError(msg, _) => {
                            return Err(classify_model_error(&msg));
                        }
                        _ => {}
                    }
                }
            }
        }

        // Prefer the engine's authoritative token usage (final chunk); fall back to the streamed
        // chunk count for completion when the engine reported none.
        let (input_tokens, output_tokens) = token_usage.unwrap_or((0, produced));
        Ok(Usage {
            input_tokens,
            output_tokens,
            // mistral.rs manages its own block-/sequence-level prefix cache internally and does not
            // surface a reused-prefix count on the streaming chunk usage, so report 0 here. The
            // engine still benefits from the reuse (lower TTFT); it is just not separately metered.
            cache_read_tokens: 0,
        })
    }

    async fn embed(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>, BackendError> {
        // mistral.rs embeddings load a distinct `EmbeddingModelBuilder` model (not the
        // `TextModelBuilder` this backend holds). The spec's preferred local-embeddings engine is
        // llama-cpp-4; wiring the mistral.rs `EmbeddingModelBuilder` path is a follow-on.
        Err(BackendError::fatal(
            "local embeddings are served by the llama engine; mistral.rs embedding models are not wired yet",
        ))
    }
}

/// Translate our protocol request into a mistral.rs [`RequestBuilder`]: the conversation with
/// native tool-call linkage preserved, the offered tools (engine-rendered through the model's own
/// chat template), and the sampling knobs (`temperature`/`top_p`/`top_k`, additive penalties,
/// stops, and the output-token cap), mirroring the llama backend's sampling semantics.
fn build_request(req: &GenerateRequest) -> RequestBuilder {
    let mut builder = RequestBuilder::new();
    if !req.system.is_empty() {
        builder = builder.add_message(TextMessageRole::System, &req.system);
    }
    for msg in &req.messages {
        match msg.role.as_str() {
            // An assistant turn that emitted tool calls keeps them attached, so the template can
            // render the call/result linkage the model was trained on.
            "assistant" if !msg.tool_calls.is_empty() => {
                let calls = msg
                    .tool_calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| tool_call_response(i, c))
                    .collect();
                builder = builder.add_message_with_tool_call(
                    TextMessageRole::Assistant,
                    &msg.content,
                    calls,
                );
            }
            // A tool result answers a specific call id (`tool_call_id` rides the message).
            "tool" => {
                let id = msg.tool_call_id.clone().unwrap_or_default();
                builder = builder.add_tool_message(&msg.content, id);
            }
            "system" => builder = builder.add_message(TextMessageRole::System, &msg.content),
            "assistant" => builder = builder.add_message(TextMessageRole::Assistant, &msg.content),
            _ => builder = builder.add_message(TextMessageRole::User, &msg.content),
        }
    }
    if !req.tools.is_empty() {
        let tools = req.tools.iter().map(tool_def).collect();
        builder = builder.set_tools(tools).set_tool_choice(ToolChoice::Auto);
    }

    // Sampling: a non-positive temperature is greedy/deterministic (matches the llama backend).
    let s = &req.sampling;
    if s.temperature <= 0.0 {
        builder = builder.set_deterministic_sampler();
    } else {
        builder = builder.set_sampler_temperature(s.temperature as f64);
        if s.top_k > 0 {
            builder = builder.set_sampler_topk(s.top_k as usize);
        }
        if s.top_p < 1.0 {
            builder = builder.set_sampler_topp(s.top_p as f64);
        }
    }
    // Additive penalties map directly; the multiplicative repeat penalty has no RequestBuilder
    // setter in mistral.rs 0.8, so it is llama-engine-only (neutral defaults add nothing here).
    if s.penalty_freq != 0.0 {
        builder = builder.set_sampler_frequency_penalty(s.penalty_freq);
    }
    if s.penalty_present != 0.0 {
        builder = builder.set_sampler_presence_penalty(s.penalty_present);
    }
    if req.max_tokens > 0 {
        builder = builder.set_sampler_max_len(req.max_tokens as usize);
    }
    if !req.stop.is_empty() {
        builder = builder.set_sampler_stop_toks(mistralrs::StopTokens::Seqs(req.stop.clone()));
    }

    // Grammar constraint: mistral.rs consumes the Lark dialect (llguidance). When only a GBNF
    // rendering is present it is for the llama engine — ignore it here rather than fail.
    if let Some(constraint) = &req.constraint {
        match &constraint.lark {
            Some(grammar) => {
                builder = builder.set_constraint(mistralrs::Constraint::Lark(grammar.clone()));
            }
            None => tracing::warn!("mistralrs: constraint carries no Lark grammar; ignoring"),
        }
    }
    builder
}

/// Map one offered [`crate::protocol::ToolDef`] into the engine's [`Tool`] (OpenAI function
/// shape). The schema string is the argument JSON-Schema; its top-level `description` doubles as
/// the tool description (same convention as the llama template renderer). An unparseable schema
/// degrades to a parameterless tool rather than failing the request.
fn tool_def(t: &crate::protocol::ToolDef) -> Tool {
    let parameters = serde_json::from_str::<HashMap<String, serde_json::Value>>(&t.schema).ok();
    let description = parameters
        .as_ref()
        .and_then(|p| p.get("description"))
        .and_then(|d| d.as_str())
        .map(str::to_string);
    Tool {
        tp: ToolType::Function,
        function: Function {
            name: t.name.clone(),
            description,
            parameters,
        },
    }
}

/// Re-shape a historical assistant [`ToolCall`] into the engine's [`ToolCallResponse`] so
/// `add_message_with_tool_call` can render the turn faithfully.
fn tool_call_response(index: usize, c: &ToolCall) -> ToolCallResponse {
    ToolCallResponse {
        index,
        id: c.call_id.clone(),
        tp: ToolCallType::Function,
        function: mistralrs::CalledFunction {
            name: c.name.clone(),
            arguments: c.args.clone(),
        },
    }
}

/// Parse a textual ISQ name (config-supplied) into a mistral.rs [`IsqType`]. Unknown names are
/// ignored (no quantization), keeping the seam forgiving.
fn parse_isq(name: &str) -> Option<IsqType> {
    match name.to_ascii_uppercase().replace('-', "_").as_str() {
        "Q4_0" => Some(IsqType::Q4_0),
        "Q4_1" => Some(IsqType::Q4_1),
        "Q5_0" => Some(IsqType::Q5_0),
        "Q5_1" => Some(IsqType::Q5_1),
        "Q8_0" => Some(IsqType::Q8_0),
        "Q2K" => Some(IsqType::Q2K),
        "Q3K" => Some(IsqType::Q3K),
        "Q4K" => Some(IsqType::Q4K),
        "Q5K" => Some(IsqType::Q5K),
        "Q6K" => Some(IsqType::Q6K),
        "Q8K" => Some(IsqType::Q8K),
        _ => None,
    }
}

/// Classify a build/internal failure: VRAM/host allocation failures map to OOM, the rest are fatal
/// (bad model id, unsupported arch, etc.).
fn classify_build(msg: &str) -> BackendError {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("out of memory")
        || lower.contains("oom")
        || lower.contains("alloc")
        || lower.contains("cuda error")
    {
        BackendError::out_of_memory(format!("mistralrs: {msg}"))
    } else {
        BackendError::fatal(format!("mistralrs: {msg}"))
    }
}

/// Classify a per-request model error: context/length overflow vs OOM vs transient.
fn classify_model_error(msg: &str) -> BackendError {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("context") || lower.contains("length") || lower.contains("too long") {
        BackendError::context_overflow(format!("mistralrs: {msg}"))
    } else if lower.contains("out of memory") || lower.contains("oom") || lower.contains("alloc") {
        BackendError::out_of_memory(format!("mistralrs: {msg}"))
    } else {
        BackendError::transient(format!("mistralrs: {msg}"))
    }
}

#[cfg(test)]
mod tests {
    use mistralrs::RequestLike;

    use super::*;
    use crate::protocol::{Msg, Sampling, ToolDef};

    fn base_request() -> GenerateRequest {
        GenerateRequest {
            system: "Be terse.".into(),
            messages: vec![Msg {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn recall_tool() -> ToolDef {
        ToolDef {
            name: "mnemosyne_recall".into(),
            schema: r#"{"type":"object","description":"Recall a memory.","properties":{"query":{"type":"string"}}}"#.into(),
        }
    }

    #[test]
    fn tools_ride_the_native_api_not_the_prompt() {
        let mut req = base_request();
        req.tools = vec![recall_tool()];
        let mut built = build_request(&req);

        // Native: set_tools + auto tool choice.
        let (tools, choice) = built.take_tools().expect("tools attached");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "mnemosyne_recall");
        assert_eq!(
            tools[0].function.description.as_deref(),
            Some("Recall a memory.")
        );
        assert!(tools[0]
            .function
            .parameters
            .as_ref()
            .is_some_and(|p| p.contains_key("properties")));
        assert!(matches!(choice, ToolChoice::Auto));

        // And NOT the prompt: no message text mentions the tool (the old preamble did).
        for m in built.messages_ref() {
            if let Some(text) = m.get("content").and_then(|c| c.as_ref().left()) {
                assert!(
                    !text.contains("mnemosyne_recall"),
                    "tool name leaked into prompt text: {text}"
                );
            }
        }
    }

    #[test]
    fn unparseable_schema_degrades_to_parameterless_tool() {
        let mut req = base_request();
        req.tools = vec![ToolDef {
            name: "broken".into(),
            schema: "not json".into(),
        }];
        let mut built = build_request(&req);
        let (tools, _) = built.take_tools().expect("tools attached");
        assert_eq!(tools[0].function.name, "broken");
        assert!(tools[0].function.parameters.is_none());
        assert!(tools[0].function.description.is_none());
    }

    #[test]
    fn tool_turns_keep_call_linkage() {
        let mut req = base_request();
        req.messages = vec![
            Msg {
                role: "user".into(),
                content: "recall x".into(),
                ..Default::default()
            },
            Msg {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![ToolCall {
                    call_id: "call-7".into(),
                    name: "mnemosyne_recall".into(),
                    args: r#"{"query":"x"}"#.into(),
                }],
                ..Default::default()
            },
            Msg {
                role: "tool".into(),
                content: "found it".into(),
                tool_call_id: Some("call-7".into()),
                ..Default::default()
            },
        ];
        let built = build_request(&req);
        let messages = built.messages_ref();
        // system + user + assistant(with calls) + tool(result).
        assert_eq!(messages.len(), 4);
        // The assistant turn carries the structured calls (rendered under the "function" key).
        assert!(
            messages[2].contains_key("function"),
            "assistant turn lost its tool calls"
        );
        // The tool result answers the exact call id.
        assert_eq!(
            messages[3]
                .get("tool_call_id")
                .and_then(|c| c.as_ref().left())
                .map(String::as_str),
            Some("call-7")
        );
    }

    #[test]
    fn sampling_penalties_stops_and_cap_reach_the_engine() {
        let mut req = base_request();
        req.sampling = Sampling {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            penalty_freq: 0.5,
            penalty_present: 0.25,
            ..Default::default()
        };
        req.max_tokens = 128;
        req.stop = vec!["</stop>".into()];
        let mut built = build_request(&req);
        let params = built.take_sampling_params();
        assert_eq!(params.frequency_penalty, Some(0.5));
        assert_eq!(params.presence_penalty, Some(0.25));
        assert_eq!(params.max_len, Some(128));
        match params.stop_toks {
            Some(mistralrs::StopTokens::Seqs(seqs)) => {
                assert_eq!(seqs, vec!["</stop>".to_string()])
            }
            other => panic!("expected stop sequences, got {other:?}"),
        }
    }
}
