// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The mistral.rs backend.
//!
//! Unlike `llama-cpp-4`, mistral.rs is async and tokio-native: [`Model`] is `Send + Sync` and
//! [`Model::stream_chat_request`] yields a stream of [`Response`] chunks, so [`generate`] forwards
//! chunks directly from the worker's tokio runtime and cancels by dropping the stream on the cancel
//! token.
//!
//! This is the Phase-1 seam: text + streaming + recovery. Native tool calls (mistral.rs supports
//! them), richer sampling, paged-attention, and CUDA/Metal perf tuning are deepened in the Phase-2
//! `mistralrs-depth` pass; the [`InferenceBackend`] contract is identical to the llama backend so
//! the daemon-side `LocalProvider` does not change.

use mistralrs::{IsqType, RequestBuilder, Response, TextMessageRole, TextModelBuilder};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendChunk, BackendError, GenerateRequest, InferenceBackend};
use crate::protocol::{Capabilities, Constraint, ModelParams, ToolCallFormat, Usage};
use crate::tooling;

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

        // With tools offered, buffer output so tool-call markup is parsed out before any text is
        // surfaced (mirrors the llama backend); without tools, stream live. Native mistral.rs
        // Tool/ToolChoice decode is the Phase-2 `mistralrs-depth` upgrade.
        let has_tools = !req.tools.is_empty();
        let mut buffered = String::new();
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
                                        if has_tools {
                                            buffered.push_str(text);
                                        } else if tx.send(BackendChunk::Text(text.clone())).is_err() {
                                            // Consumer dropped the receiver (cancel/abort upstream).
                                            break;
                                        }
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

        if has_tools {
            let (cleaned, calls) = tooling::extract_tool_calls(&buffered);
            if !cleaned.is_empty() {
                let _ = tx.send(BackendChunk::Text(cleaned));
            }
            for call in calls {
                if tx.send(BackendChunk::Tool(call)).is_err() {
                    break;
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

/// Translate our protocol request into a mistral.rs [`RequestBuilder`], applying the conversation,
/// the tool-advertisement preamble, and the sampling knobs (`temperature`/`top_p`/`top_k` and the
/// output-token cap), mirroring the llama backend's sampling semantics.
fn build_request(req: &GenerateRequest) -> RequestBuilder {
    let mut builder = RequestBuilder::new();
    let system = match tooling::tool_preamble(&req.tools) {
        Some(preamble) if req.system.is_empty() => preamble,
        Some(preamble) => format!("{preamble}\n{}", req.system),
        None => req.system.clone(),
    };
    if !system.is_empty() {
        builder = builder.add_message(TextMessageRole::System, &system);
    }
    for msg in &req.messages {
        let role = match msg.role.as_str() {
            "system" => TextMessageRole::System,
            "assistant" => TextMessageRole::Assistant,
            "tool" => TextMessageRole::Tool,
            _ => TextMessageRole::User,
        };
        builder = builder.add_message(role, &msg.content);
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
    if req.max_tokens > 0 {
        builder = builder.set_sampler_max_len(req.max_tokens as usize);
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
