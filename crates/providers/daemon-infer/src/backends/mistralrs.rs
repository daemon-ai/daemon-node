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

use mistralrs::{IsqType, Response, TextMessageRole, TextMessages, TextModelBuilder};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendChunk, BackendError, GenerateRequest, InferenceBackend};
use crate::protocol::{Capabilities, ModelParams, ToolCallFormat, Usage};
use crate::tooling;

/// A loaded mistral.rs model.
pub struct MistralRsBackend {
    model: mistralrs::Model,
    capabilities: Capabilities,
}

impl MistralRsBackend {
    /// Build a text model from `model` (an HF repo id or local path). Phase-1 seam: ISQ in-situ
    /// quantization is applied when requested; deeper builder options arrive in Phase 2.
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
            max_context: None,
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
        let messages = build_messages(&req);
        let mut stream = self
            .model
            .stream_chat_request(messages)
            .await
            .map_err(|e| BackendError::transient(format!("mistralrs request: {e}")))?;

        // With tools offered, buffer output so tool-call markup is parsed out before any text is
        // surfaced (mirrors the llama backend); without tools, stream live. Native mistral.rs
        // Tool/ToolChoice decode is the Phase-2 `mistralrs-depth` upgrade.
        let has_tools = !req.tools.is_empty();
        let mut buffered = String::new();
        let mut produced: u64 = 0;
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return Err(BackendError::cancelled()),
                next = stream.next() => {
                    let Some(resp) = next else { break };
                    match resp {
                        Response::Chunk(chunk) => {
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

        // mistral.rs streaming chunks do not carry token usage; report output chunk count.
        // (Phase 2 wires exact prompt/completion token usage.)
        Ok(Usage {
            input_tokens: 0,
            output_tokens: produced,
        })
    }
}

/// Translate our protocol messages into mistral.rs `TextMessages`.
fn build_messages(req: &GenerateRequest) -> TextMessages {
    let mut messages = TextMessages::new();
    let system = match tooling::tool_preamble(&req.tools) {
        Some(preamble) if req.system.is_empty() => preamble,
        Some(preamble) => format!("{preamble}\n{}", req.system),
        None => req.system.clone(),
    };
    if !system.is_empty() {
        messages = messages.add_message(TextMessageRole::System, &system);
    }
    for msg in &req.messages {
        let role = match msg.role.as_str() {
            "system" => TextMessageRole::System,
            "assistant" => TextMessageRole::Assistant,
            "tool" => TextMessageRole::Tool,
            _ => TextMessageRole::User,
        };
        messages = messages.add_message(role, &msg.content);
    }
    messages
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
