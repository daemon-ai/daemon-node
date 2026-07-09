// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The wire <-> engine mapping: OpenAI [`wire::ChatCompletionRequest`] into a
//! [`daemon_core::Request`], and a [`daemon_core::ModelOutput`] / [`daemon_core::StreamEvent`] back
//! into the OpenAI response / SSE-chunk shapes. This is the correctness surface (roles, params,
//! tool-call round-trip, usage), so it is heavily unit-tested.

use std::time::{SystemTime, UNIX_EPOCH};

use daemon_api::ModelDescriptor;
use daemon_common::UsageDelta;
use daemon_core::{
    ModelOutput, Request, RequestImage, RequestMsg, RequestParams, ToolCall as CoreToolCall,
    ToolDef,
};

use crate::backend::GatewayError;
use crate::wire::{
    ChatCompletionChunkResponse, ChatCompletionResponse, Choice, ChunkChoice, Delta, FunctionCall,
    MessageContent, ModelObject, ModelObjects, OpenAiTool, ResponseMessage, ToolCall, ToolKind,
    Usage,
};

/// Unix seconds now (the `created` timestamp on responses/chunks).
pub(crate) fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A short, stable response id (`chatcmpl-<unix-nanos>`); good enough for a local gateway (clients
/// only need it to be present + stable across a stream).
pub(crate) fn response_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos}")
}

/// Parse a `data:<mime>;base64,<data>` URI into a [`RequestImage`]. Returns `None` for a non-data
/// URL (an `http(s)://` image reference is not resolved here — the gateway does no network I/O).
fn parse_data_url(url: &str) -> Option<RequestImage> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    // meta is like `image/png;base64` (base64 is the only encoding we accept).
    let mime = meta.split(';').next().unwrap_or("").to_string();
    if mime.is_empty() || !meta.contains("base64") {
        return None;
    }
    Some(RequestImage {
        mime,
        data_base64: data.to_string(),
    })
}

/// Collect the image parts of a message's structured content into [`RequestImage`]s (data URLs
/// only; `http(s)` references are skipped since the gateway resolves no network I/O).
fn images_of(content: &Option<MessageContent>) -> Vec<RequestImage> {
    match content {
        Some(MessageContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                crate::wire::ContentPart::ImageUrl { image_url } => parse_data_url(&image_url.url),
                crate::wire::ContentPart::Text { .. } => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The text of a message's content (empty when absent).
fn text_of(content: &Option<MessageContent>) -> String {
    content
        .as_ref()
        .map(MessageContent::to_text)
        .unwrap_or_default()
}

/// Map a wire request tool call to the engine's canonical [`CoreToolCall`].
fn core_tool_call(tc: &ToolCall) -> CoreToolCall {
    CoreToolCall {
        call_id: tc.id.clone().unwrap_or_default(),
        name: tc.function.name.clone(),
        args: tc.function.arguments.clone(),
    }
}

/// Map an OpenAI [`ChatCompletionRequest`] into a [`daemon_core::Request`].
///
/// - `system` messages are concatenated into [`Request::system`] (newline-joined).
/// - `user`/`assistant`/`tool` messages become [`RequestMsg`]s, preserving the native tool-call
///   linkage (`assistant.tool_calls` and `tool.tool_call_id`).
/// - `tools[]` become [`ToolDef`]s (the JSON-schema parameters serialized to a string).
/// - `temperature`/`top_p`/`top_k`/`max_tokens` become [`RequestParams`].
///
/// The bearer credential ([`Request::auth`]) is *not* set here — the backend threads the acquired
/// lease secret after resolving the provider.
pub fn request_to_core(req: &crate::wire::ChatCompletionRequest) -> Result<Request, GatewayError> {
    if req.messages.is_empty() {
        return Err(GatewayError::BadRequest(
            "messages must not be empty".into(),
        ));
    }
    let mut system = String::new();
    let mut messages: Vec<RequestMsg> = Vec::new();
    for m in &req.messages {
        match m.role.as_str() {
            "system" | "developer" => {
                let text = text_of(&m.content);
                if !text.is_empty() {
                    if !system.is_empty() {
                        system.push('\n');
                    }
                    system.push_str(&text);
                }
            }
            "assistant" => messages.push(RequestMsg {
                role: "assistant".into(),
                content: text_of(&m.content),
                tool_calls: m
                    .tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().map(core_tool_call).collect())
                    .unwrap_or_default(),
                tool_call_id: None,
                cache_breakpoint: false,
                images: Vec::new(),
            }),
            "tool" => messages.push(RequestMsg {
                role: "tool".into(),
                content: text_of(&m.content),
                tool_calls: Vec::new(),
                tool_call_id: m.tool_call_id.clone(),
                cache_breakpoint: false,
                images: Vec::new(),
            }),
            // Everything else (notably "user") is a user turn.
            _ => messages.push(RequestMsg {
                role: "user".into(),
                content: text_of(&m.content),
                tool_calls: Vec::new(),
                tool_call_id: None,
                cache_breakpoint: false,
                images: images_of(&m.content),
            }),
        }
    }

    let tools = req
        .tools
        .as_ref()
        .map(|tools| tools.iter().map(tool_def).collect())
        .unwrap_or_default();

    let params = RequestParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        max_tokens: req.max_tokens,
        seed: None,
    };

    Ok(Request {
        system,
        messages,
        tools,
        auth: None,
        constraint: None,
        cache_system: false,
        cache_ttl: Default::default(),
        params,
        task: None,
    })
}

/// Map an OpenAI function tool into a [`ToolDef`] (the JSON-schema parameters serialized to a
/// string, matching the engine's `ToolDef.schema` convention).
fn tool_def(tool: &OpenAiTool) -> ToolDef {
    let schema = tool
        .function
        .parameters
        .as_ref()
        .map(|p| p.to_string())
        .unwrap_or_else(|| "{}".to_string());
    ToolDef {
        name: tool.function.name.clone(),
        schema,
    }
}

/// Map a canonical [`CoreToolCall`] into the OpenAI wire tool-call shape.
fn wire_tool_call(index: usize, call: &CoreToolCall) -> ToolCall {
    ToolCall {
        index: Some(index),
        id: Some(call.call_id.clone()),
        tp: ToolKind::Function,
        function: FunctionCall {
            name: call.name.clone(),
            arguments: call.args.clone(),
        },
    }
}

/// OpenAI usage from the engine's [`UsageDelta`].
pub(crate) fn usage_of(usage: &UsageDelta) -> Usage {
    Usage {
        prompt_tokens: usage.input_tokens,
        completion_tokens: usage.output_tokens,
        total_tokens: usage.input_tokens + usage.output_tokens,
    }
}

/// The finish reason for a completed turn: `tool_calls` when the model emitted calls, else the
/// provider-reported reason, else `stop`.
pub(crate) fn finish_reason(out: &ModelOutput) -> String {
    if !out.tool_calls.is_empty() {
        return "tool_calls".to_string();
    }
    out.meta
        .as_ref()
        .and_then(|m| m.finish_reason.clone())
        .unwrap_or_else(|| "stop".to_string())
}

/// Build a non-streaming [`ChatCompletionResponse`] from an assembled [`ModelOutput`].
pub fn output_to_response(model: &str, out: &ModelOutput) -> ChatCompletionResponse {
    let tool_calls = if out.tool_calls.is_empty() {
        None
    } else {
        Some(
            out.tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| wire_tool_call(i, c))
                .collect(),
        )
    };
    ChatCompletionResponse {
        id: response_id(),
        object: "chat.completion".to_string(),
        created: now_secs(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: (!out.text.is_empty()).then(|| out.text.clone()),
                tool_calls,
                reasoning_content: out.reasoning.clone().filter(|r| !r.is_empty()),
            },
            finish_reason: Some(finish_reason(out)),
        }],
        usage: usage_of(&out.usage),
    }
}

/// The shared frame fields of a streaming chunk (id/created/model stay stable across a stream).
#[derive(Clone)]
pub(crate) struct ChunkCtx {
    /// The stable response id.
    pub id: String,
    /// The stable creation timestamp.
    pub created: u64,
    /// The model id echoed back.
    pub model: String,
}

impl ChunkCtx {
    /// A fresh chunk context for a stream (fixes id/created for its lifetime).
    pub(crate) fn new(model: &str) -> Self {
        Self {
            id: response_id(),
            created: now_secs(),
            model: model.to_string(),
        }
    }

    fn frame(&self, choice: ChunkChoice, usage: Option<Usage>) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![choice],
            usage,
        }
    }

    /// A content-delta chunk. `role` is `Some("assistant")` only on the first emitted chunk.
    pub(crate) fn content_chunk(
        &self,
        text: String,
        role: Option<&str>,
    ) -> ChatCompletionChunkResponse {
        self.frame(
            ChunkChoice {
                index: 0,
                delta: Delta {
                    role: role.map(str::to_string),
                    content: Some(text),
                    tool_calls: None,
                    reasoning_content: None,
                },
                finish_reason: None,
            },
            None,
        )
    }

    /// A reasoning-delta chunk.
    pub(crate) fn reasoning_chunk(
        &self,
        text: String,
        role: Option<&str>,
    ) -> ChatCompletionChunkResponse {
        self.frame(
            ChunkChoice {
                index: 0,
                delta: Delta {
                    role: role.map(str::to_string),
                    content: None,
                    tool_calls: None,
                    reasoning_content: Some(text),
                },
                finish_reason: None,
            },
            None,
        )
    }

    /// The terminal chunk: any tool calls + the finish reason + usage.
    pub(crate) fn final_chunk(
        &self,
        out: &ModelOutput,
        role: Option<&str>,
    ) -> ChatCompletionChunkResponse {
        let tool_calls = if out.tool_calls.is_empty() {
            None
        } else {
            Some(
                out.tool_calls
                    .iter()
                    .enumerate()
                    .map(|(i, c)| wire_tool_call(i, c))
                    .collect(),
            )
        };
        self.frame(
            ChunkChoice {
                index: 0,
                delta: Delta {
                    role: role.map(str::to_string),
                    content: None,
                    tool_calls,
                    reasoning_content: None,
                },
                finish_reason: Some(finish_reason(out)),
            },
            Some(usage_of(&out.usage)),
        )
    }
}

/// Build the `GET /v1/models` listing from the node catalog.
pub fn catalog_to_models(catalog: &[ModelDescriptor]) -> ModelObjects {
    let created = now_secs();
    ModelObjects {
        object: "list".to_string(),
        data: catalog
            .iter()
            .map(|m| ModelObject {
                id: m.id.clone(),
                object: "model".to_string(),
                created,
                owned_by: "daemon".to_string(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{ChatCompletionRequest, ContentPart, ImageUrl};
    use daemon_api::ProviderSelector;

    fn parse_req(json: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(json).expect("request decodes")
    }

    #[test]
    fn maps_roles_system_and_params() {
        let req = parse_req(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "developer", "content": "and precise"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}
            ],
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 40,
            "max_completion_tokens": 128
        }));
        let core = request_to_core(&req).expect("maps");
        // system + developer are folded into the system prompt, newline-joined.
        assert_eq!(core.system, "be terse\nand precise");
        assert_eq!(core.messages.len(), 2);
        assert_eq!(core.messages[0].role, "user");
        assert_eq!(core.messages[0].content, "hi");
        assert_eq!(core.messages[1].role, "assistant");
        // params round-trip (max_tokens via the alias).
        assert_eq!(core.params.temperature, Some(0.3));
        assert_eq!(core.params.top_p, Some(0.9));
        assert_eq!(core.params.top_k, Some(40));
        assert_eq!(core.params.max_tokens, Some(128));
    }

    #[test]
    fn maps_tools_and_tool_call_round_trip() {
        let req = parse_req(serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"NYC\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
            ],
            "tools": [
                {"type": "function", "function": {
                    "name": "get_weather",
                    "description": "look up weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
                }}
            ]
        }));
        let core = request_to_core(&req).expect("maps");
        // The tool def carries the name + serialized JSON-schema.
        assert_eq!(core.tools.len(), 1);
        assert_eq!(core.tools[0].name, "get_weather");
        assert!(core.tools[0].schema.contains("\"city\""));
        // The assistant tool-call linkage survives.
        let asst = core
            .messages
            .iter()
            .find(|m| m.role == "assistant")
            .unwrap();
        assert_eq!(asst.tool_calls.len(), 1);
        assert_eq!(asst.tool_calls[0].call_id, "call_1");
        assert_eq!(asst.tool_calls[0].name, "get_weather");
        assert_eq!(asst.tool_calls[0].args, "{\"city\":\"NYC\"}");
        // The tool result carries the call id.
        let tool = core.messages.iter().find(|m| m.role == "tool").unwrap();
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool.content, "sunny");
    }

    #[test]
    fn extracts_data_url_images_for_user_turn() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![crate::wire::Message {
                role: "user".into(),
                content: Some(MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "look".into(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "data:image/png;base64,AAAA".into(),
                            detail: None,
                        },
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/x.png".into(),
                            detail: None,
                        },
                    },
                ])),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };
        let core = request_to_core(&req).expect("maps");
        assert_eq!(core.messages[0].content, "look");
        // Only the data URL is resolved (no network I/O for http references).
        assert_eq!(core.messages[0].images.len(), 1);
        assert_eq!(core.messages[0].images[0].mime, "image/png");
        assert_eq!(core.messages[0].images[0].data_base64, "AAAA");
    }

    #[test]
    fn empty_messages_is_bad_request() {
        let req = ChatCompletionRequest {
            model: "m".into(),
            messages: vec![],
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stream: None,
            tools: None,
            tool_choice: None,
        };
        assert!(matches!(
            request_to_core(&req),
            Err(GatewayError::BadRequest(_))
        ));
    }

    #[test]
    fn output_to_response_carries_text_and_usage() {
        let out = ModelOutput {
            text: "the answer".into(),
            usage: UsageDelta {
                input_tokens: 10,
                output_tokens: 5,
                ..Default::default()
            },
            ..Default::default()
        };
        let resp = output_to_response("m", &out);
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.model, "m");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("the answer")
        );
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(resp.choices[0].message.tool_calls.is_none());
        assert_eq!(resp.usage.prompt_tokens, 10);
        assert_eq!(resp.usage.completion_tokens, 5);
        assert_eq!(resp.usage.total_tokens, 15);
    }

    #[test]
    fn output_to_response_round_trips_tool_calls() {
        let out = ModelOutput {
            text: String::new(),
            tool_calls: vec![CoreToolCall {
                call_id: "call_9".into(),
                name: "do_thing".into(),
                args: "{\"x\":1}".into(),
            }],
            ..Default::default()
        };
        let resp = output_to_response("m", &out);
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("tool_calls"));
        // No text content on a pure tool-call turn.
        assert!(resp.choices[0].message.content.is_none());
        let calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_deref(), Some("call_9"));
        assert_eq!(calls[0].function.name, "do_thing");
        assert_eq!(calls[0].function.arguments, "{\"x\":1}");
    }

    #[test]
    fn chunk_sequence_sends_role_once_and_final_usage() {
        let ctx = ChunkCtx::new("m");
        let first = ctx.content_chunk("Hel".into(), Some("assistant"));
        assert_eq!(first.object, "chat.completion.chunk");
        assert_eq!(first.choices[0].delta.role.as_deref(), Some("assistant"));
        assert_eq!(first.choices[0].delta.content.as_deref(), Some("Hel"));
        assert!(first.choices[0].finish_reason.is_none());
        assert!(first.usage.is_none());
        // A later content chunk carries no role.
        let second = ctx.content_chunk("lo".into(), None);
        assert!(second.choices[0].delta.role.is_none());
        // The id/created are stable across the stream.
        assert_eq!(first.id, second.id);
        // The terminal chunk carries the finish reason + usage.
        let out = ModelOutput {
            text: "Hello".into(),
            usage: UsageDelta {
                input_tokens: 3,
                output_tokens: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let last = ctx.final_chunk(&out, None);
        assert_eq!(last.choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = last.usage.expect("terminal chunk carries usage");
        assert_eq!(usage.total_tokens, 5);
    }

    #[test]
    fn catalog_maps_to_openai_model_list() {
        let catalog = vec![
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
        ];
        let listing = catalog_to_models(&catalog);
        assert_eq!(listing.object, "list");
        assert_eq!(listing.data.len(), 2);
        assert_eq!(listing.data[0].id, "gpt-4o");
        assert_eq!(listing.data[0].object, "model");
        assert_eq!(listing.data[0].owned_by, "daemon");
        assert_eq!(listing.data[1].id, "local-gguf");
    }
}
