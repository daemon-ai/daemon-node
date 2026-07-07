// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope
//
// The OpenAI-compatible request/response/chunk shapes below are a slimmed port of the MIT-licensed
// mistral.rs project (request structs from `mistralrs-server-core/src/openai.rs`; response + chunk
// structs from `mistralrs-core/src/response.rs`). The mistral.rs-only fields (grammar, dry_*,
// session_id, agentic/web-search/shell tools, logprobs, image/audio) are intentionally dropped.
//
//   MIT License — Copyright (c) 2024 Eric Buehler (https://github.com/EricLBuehler/mistral.rs)
//
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
//   and associated documentation files (the "Software"), to deal in the Software without
//   restriction, including without limitation the rights to use, copy, modify, merge, publish,
//   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the standard MIT permission + warranty notice.

//! The OpenAI wire contract the gateway speaks: a slim, standard-OpenAI subset of the
//! `chat/completions` + `models` shapes (no mistral.rs extensions). These are pure serde types with
//! no dependency on the node's internal engine types — [`crate::mapping`] converts between them and
//! [`daemon_core`].

use serde::{Deserialize, Serialize};

/// The tool-call type discriminator. OpenAI currently only defines `function`; kept as an enum so
/// the `"type": "function"` tag round-trips and an unknown type is a hard deserialize error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// A function tool call (the only OpenAI tool-call kind).
    #[default]
    Function,
}

/// A function call's name + JSON-string arguments (the inner payload of a request/response tool
/// call). `arguments` is a JSON *string* per the OpenAI wire (not a parsed object).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct FunctionCall {
    /// The function name to invoke.
    pub name: String,
    /// The function arguments, as a JSON-encoded string (`"parameters"` accepted as an alias).
    #[serde(alias = "parameters", default)]
    pub arguments: String,
}

/// A tool call the assistant emitted (request-side, on an `assistant` message) or the model
/// produced (response/chunk-side). `id` correlates a call with its later `tool` result message.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ToolCall {
    /// A streaming ordinal (response/chunk side); ignored on request decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
    /// The unique id for this tool call (correlates the later `tool` result message).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The tool-call type (`"function"`).
    #[serde(rename = "type", default)]
    pub tp: ToolKind,
    /// The function call details (name + JSON-string arguments).
    pub function: FunctionCall,
}

/// One part of a structured (multimodal) message content array.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text part.
    Text {
        /// The text.
        text: String,
    },
    /// An image part (a URL or a `data:` URI).
    ImageUrl {
        /// The image reference.
        image_url: ImageUrl,
    },
}

/// An `image_url` content part payload.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct ImageUrl {
    /// The image URL — an `http(s)://…` reference or a `data:<mime>;base64,<data>` URI.
    pub url: String,
    /// The optional detail hint (`auto`/`low`/`high`); accepted and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A message's content: either a plain string or an array of structured parts (multimodal).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Structured multimodal parts.
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// The flattened text of this content (parts are joined by spaces; images contribute nothing).
    pub fn to_text(&self) -> String {
        match self {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// One message in the conversation (`system`/`user`/`assistant`/`tool`).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct Message {
    /// The role of the sender.
    pub role: String,
    /// The message content (absent for a pure tool-call assistant message).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// An optional participant name; accepted and ignored.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Tool calls this assistant message emitted (native round-trip).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For a `tool` message: the id of the call this result answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A function tool the model may call (the `chat/completions` `tools[]` shape).
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct OpenAiTool {
    /// The tool type (`"function"`).
    #[serde(rename = "type", default)]
    pub tp: ToolKind,
    /// The function definition (name + description + JSON-schema parameters).
    pub function: OpenAiFunction,
}

/// A function definition inside an [`OpenAiTool`].
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
pub struct OpenAiFunction {
    /// The function name.
    pub name: String,
    /// A human description of what the function does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON-Schema for the function's parameters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
    /// Whether to enforce strict schema adherence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A chat-completion request (a slim, standard-OpenAI subset).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    /// The conversation so far.
    pub messages: Vec<Message>,
    /// The model id (a node catalog model id).
    #[serde(default)]
    pub model: String,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus (top-p) cutoff.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Top-k cutoff (a common OpenAI-compatible extension; ignored by providers that lack it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The output-token cap (`max_completion_tokens` accepted as an alias).
    #[serde(
        default,
        alias = "max_completion_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub max_tokens: Option<u32>,
    /// Whether to stream the response as server-sent events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// The tools the model may call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    /// Controls which (if any) tool the model must call. Accepted and ignored (the node's own tool
    /// policy governs), kept so a client sending it does not get a decode error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

/// OpenAI-compatible usage accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Usage {
    /// Prompt/input tokens.
    pub prompt_tokens: u64,
    /// Completion/output tokens.
    pub completion_tokens: u64,
    /// The sum of prompt + completion tokens.
    pub total_tokens: u64,
}

/// The assistant message on a non-streaming completion choice.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ResponseMessage {
    /// The role (always `assistant`).
    pub role: String,
    /// The assistant text (absent for a pure tool-call turn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Any tool calls the model emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Separate reasoning/analysis content (Harmony/DeepSeek style), when the model produced it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// One non-streaming completion choice.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct Choice {
    /// The choice index (always 0 — the gateway returns a single choice).
    pub index: usize,
    /// The assistant message.
    pub message: ResponseMessage,
    /// The stop reason (`stop`/`tool_calls`/…), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A non-streaming `chat.completion` response.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    /// A response id.
    pub id: String,
    /// The object type (`"chat.completion"`).
    pub object: String,
    /// The creation time (unix seconds).
    pub created: u64,
    /// The response model id.
    pub model: String,
    /// The completion choices (always a single choice).
    pub choices: Vec<Choice>,
    /// Usage accounting.
    pub usage: Usage,
}

/// The incremental delta on a streaming chunk choice.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Delta {
    /// The role, sent once on the first chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The incremental assistant text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The incremental tool calls (sent on the terminal chunk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// The incremental reasoning content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

/// One streaming chunk choice.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChunkChoice {
    /// The choice index (always 0).
    pub index: usize,
    /// The incremental delta.
    pub delta: Delta,
    /// The stop reason, set only on the terminal chunk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A streaming `chat.completion.chunk` frame.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ChatCompletionChunkResponse {
    /// A response id (stable across a stream).
    pub id: String,
    /// The object type (`"chat.completion.chunk"`).
    pub object: String,
    /// The creation time (unix seconds).
    pub created: u64,
    /// The response model id.
    pub model: String,
    /// The chunk choices (always a single choice).
    pub choices: Vec<ChunkChoice>,
    /// Usage, present on the terminal chunk when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One entry in the `GET /v1/models` listing.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelObject {
    /// The model id.
    pub id: String,
    /// The object type (`"model"`).
    pub object: String,
    /// The creation time (unix seconds); a stable constant for a node catalog entry.
    pub created: u64,
    /// The owner label (`"daemon"`).
    pub owned_by: String,
}

/// The `GET /v1/models` response envelope.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct ModelObjects {
    /// The object type (`"list"`).
    pub object: String,
    /// The model entries.
    pub data: Vec<ModelObject>,
}
