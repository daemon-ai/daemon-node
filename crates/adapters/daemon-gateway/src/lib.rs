// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! `daemon-gateway` — the optional node-owned OpenAI-compatible HTTP gateway.
//!
//! It exposes `POST /v1/chat/completions` (stream + non-stream) and `GET /v1/models` backed by the
//! node's existing provider stack, so a foreign OpenAI-wire agent (codex/opencode) can be pointed
//! at a node-configured provider — local or cloud — without ever holding a real API key. The agent
//! holds only a loopback bearer token; the real provider credential is resolved node-side.
//!
//! Layering: this crate carries the OpenAI [`wire`] types (a slim port of the MIT-licensed
//! mistral.rs shapes), the wire<->engine [`mapping`], the axum [`server`] (routes + SSE + bearer
//! auth), and the [`GatewayBackend`] injection trait. The binary implements the backend (provider
//! resolution + credential brokering). The crate depends on `daemon-core`/`daemon-api`/
//! `daemon-common` only — never on the binary or `NodeApi` — so axum and the OpenAI wire types stay
//! out of the engine, the host, and the wire contract.

#![forbid(unsafe_code)]

pub mod backend;
pub mod mapping;
pub mod server;
pub mod wire;

pub use backend::{Completion, EventStream, GatewayBackend, GatewayError};
pub use mapping::{catalog_to_models, output_to_response, request_to_core};
pub use server::{router, serve};
pub use wire::{
    ChatCompletionChunkResponse, ChatCompletionRequest, ChatCompletionResponse, Choice,
    ChunkChoice, Delta, FunctionCall, Message, MessageContent, ModelObject, ModelObjects,
    OpenAiFunction, OpenAiTool, ResponseMessage, ToolCall, ToolKind, Usage,
};
