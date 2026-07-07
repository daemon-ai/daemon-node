// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! The [`GatewayBackend`] injection seam: the gateway crate carries the wire + routes, the binary
//! implements *how* a completion is served (provider resolution + credential brokering). Mirrors
//! how `daemon-node` injects `AgentDiscovery` — the adapter stays decoupled from the binary.

use async_trait::async_trait;
use daemon_api::ModelDescriptor;
use daemon_core::{ModelOutput, Request, StreamEvent};
use futures::stream::BoxStream;

/// A gateway request failure, mapped to an HTTP status by the server layer.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    /// The requested `model` is not in the node catalog (or is outside the allowlist) — `404`.
    #[error("unknown model: {0}")]
    UnknownModel(String),
    /// The request was malformed (e.g. no messages) — `400`.
    #[error("bad request: {0}")]
    BadRequest(String),
    /// A credential could not be acquired for the resolved provider — `502`.
    #[error("credential: {0}")]
    Credential(String),
    /// The underlying provider call failed — `502`.
    #[error("provider: {0}")]
    Provider(String),
}

/// A streamed model response: [`StreamEvent`]s terminating with [`StreamEvent::Done`], owned
/// (`'static`) so the backend can drive the provider on its own task.
pub type EventStream = BoxStream<'static, Result<StreamEvent, GatewayError>>;

/// The outcome of [`GatewayBackend::complete`]: a single assembled output (non-streaming) or a live
/// event stream (streaming).
pub enum Completion {
    /// A non-streaming completion: the assembled [`ModelOutput`].
    Once(ModelOutput),
    /// A streaming completion: the provider's [`StreamEvent`] stream.
    Stream(EventStream),
}

/// The node-side backing of the OpenAI-compatible surface. Implemented by the binary (which has the
/// provider resolver, credential broker, and catalog in scope); the gateway crate never depends on
/// the binary or on `NodeApi`.
#[async_trait]
pub trait GatewayBackend: Send + Sync {
    /// The discoverable model catalog, rendered by `GET /v1/models`.
    async fn catalog(&self) -> Vec<ModelDescriptor>;

    /// Run a completion for `model` from the mapped `req`. `stream` mirrors the request's `stream`
    /// flag; the backend returns the matching [`Completion`] variant.
    async fn complete(
        &self,
        model: &str,
        req: Request,
        stream: bool,
    ) -> Result<Completion, GatewayError>;
}
