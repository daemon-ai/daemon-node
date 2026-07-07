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

/// The caller resolved from a presented bearer token by [`GatewayBackend::authorize`], threaded
/// into [`GatewayBackend::complete`] so the backend enforces the token's binding.
///
/// The gateway is both a resident surface for external OpenAI clients (the `Admin` token) and the
/// per-session routing target for node-managed foreign agents (`Session` tokens). This enum lets the
/// server stay agnostic of *how* the backend resolves a token while still carrying the discriminant
/// completion needs: an `Admin` caller picks the model per request; a `Session` caller's model +
/// provider + credential are pinned node-side to the token's binding (the request model is ignored),
/// so a foreign agent's loopback token can only ever invoke its bound triple.
#[derive(Clone, Debug)]
pub enum GatewayPrincipal {
    /// The admin/global bearer (an external OpenAI client, or the node's boot-minted token): the
    /// request's `model` is honored and the provider/credential are resolved node-side.
    Admin,
    /// A per-session bearer bound node-side to a fixed provider+model+credential. The opaque token
    /// lets the backend re-resolve the binding; the request's model is ignored.
    Session(String),
}

/// The node-side backing of the OpenAI-compatible surface. Implemented by the binary (which has the
/// provider resolver, credential broker, catalog, and per-session token registry in scope); the
/// gateway crate never depends on the binary or on `NodeApi`.
#[async_trait]
pub trait GatewayBackend: Send + Sync {
    /// The discoverable model catalog, rendered by `GET /v1/models`.
    async fn catalog(&self) -> Vec<ModelDescriptor>;

    /// Resolve a presented bearer token to its caller. `None` rejects the request (`401`); the
    /// server never compares tokens itself — the backend owns the admin token and the per-session
    /// token registry.
    async fn authorize(&self, token: &str) -> Option<GatewayPrincipal>;

    /// Run a completion for the resolved `principal`. `model` is the request's model (honored for
    /// an `Admin` caller; ignored for a `Session` caller, whose model is pinned by its binding).
    /// `stream` mirrors the request's `stream` flag; the backend returns the matching [`Completion`].
    async fn complete(
        &self,
        principal: &GatewayPrincipal,
        model: &str,
        req: Request,
        stream: bool,
    ) -> Result<Completion, GatewayError>;
}
