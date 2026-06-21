//! The engine-agnostic [`InferenceBackend`] seam every local engine implements.
//!
//! llama.cpp (`llama-cpp-4`) is synchronous and `!Send`, so its impl owns the model/context on a
//! dedicated OS thread and bridges to this async trait via channels; mistral.rs is already async.
//! Both normalize to the same contract: stream [`BackendChunk`]s and return the final [`Usage`], or
//! a classified [`BackendError`]. The worker frames either onto the [`crate::protocol`] wire, so the
//! daemon never sees engine specifics.

use crate::protocol::{
    Capabilities, ErrorClass, Msg, Sampling, ToolCall, ToolCallFormat, ToolDef, Usage,
};
use tokio::sync::mpsc::UnboundedSender;
use tokio_util::sync::CancellationToken;

/// An engine-agnostic generation request (the worker's decode of [`crate::protocol::Command::Generate`]).
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    /// Correlates emitted chunks/events back to the originating request.
    pub request_id: u64,
    /// The system prompt.
    pub system: String,
    /// The flattened conversation.
    pub messages: Vec<Msg>,
    /// The tools offered this turn.
    pub tools: Vec<ToolDef>,
    /// Sampling parameters.
    pub sampling: Sampling,
    /// The output-token cap (`0` = backend default).
    pub max_tokens: u32,
}

/// One incremental output of a generation. The terminal `Done`/usage is the [`InferenceBackend::generate`]
/// return value (not a chunk); a failure is its `Err`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendChunk {
    /// Incremental assistant text.
    Text(String),
    /// Incremental reasoning text.
    Reasoning(String),
    /// A decoded tool call.
    Tool(ToolCall),
}

/// A classified backend failure (carries the wire [`ErrorClass`] so the worker forwards it verbatim).
#[derive(Clone, Debug, thiserror::Error)]
#[error("{class:?}: {message}")]
pub struct BackendError {
    /// The failure class (maps to the daemon's `Failure`).
    pub class: ErrorClass,
    /// A short human-readable detail.
    pub message: String,
}

impl BackendError {
    /// An unrecoverable failure (no backend, unloadable model, internal bug).
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::Fatal,
            message: message.into(),
        }
    }

    /// The prompt exceeded the context window.
    pub fn context_overflow(message: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::ContextOverflow,
            message: message.into(),
        }
    }

    /// A GPU/host allocator OOM.
    pub fn out_of_memory(message: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::OutOfMemory,
            message: message.into(),
        }
    }

    /// A transient/internal generation or decode error.
    pub fn transient(message: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::Transient,
            message: message.into(),
        }
    }

    /// Generation was cancelled cooperatively.
    pub fn cancelled() -> Self {
        Self {
            class: ErrorClass::Cancelled,
            message: "cancelled".into(),
        }
    }
}

/// The shared seam: a loaded local model that streams a generation.
///
/// `generate` sends [`BackendChunk`]s to `tx` as they are produced and returns the final [`Usage`]
/// on success, or a classified [`BackendError`]. It must observe `cancel` and stop promptly (mapping
/// to [`BackendError::cancelled`]).
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync {
    /// The loaded model's declared capabilities.
    fn capabilities(&self) -> Capabilities;

    /// Drive one generation to completion.
    async fn generate(
        &self,
        req: GenerateRequest,
        tx: UnboundedSender<BackendChunk>,
        cancel: CancellationToken,
    ) -> Result<Usage, BackendError>;

    /// Embed a batch of texts, returning one vector per input (same order).
    ///
    /// The default rejects embedding: only a backend loaded in embedding mode
    /// ([`crate::protocol::ModelParams::embeddings`]) overrides this. A generation-mode backend
    /// reports [`ErrorClass::Fatal`] so the daemon surfaces a clear "this model can't embed".
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, BackendError> {
        let _ = texts;
        Err(BackendError::fatal(
            "this backend was not loaded for embeddings (load with ModelParams.embeddings = true)",
        ))
    }
}

/// The fallback backend compiled when no engine feature is enabled (the default workspace gate).
///
/// It loads trivially but refuses to generate, reporting [`ErrorClass::Fatal`] "no backend" — so a
/// default-built worker is a clean, inert stub rather than a build that drags in cmake/llama.cpp.
pub struct StubBackend;

#[async_trait::async_trait]
impl InferenceBackend for StubBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            supports_native_tools: false,
            supports_streaming: false,
            tool_call_format: ToolCallFormat::Native,
            max_context: None,
        }
    }

    async fn generate(
        &self,
        _req: GenerateRequest,
        _tx: UnboundedSender<BackendChunk>,
        _cancel: CancellationToken,
    ) -> Result<Usage, BackendError> {
        Err(BackendError::fatal(
            "no inference backend compiled (build daemon-infer with --features llama or --features mistralrs)",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn stub_backend_reports_no_backend() {
        let backend = StubBackend;
        assert!(!backend.capabilities().supports_streaming);
        let (tx, _rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            request_id: 1,
            system: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            sampling: Sampling::default(),
            max_tokens: 0,
        };
        let err = backend
            .generate(req, tx, CancellationToken::new())
            .await
            .expect_err("stub refuses to generate");
        assert_eq!(err.class, ErrorClass::Fatal);
    }
}
