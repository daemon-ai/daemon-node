// SPDX-License-Identifier: MIT OR Apache-2.0
// SPDX-FileCopyrightText: 2026 Jarrad Hope

//! Embeddings — port of `embeddings.py`, over the `daemon-core` [`EmbeddingProvider`] seam.
//!
//! Mnemosyne no longer owns an embedding runtime. Instead the host injects an
//! [`EmbeddingProvider`] (remote `genai`, or a local `daemon-infer` worker — see `daemon-providers`),
//! and this thin wrapper adapts it to the call sites. With no provider the engine runs in
//! **keyword-only mode**: `embed`/`embed_query` return `None` and recall falls back to lexical
//! scoring (`embeddings.py` L206/L225 keyword-only branch).
//!
//! Embedding is async (real backends call out to a model). The synchronous BEAM [`Engine`] never
//! embeds inline: the async [`MnemosyneProvider`](crate::provider::MnemosyneProvider) hooks embed
//! here and pass the precomputed vectors into the engine's vector-aware methods.

use daemon_core::EmbeddingProvider;
use std::sync::Arc;

/// An embedding backend handle: an optional injected [`EmbeddingProvider`].
#[derive(Clone, Default)]
pub struct Embedder {
    provider: Option<Arc<dyn EmbeddingProvider>>,
}

impl Embedder {
    /// A keyword-only embedder (no provider).
    pub fn new() -> Self {
        Self::default()
    }

    /// An embedder backed by an injected provider.
    pub fn with_provider(provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self {
            provider: Some(provider),
        }
    }

    /// Whether real embeddings are available (false in keyword-only mode).
    pub fn available(&self) -> bool {
        self.provider.is_some()
    }

    /// The backing model identifier (persisted alongside stored vectors), if any.
    pub fn model(&self) -> Option<&str> {
        self.provider.as_ref().map(|p| p.model())
    }

    /// Embed a single query string. `None` in keyword-only mode or on backend error
    /// (`embeddings.py` L206).
    pub async fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        let provider = self.provider.as_ref()?;
        let owned = vec![text.to_string()];
        provider.embed(&owned).await.ok()?.into_iter().next()
    }

    /// Embed a batch of texts (`embeddings.py` L225). `None` in keyword-only mode or on error.
    pub async fn embed(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        let provider = self.provider.as_ref()?;
        provider.embed(texts).await.ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daemon_core::MockEmbedder;

    #[tokio::test]
    async fn keyword_only_returns_none() {
        let e = Embedder::new();
        assert!(!e.available());
        assert!(e.embed_query("hello").await.is_none());
        assert!(e.embed(&["hello".to_string()]).await.is_none());
    }

    #[tokio::test]
    async fn injected_provider_embeds() {
        let e = Embedder::with_provider(Arc::new(MockEmbedder::new(16)));
        assert!(e.available());
        let v = e.embed_query("hello world").await.expect("vector");
        assert_eq!(v.len(), 16);
    }
}
