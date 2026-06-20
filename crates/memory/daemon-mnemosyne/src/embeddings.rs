//! Embeddings — port of `embeddings.py`.
//!
//! Default model `BAAI/bge-small-en-v1.5` (384-dim) via the `embeddings` feature (fastembed/ONNX).
//! When the feature is off the engine runs in **keyword-only mode** (`MNEMOSYNE_NO_EMBEDDINGS`):
//! `embed_query`/`embed` return `None` and recall falls back to FTS5 + lexical scoring.
//!
//! Scaffold: the keyword-only path is implemented; the fastembed backend is a feature-gated TODO.

/// An embedding backend (keyword-only unless the `embeddings` feature is enabled).
#[derive(Default)]
pub struct Embedder {
    #[cfg(feature = "embeddings")]
    model: std::sync::Mutex<Option<fastembed::TextEmbedding>>,
}

impl Embedder {
    /// Construct an embedder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether real embeddings are available (false in keyword-only mode).
    pub fn available(&self) -> bool {
        cfg!(feature = "embeddings") && std::env::var("MNEMOSYNE_NO_EMBEDDINGS").is_err()
    }

    /// Embed a single query string. `None` in keyword-only mode (`embeddings.py` L206).
    pub fn embed_query(&self, _text: &str) -> Option<Vec<f32>> {
        // TODO(embeddings feature): lazily init fastembed TextEmbedding(BGESmallENV15) and embed.
        None
    }

    /// Embed a batch of texts (`embeddings.py` L225).
    pub fn embed(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        let _ = texts;
        None
    }
}
