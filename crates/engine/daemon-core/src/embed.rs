//! The embedding provider port — the backend-agnostic seam memory/recall use for vector similarity.
//!
//! This is a sibling of the [`Provider`](crate::Provider) port, kept separate because an embedding
//! model is a *distinct* model/capability (usually a different model entirely): a chat provider and
//! an embedder are decoupled and selected independently. Real implementations (remote `genai`, a
//! supervised local `daemon-infer` worker) live in the `daemon-providers` crate; the deterministic
//! [`MockEmbedder`] here keeps `daemon-core` and downstream tests network-free.

use crate::provider::Failure;
use std::collections::HashMap;

/// A text-embedding backend.
///
/// Embedders are async — real backends call out to a model (a networked API or a local worker
/// process). Synchronous stores (e.g. Mnemosyne's SQLite engine) therefore compute vectors at an
/// async seam and pass the precomputed vectors down, never blocking inside the store.
#[async_trait::async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// The embedding dimensionality (for store/index validation and zero-vector fallbacks).
    fn dimensions(&self) -> usize;
    /// The model identifier (persisted alongside stored vectors so a dimension/model change is
    /// detectable).
    fn model(&self) -> &str;
    /// Embed a batch of texts, returning one vector per input in the same order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Failure>;
}

/// Cosine similarity of two vectors in `[-1, 1]` (`0.0` if either is empty, length-mismatched, or
/// degenerate). The shared helper recall paths use so similarity is computed one way everywhere.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// A deterministic, network-free [`EmbeddingProvider`] for tests and the substrate.
///
/// By default it embeds via a stable bag-of-words hashing trick (token -> signed bucket), then
/// L2-normalizes — so texts that share tokens are similar and disjoint texts are near-orthogonal.
/// For tests that need an exact "semantic" relationship that token overlap can't express (a query
/// and a memory with *no* shared tokens that should still match), [`MockEmbedder::scripted`] pins
/// chosen strings to fixed vectors, falling back to hashing for anything unlisted.
pub struct MockEmbedder {
    dims: usize,
    model: String,
    scripted: HashMap<String, Vec<f32>>,
}

impl MockEmbedder {
    /// A hashing embedder producing `dims`-dimensional unit vectors.
    pub fn new(dims: usize) -> Self {
        Self {
            dims: dims.max(1),
            model: "mock-embed".to_string(),
            scripted: HashMap::new(),
        }
    }

    /// A hashing embedder that returns fixed vectors for the given `(text, vector)` pairs and hashes
    /// everything else. Pinned vectors must have length `dims`.
    pub fn scripted(dims: usize, pairs: impl IntoIterator<Item = (String, Vec<f32>)>) -> Self {
        let mut me = Self::new(dims);
        me.scripted = pairs.into_iter().collect();
        me
    }

    /// The stable bag-of-words hash embedding for `text`.
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for token in text.split_whitespace() {
            let lower = token.to_ascii_lowercase();
            let h = fnv1a(lower.as_bytes());
            let idx = (h % self.dims as u64) as usize;
            let sign = if (h >> 63) & 1 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        l2_normalize(&mut v);
        v
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for MockEmbedder {
    fn dimensions(&self) -> usize {
        self.dims
    }

    fn model(&self) -> &str {
        &self.model
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Failure> {
        Ok(texts
            .iter()
            .map(|t| {
                self.scripted
                    .get(t)
                    .cloned()
                    .unwrap_or_else(|| self.hash_embed(t))
            })
            .collect())
    }
}

/// FNV-1a (64-bit) — a small, stable, dependency-free hash for the mock embedder's hashing trick.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// L2-normalize in place (a no-op for the zero vector, avoiding NaNs).
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hashing_embedder_is_deterministic_and_normalized() {
        let e = MockEmbedder::new(32);
        let a = e.embed(&["hello world".to_string()]).await.unwrap();
        let b = e.embed(&["hello world".to_string()]).await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn shared_tokens_are_more_similar_than_disjoint() {
        let e = MockEmbedder::new(64);
        let v = e
            .embed(&[
                "authentication flow jwt".to_string(),
                "authentication flow tokens".to_string(),
                "the weather is sunny today".to_string(),
            ])
            .await
            .unwrap();
        let shared = cosine(&v[0], &v[1]);
        let disjoint = cosine(&v[0], &v[2]);
        assert!(shared > disjoint, "shared={shared} disjoint={disjoint}");
    }

    #[tokio::test]
    async fn scripted_pins_chosen_vectors() {
        let e = MockEmbedder::scripted(
            3,
            [
                ("q".to_string(), vec![1.0, 0.0, 0.0]),
                ("a".to_string(), vec![1.0, 0.0, 0.0]),
                ("b".to_string(), vec![0.0, 1.0, 0.0]),
            ],
        );
        let v = e
            .embed(&["q".to_string(), "a".to_string(), "b".to_string()])
            .await
            .unwrap();
        assert!((cosine(&v[0], &v[1]) - 1.0).abs() < 1e-6);
        assert!(cosine(&v[0], &v[2]).abs() < 1e-6);
    }
}
