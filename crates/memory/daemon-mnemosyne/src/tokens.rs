//! Token counting — port of `token_counter.py`.
//!
//! Default heuristic `len/4`; the `tiktoken` feature swaps in exact `cl100k_base` counts.

/// Estimate the token count of `text` (`token_counter.py` L20-L41).
pub fn estimate_tokens(text: &str) -> usize {
    // TODO(tiktoken feature): use tiktoken-rs cl100k_base for exact counts.
    text.len() / 4
}
