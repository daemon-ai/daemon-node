//! Token counting — port of `token_counter.py`.
//!
//! Default heuristic `len/4`; exact `cl100k_base` counting remains a feature-level follow-up.

/// Estimate the token count of `text` (`token_counter.py` L20-L41).
pub fn estimate_tokens(text: &str) -> usize {
    text.len() / 4
}
